# -*- coding: utf-8 -*-
# model_classifier_ablation.py

import torch
import torch.nn as nn

# ================= Snake Activation =================
class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        alpha = self.alpha
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(alpha * x), 2)

# ================= 1D SE-Block (通道注意力) =================
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)  # 全局平均池化
        y = self.fc(y).view(b, c, 1)
        return x * y

# ================= Residual Block (带消融开关) =================
class PeriodicResBlock(nn.Module):
    def __init__(self, dim, kernel_size=15, alpha=1.0, use_snake=True, use_se=True):
        super().__init__()
        pad = kernel_size // 2

        # ?? 消融开关 1 & 2：根据参数决定是否使用 Snake 和 SE 模块
        activation_layer = Snake(dim, alpha) if use_snake else nn.ReLU()
        se_layer = SEBlock1D(dim) if use_se else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(dim),
            activation_layer,
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(dim),
            se_layer  # 如果 use_se=False，这里就是一个无操作的 Identity 层
        )
        self.act = Snake(dim, alpha) if use_snake else nn.ReLU()

    def forward(self, x):
        return self.act(x + self.conv(x))

# ================= 完整网络 (Ablation 版本) =================
class BinaryPeriodicResNet1D_Ablation(nn.Module):
    def __init__(self, d_model=256, width_scale=4.0, use_snake=True, use_se=True, use_physics_fusion=True):
        super().__init__()
        self.use_physics_fusion = use_physics_fusion
        base = int(32 * width_scale)

        # ?? 贯穿全局的 Snake 消融：根据 use_snake 动态选择激活函数
        def get_act(dim):
            return Snake(dim) if use_snake else nn.ReLU()

        # Stem 层
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base),
            get_act(base),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            PeriodicResBlock(base, kernel_size=15, use_snake=use_snake, use_se=use_se),
            PeriodicResBlock(base, kernel_size=15, use_snake=use_snake, use_se=use_se),
            nn.Conv1d(base, base * 2, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(base * 2),
            get_act(base * 2),
            PeriodicResBlock(base * 2, kernel_size=11, use_snake=use_snake, use_se=use_se)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            PeriodicResBlock(base * 2, kernel_size=11, use_snake=use_snake, use_se=use_se),
            PeriodicResBlock(base * 2, kernel_size=11, use_snake=use_snake, use_se=use_se),
            nn.Conv1d(base * 2, base * 4, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(base * 4),
            get_act(base * 4),
            PeriodicResBlock(base * 4, kernel_size=9, use_snake=use_snake, use_se=use_se)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv1d(base * 4, base * 8, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(base * 8),
            get_act(base * 8),
            PeriodicResBlock(base * 8, kernel_size=7, use_snake=use_snake, use_se=use_se)
        )

        # 多尺度池化 (复刻您的原逻辑)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Head 层
        final_ch = base * 8 * 2      # avg 和 max 拼接后维度翻倍
        combined_ch = final_ch * 2   # 无论是 [f1*f2, abs(f1-f2)] 还是 [f1, f2]，拼接后总通道数都是 final_ch * 2

        self.head = nn.Sequential(
            nn.Linear(combined_ch, d_model),
            get_act(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x shape: [B, 2, L]
        x1 = x[:, 0:1, :]  # L1 信号 [B, 1, L]
        x2 = x[:, 1:2, :]  # L2 信号 [B, 1, L]

        # 提取特征
        out1 = self.stage3(self.stage2(self.stage1(self.stem(x1))))
        out2 = self.stage3(self.stage2(self.stage1(self.stem(x2))))

        # 降维：分别提取 Average 和 Max，再拼接
        f1 = torch.cat([self.avg_pool(out1).squeeze(-1), self.max_pool(out1).squeeze(-1)], dim=1)
        f2 = torch.cat([self.avg_pool(out2).squeeze(-1), self.max_pool(out2).squeeze(-1)], dim=1)

        # ?? 消融开关 3：融合策略
        if self.use_physics_fusion:
            # 物理融合：Hadamard 积 + 绝对差 (捕捉振幅一致性与相位干涉)
            combined = torch.cat([f1 * f2, torch.abs(f1 - f2)], dim=1)
        else:
            # 常规融合：直接暴力拼接 Concatenate
            combined = torch.cat([f1, f2], dim=1)

        # 输出匹配概率 logits
        return self.head(combined)