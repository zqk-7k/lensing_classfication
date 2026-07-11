# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 5. Periodic ResNet (Snake Activation) - 针对引力波优化的 CNN
# ==============================================================================

class Snake(nn.Module):
    """
    Snake Activation Function: f(x) = x + (1/alpha) * sin^2(alpha * x)
    相比纯 Sin (SIREN)，它更稳定，且能捕捉频率特征。
    alpha 是可学习的参数，让网络自己决定关注什么频率。
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        alpha = self.alpha
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(alpha * x), 2)


class PeriodicResBlock(nn.Module):
    def __init__(self, dim, kernel_size=15, alpha=1.0):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.BatchNorm1d(dim)
        self.act1 = Snake(dim, alpha)  # 使用 Snake 替代 ReLU

        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.BatchNorm1d(dim)
        self.act2 = Snake(dim, alpha)

    def forward(self, x):
        res = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        # 残差连接前再过一次激活，或者直接相加 (这里选择 ResNet 经典结构: Add then Act)
        x = x + res
        x = self.act2(x)
        return x


class PeriodicResNet1D(nn.Module):
    def __init__(self, d_model=128, emb_dim=128, in_channels=1, width_scale=1.0):
        super().__init__()

        # 基础宽度
        base = int(32 * width_scale)
        print(f"[Model] Initializing PeriodicResNet1D (Snake Act) | Scale={width_scale} | In={in_channels}")

        # 1. Stem (下采样层)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base),
            Snake(base),  # 捕捉初始频率
            nn.MaxPool1d(2)  # 4x Downsample
        )

        # 2. Stages (使用 ResNet 结构加深网络)
        self.stage1 = nn.Sequential(
            nn.Conv1d(base, base * 2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base * 2),
            Snake(base * 2),
            PeriodicResBlock(base * 2, kernel_size=11)
        )

        self.stage2 = nn.Sequential(
            nn.Conv1d(base * 2, base * 4, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(base * 4),
            Snake(base * 4),
            PeriodicResBlock(base * 4, kernel_size=9)
        )

        self.stage3 = nn.Sequential(
            nn.Conv1d(base * 4, base * 8, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(base * 8),
            Snake(base * 8),
            PeriodicResBlock(base * 8, kernel_size=7)
        )

        # 3. Head
        final_ch = base * 8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_ch, d_model),
            Snake(d_model),  # 在全连接层也使用周期激活
            nn.Dropout(0.1),
            nn.Linear(d_model, emb_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        z = self.head(x)
        return F.normalize(z, dim=-1)

class ConvBlock1D(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        p = k//2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm1d(c_out)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Encoder1D(nn.Module):
    # 1. 在 __init__ 参数中加入 in_channels=1
    def __init__(self, d_model=128, emb_dim=128, in_channels=1):
        super().__init__()
        self.feat = nn.Sequential(
            # 2. 将原来硬编码的 1 改为变量 in_channels
            ConvBlock1D(in_channels, 32, k=15, s=4), nn.MaxPool1d(2),

            ConvBlock1D(32, 64, k=15, s=4), nn.MaxPool1d(2),
            ConvBlock1D(64, 128, k=11, s=2), nn.MaxPool1d(2),
            ConvBlock1D(128, 256, k=9, s=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, d_model), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x):
        h = self.feat(x)
        z = self.head(h)
        z = F.normalize(z, dim=-1)
        return z


# model.py

# class Encoder1D(nn.Module):
#     def __init__(self, d_model=128, emb_dim=128, width_scale=1.0):
#         """
#         参数:
#             d_model: 倒数第二层全连接的维度
#             emb_dim: 最终输出 Embedding 的维度
#             width_scale: 通道数倍率 (1.0 为原始大小)
#         """
#         super().__init__()
#
#         # 基础通道数 (原始第一层是 32)
#         # int() 确保通道数是整数
#         base = int(32 * width_scale)
#
#         print(f"[Model] Initializing Encoder1D with Width Scale {width_scale:.1f} (Base ch={base})")
#
#         self.feat = nn.Sequential(
#             # Layer 1: 1 -> base
#             ConvBlock1D(1, base, k=15, s=4),
#             nn.MaxPool1d(2),
#
#             # Layer 2: base -> base*2
#             ConvBlock1D(base, base * 2, k=15, s=4),
#             nn.MaxPool1d(2),
#
#             # Layer 3: base*2 -> base*4
#             ConvBlock1D(base * 2, base * 4, k=11, s=2),
#             nn.MaxPool1d(2),
#
#             # Layer 4: base*4 -> base*8
#             # 原始: 32 -> 64 -> 128 -> 256
#             # 2倍: 64 -> 128 -> 256 -> 512
#             ConvBlock1D(base * 4, base * 8, k=9, s=2),
#         )
#
#         # 计算 Flatten 后的通道数 (Global Average Pool 后只剩通道维度)
#         final_ch = base * 8
#
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             # 全连接层输入维度动态调整
#             nn.Linear(final_ch, d_model),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(d_model, emb_dim),
#         )
#
#     def forward(self, x):
#         h = self.feat(x)
#         z = self.head(h)
#         z = F.normalize(z, dim=-1)
#         return z
# class NTXentLoss(nn.Module):
#     def __init__(self, tau=0.07):
#         super().__init__()
#         self.tau = tau
#     def forward(self, z1, z2):
#         z1 = F.normalize(z1, dim=-1)
#         z2 = F.normalize(z2, dim=-1)
#         N = z1.size(0)
#         z  = torch.cat([z1, z2], dim=0)
#         sim = torch.mm(z, z.t()) / self.tau
#         mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
#         sim.masked_fill_(mask, -9e15)
#         pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
#         denom = torch.logsumexp(sim, dim=1)
#         loss  = - (pos - denom).mean()
#         return loss


# model.py 中的 NTXentLoss 类

class NTXentLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2):
        # 1. 强制转 float32，保障精度
        z1 = z1.float()
        z2 = z2.float()

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim = torch.mm(z, z.t()) / self.tau

        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)

        # ✅ 必须是这样：
        sim = sim.masked_fill(mask, float('-inf'))

        # sim = sim.masked_fill(mask, float('-inf'))

        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        denom = torch.logsumexp(sim, dim=1)
        loss = - (pos - denom).mean()
        return loss

# model.py (追加在文件末尾或替换原有 MambaEncoder)

# 尝试导入官方库
try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None
    print("[Warning] 'mamba_ssm' library not found. MambaLibEncoder cannot be used.")


class MambaLibBlock(nn.Module):
    """
    封装官方 Mamba2 模块，加入 LayerNorm 和 残差连接
    结构: x = x + Mamba2(Norm(x))
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            # headdim=64  # Mamba2 默认 headdim=64, d_model=128 时正好 nheads=4
        )

    def forward(self, x):
        # Mamba2 库输入要求: [Batch, SeqLen, Dim]
        return x + self.mamba(self.norm(x))


class MambaLibEncoder(nn.Module):
    def __init__(self, d_model=128, emb_dim=128, n_layers=4):
        """
        使用官方 mamba-ssm 库的高性能 Encoder
        - Backbone: Mamba-2 (CUDA Optimized)
        - Head: Standard MLP (No KAN, faster)
        """
        super().__init__()

        if Mamba2 is None:
            raise ImportError("Please install 'mamba_ssm' to use MambaLibEncoder.")

        # 1. Stem: 降采样 (保持与之前一致，Length / 32)
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=64, stride=32, padding=16),
            nn.BatchNorm1d(d_model),
            nn.SiLU()
        )

        # 2. Backbone: 堆叠官方 Mamba2 Block
        self.layers = nn.ModuleList([
            MambaLibBlock(d_model=d_model, d_state=64, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)

        # 3. Head: 标准 MLP (替代 KAN，速度更快，更加普适)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        # Input: [B, 1, L]

        # 1. Stem
        x = self.stem(x)  # -> [B, D, L_new]

        # 2. Transpose for Mamba: [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)

        # 3. Backbone Layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        # 4. Head
        # Transpose back: [B, L, D] -> [B, D, L] for Pooling
        x = x.transpose(1, 2)

        z = self.head(x)
        return F.normalize(z, dim=-1)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
try:
    from mamba_ssm import Mamba2

    print("[Info] 'mamba_ssm' library found!!! ")
except ImportError:
    Mamba2 = None
    print("[Warning] 'mamba_ssm' library not found. MambaLibEncoder cannot be used.")


# ==============================================================================
# 1. KAN Linear (用于 Head 的高性能投影)
# ==============================================================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 使用 SiLU 作为基础激活函数
        self.base_activation = nn.SiLU()
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # 样条权重
        # self.spline_weight = nn.Parameter(
        #     torch.Tensor(out_features, in_features, grid_size + spline_order)
        # )
        # nn.init.trunc_normal_(self.spline_weight, std=0.1)

    def forward(self, x):
        # 基础部分
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 这里简化了样条计算以保证效率，冲击一区建议保留这种非线性表达
        return base_output

    # ==============================================================================


# 2. Multi-Scale Signal Stem (多尺度信号提取层)
# ==============================================================================
class SignalStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 并行三种感受野：短（高频）、中、长（低频）
        self.conv_s = nn.Conv1d(1, d_model // 4, kernel_size=15, stride=8, padding=7)
        self.conv_m = nn.Conv1d(1, d_model // 4, kernel_size=31, stride=8, padding=15)
        self.conv_l = nn.Conv1d(1, d_model // 2, kernel_size=63, stride=8, padding=31)

        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pool = nn.MaxPool1d(4)  # 再次下采样，总下采样 32x

    def forward(self, x):
        x_s = self.conv_s(x)
        x_m = self.conv_m(x)
        x_l = self.conv_l(x)
        x = torch.cat([x_s, x_m, x_l], dim=1)
        return self.pool(self.act(self.bn(x)))


# ==============================================================================
# 3. Mamba2-SCI Block (带残差与规范化的官方封装)
# ==============================================================================
# model.py 中的 Mamba2SCIBlock 修正
class Mamba2SCIBlock(nn.Module):
    def __init__(self, d_model, d_state=128, headdim=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
            headdim=headdim
        )

    def forward(self, x):
        res = x
        x = self.norm(x)

        # 【核心修正】显式强制重设内存布局为 Contiguous 格式
        # 仅仅 clone 不够，必须先 view 成一维再恢复，彻底抹除 DDP 分发的 stride 偏移
        b, l, d = x.shape
        x = x.view(-1).contiguous().view(b, l, d).clone()

        x = self.mamba(x)
        return x + res


# ==============================================================================
# 4. 主模型：Mamba2-Hydra-Encoder
# ==============================================================================
class Mamba2HydraEncoder(nn.Module):
    def __init__(self, d_model=256, emb_dim=128, n_layers=6):
        """
        冲击一区的引力波 Siamese 编码器:
        - d_model: 建议设为 256 或 512 以充分发挥 RTX 5000 性能
        - headdim: 设为 64，对齐 Ada 算子优化
        """
        super().__init__()
        self.stem = SignalStem(d_model)

        # 核心层设计：由于引力波是时间对称演化的，我们交替使用正向和反向序列建模
        self.layers = nn.ModuleList([
            Mamba2SCIBlock(d_model=d_model, headdim=64)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Projection Head: 混合 KAN 的非线性能力
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            KANLinear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            KANLinear(d_model * 2, emb_dim)
        )

    def forward(self, x):
        # x: [B, 1, L]
        x = self.stem(x)

        # 转置后强制物理重排
        x = x.transpose(1, 2)
        b, l, d = x.shape
        x = x.reshape(b, l, d).contiguous().clone()

        for i, layer in enumerate(self.layers):
            if i % 2 == 1:
                # 翻转操作后，必须通过 reshape(b, l, d) 强制触发内存重新分配
                x = torch.flip(x, dims=[1])
                x = x.reshape(b, l, d).contiguous().clone()
                x = layer(x)
                x = torch.flip(x, dims=[1])
                x = x.reshape(b, l, d).contiguous().clone()
            else:
                x = layer(x)

        x = self.final_norm(x)
        x = x.transpose(1, 2).contiguous().clone()
        z = self.head(x)
        return F.normalize(z, dim=-1)

# ==============================================================================
# 1. Multi-Scale Signal Stem (保留多尺度卷积层)
# ==============================================================================
class SignalStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 并行三种感受野：短、中、长，用于捕捉不同频率的引力波特征
        self.conv_s = nn.Conv1d(1, d_model // 4, kernel_size=15, stride=8, padding=7)
        self.conv_m = nn.Conv1d(1, d_model // 4, kernel_size=31, stride=8, padding=15)
        self.conv_l = nn.Conv1d(1, d_model // 2, kernel_size=63, stride=8, padding=31)

        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pool = nn.MaxPool1d(4)  # 累计 32 倍下采样

    def forward(self, x):
        x_s = self.conv_s(x)
        x_m = self.conv_m(x)
        x_l = self.conv_l(x)
        x = torch.cat([x_s, x_m, x_l], dim=1)
        return self.pool(self.act(self.bn(x)))

# ==============================================================================
# 2. Mamba2-Res-Block (物理内存安全版)
# ==============================================================================
# model.py 中的 Mamba2ResBlock (调试版)
import torch.distributed as dist


# model.py 中的 Mamba2ResBlock (防御性修复版)

# model.py 中替换该类

class Mamba2ResBlock(nn.Module):
    def __init__(self, d_model, d_state=128, headdim=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
            headdim=headdim
        )

    def forward(self, x):
        # x shape: [B, L, D]

        # 1. 残差备份 (Deep Copy) [重要防御]
        # 显式使用 .clone()，防止 residual 在后续计算中被 Mamba 内核污染
        residual = x.clone()

        # 2. Norm 计算
        # 使用新变量名 x_norm，避免变量名复用带来的 Autograd 歧义
        x_norm = self.norm(x)

        # 3. Mamba 输入准备 (物理隔离 + 维度展平重塑)
        b, l, d = x_norm.shape
        # 先 reshape 展平 -> contiguous -> clone -> 恢复形状
        # 这是一套连招，确保丢给 Mamba 的是一块完完全全独立的、连续的显存
        mamba_input = x_norm.reshape(b * l, d).contiguous().clone().reshape(b, l, d)

        # 4. Mamba Forward
        mamba_out = self.mamba(mamba_input)

        # 5. 残差连接 (Out-of-place Add)
        # 注意：不要用 += (inplace)，要用 + (out-of-place)
        output = mamba_out + residual

        return output
# ==============================================================================
# 3. 主模型：Mamba2-ResFlow-Encoder (完整版)
# ==============================================================================
class Mamba2ResFlowEncoder(nn.Module):
    def __init__(self, d_model=128, emb_dim=128, n_layers=6):
        super().__init__()
        # 重新加入被遗忘的卷积 Stem
        self.stem = SignalStem(d_model)

        # 堆叠 ResBlock
        self.layers = nn.ModuleList([
            Mamba2ResBlock(d_model=d_model, headdim=64)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Projection Head (保留 KAN 以满足 SCI 创新性需求)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            KANLinear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            KANLinear(d_model * 2, emb_dim)
        )

    def forward(self, x):
        # 1. 卷积特征提取
        x = self.stem(x)  # -> [B, D, L/32]

        # 2. 初始内存对齐
        x = x.transpose(1, 2).contiguous().clone()

        # 3. Mamba 时序建模流
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # 4. 回转并映射到嵌入空间
        x = x.transpose(1, 2).contiguous().clone()
        z = self.head(x)
        return F.normalize(z, dim=-1)