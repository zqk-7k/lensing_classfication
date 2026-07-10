# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# 完全导入你自己的模块
import config_classifier as cfg
import data_classifier as data_lib
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation

def train_and_evaluate(variant_name, use_snake, use_se, use_physics_fusion, device, tr_loader, va_loader):
    print(f"\n{'='*70}")
    print(f"?? 开始训练变体: {variant_name}")
    print(f"??  控制变量: Snake={use_snake} | SE={use_se} | Physics_Fusion={use_physics_fusion}")
    print(f"{'='*70}")

    # 1. 严格使用你原有的模型初始化
    model = BinaryPeriodicResNet1D_Ablation(
        d_model=256,
        width_scale=4.0,
        use_snake=use_snake,
        use_se=use_se,
        use_physics_fusion=use_physics_fusion
    ).to(device)

    # 2. 严格继承你的原版 Loss 和 Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    epochs = cfg.EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_auc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ------------------ 训练阶段 (完全保留你的逻辑) ------------------
        model.train()
        tr_loss = 0.0
        for inputs, labels in tr_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()

            # 【保留细节】防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tr_loss += loss.item() * inputs.size(0)

        tr_loss /= len(tr_loader.dataset)

        # ------------------ 验证阶段 (完全保留你的逻辑) ------------------
        model.eval()
        va_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in va_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                logits = model(inputs).squeeze(1)
                loss = criterion(logits, labels)
                va_loss += loss.item() * inputs.size(0)

                # 【保留细节】使用 sigmoid 计算概率
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        va_loss /= len(va_loader.dataset)
        va_acc = accuracy_score(all_labels, all_preds)
        va_auc = roc_auc_score(all_labels, all_probs)

        # 更新最佳结果
        if va_acc > best_val_acc: best_val_acc = va_acc
        if va_auc > best_val_auc: best_val_auc = va_auc

        print(f"Epoch [{epoch}/{epochs}] | Tr Loss: {tr_loss:.4f} | Va Loss: {va_loss:.4f} | Va Acc: {va_acc*100:.2f}% | Va AUC: {va_auc:.4f}")

        # 【保留细节】余弦退火调度
        scheduler.step()

    run_time = (time.time() - start_time) / 60
    print(f"? {variant_name} 训练完毕! 耗时: {run_time:.1f}分钟 | Best Acc: {best_val_acc*100:.2f}% | Best AUC: {best_val_auc:.4f}")

    # 释放显存，防止连续训练 OOM
    del model, optimizer, criterion
    torch.cuda.empty_cache()

    return best_val_acc, best_val_auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"?? 使用设备: {device}")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # ================= 提前加载数据 (完全继承 main_ablation 逻辑) =================
    print("\n?? 正在将所有数据加载到内存中 (仅加载一次)...")
    l1 = data_lib.load_npy_data(os.path.join(cfg.SOURCE_DIR, "l1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(cfg.SOURCE_DIR, "l2.npy"))
    unl = data_lib.load_npy_data(os.path.join(cfg.SOURCE_DIR, "unl.npy"))

    n_samples = len(l1)
    n_tr = int(n_samples * 0.8)
    n_va = n_samples - n_tr

    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    tr_ds = data_lib.GWClassifierDataset(l1, l2, unl, indices[:n_tr], mode='train')
    va_ds = data_lib.GWClassifierDataset(l1, l2, unl, indices[n_tr:n_tr+n_va], mode='val')

    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"? 数据集准备完毕! Train size: {len(tr_ds)}, Val size: {len(va_ds)}")

    # ================= 消融实验矩阵 =================
    variants = {
        "PI-ResNet (Baseline)": {"use_snake": False, "use_se": True,  "use_physics_fusion": True},
        "w/o Physics Fusion":   {"use_snake": False, "use_se": True,  "use_physics_fusion": False},
        "w/o SE Block":         {"use_snake": False, "use_se": False, "use_physics_fusion": True},
        "w/ Snake Activation":  {"use_snake": True,  "use_se": True,  "use_physics_fusion": True}
    }

    results = {}
    for name, config in variants.items():
        acc, auc = train_and_evaluate(
            variant_name=name,
            use_snake=config["use_snake"],
            use_se=config["use_se"],
            use_physics_fusion=config["use_physics_fusion"],
            device=device,
            tr_loader=tr_loader,
            va_loader=va_loader
        )
        results[name] = {"acc": acc, "auc": auc}

    # ================= 生成最终日志 =================
    summary_path = os.path.join(cfg.OUT_DIR, "ablation_summary_strictly_preserved.txt")
    with open(summary_path, "w") as f:
        header = f"\n{'='*75}\n?? 消融实验总结报告 (Ablation Study Summary)\n{'='*75}"
        print(header)
        f.write(header + "\n")

        fmt_str = "{:<25} | {:<15} | {:<10}"
        col_headers = fmt_str.format("Model Variant", "Best Acc (%)", "Best AUC")
        print(col_headers)
        f.write(col_headers + "\n")
        print("-" * 75)
        f.write("-" * 75 + "\n")

        for name, metrics in results.items():
            acc_str = f"{metrics['acc']*100:.2f}"
            auc_str = f"{metrics['auc']:.4f}"
            row = fmt_str.format(name, acc_str, auc_str)
            print(row)
            f.write(row + "\n")

    print(f"?? 实验结束！结果已保存至: {summary_path}")

if __name__ == '__main__':
    main()