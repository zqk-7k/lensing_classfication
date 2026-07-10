# -*- coding: utf-8 -*-
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import config_classifier as cfg
import data_classifier as data_lib
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return running_loss / len(dataloader.dataset), acc, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SIS', choices=['SIS', 'PM'])
    parser.add_argument('--exp_name', type=str, default='Baseline')

    parser.add_argument('--use_snake', action='store_true', help="Use Snake instead of ReLU")
    parser.add_argument('--no_se', action='store_true', help="Remove SE Block")
    parser.add_argument('--no_physics_fusion', action='store_true', help="Remove Physics Fusion")
    args = parser.parse_args()

    # ================= [修复核心] 动态分配路径并强制创建目录 =================
    if args.dataset == 'SIS':
        cfg.SOURCE_DIR = os.path.join(cfg.DATA_ROOT, "SIS_data_0222")
        cfg.OUT_DIR = "./runs/classifier_SIS_noisy_v1"
    else:
        cfg.SOURCE_DIR = os.path.join(cfg.DATA_ROOT, "PM_data_0222")
        cfg.OUT_DIR = "./runs/classifier_PM_noisy_v1"

    os.makedirs(cfg.OUT_DIR, exist_ok=True)  # 就是这句救命的代码！
    os.makedirs("./logs", exist_ok=True)
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Experiment: {args.exp_name} on {args.dataset}")
    print(f" Snake: {args.use_snake} | SE: {not args.no_se} | Fusion: {not args.no_physics_fusion}")
    print(f"{'='*60}")

    l1 = data_lib.load_npy_data(os.path.join(cfg.SOURCE_DIR, f"{args.dataset}_data_strain_1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(cfg.SOURCE_DIR, f"{args.dataset}_data_strain_2.npy"))

    unl_path = "/root/autodl-tmp/qkzhang/Unlensed_data_0222/unlensed_data_strain.npy"
    if not os.path.exists(unl_path):
        unl_path = os.path.join(cfg.SOURCE_DIR, "unlensed_data_strain.npy")
    unl = data_lib.load_npy_data(unl_path)

    n_samples = len(l1)
    n_tr = int(n_samples * 0.8)
    n_va = n_samples - n_tr
    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    tr_ds = data_lib.GWClassifierDataset(l1, l2, unl, indices[:n_tr], mode='train')
    va_ds = data_lib.GWClassifierDataset(l1, l2, unl, indices[n_tr:n_tr+n_va], mode='val')

    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = BinaryPeriodicResNet1D_Ablation(
        d_model=256,
        width_scale=4.0,
        use_snake=args.use_snake,
        use_se=not args.no_se,
        use_physics_fusion=not args.no_physics_fusion
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    epochs = cfg.EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_auc = 0.0

    safe_exp_name = args.exp_name.replace(" ", "_")
    best_model_path = os.path.join(cfg.OUT_DIR, f"best_{args.dataset}_{safe_exp_name}.pt")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc, va_auc = val_epoch(model, va_loader, criterion, device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_model_path)
        if va_auc > best_val_auc:
            best_val_auc = va_auc

        print(f"Ep {epoch:3d}: Tr_Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Va_Loss={va_loss:.4f} Acc={va_acc:.4f} AUC={va_auc:.4f}")

    run_time = (time.time() - start_time) / 60

    log_file = os.path.join(cfg.OUT_DIR, "final_ablation_results.txt")
    with open(log_file, "a") as f:
        f.write(f"{args.dataset} | {args.exp_name:25} | Acc: {best_val_acc*100:.2f}% | AUC: {best_val_auc:.4f}\n")

    print(f"Done! Best Acc: {best_val_acc*100:.2f}%. Saved to {best_model_path}")

if __name__ == '__main__':
    main()
