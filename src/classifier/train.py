# -*- coding: utf-8 -*-
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TMP_DIR = os.environ.get("GW_TMPDIR", os.path.join(PROJECT_ROOT, ".tmp"))
os.makedirs(TMP_DIR, exist_ok=True)
os.environ['TMPDIR'] = TMP_DIR

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
        # 最纯粹的训练循环，不需要任何切片补丁！
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
    parser.add_argument('--exp_name', type=str, default='Baseline')
    parser.add_argument('--use_snake', action='store_true', help="Use Snake instead of ReLU")
    parser.add_argument('--no_se', action='store_true', help="Remove SE Block")
    parser.add_argument('--no_physics_fusion', action='store_true', help="Remove Physics Fusion")
    args = parser.parse_args()

    safe_exp_name = args.exp_name.replace(" ", "_")
    final_out_dir = os.path.join(cfg.OUT_DIR, safe_exp_name)
    os.makedirs(final_out_dir, exist_ok=True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f" Target Task: [{cfg.MODEL_TYPE}] with [{cfg.DATA_MODE}] data")
    print(f" Output Dir : {final_out_dir}")
    print(f" Snake: {args.use_snake} | SE: {not args.no_se} | Fusion: {not args.no_physics_fusion}")
    print(f"{'='*60}")

    print("Loading data...")
    l1 = data_lib.load_npy_data(cfg.L1_PATH)
    l2 = data_lib.load_npy_data(cfg.L2_PATH)
    unl = data_lib.load_npy_data(cfg.UNL_PATHS[0])

    n_samples = len(l1)
    n_tr = int(n_samples * 0.8)
    n_va = n_samples - n_tr
    np.random.seed(cfg.SEED)
    indices = np.random.permutation(n_samples)

    tr_ds = data_lib.GWClassifierDataset(l1, l2, unl, indices[:n_tr], mode='train')
    va_ds = data_lib.GWClassifierDataset(l1, l2, unl, indices[n_tr:n_tr+n_va], mode='val')

    # ================= 【终极绝杀：让数据自己说话】 =================
    # 不管 Config 里写的是 1 还是 2，直接抽查第一个样本！
    sample_x, _ = tr_ds[0]
    total_channels = sample_x.shape[0]  # DataLoader 吐出的总通道数 (例如 4)
    actual_in_ch = total_channels // 2  # 孪生网络，砍掉一半就是单支路真实的物理通道数 (例如 2)
    # ================================================================

    print(f" [Auto-Detect] 数据真实总通道数: {total_channels}")
    print(f" [Auto-Detect] 模型自适应输入通道: {actual_in_ch}")

    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 用探测到的真实通道数来初始化模型，彻底杜绝报错！
    model = BinaryPeriodicResNet1D_Ablation(
        in_channels=actual_in_ch,   
        d_model=256, 
        width_scale=4.0,
        use_snake=args.use_snake,
        use_se=not args.no_se,
        use_physics_fusion=not args.no_physics_fusion
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device), reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    epochs = cfg.EPOCHS 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_auc = 0.0
    best_model_path = os.path.join(final_out_dir, f"best_{cfg.MODEL_TYPE}_{safe_exp_name}.pt")
    
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
    
    log_file = os.path.join(final_out_dir, "experiment_summary.txt")
    with open(log_file, "w") as f:
        f.write(f"Task: {cfg.MODEL_TYPE} ({cfg.DATA_MODE})\n")
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Best Val Acc: {best_val_acc*100:.2f}%\n")
        f.write(f"Best Val AUC: {best_val_auc:.4f}\n")
        f.write(f"Runtime: {run_time:.2f} mins\n")
        
    print(f"\n? Done! Best Acc: {best_val_acc*100:.2f}%. Saved to {final_out_dir}")

if __name__ == '__main__':
    main()
