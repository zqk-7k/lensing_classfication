# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ✅ 路径修复
sys.path.append(os.path.abspath(".."))
import model_classifier_ablation as model_lib
import data_classifier as data_lib
import config_classifier as cfg

# 论文级别大字体与线宽设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 22,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'axes.linewidth': 2.5
})

def force_convert_to_relu(module):
    for name, child in module.named_children():
        if 'alpha' in child._parameters or hasattr(child, 'alpha'):
            setattr(module, name, nn.ReLU())
        else:
            force_convert_to_relu(child)

def get_model_efficiency(m_type):
    run_dir = f"./runs/{m_type}_noisy_OldData_Repro"
    model_p = os.path.join(run_dir, "best_classifier.pt")
    snr_p = os.path.join(run_dir, "sample_snrs.npy")
    device = torch.device("cpu")

    # 加载模型并手术
    model = model_lib.BinaryPeriodicResNet1D_Ablation(d_model=256, width_scale=4.0).to(device)
    force_convert_to_relu(model)
    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 加载数据
    full_snr = np.load(snr_p)
    source_dir = os.path.join(cfg.DATA_ROOT, f"{m_type}_data_0222")
    l1 = data_lib.load_npy_data(os.path.join(source_dir, f"{m_type}_data_strain_1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(source_dir, f"{m_type}_data_strain_2.npy"))
    unl = data_lib.load_npy_data(os.path.join(cfg.DATA_ROOT, "Unlensed_data_0222", "unlensed_data_strain.npy"))

    n_samples = len(l1)
    indices = np.arange(n_samples)
    np.random.seed(cfg.SEED)
    np.random.shuffle(indices)
    idx_va = indices[int(n_samples * 0.8):int(n_samples * 0.9)]

    va_ds = data_lib.GWClassifierDataset(l1, l2, unl, idx_va, mode='val')
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=128, shuffle=False)

    probs, labels, snrs = [], [], []
    for pair in va_ds.fixed_pairs:
        snrs.append(full_snr[pair['idx1']])

    with torch.no_grad():
        for inputs, targets in va_loader:
            out = model(inputs).squeeze(1)
            probs.extend(torch.sigmoid(out).numpy())
            labels.extend(targets.numpy())

    snrs, probs, labels = np.array(snrs), np.array(probs), np.array(labels)

    # 计算不同 SNR 分箱下的 TPR (探测效率)
    pos_mask = (labels == 1)
    bins = np.arange(0, 22, 2)
    centers, tpr_list = [], []

    for i in range(len(bins)-1):
        mask = (snrs[pos_mask] >= bins[i]) & (snrs[pos_mask] < bins[i+1])
        if np.sum(mask) > 0:
            tpr = np.mean(probs[pos_mask][mask] >= 0.5)
            tpr_list.append(tpr)
            centers.append((bins[i] + bins[i+1]) / 2)

    return centers, tpr_list

if __name__ == "__main__":
    print("⏳ 正在提取 SIS 模型效率数据...")
    c_sis, e_sis = get_model_efficiency("SIS")
    print("⏳ 正在提取 PM 模型效率数据...")
    c_pm, e_pm = get_model_efficiency("PM")

    plt.figure(figsize=(12, 8))
    plt.plot(c_sis, e_sis, label='SIS Model', color='#1f77b4', marker='o', lw=4, ms=12)
    plt.plot(c_pm, e_pm, label='PM Model', color='#d62728', marker='s', lw=4, ms=12)

    plt.axvline(x=8.0, color='black', linestyle='--', lw=3, label='Threshold (SNR=8)')
    plt.xlabel('Optimal Matched-filter SNR', fontweight='bold')
    plt.ylabel('Detection Efficiency (TPR)', fontweight='bold')
    plt.title('Efficiency Sweep: Signal Strength Analysis')
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 20)
    plt.grid(True, which="both", alpha=0.3, linestyle=':')
    plt.legend(loc='lower right', frameon=True, edgecolor='black')

    os.makedirs('pic', exist_ok=True)
    plt.tight_layout()
    plt.savefig('pic/fig12_efficiency_sweep_new.png', dpi=300)
    print("✅ 终极大图已生成: pic/fig12_efficiency_sweep_new.png")
