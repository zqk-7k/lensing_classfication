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

# 论文级别大字体设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 2.5
})

# 🛠️ 核心函数：强行移除 Snake 并注入 ReLU
def force_convert_to_relu(module):
    for name, child in module.named_children():
        if 'alpha' in child._parameters or hasattr(child, 'alpha'):
            setattr(module, name, nn.ReLU())
        else:
            force_convert_to_relu(child)

def plot_pm_error_hist():
    # 指向 PM 的新复现目录
    run_dir = "./runs/PM_noisy_OldData_Repro"
    model_p = os.path.join(run_dir, "best_classifier.pt")
    snr_p = os.path.join(run_dir, "sample_snrs.npy")

    device = torch.device("cpu")

    # 1. 初始化模型并进行架构手术
    model = model_lib.BinaryPeriodicResNet1D_Ablation(d_model=256, width_scale=4.0).to(device)
    force_convert_to_relu(model)

    # 2. 加载权重
    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. 加载 SNR 和 PM 的 0222 原始数据
    full_snr = np.load(snr_p)
    source_dir = os.path.join(cfg.DATA_ROOT, "PM_data_0222")
    l1 = data_lib.load_npy_data(os.path.join(source_dir, "PM_data_strain_1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(source_dir, "PM_data_strain_2.npy"))
    unl = data_lib.load_npy_data(os.path.join(cfg.DATA_ROOT, "Unlensed_data_0222", "unlensed_data_strain.npy"))

    # 验证集划分 (保持 0.8-0.9 的区间)
    n_samples = len(l1)
    indices = np.arange(n_samples)
    np.random.seed(cfg.SEED)
    np.random.shuffle(indices)
    idx_va = indices[int(n_samples * 0.8):int(n_samples * 0.9)]

    va_ds = data_lib.GWClassifierDataset(l1, l2, unl, idx_va, mode='val')
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=128, shuffle=False)

    # 4. 推理
    probs, labels, snrs = [], [], []
    for pair in va_ds.fixed_pairs:
        snrs.append(full_snr[pair['idx1']])

    print("⏳ 正在为 PM 模型计算预测分布...")
    with torch.no_grad():
        for inputs, targets in va_loader:
            out = model(inputs).squeeze(1)
            probs.extend(torch.sigmoid(out).numpy())
            labels.extend(targets.numpy())

    # 5. 准备柱状图数据 (仅分析正样本 Lensed)
    snrs, probs, labels = np.array(snrs), np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(int)

    lensed_mask = (labels == 1)
    tp_snrs = snrs[lensed_mask & (preds == 1)]
    fn_snrs = snrs[lensed_mask & (preds == 0)]

    # 6. 绘图
    plt.figure(figsize=(9, 7))
    bins = np.linspace(0, 20, 21)

    plt.hist(tp_snrs, bins=bins, alpha=0.7, label='True Positives', color='#1f77b4', edgecolor='white')
    plt.hist(fn_snrs, bins=bins, alpha=0.9, label='False Negatives', color='#ff7f0e', hatch='///', edgecolor='white')

    plt.axvline(x=8.0, color='black', linestyle='--', lw=3, label='Threshold (SNR=8)')

    plt.yscale('log')
    plt.xlim([-0.5, 20.5])
    plt.grid(True, axis='y', ls="--", alpha=0.3)
    plt.xlabel('Optimal Matched-filter SNR', fontweight='bold')
    plt.ylabel('Event Count (Log Scale)', fontweight='bold')
    plt.title('Error Attribution: PM') # 统一格式标题
    plt.legend(loc='upper right', frameon=True, edgecolor='black')

    os.makedirs('pic', exist_ok=True)
    plt.tight_layout()
    plt.savefig('pic/fig11_pm_hist_new.png', dpi=300)
    print("✅ PM 误差分布图已生成: pic/fig11_pm_hist_new.png")

if __name__ == "__main__":
    plot_pm_error_hist()
