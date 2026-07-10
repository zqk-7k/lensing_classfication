# -*- coding: utf-8 -*-
import os
import sys

# ✅ 让 Python 既能找当前目录，也能找上一级主目录
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

import config_classifier as cfg
import data_classifier as data_lib
import model_classifier_ablation as model_lib  # <--- ✅ 核心修复：指向你真实的模型文件名

# ========================================================
# ✅ 独立复现文件夹路径
# ========================================================
RUNS_DIR_SIS = "./runs/SIS_noisy_OldData_Repro"
RUNS_DIR_PM  = "./runs/PM_noisy_OldData_Repro"

PATHS = {
    "SIS": {
        "model": os.path.join(RUNS_DIR_SIS, "best_classifier.pt"),
        "snr": os.path.join(RUNS_DIR_SIS, "sample_snrs.npy")
    },
    "PM": {
        "model": os.path.join(RUNS_DIR_PM, "best_classifier.pt"),
        "snr": os.path.join(RUNS_DIR_PM, "sample_snrs.npy")
    }
}

# ========================================================
# 核心视觉优化：论文级高对比度、大字体设置
# ========================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 2.5,
    'figure.facecolor': 'white'
})

def get_predictions(m_type, device):
    """加载 ReLU 版本的新权重并获取测试集的预测结果"""
    print(f"\n[任务] 正在处理 {m_type} 模型的推理...")

    model_p = PATHS[m_type]["model"]
    snr_p = PATHS[m_type]["snr"]

    if not (os.path.exists(model_p) and os.path.exists(snr_p)):
        print(f"  --> [错误] 找不到路径: {model_p} 或 {snr_p}。")
        return None, None, None

    # 加载模型 - 确保兼容你代码里的类名
    try:
        model = model_lib.BinaryResNet1D(d_model=256, width_scale=4.0).to(device)
    except AttributeError:
        model = model_lib.BinaryPeriodicResNet1D(d_model=256, width_scale=4.0).to(device)

    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    full_snr_array = np.load(snr_p)

    # 加载测试数据 (0222版)
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
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=2)

    all_probs, all_labels, va_sample_snrs = [], [], []
    for pair in va_ds.fixed_pairs:
        va_sample_snrs.append(full_snr_array[pair['idx1']])

    with torch.no_grad():
        for inputs, labels in va_loader:
            logits = model(inputs.to(device)).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs), np.array(va_sample_snrs)

def plot_log_roc(y_true, y_prob, snrs, m_type):
    high_mask = (snrs >= 10.0) | (y_true == 0)
    low_mask = (snrs < 10.0) | (y_true == 0)

    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#d62728']

    for mask, label_str, style, clr in zip([high_mask, low_mask], [r'High SNR ($\geq$10.0)', r'Low SNR (<10.0)'], ['-', '--'], colors):
        if np.sum(mask) > 0:
            fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
            plt.plot(fpr, tpr, label=label_str, linestyle=style, lw=4, color=clr)

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", alpha=0.3, linestyle=':')
    plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
    plt.title(f'Log-ROC Sensitivity: {m_type}', fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True, edgecolor='black')
    plt.xlim(1e-4, 1.0)
    plt.ylim(1e-2, 1.05)

    os.makedirs('pic', exist_ok=True)
    save_path = f"pic/fig_roc_by_snr_log_{m_type}_large.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  --> 已保存: {save_path}")

def plot_error_attribution(y_true, y_prob, snrs, m_type):
    preds = (y_prob >= 0.5).astype(int)
    lensed_mask = (y_true == 1)

    tp_snrs = snrs[lensed_mask & (preds == 1)]
    fn_snrs = snrs[lensed_mask & (preds == 0)]

    plt.figure(figsize=(9, 7))
    bins = np.linspace(0, 20, 20)

    plt.hist(tp_snrs, bins=bins, alpha=0.7, label='True Positives', color='#1f77b4', edgecolor='white')
    plt.hist(fn_snrs, bins=bins, alpha=0.9, label='False Negatives', color='#ff7f0e', hatch='///', edgecolor='white')
    plt.axvline(x=8.0, color='black', linestyle='--', lw=3, label='Threshold (SNR=8)')

    plt.yscale('log')
    plt.xlim([-1, 21])
    plt.grid(True, axis='y', ls="--", alpha=0.3)
    plt.xlabel('Optimal Matched-filter SNR', fontweight='bold')
    plt.ylabel('Event Count (Log Scale)', fontweight='bold')
    plt.title(f'Error Attribution: {m_type} Model', fontweight='bold', pad=15)
    plt.legend(loc='upper right', edgecolor='black')

    save_path = f"pic/fig_error_by_snr_hist_{m_type}_large.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  --> 已保存: {save_path}")

def plot_efficiency_sweep(y_t_sis, y_p_sis, s_sis, y_t_pm, y_p_pm, s_pm):
    def get_eff(y_t, y_p, snrs):
        preds = (y_p >= 0.5).astype(int)
        pos_mask = (y_t == 1)
        bins = np.arange(0, 22, 2)
        centers, tprs = [], []
        for i in range(len(bins)-1):
            mask = (snrs[pos_mask] >= bins[i]) & (snrs[pos_mask] < bins[i+1])
            if np.sum(mask) > 0:
                tprs.append(np.mean(preds[pos_mask][mask] == 1))
                centers.append((bins[i] + bins[i+1])/2)
        return centers, tprs

    bc_sis, eff_sis = get_eff(y_t_sis, y_p_sis, s_sis)
    bc_pm, eff_pm = get_eff(y_t_pm, y_p_pm, s_pm)

    plt.figure(figsize=(10, 8))
    plt.plot(bc_sis, eff_sis, marker='o', markersize=10, label='SIS Model', color='#1f77b4', lw=4)
    plt.plot(bc_pm, eff_pm, marker='s', markersize=10, label='PM Model', color='#d62728', lw=4)

    plt.axvline(x=8.0, color='black', linestyle='--', lw=3, label='Threshold (SNR=8)')
    plt.xlim(0, 20)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", alpha=0.3, linestyle=':')
    plt.xlabel('Optimal Matched-filter SNR', fontweight='bold')
    plt.ylabel('Detection Efficiency (TPR)', fontweight='bold')
    plt.title('Efficiency Sweep: Signal Strength Analysis', fontweight='bold', pad=15)
    plt.legend(loc='lower right', edgecolor='black')

    save_path = "pic/fig_snr_efficiency_sweep_large.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  --> 已保存: {save_path}")

if __name__ == "__main__":
    device = torch.device("cpu")
    print("=== 开始生成最新版 (ReLU/No-Snake) 高清大字插图 ===")

    y_t_sis, y_p_sis, snr_sis = get_predictions("SIS", device)
    if y_t_sis is not None:
        plot_log_roc(y_t_sis, y_p_sis, snr_sis, "SIS")
        plot_error_attribution(y_t_sis, y_p_sis, snr_sis, "SIS")

    y_t_pm, y_p_pm, snr_pm = get_predictions("PM", device)
    if y_t_pm is not None:
        plot_log_roc(y_t_pm, y_p_pm, snr_pm, "PM")
        plot_error_attribution(y_t_pm, y_p_pm, snr_pm, "PM")

    if y_t_sis is not None and y_t_pm is not None:
        plot_efficiency_sweep(y_t_sis, y_p_sis, snr_sis, y_t_pm, y_p_pm, snr_pm)

    print("\n✅ 所有新版插图已成功生成在 './pic' 目录下。")
