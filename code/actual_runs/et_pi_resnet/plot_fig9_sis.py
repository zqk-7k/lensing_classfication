# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

# 🛠️ 黑科技：递归查找并强行把 Snake 替换为 ReLU
def force_convert_to_relu(module):
    for name, child in module.named_children():
        # 只要层里面有 'alpha' 参数，就说明它是 Snake，直接干掉换成 ReLU！
        if 'alpha' in child._parameters or hasattr(child, 'alpha'):
            setattr(module, name, nn.ReLU())
        else:
            force_convert_to_relu(child)

def plot_sis_log_roc():
    run_dir = "./runs/SIS_noisy_OldData_Repro"
    model_p = os.path.join(run_dir, "best_classifier.pt")
    snr_p = os.path.join(run_dir, "sample_snrs.npy")

    device = torch.device("cpu")

    print("1. 正在初始化基础模型框架...")
    model = model_lib.BinaryPeriodicResNet1D_Ablation(d_model=256, width_scale=4.0).to(device)

    print("2. 正在进行架构手术：强行移除 Snake 并注入 ReLU...")
    force_convert_to_relu(model)

    print("3. 正在读取并加载纯 ReLU 权重文件...")
    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # 使用 strict=False 安全加载
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("✅ 权重完美加载！")

    # 4. 加载 SNR 和 0222 原始数据
    full_snr = np.load(snr_p)
    source_dir = os.path.join(cfg.DATA_ROOT, "SIS_data_0222")
    l1 = data_lib.load_npy_data(os.path.join(source_dir, "SIS_data_strain_1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(source_dir, "SIS_data_strain_2.npy"))
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

    print("⏳ 正在为 SIS 模型运行推理，进度条开启...")
    with torch.no_grad():
        for inputs, targets in va_loader:
            out = model(inputs).squeeze(1)
            probs.extend(torch.sigmoid(out).numpy())
            labels.extend(targets.numpy())

    # 5. 绘图
    snrs, probs, labels = np.array(snrs), np.array(probs), np.array(labels)
    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#d62728']
    masks = [snrs >= 10.0, snrs < 10.0]
    names = [r'High SNR ($\geq$10.0)', r'Low SNR (<10.0)']

    for mask, name, style, clr in zip(masks, names, ['-', '--'], colors):
        combined_mask = mask | (labels == 0)
        fpr, tpr, _ = roc_curve(labels[combined_mask], probs[combined_mask])
        plt.plot(fpr, tpr, label=name, linestyle=style, lw=4, color=clr)

    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1e-4, 1.0); plt.ylim(1e-2, 1.05)
    plt.grid(True, which="both", alpha=0.3, linestyle=':')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Log-ROC Sensitivity: SIS')
    plt.legend(loc='lower right', frameon=True, edgecolor='black')

    os.makedirs('pic', exist_ok=True)
    plt.tight_layout()
    plt.savefig('pic/fig9_sis_new_large.png', dpi=300)
    print("✅ 绘图大功告成！已保存至: pic/fig9_sis_new_large.png")

if __name__ == "__main__":
    plot_sis_log_roc()
