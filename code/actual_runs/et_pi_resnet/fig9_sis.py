# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# 路径修复
sys.path.append(os.path.abspath(".."))
import model_classifier_ablation as model_lib
import data_classifier as data_lib
import config_classifier as cfg

# 论文风格绘图参数
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


def force_convert_to_relu(module):
    """
    递归把 Snake 激活替换成 ReLU
    """
    for name, child in module.named_children():
        if 'alpha' in child._parameters or hasattr(child, 'alpha'):
            setattr(module, name, nn.ReLU())
        else:
            force_convert_to_relu(child)


def reduce_snr_to_1d(full_snr):
    """
    把可能的二维 SNR 压成一维，保证每个样本只有一个 SNR。
    优先规则：
    - (N,) 直接返回
    - (N, 1) squeeze 后返回
    - (N, 2) 取每个样本两路中的最大值，作为 network-level SNR
    """
    full_snr = np.asarray(full_snr)
    full_snr = np.squeeze(full_snr)

    if full_snr.ndim == 1:
        return full_snr

    if full_snr.ndim == 2:
        # 对于 (N, 2)，取最大值更合理，表示双路/双像中的更强响应
        return np.max(full_snr, axis=1)

    raise ValueError(f"Unsupported SNR shape after squeeze: {full_snr.shape}")


def plot_sis_log_roc():
    run_dir = "./runs/SIS_noisy_OldData_Repro"
    model_p = os.path.join(run_dir, "best_classifier.pt")

    source_dir = "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/ET_vs_LIGO/LIGO_data/SIS_data"
    unlensed_dir = "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/ET_vs_LIGO/LIGO_data/Unlensed_data"

    # 优先用 network-level SNR
    snr_p = os.path.join(source_dir, "SIS_optimal_SNR_network_1.npy")
    unl_p = os.path.join(unlensed_dir, "unlensed_data_strain.npy")

    device = torch.device("cpu")

    print("1. Initializing base model...")
    model = model_lib.BinaryPeriodicResNet1D_Ablation(
        d_model=256,
        width_scale=4.0
    ).to(device)

    print("2. Replacing Snake activations with ReLU...")
    force_convert_to_relu(model)

    print("3. Loading checkpoint...")
    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Weights loaded successfully.")

    print("SIS dir:", source_dir)
    print("Unlensed dir:", unlensed_dir)
    print("SNR file:", snr_p)

    if not os.path.exists(model_p):
        raise FileNotFoundError(f"Model file not found: {model_p}")
    if not os.path.exists(snr_p):
        raise FileNotFoundError(f"SNR file not found: {snr_p}")
    if not os.path.exists(os.path.join(source_dir, "SIS_data_strain_1.npy")):
        raise FileNotFoundError("Missing SIS_data_strain_1.npy")
    if not os.path.exists(os.path.join(source_dir, "SIS_data_strain_2.npy")):
        raise FileNotFoundError("Missing SIS_data_strain_2.npy")
    if not os.path.exists(unl_p):
        raise FileNotFoundError(
            f"Unlensed file not found: {unl_p}\n"
            f"If the filename differs, update unl_p in this script."
        )

    # 4. 加载数据
    full_snr_raw = np.load(snr_p)
    full_snr = reduce_snr_to_1d(full_snr_raw)

    l1 = data_lib.load_npy_data(os.path.join(source_dir, "SIS_data_strain_1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(source_dir, "SIS_data_strain_2.npy"))
    unl = data_lib.load_npy_data(unl_p)

    print("Raw SNR shape:", np.asarray(full_snr_raw).shape)
    print("Reduced SNR shape:", full_snr.shape)
    print("l1 shape:", l1.shape)
    print("l2 shape:", l2.shape)
    print("unl shape:", unl.shape)

    n_samples = len(l1)
    if len(full_snr) != n_samples:
        raise ValueError(
            f"SNR length ({len(full_snr)}) does not match number of SIS samples ({n_samples})."
        )

    indices = np.arange(n_samples)
    np.random.seed(cfg.SEED)
    np.random.shuffle(indices)

    # 改回 10%
    idx_va = indices[int(n_samples * 0.8):int(n_samples * 0.9)]

    va_ds = data_lib.GWClassifierDataset(l1, l2, unl, idx_va, mode='val')
    va_loader = torch.utils.data.DataLoader(
        va_ds,
        batch_size=128,
        shuffle=False
    )

    probs, labels, snrs = [], [], []

    for pair in va_ds.fixed_pairs:
        snrs.append(full_snr[pair['idx1']])

    print("Running inference for SIS model...")
    with torch.no_grad():
        for inputs, targets in va_loader:
            out = model(inputs).squeeze(1)
            probs.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    # 5. 转成 numpy
    snrs = np.array(snrs)
    probs = np.array(probs)
    labels = np.array(labels)

    print("Total samples used for plotting:", len(labels))
    print("Positive labels:", np.sum(labels == 1))
    print("Negative labels:", np.sum(labels == 0))

    # 检查每个分组里有没有正负样本
    masks = [snrs >= 10.0, snrs < 10.0]
    group_names_debug = ['High SNR', 'Low SNR']

    for mask, name in zip(masks, group_names_debug):
        combined_mask = mask | (labels == 0)
        y = labels[combined_mask]
        print(name)
        print("  total:", len(y))
        print("  positives:", np.sum(y == 1))
        print("  negatives:", np.sum(y == 0))

    # 6. 绘图
    plt.figure(figsize=(9, 7))

    # 先画低 SNR，再画高 SNR，让蓝线压在最上面
    plot_items = [
        (snrs < 10.0,  r'Low SNR (<10.0)', '--', '#d62728', 2, None),
        (snrs >= 10.0, r'High SNR ($\geq$10.0)', '-', '#1f77b4', 3, 'o'),
    ]

    plotted_any = False

    for mask, name, style, clr, zord, marker in plot_items:
        combined_mask = mask | (labels == 0)
        y_true = labels[combined_mask]
        y_score = probs[combined_mask]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        print(f"{name}: positives={n_pos}, negatives={n_neg}")

        if n_pos == 0 or n_neg == 0:
            print(f"Skip {name}: not enough class diversity for ROC.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)

        # log 坐标显示裁剪，避免 0 在 log 坐标下消失
        fpr_plot = np.clip(fpr, 1e-4, 1.0)
        tpr_plot = np.clip(tpr, 1e-2, 1.0)

        order = np.argsort(fpr_plot)
        fpr_plot = fpr_plot[order]
        tpr_plot = tpr_plot[order]

        if marker is None:
            plt.plot(
                fpr_plot, tpr_plot,
                label=name,
                linestyle=style,
                lw=3.2,
                color=clr,
                alpha=0.95,
                zorder=zord
            )
        else:
            plt.plot(
                fpr_plot, tpr_plot,
                label=name,
                linestyle=style,
                lw=3.8,
                color=clr,
                zorder=zord,
                marker=marker,
                markersize=4,
                markevery=max(1, len(fpr_plot) // 10)
            )

        plotted_any = True

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-4, 1.0)
    plt.ylim(1e-2, 1.15)

    plt.grid(True, which="both", alpha=0.3, linestyle=':')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Log-ROC Sensitivity: SIS')

    if plotted_any:
        plt.legend(loc='lower right', frameon=True, edgecolor='black')
    else:
        print("No valid ROC curve was plotted.")

    os.makedirs('pic', exist_ok=True)
    plt.tight_layout()
    plt.savefig('pic/fig9_sis_new_large.png', dpi=300)
    plt.show()

    print("Plot saved to pic/fig9_sis_new_large.png")


if __name__ == "__main__":
    plot_sis_log_roc()