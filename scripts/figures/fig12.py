# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

# 路径修复
sys.path.append(os.path.abspath(".."))

import model_classifier_ablation as model_lib
import data_classifier as data_lib
import config_classifier as cfg

# =========================
# 全局绘图风格
# =========================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'axes.linewidth': 2.2
})

# =========================
# 可调参数
# =========================
N_LOG_BINS = 14
MIN_BIN_COUNT = 3
USE_PERCENTILE_XMAX = True
XMAX_PERCENTILE = 99.0

THRESHOLD = 0.5
BATCH_SIZE = 128

# 关键：这里要和你前面所有图保持一致
VAL_START_FRAC = 0.8
VAL_END_FRAC = 1.0   # 如果你之前那张图用的是最后10%，改成 0.9

SAVE_PATH = 'pic/fig12_efficiency_sweep_consistent.png'

COLOR_SIS = '#1f77b4'
COLOR_PM  = '#d62728'


# =========================
# 工具函数
# =========================
def force_convert_to_relu(module):
    for name, child in module.named_children():
        if 'alpha' in child._parameters or hasattr(child, 'alpha'):
            setattr(module, name, nn.ReLU())
        else:
            force_convert_to_relu(child)


def load_original_lensed_snr(m_type, source_dir, n_samples):
    """
    读取原始 optimal SNR，并合成为总 SNR:
    total_snr = sqrt(snr1^2 + snr2^2)

    优先:
      {m_type}_optimal_SNR_1.npy
      {m_type}_optimal_SNR_2.npy

    否则尝试:
      test_{m_type}_optimal_SNR_1.npy
      test_{m_type}_optimal_SNR_2.npy
    """
    normal_snr1 = os.path.join(source_dir, f"{m_type}_optimal_SNR_1.npy")
    normal_snr2 = os.path.join(source_dir, f"{m_type}_optimal_SNR_2.npy")

    test_snr1 = os.path.join(source_dir, f"test_{m_type}_optimal_SNR_1.npy")
    test_snr2 = os.path.join(source_dir, f"test_{m_type}_optimal_SNR_2.npy")

    if os.path.exists(normal_snr1) and os.path.exists(normal_snr2):
        snr1 = np.load(normal_snr1).reshape(-1)
        snr2 = np.load(normal_snr2).reshape(-1)

        if len(snr1) == n_samples and len(snr2) == n_samples:
            full_snr = np.sqrt(snr1 ** 2 + snr2 ** 2)
            print(f"\n========== 使用原始 {m_type} 合成 SNR ==========")
            print(f"SNR 文件 1: {normal_snr1}")
            print(f"SNR 文件 2: {normal_snr2}")
            return full_snr

        print(
            f"[Warning] 普通 {m_type} SNR 长度与数据长度不一致: "
            f"len(snr1)={len(snr1)}, len(snr2)={len(snr2)}, n_samples={n_samples}"
        )

    if os.path.exists(test_snr1) and os.path.exists(test_snr2):
        snr1 = np.load(test_snr1).reshape(-1)
        snr2 = np.load(test_snr2).reshape(-1)

        if len(snr1) == n_samples and len(snr2) == n_samples:
            full_snr = np.sqrt(snr1 ** 2 + snr2 ** 2)
            print(f"\n========== 使用 test {m_type} 合成 SNR ==========")
            print(f"SNR 文件 1: {test_snr1}")
            print(f"SNR 文件 2: {test_snr2}")
            return full_snr

        print(
            f"[Warning] test {m_type} SNR 长度与数据长度不一致: "
            f"len(snr1)={len(snr1)}, len(snr2)={len(snr2)}, n_samples={n_samples}"
        )

    raise FileNotFoundError(
        f"没有找到与当前 {m_type}_data_strain 数量匹配的 SNR 文件。"
    )


def print_snr_stats(name, arr):
    arr = np.asarray(arr).reshape(-1)

    print(f"\n========== {name} ==========")
    print(f"数量: {len(arr)}")
    print(f"min : {arr.min():.6f}")
    print(f"p1  : {np.percentile(arr, 1):.6f}")
    print(f"p5  : {np.percentile(arr, 5):.6f}")
    print(f"p25 : {np.percentile(arr, 25):.6f}")
    print(f"p50 : {np.percentile(arr, 50):.6f}")
    print(f"p75 : {np.percentile(arr, 75):.6f}")
    print(f"p90 : {np.percentile(arr, 90):.6f}")
    print(f"p95 : {np.percentile(arr, 95):.6f}")
    print(f"p99 : {np.percentile(arr, 99):.6f}")
    print(f"max : {arr.max():.6f}")
    print(f"mean: {arr.mean():.6f}")
    print(f"> 5  数量: {np.sum(arr > 5)}")
    print(f"> 8  数量: {np.sum(arr > 8)}")
    print(f"> 10 数量: {np.sum(arr > 10)}")


def build_log_efficiency_bins(pos_snrs):
    pos_snrs = np.asarray(pos_snrs).reshape(-1)
    pos_snrs = pos_snrs[pos_snrs > 0]

    if len(pos_snrs) == 0:
        raise ValueError("正样本 SNR 全部小于等于 0，无法构造 log 分箱。")

    x_min = max(1.0, np.floor(np.percentile(pos_snrs, 1)))

    if USE_PERCENTILE_XMAX:
        x_max = np.ceil(np.percentile(pos_snrs, XMAX_PERCENTILE) / 50.0) * 50.0
    else:
        x_max = np.ceil(pos_snrs.max() / 100.0) * 100.0

    if x_max <= x_min:
        x_max = np.ceil(pos_snrs.max() / 10.0) * 10.0
    if x_max <= x_min:
        x_max = x_min + 10.0

    bins = np.logspace(np.log10(x_min), np.log10(x_max), N_LOG_BINS + 1)
    return bins, x_min, x_max


def get_split_indices(n_samples, seed, start_frac=0.8, end_frac=1.0):
    """
    单独抽出来，保证所有图都能共用同一套 split 逻辑
    """
    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    i0 = int(n_samples * start_frac)
    i1 = int(n_samples * end_frac)
    return indices[i0:i1]


def get_model_efficiency(m_type):
    run_dir = f"./runs/{m_type}_noisy_OldData_Repro"
    model_p = os.path.join(run_dir, "best_classifier.pt")
    device = torch.device("cpu")

    # 1. 模型
    model = model_lib.BinaryPeriodicResNet1D_Ablation(
        d_model=256,
        width_scale=4.0
    ).to(device)

    force_convert_to_relu(model)

    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    new_state_dict = {
        k.replace('module.', ''): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. 数据：保持和前面图完全一致
    source_dir = f"/root/autodl-tmp/qkzhang/{m_type}_data_0222"
    unlensed_dir = "/root/autodl-tmp/qkzhang/Unlensed_data_0222"

    l1 = data_lib.load_npy_data(
        os.path.join(source_dir, f"{m_type}_data_strain_1.npy")
    )
    l2 = data_lib.load_npy_data(
        os.path.join(source_dir, f"{m_type}_data_strain_2.npy")
    )
    unl = data_lib.load_npy_data(
        os.path.join(unlensed_dir, "unlensed_data_strain.npy")
    )

    n_samples = len(l1)

    # 3. 同样的 event-level total SNR
    full_snr = load_original_lensed_snr(m_type, source_dir, n_samples)

    print(f"\n========== {m_type} 原始数据检查 ==========")
    print(f"source_dir: {source_dir}")
    print(f"unlensed_dir: {unlensed_dir}")
    print(f"{m_type}_data_strain_1 数量: {len(l1)}")
    print(f"{m_type}_data_strain_2 数量: {len(l2)}")
    print(f"Unlensed_data_strain 数量: {len(unl)}")
    print(f"合成 SNR 数组数量: {len(full_snr)}")

    if len(l1) != len(l2):
        print(f"[Warning] {m_type} 的 l1 和 l2 数量不一致。")

    if len(full_snr) != len(l1):
        raise ValueError(
            f"{m_type} SNR 数组长度 {len(full_snr)} 与样本数量 {len(l1)} 不一致，"
            "不能直接使用 pair['idx1'] 映射。"
        )

    print_snr_stats(f"全量 {m_type} 合成 SNR", full_snr)

    # 4. 验证集划分：这里和前面图统一
    idx_va = get_split_indices(
        n_samples=n_samples,
        seed=cfg.SEED,
        start_frac=VAL_START_FRAC,
        end_frac=VAL_END_FRAC
    )

    print(f"\n========== {m_type} 验证集划分 ==========")
    print(f"总 {m_type} 样本数: {n_samples}")
    print(f"验证集 idx_va 数量: {len(idx_va)}")
    print(f"验证集比例: {len(idx_va) / n_samples:.2%}")
    print(f"split = [{VAL_START_FRAC:.2f}, {VAL_END_FRAC:.2f}]")

    print_snr_stats(f"当前验证集 {m_type} 合成 SNR", full_snr[idx_va])

    va_ds = data_lib.GWClassifierDataset(
        l1,
        l2,
        unl,
        idx_va,
        mode='val'
    )

    va_loader = torch.utils.data.DataLoader(
        va_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"\n========== {m_type} Dataset 检查 ==========")
    print(f"va_ds 实际样本数量: {len(va_ds)}")

    if hasattr(va_ds, "fixed_pairs"):
        print(f"va_ds.fixed_pairs 数量: {len(va_ds.fixed_pairs)}")
    else:
        raise AttributeError("va_ds 中没有 fixed_pairs 属性，无法提取对应 SNR。")

    # 5. 推理
    probs, labels, snrs = [], [], []

    # 和前面 error attribution 一样，用 pair['idx1'] 对应 event-level SNR
    for pair in va_ds.fixed_pairs:
        snrs.append(full_snr[pair['idx1']])

    print(f"\n正在提取 {m_type} 模型效率数据...")

    with torch.no_grad():
        for inputs, targets in va_loader:
            inputs = inputs.to(device)
            out = model(inputs).squeeze(1)

            probs.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    snrs = np.array(snrs)
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= THRESHOLD).astype(int)

    print(f"\n========== {m_type} 推理结果检查 ==========")
    print(f"probs 数量: {len(probs)}")
    print(f"labels 数量: {len(labels)}")
    print(f"snrs 数量: {len(snrs)}")
    print(f"preds 数量: {len(preds)}")

    if len(snrs) != len(labels):
        print("[Warning] snrs 和 labels 数量不一致，请检查 fixed_pairs 与 DataLoader 返回是否一一对应。")

    label_counter = Counter(labels.astype(int))
    print(f"标签分布: {dict(label_counter)}")
    print(f"负样本数量 label=0: {label_counter.get(0, 0)}")
    print(f"正样本数量 label=1: {label_counter.get(1, 0)}")

    # 6. 只对正样本算 TPR
    pos_mask = (labels == 1)

    pos_snrs = snrs[pos_mask]
    pos_probs = probs[pos_mask]

    print(f"\n========== {m_type} 正样本效率统计 ==========")
    print(f"用于效率曲线的正样本数量: {len(pos_snrs)}")

    if len(pos_snrs) == 0:
        raise ValueError(f"{m_type} 没有正样本，无法计算 TPR。")

    print_snr_stats(f"{m_type} 用于效率曲线的正样本 SNR", pos_snrs)

    # 7. log 分箱
    bins, x_min, x_max = build_log_efficiency_bins(pos_snrs)

    centers, tpr_list, bin_counts = [], [], []

    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]

        if i == len(bins) - 2:
            mask = (pos_snrs >= left) & (pos_snrs <= right)
        else:
            mask = (pos_snrs >= left) & (pos_snrs < right)

        count_i = int(np.sum(mask))

        if count_i >= MIN_BIN_COUNT:
            tpr = np.mean(pos_probs[mask] >= THRESHOLD)
            center = np.sqrt(left * right)

            centers.append(center)
            tpr_list.append(tpr)
            bin_counts.append(count_i)

    print(f"\n{m_type} log 分箱区间:")
    print(bins)

    for c, tpr, cnt in zip(centers, tpr_list, bin_counts):
        print(f"SNR bin center={c:.2f}, 样本数={cnt}, TPR={tpr:.4f}")

    if len(centers) == 0:
        raise ValueError(
            f"{m_type} 所有 SNR bin 的样本数都小于 MIN_BIN_COUNT={MIN_BIN_COUNT}，"
            "请降低 MIN_BIN_COUNT 或减少 N_LOG_BINS。"
        )

    return (
        np.array(centers),
        np.array(tpr_list),
        np.array(bin_counts),
        x_min,
        x_max
    )


def pretty_y_limits(*eff_arrays):
    vals = np.concatenate(eff_arrays)

    y_min_raw = vals.min()
    y_max_raw = vals.max()

    y_min = max(0.4, np.floor((y_min_raw - 0.03) / 0.05) * 0.05)
    y_max = min(1.02, np.ceil((y_max_raw + 0.01) / 0.05) * 0.05)

    if y_max < 1.01:
        y_max = 1.01

    return y_min, y_max


def plot_efficiency_curve(c_sis, e_sis, c_pm, e_pm, xmin_sis, xmax_sis, xmin_pm, xmax_pm):
    plot_xmin = min(xmin_sis, xmin_pm)
    plot_xmax = max(xmax_sis, xmax_pm)

    # 稍微留一点边
    plot_xmin *= 0.95
    plot_xmax *= 1.05

    y_min, y_max = pretty_y_limits(e_sis, e_pm)

    fig, ax = plt.subplots(figsize=(11.5, 7.3))

    ax.plot(
        c_sis, e_sis,
        label='SIS Model',
        color=COLOR_SIS,
        marker='o',
        lw=3.2,
        ms=10,
        mec=COLOR_SIS,
        mfc=COLOR_SIS
    )

    ax.plot(
        c_pm, e_pm,
        label='PM Model',
        color=COLOR_PM,
        marker='s',
        lw=3.2,
        ms=9.5,
        mec=COLOR_PM,
        mfc=COLOR_PM
    )

    ax.set_xscale('log')
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Optimal Matched-filter SNR', fontweight='bold')
    ax.set_ylabel('Detection Efficiency (TPR)', fontweight='bold')
    ax.set_title('Efficiency Sweep: Signal Strength Analysis')

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

    ax.grid(True, which="major", alpha=0.28, linestyle=':')
    ax.grid(True, which="minor", axis='x', alpha=0.15, linestyle=':')

    leg = ax.legend(loc='lower right', frameon=True, edgecolor='black')
    leg.get_frame().set_linewidth(1.0)

    os.makedirs('pic', exist_ok=True)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.show()

    print(f"\n效率曲线图已生成: {SAVE_PATH}")


if __name__ == "__main__":
    print("正在提取 SIS 模型效率数据...")
    c_sis, e_sis, n_sis, xmin_sis, xmax_sis = get_model_efficiency("SIS")

    print("\n正在提取 PM 模型效率数据...")
    c_pm, e_pm, n_pm, xmin_pm, xmax_pm = get_model_efficiency("PM")

    print("\n========== 最终绘图范围 ==========")
    print("横轴使用 log scale")
    print(f"SIS: xmin={xmin_sis:.4f}, xmax={xmax_sis:.4f}")
    print(f"PM : xmin={xmin_pm:.4f}, xmax={xmax_pm:.4f}")
    print(f"Validation split = [{VAL_START_FRAC:.2f}, {VAL_END_FRAC:.2f}]")
    print("SNR definition = event-level total SNR = sqrt(SNR1^2 + SNR2^2)")

    plot_efficiency_curve(
        c_sis, e_sis,
        c_pm, e_pm,
        xmin_sis, xmax_sis,
        xmin_pm, xmax_pm
    )