# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.abspath(".."))

import model_classifier_ablation as model_lib
import data_classifier as data_lib
import config_classifier as cfg


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
    for name, child in module.named_children():
        if 'alpha' in child._parameters or hasattr(child, 'alpha'):
            setattr(module, name, nn.ReLU())
        else:
            force_convert_to_relu(child)


def load_original_sis_snr(source_dir, n_samples):
    """
    读取 SIS 原始 optimal SNR。

    优先使用：
    SIS_optimal_SNR_1.npy
    SIS_optimal_SNR_2.npy

    如果当前数据长度是 5000，并且普通 SNR 文件长度不匹配，
    再尝试使用 test_SIS_optimal_SNR_1.npy / test_SIS_optimal_SNR_2.npy。
    """
    normal_snr1 = os.path.join(source_dir, "SIS_optimal_SNR_1.npy")
    normal_snr2 = os.path.join(source_dir, "SIS_optimal_SNR_2.npy")

    test_snr1 = os.path.join(source_dir, "test_SIS_optimal_SNR_1.npy")
    test_snr2 = os.path.join(source_dir, "test_SIS_optimal_SNR_2.npy")

    if os.path.exists(normal_snr1) and os.path.exists(normal_snr2):
        snr1 = np.load(normal_snr1).reshape(-1)
        snr2 = np.load(normal_snr2).reshape(-1)

        if len(snr1) == n_samples and len(snr2) == n_samples:
            full_snr = np.sqrt(snr1 ** 2 + snr2 ** 2)
            print("\n========== 使用原始 SIS 合成 SNR ==========")
            print(f"SNR 文件 1: {normal_snr1}")
            print(f"SNR 文件 2: {normal_snr2}")
            return full_snr

        print(
            f"[Warning] 普通 SIS SNR 长度与数据长度不一致: "
            f"len(snr1)={len(snr1)}, len(snr2)={len(snr2)}, n_samples={n_samples}"
        )

    if os.path.exists(test_snr1) and os.path.exists(test_snr2):
        snr1 = np.load(test_snr1).reshape(-1)
        snr2 = np.load(test_snr2).reshape(-1)

        if len(snr1) == n_samples and len(snr2) == n_samples:
            full_snr = np.sqrt(snr1 ** 2 + snr2 ** 2)
            print("\n========== 使用 test SIS 合成 SNR ==========")
            print(f"SNR 文件 1: {test_snr1}")
            print(f"SNR 文件 2: {test_snr2}")
            return full_snr

        print(
            f"[Warning] test SIS SNR 长度与数据长度不一致: "
            f"len(snr1)={len(snr1)}, len(snr2)={len(snr2)}, n_samples={n_samples}"
        )

    raise FileNotFoundError("没有找到与当前 SIS_data_strain 数量匹配的 SNR 文件。")


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


def plot_sis_error_hist():
    run_dir = "./runs/SIS_noisy_OldData_Repro"
    model_p = os.path.join(run_dir, "best_classifier.pt")

    device = torch.device("cpu")

    # 1. 加载模型
    model = model_lib.BinaryPeriodicResNet1D_Ablation(
        d_model=256,
        width_scale=4.0
    ).to(device)

    force_convert_to_relu(model)

    state_dict = torch.load(model_p, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. 统一使用 0222 数据
    source_dir = "/root/autodl-tmp/qkzhang/SIS_data_0222"
    unlensed_dir = "/root/autodl-tmp/qkzhang/Unlensed_data_0222"

    l1 = data_lib.load_npy_data(
        os.path.join(source_dir, "SIS_data_strain_1.npy")
    )

    l2 = data_lib.load_npy_data(
        os.path.join(source_dir, "SIS_data_strain_2.npy")
    )

    unl = data_lib.load_npy_data(
        os.path.join(unlensed_dir, "unlensed_data_strain.npy")
    )

    n_samples = len(l1)

    # 3. 读取原始 SIS optimal SNR，并合成为总 SNR
    full_snr = load_original_sis_snr(source_dir, n_samples)

    print("\n========== 原始数据检查 ==========")
    print(f"source_dir: {source_dir}")
    print(f"unlensed_dir: {unlensed_dir}")
    print(f"SIS_data_strain_1 数量: {len(l1)}")
    print(f"SIS_data_strain_2 数量: {len(l2)}")
    print(f"Unlensed_data_strain 数量: {len(unl)}")
    print(f"合成 SNR 数组数量: {len(full_snr)}")

    if len(l1) != len(l2):
        print("[Warning] l1 和 l2 数量不一致，请检查 SIS 双像数据是否对应。")

    if len(full_snr) != len(l1):
        raise ValueError(
            f"SNR 数组长度 {len(full_snr)} 与 SIS 样本数量 {len(l1)} 不一致，"
            "不能直接使用 pair['idx1'] 映射。"
        )

    print_snr_stats("全量 SIS 合成 SNR", full_snr)

    # 4. 验证集划分：改回最后 10%
    indices = np.arange(n_samples)
    np.random.seed(cfg.SEED)
    np.random.shuffle(indices)

    idx_va = indices[int(n_samples * 0.8):int(n_samples * 0.9)]

    print("\n========== 验证集划分 ==========")
    print(f"总 SIS 样本数: {n_samples}")
    print(f"验证集 idx_va 数量: {len(idx_va)}")
    print(f"验证集比例: {len(idx_va) / n_samples:.2%}")

    print_snr_stats("当前 10% 验证集 SIS 合成 SNR", full_snr[idx_va])

    va_ds = data_lib.GWClassifierDataset(
        l1,
        l2,
        unl,
        idx_va,
        mode='val'
    )

    va_loader = torch.utils.data.DataLoader(
        va_ds,
        batch_size=128,
        shuffle=False
    )

    print("\n========== Dataset 检查 ==========")
    print(f"va_ds 实际样本数量: {len(va_ds)}")

    if hasattr(va_ds, "fixed_pairs"):
        print(f"va_ds.fixed_pairs 数量: {len(va_ds.fixed_pairs)}")
    else:
        raise AttributeError("va_ds 中没有 fixed_pairs 属性，无法提取 idx1 对应的 SNR。")

    # 5. 推理
    probs, labels, snrs = [], [], []

    for pair in va_ds.fixed_pairs:
        snrs.append(full_snr[pair['idx1']])

    print("\n正在为 SIS 模型计算预测分布...")

    with torch.no_grad():
        for inputs, targets in va_loader:
            inputs = inputs.to(device)

            out = model(inputs).squeeze(1)
            batch_probs = torch.sigmoid(out).cpu().numpy()

            probs.extend(batch_probs)
            labels.extend(targets.cpu().numpy())

    snrs = np.array(snrs)
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)

    print("\n========== 推理结果检查 ==========")
    print(f"probs 数量: {len(probs)}")
    print(f"labels 数量: {len(labels)}")
    print(f"snrs 数量: {len(snrs)}")

    if len(snrs) != len(labels):
        print("[Warning] snrs 和 labels 数量不一致，说明 fixed_pairs 与 DataLoader 返回顺序或数量可能不匹配。")

    label_counter = Counter(labels.astype(int))
    print(f"标签分布: {dict(label_counter)}")
    print(f"负样本数量 label=0: {label_counter.get(0, 0)}")
    print(f"正样本数量 label=1: {label_counter.get(1, 0)}")

    # 6. 只分析 lensed 正样本
    lensed_mask = (labels == 1)

    tp_mask = lensed_mask & (preds == 1)
    fn_mask = lensed_mask & (preds == 0)

    tp_snrs = snrs[tp_mask]
    fn_snrs = snrs[fn_mask]

    print("\n========== 最终画图数据 ==========")
    print(f"用于画图的 lensed 样本数量: {lensed_mask.sum()}")
    print(f"True Positives 数量: {len(tp_snrs)}")
    print(f"False Negatives 数量: {len(fn_snrs)}")

    if lensed_mask.sum() > 0:
        lensed_snrs = snrs[lensed_mask]
        print_snr_stats("实际画图 lensed SIS 合成 SNR", lensed_snrs)

    if len(tp_snrs) > 0:
        print_snr_stats("True Positives SIS 合成 SNR", tp_snrs)

    if len(fn_snrs) > 0:
        print_snr_stats("False Negatives SIS 合成 SNR", fn_snrs)

    # 额外导出 SNR≈100 的 FN，方便你分析
    fn_indices = np.where(fn_mask)[0]
    suspect_fn = [i for i in fn_indices if 90 <= snrs[i] <= 110]

    print("\n========== 可疑 FN 样本（90 <= SNR <= 110） ==========")
    print(f"数量: {len(suspect_fn)}")

    for i in suspect_fn:
        pair = va_ds.fixed_pairs[i]
        print(
            f"local_idx={i}, "
            f"idx1={pair['idx1']}, "
            f"idx2={pair.get('idx2', 'N/A')}, "
            f"snr={snrs[i]:.4f}, "
            f"prob={probs[i]:.4f}, "
            f"label={labels[i]}, "
            f"pred={preds[i]}"
        )

    # 7. 绘图
    plt.figure(figsize=(9, 7))

    if len(tp_snrs) + len(fn_snrs) > 0:
        all_plot_snrs = np.concatenate([tp_snrs, fn_snrs])
        positive_snrs = all_plot_snrs[all_plot_snrs > 0]

        if len(positive_snrs) == 0:
            raise ValueError("所有 SNR 都小于等于 0，无法使用 log 横轴绘图。")

        x_min = max(1.0, np.floor(positive_snrs.min()))
        x_max = np.ceil(positive_snrs.max() / 100.0) * 100.0

        if x_max <= x_min:
            x_max = x_min + 10.0

        bins = np.logspace(
            np.log10(x_min),
            np.log10(x_max),
            30
        )
    else:
        x_min = 1.0
        x_max = 10.0
        bins = np.logspace(np.log10(x_min), np.log10(x_max), 10)

    print("\n========== 绘图范围 ==========")
    print(f"横轴使用 log scale")
    print(f"x_min: {x_min:.4f}")
    print(f"x_max: {x_max:.4f}")
    print(f"bin 数量: {len(bins) - 1}")

    plt.hist(
        tp_snrs,
        bins=bins,
        alpha=0.7,
        label='True Positives',
        color='#1f77b4',
        edgecolor='white'
    )

    plt.hist(
        fn_snrs,
        bins=bins,
        alpha=0.9,
        label='False Negatives',
        color='#ff7f0e',
        hatch='///',
        edgecolor='white'
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim([x_min, x_max])
    plt.grid(True, axis='y', ls='--', alpha=0.3)

    plt.xlabel('Optimal Matched-filter SNR', fontweight='bold')
    plt.ylabel('Event Count (Log Scale)', fontweight='bold')
    plt.title('Error Attribution: SIS')

    plt.legend(
        loc='upper right',
        frameon=True,
        edgecolor='black'
    )

    os.makedirs('pic', exist_ok=True)
    plt.tight_layout()
    plt.savefig('pic/fig11_sis_hist_new.png', dpi=300)
    plt.show()

    print("\nSIS 误差分布图已生成: pic/fig11_sis_hist_new.png")


if __name__ == "__main__":
    plot_sis_error_hist()