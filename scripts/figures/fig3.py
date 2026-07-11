# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.signal.windows import tukey

sys.path.append(os.path.abspath(".."))

import data_classifier as data_lib
import config_classifier as cfg


# =========================
# Global style
# =========================
ACADEMIC_BLUE = '#2e81b7'

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'font.family': 'serif'
})


# =========================
# Config
# =========================
M_TYPE = "SIS"          # 可改成 "PM"
fs = 2048.0             # 采样率
USE_VAL_20_PERCENT = True

# 样本选择方式：
# "max_snr"：从当前 20% 验证集中选 SNR 最大的样本
# "median_snr"：选接近中位数 SNR 的样本
# "first"：选验证集第一个样本
# 也可以直接填整数，例如 SAMPLE_MODE = 123
SAMPLE_MODE = "max_snr"

# 和你参考图保持一致：峰值前 0.2 s 到峰值后 0.05 s
offset_start = -0.2
offset_end = 0.05

# 展示用白化频段
F_LOW = 20.0
F_HIGH = 512.0

SAVE_DIR = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)


def to_1d(x):
    x = np.asarray(x).squeeze()

    if x.ndim == 1:
        return x.astype(np.float64)

    if x.ndim == 2:
        if 1 in x.shape:
            return x.reshape(-1).astype(np.float64)

        if x.shape[0] < x.shape[1]:
            return x[0].astype(np.float64)
        else:
            return x[:, 0].astype(np.float64)

    raise ValueError(f"无法转换为一维 strain，当前 shape={x.shape}")


def load_snr_if_available(m_type, source_dir):
    """
    优先读取当前数据目录下的 optimal SNR。
    如果有 single_1 和 single_2，则使用 sqrt(snr1^2 + snr2^2) 合成。
    找不到时，退回 runs/.../sample_snrs.npy。
    """
    if not os.path.isdir(source_dir):
        return None, "source_dir not found"

    files = os.listdir(source_dir)
    lower_map = {fn.lower(): fn for fn in files}

    key1_list = [
        f"{m_type.lower()}_optimal_snr_single_1.npy",
        f"{m_type.lower()}_optimal_snr_1.npy",
        f"{m_type.lower()}_snr_single_1.npy",
    ]

    key2_list = [
        f"{m_type.lower()}_optimal_snr_single_2.npy",
        f"{m_type.lower()}_optimal_snr_2.npy",
        f"{m_type.lower()}_snr_single_2.npy",
    ]

    snr1_path = None
    snr2_path = None

    for key in key1_list:
        if key in lower_map:
            snr1_path = os.path.join(source_dir, lower_map[key])
            break

    for key in key2_list:
        if key in lower_map:
            snr2_path = os.path.join(source_dir, lower_map[key])
            break

    if snr1_path is not None and snr2_path is not None:
        snr1 = np.load(snr1_path).reshape(-1)
        snr2 = np.load(snr2_path).reshape(-1)

        n = min(len(snr1), len(snr2))
        snr_total = np.sqrt(snr1[:n] ** 2 + snr2[:n] ** 2)

        return snr_total, f"combined: {os.path.basename(snr1_path)} + {os.path.basename(snr2_path)}"

    snr_candidates = []
    for fn in files:
        low = fn.lower()
        if fn.endswith(".npy") and "snr" in low:
            snr_candidates.append(fn)

    if len(snr_candidates) > 0:
        snr_candidates = sorted(
            snr_candidates,
            key=lambda x: 0 if "optimal" in x.lower() else 1
        )

        p = os.path.join(source_dir, snr_candidates[0])
        return np.load(p).reshape(-1), f"source_dir: {snr_candidates[0]}"

    run_snr_path = os.path.join(
        "./runs",
        f"{m_type}_noisy_OldData_Repro",
        "sample_snrs.npy"
    )

    if os.path.exists(run_snr_path):
        return np.load(run_snr_path).reshape(-1), f"runs: {run_snr_path}"

    return None, "no snr file found"


def whiten_strain_for_display(strain, fs, f_low=20.0, f_high=512.0):
    """
    当前 .npy strain 的展示用白化。

    注意：
    这里不是 bilby ifo.whitened_time_domain_strain 的严格复现，
    因为当前数据文件里没有 bilby interferometer 对象和原始 PSD 元数据。
    这里使用 Welch 从当前 strain 估计 PSD，然后频域除以 sqrt(PSD)。
    """
    x = np.asarray(strain, dtype=np.float64)
    x = x - np.mean(x)

    n = len(x)
    dt = 1.0 / fs

    window = tukey(n, alpha=0.1)
    x_win = x * window

    nperseg = min(n, int(4 * fs))
    nperseg = max(256, nperseg)
    nperseg = min(n, nperseg)

    freqs_psd, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant"
    )

    freqs = np.fft.rfftfreq(n, d=dt)
    psd_interp = np.interp(freqs, freqs_psd, psd)

    positive_psd = psd_interp[psd_interp > 0]
    if len(positive_psd) == 0:
        raise ValueError("PSD 全部为 0，无法进行白化。")

    eps = np.median(positive_psd) * 1e-12
    psd_interp = np.maximum(psd_interp, eps)

    hf = np.fft.rfft(x_win)
    white_hf = hf / np.sqrt(psd_interp)

    band_mask = (freqs >= f_low) & (freqs <= f_high)
    white_hf[~band_mask] = 0.0

    white = np.fft.irfft(white_hf, n=n)

    std = np.std(white)
    if std > 0:
        white = white / std

    return white


def choose_sample_index(indices, snr_arr=None, mode="max_snr"):
    if isinstance(mode, int):
        return int(mode)

    if mode == "first":
        return int(indices[0])

    if snr_arr is None:
        print("[Warning] 未找到 SNR 文件，默认选择第一个样本。")
        return int(indices[0])

    valid_indices = indices[indices < len(snr_arr)]

    if len(valid_indices) == 0:
        print("[Warning] 验证集索引超出 SNR 数组范围，默认选择第一个样本。")
        return int(indices[0])

    valid_snrs = snr_arr[valid_indices]

    if mode == "max_snr":
        return int(valid_indices[np.argmax(valid_snrs)])

    if mode == "median_snr":
        median_snr = np.median(valid_snrs)
        return int(valid_indices[np.argmin(np.abs(valid_snrs - median_snr))])

    raise ValueError(f"未知 SAMPLE_MODE: {mode}")


def get_peak_zoom_indices(d_raw, h_raw, fs, offset_start=-0.2, offset_end=0.05):
    """
    当前数据没有 trigger_time，所以改为自动找峰值。
    为了保持参考图的局部放大效果，这里以 Data 和 Template 中更强的峰值作为中心。
    """
    score = np.maximum(np.abs(d_raw), np.abs(h_raw))
    peak_idx = int(np.argmax(score))

    idx_start = peak_idx + int(offset_start * fs)
    idx_end = peak_idx + int(offset_end * fs)

    idx_start = max(0, idx_start)
    idx_end = min(len(d_raw), idx_end)

    return idx_start, idx_end, peak_idx


def make_fig3_zoomed_current_data(m_type="SIS"):
    print(f"Generating Zoomed Figure 3 style plots for current {m_type} data...")

    source_dir = os.path.join(cfg.DATA_ROOT, f"{m_type}_data_0222")

    l1_path = os.path.join(source_dir, f"{m_type}_data_strain_1.npy")
    l2_path = os.path.join(source_dir, f"{m_type}_data_strain_2.npy")

    l1 = data_lib.load_npy_data(l1_path)
    l2 = data_lib.load_npy_data(l2_path)

    n_samples = len(l1)

    print("\n========== Data Check ==========")
    print(f"source_dir: {source_dir}")
    print(f"{m_type}_data_strain_1: {len(l1)}")
    print(f"{m_type}_data_strain_2: {len(l2)}")

    snr_arr, snr_source = load_snr_if_available(m_type, source_dir)

    print("\n========== SNR Check ==========")
    print(f"SNR source: {snr_source}")

    if snr_arr is not None:
        print(f"SNR count: {len(snr_arr)}")
        print(f"SNR min : {np.min(snr_arr):.6f}")
        print(f"SNR p50 : {np.percentile(snr_arr, 50):.6f}")
        print(f"SNR p95 : {np.percentile(snr_arr, 95):.6f}")
        print(f"SNR max : {np.max(snr_arr):.6f}")

    indices = np.arange(n_samples)
    np.random.seed(cfg.SEED)
    np.random.shuffle(indices)

    if USE_VAL_20_PERCENT:
        sample_pool = indices[int(n_samples * 0.8):]
        print(f"\nUsing last 20% shuffled samples: {len(sample_pool)}")
    else:
        sample_pool = indices
        print(f"\nUsing all samples: {len(sample_pool)}")

    sample_idx = choose_sample_index(sample_pool, snr_arr, SAMPLE_MODE)

    print("\n========== Selected Sample ==========")
    print(f"m_type: {m_type}")
    print(f"sample_idx: {sample_idx}")

    if snr_arr is not None and sample_idx < len(snr_arr):
        print(f"sample SNR: {snr_arr[sample_idx]:.6f}")

    # 这里为了保持参考图样式：
    # strain_1 作为 Data
    # strain_2 作为 Template
    d_raw = to_1d(l1[sample_idx])
    h_raw = to_1d(l2[sample_idx])

    if len(d_raw) != len(h_raw):
        min_len = min(len(d_raw), len(h_raw))
        d_raw = d_raw[:min_len]
        h_raw = h_raw[:min_len]
        print(f"[Warning] d_raw 和 h_raw 长度不一致，已截断到 {min_len}")

    duration = len(d_raw) / fs

    print(f"strain length: {len(d_raw)}")
    print(f"duration: {duration:.3f} s")
    print(f"fs: {fs:.1f} Hz")

    # 白化
    d_whitened = whiten_strain_for_display(d_raw, fs, F_LOW, F_HIGH)
    h_whitened = whiten_strain_for_display(h_raw, fs, F_LOW, F_HIGH)

    # 自动峰值切片
    idx_start, idx_end, peak_idx = get_peak_zoom_indices(
        d_raw,
        h_raw,
        fs,
        offset_start=offset_start,
        offset_end=offset_end
    )

    t = np.arange(len(d_raw)) / fs

    # 当前数据没有真实 trigger_time，这里使用相对峰值时间，更适合当前数据。
    t_zoom = t[idx_start:idx_end] - t[peak_idx]

    d_raw_zoom = d_raw[idx_start:idx_end]
    h_raw_zoom = h_raw[idx_start:idx_end]

    d_whitened_zoom = d_whitened[idx_start:idx_end]
    h_whitened_zoom = h_whitened[idx_start:idx_end]

    print("\n========== Zoom Window ==========")
    print(f"peak_idx: {peak_idx}")
    print(f"zoom start: {t_zoom[0]:.6f} s")
    print(f"zoom end  : {t_zoom[-1]:.6f} s")

    # =========================
    # Figure A: Before Whitening
    # =========================
    plt.figure(figsize=(12, 6))

    plt.plot(
        t_zoom,
        d_raw_zoom,
        color='gray',
        alpha=0.5,
        label='Data (Raw)'
    )

    plt.plot(
        t_zoom,
        h_raw_zoom,
        color=ACADEMIC_BLUE,
        linewidth=2.5,
        label='Template'
    )

    plt.ylabel('Strain', fontweight='bold')
    plt.xlabel('Time (s)', fontweight='bold')
    plt.xlim(t_zoom[0], t_zoom[-1])
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    before_path = os.path.join(
        SAVE_DIR,
        f"Figure3a_{m_type}_Before_Zoom_current_idx{sample_idx}.jpg"
    )

    plt.savefig(before_path, dpi=300)
    plt.close()

    # =========================
    # Figure B: After Whitening
    # =========================
    plt.figure(figsize=(12, 6))

    plt.plot(
        t_zoom,
        d_whitened_zoom,
        color='black',
        alpha=0.3,
        label='Data (Whitened)'
    )

    plt.plot(
        t_zoom,
        h_whitened_zoom,
        color=ACADEMIC_BLUE,
        linewidth=2.5,
        label='Template'
    )

    plt.ylabel('Whitened Strain', fontweight='bold')
    plt.xlabel('Time (s)', fontweight='bold')
    plt.xlim(t_zoom[0], t_zoom[-1])
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()

    after_path = os.path.join(
        SAVE_DIR,
        f"Figure3b_{m_type}_After_Zoom_current_idx{sample_idx}.jpg"
    )

    plt.savefig(after_path, dpi=300)
    plt.close()

    print(f"\nSaved: {before_path}")
    print(f"Saved: {after_path}")


if __name__ == "__main__":
    make_fig3_zoomed_current_data(M_TYPE)