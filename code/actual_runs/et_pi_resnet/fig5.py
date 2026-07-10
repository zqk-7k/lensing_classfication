# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =========================
# 全局绘图风格
# =========================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 28,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 2.0
})

# =========================
# 路径与参数
# =========================
DATA_ROOT = "/root/autodl-tmp/qkzhang"   # 改成你的根目录

SIS_DIR = os.path.join(DATA_ROOT, "SIS_data_0222")
PM_DIR  = os.path.join(DATA_ROOT, "PM_data_0222")
UNL_DIR = os.path.join(DATA_ROOT, "Unlensed_data_0222")

SEED = 42
VAL_START_FRAC = 0.8
VAL_END_FRAC   = 1.0
USE_VAL_SPLIT  = True   # True: 用验证集；False: 用全部数据

N_BINS_UNL = 22
N_BINS_LEN = 22

HIST_COLOR = "#4C92C3"
SAVE_PATH = "pic/snr_distribution_unlensed_vs_lensed_clean.png"


# =========================
# 工具函数
# =========================
def positive_only(x):
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x) & (x > 0)]


def get_split_indices(n, seed=42, start_frac=0.8, end_frac=1.0):
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    i0 = int(n * start_frac)
    i1 = int(n * end_frac)
    return idx[i0:i1]


def load_unlensed_snr(source_dir):
    """
    读取 unlensed 事件级 SNR
    """
    candidates = [
        "unlensed_optimal_SNR.npy",
        "unlensed_optimal_snr.npy",
        "optimal_SNR.npy",
        "optimal_snr.npy",
        "Unlensed_optimal_SNR.npy",
        "Unlensed_optimal_snr.npy",
    ]

    for name in candidates:
        p = os.path.join(source_dir, name)
        if os.path.exists(p):
            arr = np.load(p).reshape(-1)
            print(f"Using unlensed SNR file: {p}")
            return arr

    raise FileNotFoundError(
        f"Cannot find unlensed SNR file in {source_dir}.\nTried:\n" + "\n".join(candidates)
    )


def load_lensed_total_snr(source_dir, prefix):
    """
    读取透镜事件的两个分量 SNR，并合成为事件级 total SNR:
        total_snr = sqrt(snr1^2 + snr2^2)
    """
    normal_snr1 = os.path.join(source_dir, f"{prefix}_optimal_SNR_1.npy")
    normal_snr2 = os.path.join(source_dir, f"{prefix}_optimal_SNR_2.npy")

    test_snr1 = os.path.join(source_dir, f"test_{prefix}_optimal_SNR_1.npy")
    test_snr2 = os.path.join(source_dir, f"test_{prefix}_optimal_SNR_2.npy")

    if os.path.exists(normal_snr1) and os.path.exists(normal_snr2):
        snr1 = np.load(normal_snr1).reshape(-1)
        snr2 = np.load(normal_snr2).reshape(-1)
        total = np.sqrt(snr1**2 + snr2**2)
        print(f"Using normal lensed SNR files for {prefix}:")
        print(normal_snr1)
        print(normal_snr2)
        return total

    if os.path.exists(test_snr1) and os.path.exists(test_snr2):
        snr1 = np.load(test_snr1).reshape(-1)
        snr2 = np.load(test_snr2).reshape(-1)
        total = np.sqrt(snr1**2 + snr2**2)
        print(f"Using test lensed SNR files for {prefix}:")
        print(test_snr1)
        print(test_snr2)
        return total

    raise FileNotFoundError(
        f"Cannot find matched lensed SNR files for {prefix} in {source_dir}"
    )


def make_log_bins(data, n_bins=22):
    """
    根据当前数据单独生成 log 分箱
    """
    data = positive_only(data)
    dmin = data.min()
    dmax = data.max()

    # 给一点边界余量，但不要太大，避免空白过多
    left  = max(1.0, dmin * 0.90)
    right = dmax * 1.08

    bins = np.logspace(np.log10(left), np.log10(right), n_bins + 1)
    return bins, left, right


def style_log_axis(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(True, which="both", axis="y", linestyle='-', alpha=0.25)


# =========================
# 主流程
# =========================
def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 1. 读取原始数据
    unl_snr = positive_only(load_unlensed_snr(UNL_DIR))
    sis_total_snr = positive_only(load_lensed_total_snr(SIS_DIR, "SIS"))
    pm_total_snr  = positive_only(load_lensed_total_snr(PM_DIR, "PM"))

    # 2. 按和之前一致的 validation split 取数据
    if USE_VAL_SPLIT:
        idx_unl = get_split_indices(len(unl_snr), seed=SEED,
                                    start_frac=VAL_START_FRAC, end_frac=VAL_END_FRAC)
        idx_sis = get_split_indices(len(sis_total_snr), seed=SEED,
                                    start_frac=VAL_START_FRAC, end_frac=VAL_END_FRAC)
        idx_pm  = get_split_indices(len(pm_total_snr), seed=SEED,
                                    start_frac=VAL_START_FRAC, end_frac=VAL_END_FRAC)

        unl_plot = positive_only(unl_snr[idx_unl])
        sis_plot = positive_only(sis_total_snr[idx_sis])
        pm_plot  = positive_only(pm_total_snr[idx_pm])
    else:
        unl_plot = unl_snr
        sis_plot = sis_total_snr
        pm_plot  = pm_total_snr

    # 3. Lensed 合并 SIS + PM
    lensed_plot = positive_only(np.concatenate([sis_plot, pm_plot]))

    print("\nData summary:")
    print(f"Unlensed count = {len(unl_plot)}")
    print(f"SIS count      = {len(sis_plot)}")
    print(f"PM count       = {len(pm_plot)}")
    print(f"Lensed total   = {len(lensed_plot)}")

    # 4. 分别为两个子图生成自己的 bins 和范围（不共用 x 轴）
    bins_unl, xleft_unl, xright_unl = make_log_bins(unl_plot, n_bins=N_BINS_UNL)
    bins_len, xleft_len, xright_len = make_log_bins(lensed_plot, n_bins=N_BINS_LEN)

    # 5. 绘图
    fig, axes = plt.subplots(
        2, 1,
        figsize=(12.5, 9.0),
        sharex=False
    )

    # -------------------------
    # 上图：Unlensed
    # -------------------------
    ax = axes[0]
    ax.hist(
        unl_plot,
        bins=bins_unl,
        color=HIST_COLOR,
        alpha=0.88,
        edgecolor='white',
        linewidth=1.0
    )
    style_log_axis(ax)
    ax.set_xlim(xleft_unl, xright_unl)
    ax.set_title("Unlensed GW signals", pad=8)
    ax.set_ylabel("Count (Log Scale)")
    # 不给上图加 xlabel，避免和下图标题重叠
    ax.tick_params(axis='x', which='both', labelbottom=True)

    # -------------------------
    # 下图：Lensed = SIS + PM
    # -------------------------
    ax = axes[1]
    ax.hist(
        lensed_plot,
        bins=bins_len,
        color=HIST_COLOR,
        alpha=0.88,
        edgecolor='white',
        linewidth=1.0
    )
    style_log_axis(ax)
    ax.set_xlim(xleft_len, xright_len)
    ax.set_title("Lensed GW signals", pad=8)
    ax.set_ylabel("Count (Log Scale)")
    ax.set_xlabel("Optimal SNR")
    ax.tick_params(axis='x', which='both', labelbottom=True)

    # 6. 压缩空白，避免重叠
    plt.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.96,
        bottom=0.10,
        hspace=0.25
    )

    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.show()

    print(f"\nSaved figure to: {SAVE_PATH}")


if __name__ == "__main__":
    main()