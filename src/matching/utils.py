# utils.py
import numpy as np
import math
import os
import networkx as nx
from openpyxl import Workbook
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

def pad_or_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    n = x.shape[0]
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    y = np.zeros((target_len,), dtype=x.dtype)
    y[:n] = x
    return y

# --- 数学与匹配 ---
def gcc_phat_corr(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0, 0
    n = max(x.size, y.size)
    if x.size != n: x = pad_or_trim(x, n)
    if y.size != n: y = pad_or_trim(y, n)
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    nlin = 2*n - 1
    nfft = 1
    while nfft < nlin:
        nfft <<= 1
    Xf = np.fft.rfft(x, nfft)
    Yf = np.fft.rfft(y, nfft)
    R = Xf * np.conj(Yf)
    denom = np.abs(R)
    R = np.where(denom > 1e-12, R/denom, 0.0)
    r = np.fft.irfft(R, nfft)
    r_lin = np.concatenate([r[-(n-1):], r[:n]])
    k = int(np.argmax(np.abs(r_lin)))
    best = float(np.abs(r_lin[k]) / n)
    lag = k - (n - 1)
    return best, lag


def refine_scores(X, S0, topk=50):
    N = len(X)
    S = np.zeros((N, N), dtype=np.float32)
    L = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        k = min(topk, N)
        idx = np.argpartition(-S0[i], k-1)[:k]
        idx = idx[idx != i]
        for j in idx:
            c, lag = gcc_phat_corr(X[i], X[j])
            S[i, j] = c; L[i, j] = lag
    return S, L


def refine_scores_fast(X_list, S0, topk=50):
    N = len(X_list)
    # 1. 统一 Padding 长度并转为 Tensor 或高性能 Numpy
    T = max(len(x) for x in X_list)
    nfft = 1 << (2 * T - 1).bit_length()

    # 2. 预计算所有序列的 FFT (使用 rfft)
    X_f_all = []
    print(f"[SpeedUp] Pre-calculating FFT for {N} samples...")
    for x in X_list:
        # 归一化
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
        X_f_all.append(np.fft.rfft(x_norm, nfft))
    X_f_all = np.array(X_f_all)  # [N, nfft_half]

    S = np.zeros((N, N), dtype=np.float32)
    L = np.zeros((N, N), dtype=np.int32)

    # 3. 仅在 Top-K 对中进行频域乘法
    for i in range(N):
        k = min(topk, N)
        idx = np.argpartition(-S0[i], k - 1)[:k]
        idx = idx[idx != i]

        Xi_f = X_f_all[i]
        for j in idx:
            # 频域交叉相乘
            R = Xi_f * np.conj(X_f_all[j])
            denom = np.abs(R)
            R_phase = np.where(denom > 1e-12, R / denom, 0.0)

            # 逆 FFT 得到相关函数
            r = np.fft.irfft(R_phase, nfft)
            # 查找最大值 (等同于原 gcc_phat_corr 逻辑)
            # 注意：只需处理线性卷积对应的部分
            r_max_idx = np.argmax(np.abs(r))
            S[i, j] = np.abs(r[r_max_idx]) / T
            # 计算 lag (需处理 FFT shift)
            lag = r_max_idx if r_max_idx < nfft // 2 else r_max_idx - nfft
            L[i, j] = lag

    return S, L


def apply_time_delay_constraint(S, t_starts, max_dt):
    """
    在相似度矩阵上应用时间延迟约束。
    如果 |t_i - t_j| > max_dt，则认为不可能匹配，将 S[i,j] 置为 -1。

    S: 相似度矩阵 (N, N)
    t_starts: 每个样本的起始 GPS 时间 (N,)
    max_dt: 最大允许时间差 (秒)
    """
    print(f"[Constraint] Applying Time Delay Filter (Max DT = {max_dt / 3600 / 24:.1f} days)...")

    # 1. 计算时间差矩阵 (利用广播机制)
    # shape: (N, N)
    # dt_matrix[i, j] = |t[i] - t[j]|
    dt_matrix = np.abs(t_starts[:, None] - t_starts[None, :])

    # 2. 创建掩码：时间差超标的地方为 True
    mask = dt_matrix > max_dt

    # 3. 统计被过滤掉的高分对子 (仅用于调试观察)
    # 比如原来相似度 > 0.5 但因为时间不对被杀掉的数量
    n_filtered = np.sum((S > 0.5) & mask) / 2  # 除以2是因为矩阵对称
    print(f"  -> Filtered out {int(n_filtered)} candidate pairs due to time delay.")

    # 4. 修改相似度矩阵
    # 将不满足条件的置为 -1 (确保匹配算法绝不选中)
    S_filtered = S.copy()
    S_filtered[mask] = -1.0

    return S_filtered

# utils.py 追加或替换相关函数

import torch

# utils.py 补充内容
import torch


def refine_scores_gpu(X_list, S0, topk=50, device='cuda'):
    """
    使用 GPU 加速的批量 GCC-PHAT。

    参数:
        X_list: 信号列表 (list of numpy arrays)
        S0: 粗匹配的相似度矩阵 (N x N)
        topk: 每行处理的前 K 个最相似对
        device: 'cuda' 或 'cpu'
    """
    N = len(X_list)
    # 计算 FFT 所需长度
    T = max(len(x) for x in X_list)
    nfft = 1 << (2 * T - 1).bit_length()

    # 1. 数据准备：Pad 并搬运到 GPU
    X_padded = np.zeros((N, T), dtype=np.float32)
    for i, x in enumerate(X_list):
        # 确保长度一致，不足补0
        cur_len = min(len(x), T)
        X_padded[i, :cur_len] = x[:cur_len]

    X_tensor = torch.from_numpy(X_padded).to(device)

    # 2. 归一化 (减均值，除标准差)
    X_mean = X_tensor.mean(dim=1, keepdim=True)
    X_std = X_tensor.std(dim=1, keepdim=True) + 1e-8
    X_tensor = (X_tensor - X_mean) / X_std

    # 3. 预计算所有信号的 FFT
    X_f = torch.fft.rfft(X_tensor, n=nfft)  # [N, nfft_half]

    S_ref = np.zeros((N, N), dtype=np.float32)
    L_ref = np.zeros((N, N), dtype=np.int32)

    # 4. 批量处理 Top-K 对
    for i in range(N):
        # 选取 Top-K 索引 (排除自身)
        k_indices = np.argpartition(-S0[i], topk)[:topk]
        k_indices = k_indices[k_indices != i]

        # 频谱交叉相乘: R = Xi_f * conj(Xj_f)
        Xi_f = X_f[i:i + 1]  # [1, nfft_half]
        Xj_f = X_f[k_indices]  # [K-1, nfft_half]
        R = Xi_f * torch.conj(Xj_f)

        # PHAT 加权: R / |R|
        denom = torch.abs(R)
        R_phase = torch.where(denom > 1e-12, R / denom, torch.zeros_like(R))

        # 逆 FFT 得到互相关曲线
        r = torch.fft.irfft(R_phase, n=nfft)  # [K-1, nfft]

        # 寻找峰值和滞后 (Lag)
        vals, idxs = torch.max(torch.abs(r), dim=1)

        # 处理滞后时间: 对应原代码中的 r_lin 拼接逻辑
        lags = torch.where(idxs < nfft // 2, idxs, idxs - nfft)

        # 存回 CPU
        S_ref[i, k_indices] = (vals / T).cpu().numpy()
        L_ref[i, k_indices] = lags.cpu().numpy()

    return S_ref, L_ref


def fuse_scores(S_emb, S_gcc, alpha=0.5):
    """
    分数融合：结合 Embedding 的语义信息和 GCC 的波形对齐信息
    alpha 为 Embedding 分数的权重
    """

    # 归一化到 0-1
    def norm(s):
        m = s.max()
        return s / (m + 1e-12) if m > 0 else s

    S_total = alpha * norm(S_emb) + (1 - alpha) * norm(S_gcc)
    # 只保留双向都有值的部分（可选）
    return S_total

def auto_threshold(best_scores: np.ndarray) -> float:
    best_scores = np.asarray(best_scores, dtype=np.float64)
    thr = None
    try:
        from sklearn.mixture import GaussianMixture
        x = best_scores.reshape(-1,1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(x)
        m = gmm.means_.ravel(); v = gmm.covariances_.ravel(); w = gmm.weights_.ravel()
        mu1, mu2 = m[0], m[1]; v1, v2 = v[0], v[1]; w1, w2 = w[0], w[1]
        A = 1/(2*v2) - 1/(2*v1)
        B = mu1/v1 - mu2/v2
        C = (mu2*mu2)/(2*v2) - (mu1*mu1)/(2*v1) + math.log((w2*math.sqrt(v1))/(w1*math.sqrt(v2)) + 1e-12)
        roots = []
        if abs(A) < 1e-12:
            if abs(B) > 1e-12:
                roots = [-C/B]
        else:
            disc = B*B - 4*A*C
            if disc >= 0:
                r1 = (-B - math.sqrt(disc)) / (2*A)
                r2 = (-B + math.sqrt(disc)) / (2*A)
                roots = [r1, r2]
        lo, hi = float(best_scores.min()), float(best_scores.max())
        cand = [r for r in roots if lo-1e-6 <= r <= hi+1e-6 and min(mu1,mu2) <= r <= max(mu1,mu2)]
        thr = float(cand[0]) if cand else float((mu1+mu2)/2)
    except Exception:
        x = np.clip(best_scores, 0.0, 1.0)
        hist, edges = np.histogram(x, bins=256, range=(0,1))
        p = hist.astype(np.float64); p /= (p.sum()+1e-12)
        omega = np.cumsum(p)
        centers = (edges[:-1] + edges[1:]) / 2.0
        mu = np.cumsum(p * centers)
        mu_t = mu[-1]
        sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1-omega) + 1e-12)
        k = int(np.argmax(sigma_b2))
        thr = float(centers[k])
    return thr


def build_edges_from_scores(S: np.ndarray, thr: float):
    N = S.shape[0]
    best_j = np.argmax(S, axis=1)
    best_s = S[np.arange(N), best_j]
    rev_best_i = np.argmax(S, axis=0)
    edge_best = {}
    for i in range(N):
        j = int(best_j[i]); s = float(best_s[i])
        if s >= thr and rev_best_i[j] == i and i != j:
            a, b = (i, j) if i < j else (j, i)
            k = (a, b)
            if (k not in edge_best) or (s > edge_best[k]):
                edge_best[k] = s
    edges = [(a,b,s) for (a,b),s in edge_best.items()]
    return edges, best_s


def max_weight_matching(edges, N):
    try:
        import networkx as nx
        G = nx.Graph(); G.add_nodes_from(range(N))
        for a,b,s in edges:
            G.add_edge(a,b,weight=float(s))
        M = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight='weight')
        pairs = [(min(u,v), max(u,v)) for (u,v) in M]
        used = set([i for p in pairs for i in p])
        unmatched = [i for i in range(N) if i not in used]
        return pairs, unmatched
    except Exception:
        items = sorted(edges, key=lambda e: e[2], reverse=True)
        used = np.zeros(N, dtype=bool)
        pairs = []
        for a,b,s in items:
            if (not used[a]) and (not used[b]):
                pairs.append((a,b)); used[a]=True; used[b]=True
        unmatched = np.where(~used)[0].tolist()
        return pairs, unmatched



def build_ground_truth(meta_all):
    N = len(meta_all)
    gt = np.full(N, -1, dtype=int)
    pos = {}
    for i, m in enumerate(meta_all):
        # 使用 unique_id 更安全，或者 tag+idx 组合
        pos[(m["tag"], m["idx_in_file"])] = i

    for i, m in enumerate(meta_all):
        target_tag = "L2" if m["tag"] == "L1" else "L1"
        if m["tag"] in ["L1", "L2"]:
            key = (target_tag, m["idx_in_file"])
            if key in pos: gt[i] = pos[key]
    return gt


# --- 导出 ---
def write_summary_xlsx(file_path: str, summary: dict):
    from openpyxl import Workbook
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    wb = Workbook(); ws = wb.active; ws.title = "summary"
    ws.append(["key", "value"])
    for k, v in summary.items():
        ws.append([str(k), str(v)])
    wb.save(file_path)


def write_pairs_detailed_xlsx(file_path: str, sheet_name: str, rows: list):
    from openpyxl import Workbook
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    wb = Workbook(); ws = wb.active; ws.title = sheet_name
    ws.append(["i","j","tag_i","idx_i","tag_j","idx_j","score","lag","is_true","pair_type"])
    for r in rows:
        ws.append([
            int(r["i"]), int(r["j"]), r["tag_i"], int(r["idx_i"]), r["tag_j"], int(r["idx_j"]),
            float(r["score"]) if r["score"] is not None else None,
            int(r["lag"]) if r.get("lag") is not None else None,
            int(r["is_true"]), r["pair_type"],
        ])
    wb.save(file_path)


# --- 绘图 (封装为一个类或大函数) ---
def plot_all_figures(OUT_DIR, S0, S_ref, meta_te, gt_partner, Z):
    """
    将原代码末尾那一堆 matplotlib 代码封装到这里。
    参数传入需要的矩阵和元数据。
    """
    FIG_DIR = os.path.join(OUT_DIR, "figs")
    os.makedirs(FIG_DIR, exist_ok=True)

    # ... 在这里粘贴所有 plt 相关的代码 ...
    print(f"[figs] Figures saved to {FIG_DIR}")


# utils.py (追加或修改以下内容)

def split_tp_fp(pairs, gt_partner):
    """区分真阳性 (TP) 和 假阳性 (FP)"""
    tp, fp = [], []
    for (i, j) in pairs:
        # 双方互为真值伙伴才算 TP
        if gt_partner[i] == j and gt_partner[j] == i:
            tp.append((i, j))
        else:
            fp.append((i, j))
    return tp, fp


def evaluate_and_print(pairs, name, gt_partner, N_true_pairs, meta_te, S_used, USE_GCC, L_matrix=None):
    """计算指标，打印日志，并返回详细行用于导出"""
    tp, fp = split_tp_fp(pairs, gt_partner)
    P = len(pairs)

    # 防止除零错误
    precision = len(tp) / P if P > 0 else 0.0
    recall = len(tp) / N_true_pairs if N_true_pairs > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # === 关键：这里就是你要的打印输出 ===
    print(
        f"[{name}] pairs={P} | TP={len(tp)} | FP={len(fp)} | precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")

    # 生成详细数据用于 Excel
    rows = []
    for (i, j) in pairs:
        rows.append({
            "i": i, "j": j,
            "tag_i": meta_te[i]["tag"], "idx_i": meta_te[i]["idx_in_file"],
            "tag_j": meta_te[j]["tag"], "idx_j": meta_te[j]["idx_in_file"],
            "score": float(S_used[i, j]),
            "lag": int(L_matrix[i, j]) if (USE_GCC and L_matrix is not None) else None,
            "is_true": 1 if (gt_partner[i] == j and gt_partner[j] == i) else 0,
            "pair_type": f"{min(meta_te[i]['tag'], meta_te[j]['tag'])}-{max(meta_te[i]['tag'], meta_te[j]['tag'])}",
        })

    stats = dict(P=P, TP=len(tp), FP=len(fp), precision=precision, recall=recall, f1=f1)
    return stats, rows


import csv


def analyze_and_save_dist(pairs, S_matrix, gt_partner, output_path):
    """
    通用分布分析函数：
    1. 统计 TP/FP 的分数分布 (Min/Max/Avg)
    2. 自动推荐最佳阈值
    3. 将所有配对按分数降序保存到 CSV 文件

    参数:
        pairs: 配对列表 [(i, j), ...]
        S_matrix: 相似度矩阵 (S0 或 S_ref)
        gt_partner: 真值表
        output_path: 结果保存路径 (如 "coarse_dist.csv")
    """
    print(f"\n[Analysis] Analyzing {len(pairs)} pairs for threshold selection...")

    # 1. 收集数据
    data_points = []
    tp_scores = []
    fp_scores = []

    for i, j in pairs:
        score = float(S_matrix[i, j])
        # 双向验证是否为真值
        is_tp = (gt_partner[i] == j and gt_partner[j] == i)

        data_points.append({
            "i": i, "j": j, "score": score, "is_tp": is_tp
        })

        if is_tp:
            tp_scores.append(score)
        else:
            fp_scores.append(score)

    # 2. 按分数从高到低排序 (关键步骤)
    data_points.sort(key=lambda x: x["score"], reverse=True)

    # 3. 终端打印统计摘要
    print("-" * 60)
    print(f"{'Type':<8} | {'Count':<6} | {'Min Score':<10} | {'Max Score':<10} | {'Avg Score':<10}")
    print("-" * 60)

    if tp_scores:
        print(
            f"{'✅ TP':<8} | {len(tp_scores):<6} | {min(tp_scores):.4f}     | {max(tp_scores):.4f}     | {sum(tp_scores) / len(tp_scores):.4f}")
    else:
        print(f"{'✅ TP':<8} | 0      | N/A        | N/A        | N/A")

    if fp_scores:
        print(
            f"{'❌ FP':<8} | {len(fp_scores):<6} | {min(fp_scores):.4f}     | {max(fp_scores):.4f}     | {sum(fp_scores) / len(fp_scores):.4f}")
    else:
        print(f"{'❌ FP':<8} | 0      | N/A        | N/A        | N/A")
    print("-" * 60)

    # 4. 自动推荐阈值
    suggested_thr = None
    if tp_scores and fp_scores:
        min_tp = min(tp_scores)
        max_fp = max(fp_scores)

        if min_tp > max_fp:
            suggested_thr = (min_tp + max_fp) / 2
            print(f"\n🚀 [Perfect Cut] TP min > FP max.")
            print(f"   -> Suggest Threshold = {suggested_thr:.4f}")
        else:
            print(f"\n⚠️ [Overlap Detected] TP min < FP max.")
            print(f"   -> To keep ALL TPs (Recall=1.0), set Threshold <= {min_tp - 0.0001:.4f}")
            print(f"   -> To kill ALL FPs (Precision=1.0), set Threshold >= {max_fp + 0.0001:.4f}")
            print(f"   -> (Check the CSV file to find the best trade-off)")

    elif tp_scores:
        print(f"\n[Info] Only TPs found. Set Threshold <= {min(tp_scores):.4f} to keep them.")

    # 5. 保存到 CSV
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Type", "Score", "Index_i", "Index_j"])
            for rank, item in enumerate(data_points):
                t_str = "TP" if item["is_tp"] else "FP"
                writer.writerow([rank + 1, t_str, f"{item['score']:.6f}", item['i'], item['j']])
        print(f"\n[Saved] Detailed distribution saved to: {output_path}")
    except Exception as e:
        print(f"[Error] Failed to save CSV: {e}")

    print("=" * 60)

# 在 utils.py 中添加
# def apply_time_delay_constraint(S, t_starts, dt_min=0.1, dt_max=1e7):
#     """
#     S: 相似度矩阵 (N, N)
#     t_starts: 每个样本的起始 GPS 时间 (N,)
#     """
#     N = S.shape[0]
#     S_filtered = S.copy()
#     mask_count = 0
#
#     # 计算所有样本间的时间差绝对值矩阵
#     # dt_matrix[i, j] = |t[i] - t[j]|
#     dt_matrix = np.abs(t_starts[:, None] - t_starts[None, :])
#
#     # 定义物理掩码：时间差必须在 [dt_min, dt_max] 之间
#     # 注意：自身对比的时间差是0，也会被置为0分数，这符合逻辑
#     invalid_mask = (dt_matrix < dt_min) | (dt_matrix > dt_max)
#
#     # 记录被过滤掉的潜在高分对子数（仅调试用）
#     mask_count = np.sum((S > 0) & invalid_mask)
#
#     # 应用掩码
#     S_filtered[invalid_mask] = 0
#
#     print(f"[Constraint] Time filter masked {mask_count} pairs outside [{dt_min}s, {dt_max:.1e}s]")
#     return S_filtered