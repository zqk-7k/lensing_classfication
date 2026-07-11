# test.py
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

# 导入原有模块
import config as cfg
import data
import model
import utils


def embed_all(encoder, X, device, target_len):
    """提取所有数据的 Embedding 并进行 L2 归一化"""
    ds = data.SeriesDataset(X, target_len)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    encoder.eval()
    Z = []
    print(f"[Info] Extracting embeddings for {len(X)} samples...")
    with torch.no_grad():
        for x in dl:
            Z.append(encoder(x.to(device)).cpu())
    Z = torch.cat(Z, dim=0).numpy()
    # L2 normalize
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def main():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 确保权重文件存在
    model_path = os.path.join(cfg.OUT_DIR, "best.pt")
    if not os.path.exists(model_path):
        print(f"[Error] Pretrained model not found at {model_path}")
        return

    # -------------------------------------------------------------------------
    # 2. 加载数据 (逻辑与训练脚本保持高度一致)
    # -------------------------------------------------------------------------
    print(f"[Info] Loading data... Mode: {cfg.MODEL_TYPE} | {cfg.DATA_MODE}")

    # 加载透镜数据
    X1, m1 = data.load_npy_series(cfg.L1_PATH, "L1", cfg.LENSED_LIMIT, file_suffix="PM")
    X2, m2 = data.load_npy_series(cfg.L2_PATH, "L2", cfg.LENSED_LIMIT, file_suffix="PM")

    # 加载非透镜数据
    Xu, mu = [], []
    curr_u = 0
    for i, path in enumerate(cfg.UNL_PATHS):
        if cfg.UNL_LIMIT and curr_u >= cfg.UNL_LIMIT: break
        to_read = cfg.UNL_LIMIT - curr_u if cfg.UNL_LIMIT else None
        x, m = data.load_npy_series(path, "U", to_read, file_suffix=str(i))
        Xu.extend(x)
        mu.extend(m)
        curr_u += len(x)

    N1 = len(X1)
    # 获取最大长度用于 Padding/Truncate
    T = max(len(X1[0]), len(X2[0]), len(Xu[0]) if Xu else 0)

    # 3. 数据切分 (使用 config 中的 SEED 确保拿到同样的 Test Set)
    (tr_p, va_p, te_p), (tr_u, va_u, te_u) = data.event_level_split(N1, len(Xu), cfg.SPLIT, cfg.SEED)

    # 构造测试集列表
    X_te = [X1[i] for i in te_p] + [X2[i] for i in te_p] + [Xu[i] for i in te_u]
    m_te = [m1[i] for i in te_p] + [m2[i] for i in te_p] + [mu[i] for i in te_u]
    t_starts = data.load_test_time_arrays(cfg)  # 使用新函数
    N_te = len(X_te)

    # 计算真值
    gt_partner = utils.build_ground_truth(m_te)
    N_true_pairs = sum(1 for x in gt_partner if x >= 0) // 2
    print(f"[Info] Test set size: {N_te}, True Pairs in Test Set: {N_true_pairs}")

    # -------------------------------------------------------------------------
    # 4. 加载模型
    # -------------------------------------------------------------------------
    # 注意：如果训练时用了 Mamba2ResFlowEncoder，这里也要对应更换
    net = model.Encoder1D(d_model=cfg.D_MODEL, emb_dim=cfg.EMB_DIM).to(device)

    print(f"[Info] Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # -------------------------------------------------------------------------
    # 5. 执行评估 (Coarse & Refined)
    # -------------------------------------------------------------------------
    # --- 1. 粗匹配 (Coarse) ---
    Z = embed_all(net, X_te, device, T)
    S0 = Z @ Z.T
    np.fill_diagonal(S0, -1)  # 屏蔽自身匹配
    S0 = utils.apply_time_delay_constraint(S0, t_starts)

    # 自动或手动阈值
    thr0 = cfg.THRESHOLD if cfg.THRESHOLD else utils.auto_threshold(S0.max(1))
    edges0, _ = utils.build_edges_from_scores(S0, thr0)
    pairs0, _ = utils.max_weight_matching(edges0, N_te)

    print("\n" + "=" * 30 + " COARSE RESULTS " + "=" * 30)
    stats0, rows0 = utils.evaluate_and_print(
        pairs0, "coarse", gt_partner, N_true_pairs, m_te, S0, cfg.USE_GCC, L_matrix=None
    )


    # --- 2. 精细匹配与融合 (Refined / Fusion) ---
    if cfg.USE_GCC:
        print(f"\n[Info] Refining scores with GPU-accelerated GCC-PHAT...")

        # 1. 获取 GCC 分数 (使用你之前的 GPU 版本)
        S_ref, L = utils.refine_scores_gpu(X_te, S0, topk=cfg.TOPK, device=device)

        # 2. 调用融合函数 (在这里改进)
        # alpha 建议设为 0.7 左右，因为目前你的神经网络模型比 GCC 更可靠
        alpha = 0.7
        S_fused = utils.fuse_scores(S0, S_ref, alpha=alpha)

        # 【新增】在 Refined 阶段再次强化时间约束
        S_fused = utils.apply_time_delay_constraint(S_fused, t_starts)

        # 3. 使用融合后的分数 S_fused 进行后续匹配逻辑
        # 自动阈值现在作用于融合分数
        thr1 = cfg.THRESHOLD if cfg.THRESHOLD else utils.auto_threshold(S_fused.max(1))

        # 建立边和执行最大权重匹配
        edges1, _ = utils.build_edges_from_scores(S_fused, thr1)
        pairs1, _ = utils.max_weight_matching(edges1, N_te)

        print("\n" + "=" * 30 + " FUSED RESULTS (S0 + S_ref) " + "=" * 30)
        stats1, rows1 = utils.evaluate_and_print(
            pairs1, "fused", gt_partner, N_true_pairs, m_te, S_fused, cfg.USE_GCC, L_matrix=L
        )

    # --- 3. 结果保存 ---
    if cfg.SAVE_XLSX:
        out_file = os.path.join(cfg.OUT_DIR, "test_results.xlsx")
        utils.write_pairs_detailed_xlsx(out_file, "coarse", rows0)
        print(f"[Info] Results saved to {out_file}")

    print(f"\n[Done] Test finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()