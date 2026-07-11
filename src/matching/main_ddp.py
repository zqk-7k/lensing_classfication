# main_ddp.py
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 导入模块
import config as cfg
import data
import model
import utils


# ==========================================
# 1. DDP 辅助函数 (新增)
# ==========================================
def setup_ddp():
    if not dist.is_initialized():
        # torchrun 会自动设置这些环境变量
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ==========================================
# 2. 训练/验证逻辑 (适配 DDP)
# ==========================================
def train_epoch(encoder, loader, optimizer, loss_fn, device, is_ddp=False, epoch=0):
    encoder.train()

    # DDP 必须在每个 epoch 开始前设置 sampler 的 epoch，以保证 shuffle 随机性
    if is_ddp and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    current_rank = dist.get_rank() if is_ddp else 0
    # 为了避免打印太多，只计算 loss 均值
    for i, (xa, xb) in enumerate(loader):
        xa, xb = xa.to(device), xb.to(device)
        # --- 新增打印逻辑 ---
        # 只在第一轮（Epoch 1）的第一个 Batch（i == 0）且只在主卡（Rank 0）打印
        if epoch == 1 and i == 0 and current_rank == 0:
            print(f"\n[Check] Input xa size: {xa.size()}")
            print(f"[Check] Input xb size: {xb.size()}\n")

        # 混合精度训练 (可选，为了速度推荐加上，不加也不影响逻辑)
        # with torch.cuda.amp.autocast():
        z1, z2 = encoder(xa), encoder(xb)
        loss = loss_fn(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xa.size(0)

    # 计算平均 Loss
    avg_loss = total_loss / len(loader.dataset)

    # DDP 模式下，只是为了打印好看，可以将所有卡的 loss 聚合（可选）
    if is_ddp:
        loss_tensor = torch.tensor(avg_loss).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()

    return avg_loss


def val_epoch(encoder, loader, loss_fn, device, is_ddp=False):
    encoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xa, xb in loader:
            xa, xb = xa.to(device), xb.to(device)
            z1, z2 = encoder(xa), encoder(xb)
            loss = loss_fn(z1, z2)
            total_loss += loss.item() * xa.size(0)

    avg_loss = total_loss / len(loader.dataset)

    if is_ddp:
        loss_tensor = torch.tensor(avg_loss).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()

    return avg_loss



def embed_all(encoder, X, device, target_len, use_flip=False, use_hilbert=False):  # <--- [新增参数]
    # 将 use_hilbert 传给 Dataset，这样测试集也会生成双通道数据
    ds = data.SeriesDataset(X, target_len, use_flip=use_flip, use_hilbert=use_hilbert)  # <--- [传入]

    # 推理时不需要 shuffle
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    encoder.eval()
    Z = []
    with torch.no_grad():
        for x in dl:
            Z.append(encoder(x.to(device)).cpu())
    Z = torch.cat(Z, dim=0).numpy()
    # L2 normalize
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- DDP 初始化 ---
    # 检查是否通过 torchrun 启动
    is_ddp = "LOCAL_RANK" in os.environ
    local_rank = 0

    if is_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 标记主进程：只有 Rank 0 负责打印日志和保存文件
    is_main_process = (not is_ddp) or (dist.get_rank() == 0)

    if is_main_process:
        print(f"[Info] DDP Mode: {is_ddp}, Local Rank: {local_rank}, Device: {device}")
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        t0 = time.time()

    # 开启异常检测 (调试用，稳定后可注释)
    torch.autograd.set_detect_anomaly(True)


    # -------------------------------------------------------------------------
    # 1. 加载数据 (保留原 main.py 逻辑)
    # -------------------------------------------------------------------------
    if is_main_process:
        print(f"[Info] Loading data (NPY mode)... Limit L={cfg.LENSED_LIMIT}, U={cfg.UNL_LIMIT}")

    # 读取透镜数据 (使用 load_npy_series)
    # 注意：所有进程都需要加载数据。如果内存不够，需要换成 mmap 懒加载。
    # 目前假设内存充足 (124G)，每个进程独立加载一份。
    X1, m1 = data.load_npy_series(cfg.L1_PATH, "L1", cfg.LENSED_LIMIT, file_suffix="PM")
    X2, m2 = data.load_npy_series(cfg.L2_PATH, "L2", cfg.LENSED_LIMIT, file_suffix="PM")

    Xu, mu = [], []
    curr_u = 0
    for i, path in enumerate(cfg.UNL_PATHS):
        if cfg.UNL_LIMIT and curr_u >= cfg.UNL_LIMIT: break
        to_read = cfg.UNL_LIMIT - curr_u if cfg.UNL_LIMIT else None

        # 使用 load_npy_series 读取背景/非透镜数据
        x, m = data.load_npy_series(path, "U", to_read, file_suffix=str(i))

        Xu.extend(x)
        mu.extend(m)
        curr_u += len(x)

    # 简单的检查
    if len(X1) == 0:
        if is_main_process: print("[Error] No Lensed data loaded! Check your paths in config.py.")
        if is_ddp: cleanup_ddp()
        return

    N1 = len(X1)
    # 修改这里：不要使用数据的原始最大长度，使用 config 中的裁剪长度



    # ========================== 修复开始 ==========================
    # T 代表“传递给 Dataset 进行截断的目标长度” (9830)
    T = cfg.TARGET_LEN

    # 仅用于打印显示：最终进模型的实际长度 (4915)
    T_final = T // cfg.STRIDE
    print(f"T's size is: {T}")
    print(f"T's size is: {T_final}")

    if is_main_process:
        print(f"[Info] Loaded. L1={len(X1)}, L2={len(X2)}, U={len(Xu)}, Len={T}")

    # 2. 切分 (使用固定种子，保证所有 Rank 切分一致)
    (tr_p, va_p, te_p), (tr_u, va_u, te_u) = data.event_level_split(N1, len(Xu), cfg.SPLIT, cfg.SEED)

    # 3. Dataloader (增加 DistributedSampler)
    # tr_ds = data.PairDataset(X1, X2, tr_p, Xu, tr_u, T, cfg)
    # va_ds = data.PairDataset(X1, X2, va_p, Xu, va_u, T, cfg)

    tr_ds = data.PairDataset(X1, X2, tr_p, Xu, tr_u, T, cfg)
    va_ds = data.PairDataset(X1, X2, va_p, Xu, va_u, T, cfg)

    # DDP 关键：训练集需要 Sampler
    if is_ddp:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True)
        # 如果用了 sampler，DataLoader 的 shuffle 必须为 False
        tr_loader = DataLoader(tr_ds, cfg.BATCH_SIZE, sampler=tr_sampler, shuffle=False, num_workers=4, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 验证集通常不需要 DDP Sampler (或者为了简单，让每个卡跑一部分，最后算平均)
    va_loader = DataLoader(va_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # -------------------------------------------------------------------------
    # 4. 模型训练
    # -------------------------------------------------------------------------
    use_hilbert = getattr(cfg, 'USE_HILBERT', False)
    IN_CHANS = 2 if use_hilbert else 1
    if is_main_process:
        print(f"[Info] Input Channels: {IN_CHANS} (Hilbert={use_hilbert})")

        # === [修改] 实例化模型，传入 in_channels ===
    # net = model.Encoder1D(
    #     d_model=cfg.D_MODEL,
    #     emb_dim=cfg.EMB_DIM,
    #     in_channels=IN_CHANS  # <--- [新增参数]
    # ).to(device)

    # === 替换为新的 PeriodicResNet1D ===
    net = model.PeriodicResNet1D(
        d_model=cfg.D_MODEL,
        emb_dim=cfg.EMB_DIM,
        in_channels=IN_CHANS,
        width_scale=getattr(cfg, 'WIDTH_SCALE', 2.0)  # 建议 Scale 设为 2.0
    ).to(device)

    # net = model.Encoder1D(d_model=cfg.D_MODEL, emb_dim=cfg.EMB_DIM).to(device)

    # net = model.Encoder1D(
    #     d_model=cfg.D_MODEL,
    #     emb_dim=cfg.EMB_DIM,
    #     width_scale=getattr(cfg, 'WIDTH_SCALE', 1.0)  # 兼容旧配置，默认1.0
    # ).to(device)

    # 使用您注释掉的更强模型 (如果需要请取消注释)
    # net = model.Mamba2ResFlowEncoder(
    #     d_model=cfg.D_MODEL,
    #     emb_dim=cfg.EMB_DIM,
    #     n_layers=cfg.N_LAYERS
    # ).to(device)

    if is_main_process:
        print(f"\n[System] Using device: {str(device).upper()}")
        print(f"[Model] model Initialized. Params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    # DDP 包装
    if is_ddp:
        # 转换 BatchNorm 为 SyncBatchNorm (提升多卡训练效果)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        # 包装模型
        net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    crit = model.NTXentLoss(cfg.TAU)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_loss = float('inf')

    for ep in range(1, cfg.EPOCHS + 1):
        # 传入 is_ddp 和 epoch 以设置 sampler
        tr_l = train_epoch(net, tr_loader, opt, crit, device, is_ddp=is_ddp, epoch=ep)
        va_l = val_epoch(net, va_loader, crit, device, is_ddp=is_ddp)

        # --- [新增] 每个 Epoch 结束后手动清理 ---
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # ------------------------------------

        # 只在主进程保存和打印
        if is_main_process:
            print(f"Epoch {ep}: tr={tr_l:.4f}, va={va_l:.4f}")
            if va_l < best_loss:
                best_loss = va_l
                # DDP 保存时要存 net.module
                state_to_save = net.module.state_dict() if is_ddp else net.state_dict()
                torch.save(state_to_save, os.path.join(cfg.OUT_DIR, "best.pt"))

    # -------------------------------------------------------------------------
    # 5. 测试与评估 (只在主进程执行，避免重复计算)
    # -------------------------------------------------------------------------
    if is_main_process:
        print("\n[Info] Starting evaluation on Test set...")

        # 加载最佳模型 (需要重新初始化一个非 DDP 的模型对象来加载)
        # net_eval = model.Encoder1D(
        #     d_model=cfg.D_MODEL,
        #     emb_dim=cfg.EMB_DIM,
        #     in_channels=IN_CHANS  # <--- [新增参数] 必须与训练时一致！
        # ).to(device)

        net_eval = model.PeriodicResNet1D(
            d_model=cfg.D_MODEL,
            emb_dim=cfg.EMB_DIM,
            in_channels=IN_CHANS,  # 必须与训练时一致 (2 if use_hilbert else 1)
            width_scale=getattr(cfg, 'WIDTH_SCALE', 2.0)  # 必须与训练时一致 (建议 2.0)
        ).to(device)
        # net_eval = model.Encoder1D(d_model=cfg.D_MODEL, emb_dim=cfg.EMB_DIM).to(device)
        # net_eval = model.Encoder1D(
        #     d_model=cfg.D_MODEL,
        #     emb_dim=cfg.EMB_DIM,
        #     width_scale=getattr(cfg, 'WIDTH_SCALE', 1.0)  # <--- 这里也要加
        # ).to(device)
        # net_eval = model.Mamba2ResFlowEncoder(d_model=cfg.D_MODEL, emb_dim=cfg.EMB_DIM, n_layers=cfg.N_LAYERS).to(
        #     device)

        net_eval.load_state_dict(torch.load(os.path.join(cfg.OUT_DIR, "best.pt")))
        net_eval.eval()

        # 构造测试集数据
        X_te = [X1[i] for i in te_p] + [X2[i] for i in te_p] + [Xu[i] for i in te_u]
        m_te = [m1[i] for i in te_p] + [m2[i] for i in te_p] + [mu[i] for i in te_u]
        N_te = len(X_te)

        # 计算真值表 (Ground Truth)
        gt_partner = utils.build_ground_truth(m_te)
        # 计算测试集中真实的透镜对数量（用于计算 Recall）
        N_true_pairs = sum(1 for x in gt_partner if x >= 0) // 2
        print(f"[Info] Test set size: {N_te}, True Pairs: {N_true_pairs}")

        # =======================================================
        # 【新增】加载测试集时间序列并应用约束
        # =======================================================
        t_te = data.load_test_time_arrays(cfg, te_p, te_u)

        # --- 1. 粗匹配 (Coarse) ---
        # Z = embed_all(net_eval, X_te, device, T, use_flip=cfg.AUG_FLIP)
        Z = embed_all(
            net_eval, X_te, device, T,
            use_flip=cfg.AUG_FLIP,
            use_hilbert=use_hilbert  # <--- [新增参数] 确保测试数据也是双通道
        )
        S0 = Z @ Z.T
        np.fill_diagonal(S0, -1)

        # =======================================================
        # 【新增功能】Top-K Recall (检索排位分析)
        # =======================================================
        print("\n[Eval] Calculating Top-K Recall...")
        k_list = [1, 3, 5, 10, 20, 50]
        hits = {k: 0 for k in k_list}
        ranks = []

        # 找出所有包含真实伙伴的查询 (即透镜信号)
        queries = [i for i in range(N_te) if gt_partner[i] != -1]

        for i in queries:
            true_j = gt_partner[i]
            # 对当前信号 i 与其他所有信号的相似度进行降序排序，得到排序后的索引
            ranked_indices = np.argsort(S0[i])[::-1]

            # 找到它的真实伙伴 true_j 在排名中的位置 (1-based index)
            rank = np.where(ranked_indices == true_j)[0][0] + 1
            ranks.append(rank)

            # 统计命中率
            for k in k_list:
                if rank <= k:
                    hits[k] += 1

        total_queries = len(queries)
        if total_queries > 0:
            print(f"Total Lensed Queries: {total_queries}")
            for k in k_list:
                print(f"  -> Recall@{k:<2}: {hits[k] / total_queries * 100:.2f}% ({hits[k]}/{total_queries})")

            print(f"  -> 平均排位 (Mean Rank): {np.mean(ranks):.2f}")
            print(f"  -> 找到全部正确配对所需的 Top-K (Max Rank): {np.max(ranks)}")
        print("=======================================================\n")
        # =======================================================

        # ----------------------------------------------------
        # 【核心修改】在此处应用时间约束
        # ----------------------------------------------------
        if t_te is not None:
            # 应用硬阈值，过滤掉时间差过大的对子
            S0 = utils.apply_time_delay_constraint(S0, t_te, cfg.MAX_DT)
        else:
            print("[Warn] Time arrays not loaded, skipping time constraint.")
        # ----------------------------------------------------

        # 自动阈值
        thr0 = cfg.THRESHOLD if cfg.THRESHOLD else utils.auto_threshold(S0.max(1))
        edges0, _ = utils.build_edges_from_scores(S0, thr0)
        pairs0, _ = utils.max_weight_matching(edges0, N_te)

        # 【打印 Coarse 结果】
        stats0, rows0 = utils.evaluate_and_print(
            pairs0, "coarse", gt_partner, N_true_pairs, m_te, S0, cfg.USE_GCC, L_matrix=None
        )

        # main_ddp.py

        # ... (在 Coarse 或 Refined 评估之后添加) ...

        if t_te is not None:
            print("\n" + "=" * 20 + " Time Delay Statistics " + "=" * 20)

            # 1. 提取所有配对的时间差数据
            tp_dt = []
            fp_dt = []

            for (i, j) in pairs0:  # pairs0 是匹配算法输出的结果
                dt = abs(t_te[i] - t_te[j])
                # 检查是否为真值对 (TP)
                if gt_partner[i] == j and gt_partner[j] == i:
                    tp_dt.append(dt)
                else:
                    fp_dt.append(dt)

            # 2. 定义统计函数
            def print_stats(data, name):
                if not data:
                    print(f"[{name}] No pairs found for statistics.")
                    return
                data = np.array(data)
                days = data / (24 * 3600)
                print(f"【{name}】 (Count: {len(data)})")
                print(f"  - Mean:   {np.mean(data):12.2f} s ({np.mean(days):.4f} days)")
                print(f"  - Median: {np.median(data):12.2f} s ({np.median(days):.4f} days)")
                print(f"  - Max:    {np.max(data):12.2f} s ({np.max(days):.4f} days)")
                print(f"  - Min:    {np.min(data):12.2f} s ({np.min(days):.4f} days)")
                print("-" * 50)

            # 3. 输出结果
            print_stats(tp_dt, "True Pairs (TP) Time Delay")
            print_stats(fp_dt, "False Positives (FP) Time Delay")
            print("=" * 60)

        # =======================================================
        # 【新增功能】输出 Recall 最大时的指标
        # =======================================================
        print("\n[Eval] Calculating metrics for Max Recall (Threshold = Min TP Score)...")

        # 1. 找到所有真值对（True Pairs）中的最小分数
        tp_scores = []
        for i in range(len(gt_partner)):
            j = gt_partner[i]
            # gt_partner[i] = j 且 j != -1 表示 i 和 j 是真值对
            # j > i 是为了避免 (i, j) 和 (j, i) 重复统计
            if j != -1 and j > i:
                score = S0[i, j]
                tp_scores.append(score)

        if len(tp_scores) > 0:
            min_tp_score = min(tp_scores)
            # 设定阈值为 最小真值分数 - 极小值 (确保该对子能被选中)
            # 注意：如果真值分数极低（比如负数），这会导致引入大量 FP
            thr_max_recall = min_tp_score - 1e-6

            print(f"  -> Found Min True Pair Score: {min_tp_score:.6f}")
            print(f"  -> Re-running matching with Threshold: {thr_max_recall:.6f}")

            # 2. 使用这个超低阈值重新构建图和匹配
            edges_recall, _ = utils.build_edges_from_scores(S0, thr_max_recall)
            pairs_recall, _ = utils.max_weight_matching(edges_recall, N_te)

            # 3. 评估并打印新结果
            stats_recall, _ = utils.evaluate_and_print(
                pairs_recall, "coarse_max_recall", gt_partner, N_true_pairs, m_te, S0, cfg.USE_GCC, L_matrix=None
            )
        else:
            print("  [Warn] No True Pairs found in Ground Truth.")

        # =======================================================

        # --- 2. 精细匹配 (Refined / GCC-PHAT) ---
        if cfg.USE_GCC:
            print("[Info] Refining scores with GCC-PHAT...")
            # S_ref, L = utils.refine_scores(X_te, S0, topk=cfg.TOPK)
            S_ref, L = utils.refine_scores_gpu(X_te, S0, topk=cfg.TOPK, device=device)

            s_max = S_ref.max()
            if s_max > 1e-12:
                print(f"  -> Normalizing S_ref (Max raw score: {s_max:.4e})")
                S_ref = S_ref / s_max
            else:
                print("  [Warn] S_ref scores are all zero!")

            # Refined 阈值 (通常重新计算)
            thr1 = cfg.THRESHOLD if cfg.THRESHOLD else utils.auto_threshold(S_ref.max(1))
            edges1, _ = utils.build_edges_from_scores(S_ref, thr1)
            pairs1, _ = utils.max_weight_matching(edges1, N_te)

            # 【打印 Refined 结果】
            stats1, rows1 = utils.evaluate_and_print(
                pairs1, "refined", gt_partner, N_true_pairs, m_te, S_ref, cfg.USE_GCC, L_matrix=L
            )
        else:
            S_ref, L = None, None
            stats1, rows1 = None, []
            thr1 = None

        # --- 3. 绘图与导出 ---
        # 调用 utils 里的绘图函数 (如果 utils 里封装好了的话)
        # utils.plot_all_figures(...)

        # 导出 Excel (如果你需要)
        if cfg.SAVE_XLSX:
            utils.write_pairs_detailed_xlsx(os.path.join(cfg.OUT_DIR, "pairs_coarse.xlsx"), "coarse", rows0)
            if cfg.USE_GCC:
                utils.write_pairs_detailed_xlsx(os.path.join(cfg.OUT_DIR, "pairs_refined.xlsx"), "refined", rows1)

        print(f"[Done] Finished in {time.time() - t0:.1f}s")

    # DDP 清理
    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
