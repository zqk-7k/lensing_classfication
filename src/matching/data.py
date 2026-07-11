# data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config as cfg
from scipy.signal import hilbert


def _to_list(obj_array):
    if isinstance(obj_array, (list, tuple)): return [np.asarray(x) for x in obj_array]
    if isinstance(obj_array, np.ndarray):
        return [obj_array[i] for i in range(obj_array.shape[0])] if obj_array.ndim >= 1 else [obj_array]
    return [np.asarray(obj_array)]


# def pad_or_trim(x, target_len):
#     n = x.shape[0]
#     if n == target_len: return x
#     if n > target_len: return x[:target_len]
#     y = np.zeros((target_len,), dtype=x.dtype)
#     y[:n] = x
#     return y

def load_test_time_arrays(cfg, te_p, te_u):
    """
    根据测试集索引列表，构造与 X_te 完全对应的起始时间轴 (GPS Start Time)。
    对应 main_ddp.py 中 X_te 的拼接顺序：[L1_te] + [L2_te] + [U_te]
    """
    import os
    import numpy as np

    # 检查文件是否存在
    if not (os.path.exists(cfg.TIME_PATH_1) and os.path.exists(cfg.TIME_PATH_2) and os.path.exists(cfg.TIME_PATH_U)):
        print("[Error] Time array files not found!")
        return None

    print("[Info] Loading time arrays for Time Delay constraint...")

    # 1. 加载原始时间数据
    # 假设 npy shape 为 [N, Length]，我们只取 [:, 0] 即起始时间
    # 使用 mmap_mode='r' 避免一次性加载巨大文件到内存
    T1_full = np.load(cfg.TIME_PATH_1, mmap_mode='r')
    T2_full = np.load(cfg.TIME_PATH_2, mmap_mode='r')
    Tu_full = np.load(cfg.TIME_PATH_U, mmap_mode='r')

    # 提取起始时间 (Start GPS Time)
    # 注意：如果数据是 (N, L)，切片 [:, 0] 会返回 (N,)
    # t1_starts = T1_full[:, 0]
    # t2_starts = T2_full[:, 0]
    # tu_starts = Tu_full[:, 0]
    t1_starts = T1_full[:, 0].flatten()
    t2_starts = T2_full[:, 0].flatten()
    tu_starts = Tu_full[:, 0].flatten()

    # 2. 根据测试集索引提取
    # 注意要转为 numpy array 方便后续计算
    t1_te = t1_starts[te_p]
    t2_te = t2_starts[te_p]
    tu_te = tu_starts[te_u]

    # 3. 严格按照 X_te 的顺序拼接: L1 -> L2 -> U
    # 结果 shape 应为 (N_te,)
    t_te_all = np.concatenate([t1_te, t2_te, tu_te], axis=0)

    return t_te_all

# 在 data.py 中修改或替换
# data.py
def make_channels(x, use_hilbert=False):
    """
    根据配置动态生成多通道数据
    Input: (L,) 1D array
    Output: (C, L) 2D array
    """
    if use_hilbert:
        # [双通道模式]: 实部 + 希尔伯特虚部
        x_analytic = hilbert(x)
        real = x_analytic.real
        imag = x_analytic.imag
        # Stack -> [2, L]
        return np.stack([real, imag], axis=0).astype(np.float32)
    else:
        # [单通道模式]: 保持原样，增加一个维度以适配 Conv1d
        # Reshape -> [1, L]
        return x[None, :].astype(np.float32)



def pad_or_trim(x, target_len):
    """
    逻辑：
    1. 截断/补零：将数据强制对齐到 target_len (例如 9830)
    2. 下采样：按照 cfg.STRIDE 进行稀疏化 (例如 /2)
    最终返回长度: target_len // stride
    """

    # 获取数据当前长度
    n = x.shape[-1]

    # --- 步骤 1: 截断或补零到 target_len ---
    if n >= target_len:
        # 数据足够长，取末尾 target_len 个点
        y = x[..., -target_len:]
    else:
        # 数据不足，前补零
        shape = list(x.shape)
        shape[-1] = target_len
        y = np.zeros(tuple(shape), dtype=x.dtype)
        y[..., -n:] = x

    # --- 步骤 2: 下采样 ---
    # 错误修正：必须用 cfg.STRIDE，不能用 cfg.TARGET_LEN
    stride = cfg.STRIDE
    if stride > 1:
        y = y[..., ::stride]

    return y

def load_npz_series(path, tag, max_count=None, file_suffix=""):
    """读取 NPZ 并支持截断和 ID 唯一化"""
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return [], []
    z = np.load(path, allow_pickle=True)
    t_list = _to_list(z["time_array"])
    x_list = _to_list(z["data_strain"])

    if max_count is not None and max_count > 0:
        x_list = x_list[:max_count]
        t_list = t_list[:max_count]

    X, meta = [], []
    for idx, (t, x) in enumerate(zip(t_list, x_list)):
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float32)
        x = x - float(x.mean())
        dt = float(np.median(np.diff(t))) if t.size >= 2 else None
        X.append(x)
        meta.append({
            "tag": tag,
            "file": os.path.basename(path),
            "idx_in_file": idx,
            "unique_id": f"{tag}_{file_suffix}_{idx}",
            "dt": dt
        })
    return X, meta

def load_npy_series(path, tag, max_count=None, file_suffix=""):
    """
    【新增】读取 .npy 文件 (假设只包含 data_strain 数据)
    """
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return [], []

    # 加载 npy 数据
    data_raw = np.load(path, allow_pickle=True)
    x_list = _to_list(data_raw)  # 复用已有的转换函数

    # 截断数量
    if max_count is not None and max_count > 0:
        x_list = x_list[:max_count]

    X, meta = [], []
    for idx, x in enumerate(x_list):
        # 数据预处理
        x = np.asarray(x, dtype=np.float32)
        x = x - float(x.mean())

        # .npy 通常只有数据没有时间轴，所以 dt 设为 None
        dt = None

        X.append(x)
        # 构建元数据
        meta.append({
            "tag": tag,
            "file": os.path.basename(path),
            "idx_in_file": idx,
            "unique_id": f"{tag}_{file_suffix}_{idx}",
            "dt": dt
        })
    return X, meta

# def augment_pair(x, y, roll_max=128, scale=0.1, noise=0.01):
#     def aug(z):
#         z = z.copy()
#         if roll_max > 0:
#             r = np.random.randint(-roll_max, roll_max + 1)
#             z = np.roll(z, r)
#         if scale > 0:
#             s = 1.0 + np.random.uniform(-scale, scale)
#             z = z * s
#         if noise > 0:
#             std = z.std() + 1e-8
#             z = z + np.random.normal(0.0, noise * std, size=z.shape)
#         return z
#
#     return aug(x), aug(y)

# data.py

def augment_pair(x, y, roll_max=128, scale=0.1, noise=0.01, use_phase=False, use_flip=False):
    # === 【新增 Debug 1】 定义一个静态变量来控制打印次数 ===
    # 这种写法利用了 Python 函数对象的属性，充当“静态变量”
    if not hasattr(augment_pair, "_debug_counter"):
        augment_pair._debug_counter = 0

    def apply_random_phase(z, tag=""):
        z_analytic = hilbert(z)
        theta = np.random.uniform(0, 2 * np.pi)

        # === 【新增 Debug 2】 打印调试信息 (前 5 次调用时打印) ===
        # 我们限制打印前 5 次，这样你可以看到几个不同的 theta 值，确认它是随机的
        if use_phase and augment_pair._debug_counter < 0:
            print(
                f"[Debug Phase] PID={os.getpid()} | Tag={tag} | Applied Theta={theta:.4f} rad ({theta * 180 / np.pi:.1f}°)")
            # 注意：这里我们不立即增加 counter，要在外层增加，避免 x 和 y 打印两次计数混乱

        z_rotated = z_analytic * np.exp(1j * theta)
        return np.real(z_rotated).astype(z.dtype)

    # 内部辅助函数：执行峰值翻转
    # data.py

    def apply_peak_flip(z):
        # 寻找绝对值最大的点
        peak_idx = np.argmax(np.abs(z))
        peak_val = z[peak_idx]

        # 如果该点的值为负，则翻转
        if peak_val < 0:
            # 【添加 Debug 打印】
            # 使用 getattr 确保只在训练初期打印几次，避免刷屏
            if not hasattr(apply_peak_flip, "_count"): apply_peak_flip._count = 0
            if apply_peak_flip._count < 1:
                print(f"[Debug Flip] Peak at {peak_idx} is {peak_val:.6f} -> Flipping to {-peak_val:.6f}")
                apply_peak_flip._count += 1

            return -z

        return z  # 确保原样返回

    def aug(z, tag=""):
        z = z.copy()

        # --- 0. 峰值翻转 (在所有随机增强之前进行，保证确定性) ---
        if use_flip:
            z = apply_peak_flip(z)

        # --- A. 随机相位 ---
        if use_phase:
            z = apply_random_phase(z, tag)  # 传入 tag 方便区分

        # ... (其他增强逻辑保持不变) ...
        if roll_max > 0:
            r = np.random.randint(-roll_max, roll_max + 1)
            z = np.roll(z, r)
        if scale > 0:
            s = 1.0 + np.random.uniform(-scale, scale)
            z = z * s
        if noise > 0:
            std = z.std() + 1e-8
            z = z + np.random.normal(0.0, noise * std, size=z.shape)
        return z

    # === 【新增 Debug 3】 更新计数器 ===
    if use_phase and augment_pair._debug_counter < 5:
        augment_pair._debug_counter += 1

    # 分别调用，传入标签方便 Debug 区分
    return aug(x, "Image1"), aug(y, "Image2")

class PairDataset(Dataset):
    def __init__(self, X_l1, X_l2, idx_pairs, X_u, idx_u, target_len, cfg):
        self.l1 = X_l1;
        self.l2 = X_l2;
        self.idx_pairs = list(idx_pairs)
        self.u = X_u;
        self.idx_u = list(idx_u)
        self.target_len = target_len
        self.cfg = cfg
        self.items = [("L", i) for i in self.idx_pairs] + [("U", j) for j in self.idx_u]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        tag, idx = self.items[k]
        # 获取是否启用相位增强 (兼容旧配置文件，默认 False)
        use_phase = getattr(self.cfg, 'AUG_PHASE', False)
        use_flip = getattr(self.cfg, 'AUG_FLIP', False)
        use_hilbert = getattr(self.cfg, 'USE_HILBERT', False)

        if tag == "L":
            x = pad_or_trim(self.l1[idx], self.target_len)
            y = pad_or_trim(self.l2[idx], self.target_len)
            # xa, xb = augment_pair(x, y, self.cfg.AUG_ROLL, self.cfg.AUG_SCALE, self.cfg.AUG_NOISE)

            # 【修改】传入 use_phase 参数
            xa, xb = augment_pair(
                x, y,
                self.cfg.AUG_ROLL,
                self.cfg.AUG_SCALE,
                self.cfg.AUG_NOISE,
                use_phase=use_phase,   # <--- 新增
                use_flip=use_flip
            )
        else:
            u = pad_or_trim(self.u[idx], self.target_len)
            # xa, xb = augment_pair(u, u, self.cfg.AUG_ROLL, self.cfg.AUG_SCALE, self.cfg.AUG_NOISE)

            # 【修改】传入 use_phase 参数
            xa, xb = augment_pair(
                u, u,
                self.cfg.AUG_ROLL,
                self.cfg.AUG_SCALE,
                self.cfg.AUG_NOISE,
                use_phase=use_phase,
                use_flip=use_flip
            )

        # Z-score normalization
        xa = (xa - xa.mean()) / (xa.std() + 1e-8)
        xb = (xb - xb.mean()) / (xb.std() + 1e-8)

        # 4. 【核心修改】根据开关生成通道
        xa_out = make_channels(xa, use_hilbert)  # -> [1, L] or [2, L]
        xb_out = make_channels(xb, use_hilbert)
        return torch.from_numpy(xa_out), torch.from_numpy(xb_out)
        # return torch.from_numpy(xa[None, :].astype(np.float32)), torch.from_numpy(xb[None, :].astype(np.float32))



class SeriesDataset(Dataset):
    def __init__(self, X, target_len=None, use_flip=False, use_hilbert=False):
        self.X = X;
        self.target_len = target_len
        self.use_flip = use_flip
        self.use_hilbert = use_hilbert

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.target_len: x = pad_or_trim(x, self.target_len)

        if self.use_flip:
            peak_idx = np.argmax(np.abs(x))
            if x[peak_idx] < 0:
                x = -x

        x = (x - x.mean()) / (x.std() + 1e-8)
        x_out = make_channels(x, self.use_hilbert)
        return torch.from_numpy(x_out)


def event_level_split(N_lensed, N_unl, split_ratios, seed=42):
    rng = np.random.default_rng(seed)
    # Lensed
    idx_pairs = rng.permutation(N_lensed)
    n_tr = int(split_ratios['train'] * N_lensed)
    n_va = int(split_ratios['val'] * N_lensed)
    tr_p, va_p, te_p = idx_pairs[:n_tr], idx_pairs[n_tr:n_tr + n_va], idx_pairs[n_tr + n_va:]
    # Unlensed
    idx_u = rng.permutation(N_unl)
    m_tr = int(split_ratios['train'] * N_unl)
    m_va = int(split_ratios['val'] * N_unl)
    tr_u, va_u, te_u = idx_u[:m_tr], idx_u[m_tr:m_tr + m_va], idx_u[m_tr + m_va:]
    return (tr_p, va_p, te_p), (tr_u, va_u, te_u)


# def embed_all(encoder, X, device, target_len):
#     """提取所有数据的嵌入向量 (Embedding)"""
#     ds = data.SeriesDataset(X, target_len)
#     # 推理时不需要 Shuffle
#     dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
#     encoder.eval()
#     Z = []
#     with torch.no_grad():
#         for x in dl:
#             # 确保内存连续性以适配 Mamba2 算子
#             x = x.to(device).contiguous()
#             Z.append(encoder(x).cpu())
#     Z = torch.cat(Z, dim=0).numpy()
#     # L2 归一化，确保余弦相似度计算正确
#     return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def load_mixed_lensed(paths_l1, paths_l2, limit, offset=0):
    """加载混合透镜数据并确保 L1/L2 唯一匹配"""
    X1_all, m1_all, X2_all, m2_all = [], [], [], []
    for p1, p2 in zip(paths_l1, paths_l2):
        source_type = "PM" if "pm" in p1.lower() else "SIS"
        # max_count 设为 offset + limit 以确保能切到正确区间
        curr_x1, curr_m1 = load_npz_series(p1, "L1", max_count=offset + limit, file_suffix=source_type)
        curr_x2, curr_m2 = load_npz_series(p2, "L2", max_count=offset + limit, file_suffix=source_type)

        if len(curr_x1) > offset:
            X1_all.extend(curr_x1[offset:offset + limit])
            m1_all.extend(curr_m1[offset:offset + limit])
            X2_all.extend(curr_x2[offset:offset + limit])
            m2_all.extend(curr_m2[offset:offset + limit])
    return X1_all, m1_all, X2_all, m2_all


def build_ground_truth_fixed(meta_all):
    """基于 unique_id 的精确真值构建，避免索引冲突"""
    N = len(meta_all)
    gt = np.full(N, -1, dtype=int)
    id_to_idx = {m["unique_id"]: i for i, m in enumerate(meta_all) if "unique_id" in m}

    for i, m in enumerate(meta_all):
        uid = m.get("unique_id", "")
        if not uid: continue
        # 转换 ID 以寻找伙伴 (L1 <-> L2)
        target_id = uid.replace("L1", "L2", 1) if "L1" in uid else uid.replace("L2", "L1", 1)
        if target_id in id_to_idx:
            gt[i] = id_to_idx[target_id]
    return gt


def load_unlensed(paths, limit, offset=0):
    """加载背景噪声，支持多种格式及偏移"""
    X_all, m_all = [], []
    current_skip = offset
    for i, path in enumerate(paths):
        if len(X_all) >= limit: break
        # 兼容 npy 和 npz 格式
        loader_fn = load_npy_series if path.endswith('.npy') else load_npz_series
        x, m = loader_fn(path, "U", max_count=None, file_suffix=str(i))

        if current_skip > 0:
            if len(x) > current_skip:
                x, m = x[current_skip:], m[current_skip:]
                current_skip = 0
            else:
                current_skip -= len(x)
                continue

        needed = limit - len(X_all)
        X_all.extend(x[:needed]);
        m_all.extend(m[:needed])
    return X_all, m_all


# 在 data.py 中添加
# def load_test_time_arrays(cfg, te_p, te_u):
#     """
#     根据测试集索引列表，构造与 X_te 完全对应的起始时间轴
#     """
#     # 1. 构造文件路径
#     t1_path = os.path.join(cfg.SOURCE_DIR, f"{cfg.FILE_PREFIX}_time_array_1.npy")
#     t2_path = os.path.join(cfg.SOURCE_DIR, f"{cfg.FILE_PREFIX}_time_array_2.npy")
#     unl_t_path = os.path.join(cfg.UNL_DIR, "unlensed_time_array.npy")
#
#     # 2. 读取原始数据 (取第一列作为 GPS 起始时间)
#     print(f"[Info] Loading time arrays for consistency check...")
#     T1_all = np.load(t1_path)[:, 0]
#     T2_all = np.load(t2_path)[:, 0]
#     Tu_all = np.load(unl_t_path)[:, 0]
#
#     # 3. 严格按照 X_te 的拼接顺序进行索引抽取:
#     # 顺序: [L1 的测试部] + [L2 的测试部] + [Unlensed 的测试部]
#     t_te = [T1_all[i] for i in te_p] + [T2_all[i] for i in te_p] + [Tu_all[i] for i in te_u]
#
#     return np.array(t_te)