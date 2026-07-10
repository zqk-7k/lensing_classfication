# data_classifier.py
import numpy as np
import torch
from torch.utils.data import Dataset
import config_classifier as cfg

# ================= 工具函数 =================
def pad_or_trim(x, target_len, stride):
    # 【核心修复】强制转换为 numpy 数组并展平为 1D，消除 (1, L) 等维度差异
    x = np.array(x).flatten().astype(np.float32)
    n = x.shape[0]

    if n >= target_len:
        y = x[-target_len:]  # 取末尾信号
    else:
        pad_width = target_len - n
        y = np.pad(x, (pad_width, 0), mode='constant')

    if stride > 1:
        y = y[::stride]
    return y

def load_npy_data(path, limit=None):
    print(f"[Data] Mapping {path} ...")
    data = np.load(path, mmap_mode='r')
    if limit:
        data = data[:limit]
    return data

def shift_signal_zeros(x, shift):
    """真正的物理平移（空缺处补零，绝不环形相接）"""
    if shift == 0:
        return x
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y

# ================= 增强逻辑 =================
def apply_augmentations(c1, c2):
    c1 = c1.copy()
    c2 = c2.copy()

    # 1. 峰值翻转 (Peak Flip)
    if cfg.AUG_FLIP:
        if c1[np.argmax(np.abs(c1))] < 0: c1 *= -1
        if c2[np.argmax(np.abs(c2))] < 0: c2 *= -1

    if np.random.rand() > cfg.AUG_PROB:
        return c1, c2

    # 2. 真实物理平移
    roll_max = cfg.AUG_ROLL_MAX
    if roll_max > 0:
        shift1 = np.random.randint(-roll_max, roll_max)
        if cfg.AUG_INDEPENDENT_ROLL:
            shift2 = np.random.randint(-roll_max, roll_max)
        else:
            shift2 = shift1

        c1 = shift_signal_zeros(c1, shift1)
        c2 = shift_signal_zeros(c2, shift2)

    return c1, c2

# ================= 数据集类 =================
class GWClassifierDataset(Dataset):
    def __init__(self, l1_data, l2_data, unl_data, indices, mode='train'):
        self.l1 = l1_data
        self.l2 = l2_data
        self.unl = unl_data
        self.indices = indices
        self.mode = mode

        ratios = cfg.NEG_RATIO
        total = ratios['diff_event'] + ratios['noise']
        self.p_diff = ratios['diff_event'] / total

        if self.mode != 'train':
            self.fixed_pairs = self._generate_fixed_pairs()

    def _generate_fixed_pairs(self):
        pairs = []
        rng = np.random.RandomState(cfg.SEED)
        for idx in self.indices:
            # 正样本
            pairs.append({'type': 'pos', 'idx1': idx, 'idx2': idx, 'src2': 'l2'})
            # 负样本
            if rng.rand() < self.p_diff:
                rand_idx = rng.choice(self.indices)
                while rand_idx == idx: rand_idx = rng.choice(self.indices)
                pairs.append({'type': 'neg', 'idx1': idx, 'idx2': rand_idx, 'src2': 'l2'})
            else:
                rand_unl = rng.randint(0, len(self.unl))
                pairs.append({'type': 'neg', 'idx1': idx, 'idx2': rand_unl, 'src2': 'unl'})
        return pairs

    def __len__(self):
        if self.mode == 'train':
            return len(self.indices) * 2
        else:
            return len(self.fixed_pairs)

    def __getitem__(self, item):
        if self.mode == 'train':
            real_idx = self.indices[item // 2]
            if item % 2 == 0:
                c1 = self.l1[real_idx]
                c2 = self.l2[real_idx]
                label = 1.0
            else:
                c1 = self.l1[real_idx]
                label = 0.0
                if np.random.rand() < self.p_diff:
                    rand_idx = np.random.choice(self.indices)
                    while rand_idx == real_idx: rand_idx = np.random.choice(self.indices)
                    c2 = self.l2[rand_idx]
                else:
                    rand_unl = np.random.randint(0, len(self.unl))
                    c2 = self.unl[rand_unl]
        else:
            pair_info = self.fixed_pairs[item]
            c1 = self.l1[pair_info['idx1']]
            if pair_info['src2'] == 'l2':
                c2 = self.l2[pair_info['idx2']]
            else:
                c2 = self.unl[pair_info['idx2']]
            label = 1.0 if pair_info['type'] == 'pos' else 0.0

        # --- B. 统一截断与预处理 ---
        c1 = pad_or_trim(c1, cfg.TARGET_LEN, cfg.STRIDE)
        c2 = pad_or_trim(c2, cfg.TARGET_LEN, cfg.STRIDE)

        if self.mode == 'train':
            c1, c2 = apply_augmentations(c1, c2)

        # --- C. 归一化 ---
        c1 = (c1 - c1.mean()) / (c1.std() + 1e-8)
        c2 = (c2 - c2.mean()) / (c2.std() + 1e-8)

        input_tensor = np.stack([c1, c2], axis=0).astype(np.float32)
        return torch.from_numpy(input_tensor), torch.tensor(label, dtype=torch.float32)