"""Pair construction and preprocessing for time-domain classification."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import config as cfg


def pad_or_trim(x, target_len: int, stride: int) -> np.ndarray:
    """Left-pad or keep trailing samples, then downsample the time axis."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        length = x.shape[1]
        y = x[:, -target_len:] if length >= target_len else np.pad(
            x, ((0, 0), (target_len - length, 0)), mode="constant"
        )
        return y[:, ::stride] if stride > 1 else y
    length = x.shape[0]
    y = x[-target_len:] if length >= target_len else np.pad(
        x, (target_len - length, 0), mode="constant"
    )
    y = y[::stride] if stride > 1 else y
    return y[None, :]


def load_npy_data(path, limit=None):
    """Memory-map a NumPy array without loading the full catalog into RAM."""
    print(f"[data] mapping {path}")
    data = np.load(path, mmap_mode="r")
    return data[:limit] if limit else data


def shift_with_zeros(x: np.ndarray, shift: int) -> np.ndarray:
    """Translate a one- or multi-channel time series without circular wrapping."""
    if shift == 0:
        return x
    y = np.zeros_like(x)
    if x.ndim == 2:
        if shift > 0:
            y[:, shift:] = x[:, :-shift]
        else:
            y[:, :shift] = x[:, -shift:]
    elif shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y


def apply_augmentations(first: np.ndarray, second: np.ndarray):
    """Apply peak-sign normalization and optional translations."""
    first, second = first.copy(), second.copy()
    if cfg.AUG_FLIP:
        if first.flat[np.argmax(np.abs(first))] < 0:
            first *= -1
        if second.flat[np.argmax(np.abs(second))] < 0:
            second *= -1
    if np.random.rand() > cfg.AUG_PROB or cfg.AUG_ROLL_MAX <= 0:
        return first, second
    first_shift = np.random.randint(-cfg.AUG_ROLL_MAX, cfg.AUG_ROLL_MAX)
    second_shift = (
        np.random.randint(-cfg.AUG_ROLL_MAX, cfg.AUG_ROLL_MAX)
        if cfg.AUG_INDEPENDENT_ROLL else first_shift
    )
    return shift_with_zeros(first, first_shift), shift_with_zeros(second, second_shift)


class PairDataset(Dataset):
    """Create positive and mixed hard/easy-negative event pairs.

    Training negatives are sampled online. Evaluation pairs are deterministic.
    The caller must provide disjoint source pools for training and validation.
    """

    def __init__(self, image_1, image_2, unlensed, indices, mode="train"):
        self.image_1, self.image_2, self.unlensed = image_1, image_2, unlensed
        self.indices, self.mode = np.asarray(indices), mode
        total = cfg.NEG_RATIO["diff_event"] + cfg.NEG_RATIO["noise"]
        self.hard_probability = cfg.NEG_RATIO["diff_event"] / total
        if mode != "train":
            self.fixed_pairs = self._build_fixed_pairs()

    def _build_fixed_pairs(self):
        pairs, rng = [], np.random.RandomState(cfg.SEED)
        for index in self.indices:
            pairs.append((index, index, "image_2", 1.0))
            if rng.rand() < self.hard_probability:
                other = rng.choice(self.indices)
                while other == index:
                    other = rng.choice(self.indices)
                pairs.append((index, other, "image_2", 0.0))
            else:
                pairs.append((index, rng.randint(len(self.unlensed)), "unlensed", 0.0))
        return pairs

    def __len__(self):
        return len(self.indices) * 2 if self.mode == "train" else len(self.fixed_pairs)

    def __getitem__(self, item):
        if self.mode == "train":
            first_index = self.indices[item // 2]
            if item % 2 == 0:
                second_index, source, label = first_index, "image_2", 1.0
            elif np.random.rand() < self.hard_probability:
                second_index = np.random.choice(self.indices)
                while second_index == first_index:
                    second_index = np.random.choice(self.indices)
                source, label = "image_2", 0.0
            else:
                second_index, source, label = np.random.randint(len(self.unlensed)), "unlensed", 0.0
        else:
            first_index, second_index, source, label = self.fixed_pairs[item]

        first = self.image_1[first_index]
        second = self.image_2[second_index] if source == "image_2" else self.unlensed[second_index]
        first = pad_or_trim(first, cfg.TARGET_LEN, cfg.STRIDE)
        second = pad_or_trim(second, cfg.TARGET_LEN, cfg.STRIDE)
        if self.mode == "train":
            first, second = apply_augmentations(first, second)
        first = (first - first.mean(axis=-1, keepdims=True)) / (first.std(axis=-1, keepdims=True) + 1e-8)
        second = (second - second.mean(axis=-1, keepdims=True)) / (second.std(axis=-1, keepdims=True) + 1e-8)
        pair = np.concatenate([first, second], axis=0).astype(np.float32)
        return torch.from_numpy(pair), torch.tensor(label, dtype=torch.float32)
