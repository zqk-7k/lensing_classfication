import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

# 读取你的配置
import config_preprocess as cfg


def pad_or_trim(x, target_len, stride):
    x = x.astype(np.float32)
    n = x.shape[-1]
    if n >= target_len:
        y = x[..., -target_len:]
    else:
        pad_width = target_len - n
        y = np.pad(x, (pad_width, 0), mode='constant')
    if stride > 1:
        y = y[..., ::stride]
    return y

"""---------------------

def convert_to_2d_and_save(c1, c2, method, sr, save_path):
    # 白化 (Instance Norm)
    c1 = (c1 - c1.mean()) / (c1.std() + 1e-8)
    c2 = (c2 - c2.mean()) / (c2.std() + 1e-8)

    if method == 'cqt':
        # 真正的 Q-Transform
        cqt1 = librosa.cqt(c1, sr=sr, fmin=20.0, n_bins=112, bins_per_octave=24, hop_length=16)
        cqt2 = librosa.cqt(c2, sr=sr, fmin=20.0, n_bins=112, bins_per_octave=24, hop_length=16)
        mag1 = librosa.amplitude_to_db(np.abs(cqt1), ref=np.max)
        mag2 = librosa.amplitude_to_db(np.abs(cqt2), ref=np.max)
    elif method == 'mel':
        # Mel 频谱
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=16, n_mels=112, f_min=20.0, f_max=500.0)
        mel1 = mel_transform(torch.tensor(c1))
        mel2 = mel_transform(torch.tensor(c2))
        mag1 = 10 * torch.log10(mel1 + 1e-8).numpy()
        mag2 = 10 * torch.log10(mel2 + 1e-8).numpy()
    else:
        raise ValueError("TRANSFORM_METHOD 必须是 'cqt' 或 'mel'")

    # 转为 Tensor 以便进行时间轴的强制对齐
    mag1_ts = torch.tensor(mag1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mag2_ts = torch.tensor(mag2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 强行插值对齐到宽 224
    mag1_resized = F.interpolate(mag1_ts, size=(112, 224), mode='bilinear', align_corners=False).squeeze()
    mag2_resized = F.interpolate(mag2_ts, size=(112, 224), mode='bilinear', align_corners=False).squeeze()

    # 上下拼接，得到 224 * 224 的矩阵
    concat_matrix = torch.cat([mag1_resized, mag2_resized], dim=0).numpy()

    # ★ 核心技巧：利用 matplotlib 直接保存带有 viridis 伪彩色映射的图片
    # 这样保存下来的图片，原生的像素大小就是 224x224，且自带 3 通道 (RGB)
    plt.imsave(save_path, concat_matrix, cmap='viridis')

"""
def convert_to_2d_and_save(c1, c2, method, sr, save_path):
    # 白化 (针对最后一个时间维度求均值和方差)
    c1 = (c1 - c1.mean(axis=-1, keepdims=True)) / (c1.std(axis=-1, keepdims=True) + 1e-8)
    c2 = (c2 - c2.mean(axis=-1, keepdims=True)) / (c2.std(axis=-1, keepdims=True) + 1e-8)

    # 统一降维/升维，确保输入是 2D: (Channels, Length)
    if c1.ndim == 1:
        c1 = np.expand_dims(c1, 0)
        c2 = np.expand_dims(c2, 0)

    mags = []

    # 提取信号 1 的所有探测器频谱
    for i in range(c1.shape[0]):
        if method == 'cqt':
            cqt = librosa.cqt(c1[i], sr=sr, fmin=20.0, n_bins=112, bins_per_octave=24, hop_length=16)
            mag = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        elif method == 'mel':
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=16, n_mels=112, f_min=20.0, f_max=500.0)
            mel = mel_transform(torch.tensor(c1[i], dtype=torch.float32))
            mag = 10 * torch.log10(mel + 1e-8).numpy()
        mags.append(mag)

    # 提取信号 2 的所有探测器频谱
    for i in range(c2.shape[0]):
        if method == 'cqt':
            cqt = librosa.cqt(c2[i], sr=sr, fmin=20.0, n_bins=112, bins_per_octave=24, hop_length=16)
            mag = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        elif method == 'mel':
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=16, n_mels=112, f_min=20.0, f_max=500.0)
            mel = mel_transform(torch.tensor(c2[i], dtype=torch.float32))
            mag = 10 * torch.log10(mel + 1e-8).numpy()
        mags.append(mag)

    # 将所有的频谱强行插值并上下拼接
    resized_mags = []
    for mag in mags:
        mag_ts = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # 将每个频谱的宽度强制对齐到 224，高度 112
        mag_resized = F.interpolate(mag_ts, size=(112, 224), mode='bilinear', align_corners=False).squeeze()
        resized_mags.append(mag_resized)

    # 上下垂直拼接。如果是单探测器 (PM) 会拼 2 块，双探测器 (SIS) 会拼 4 块
    concat_matrix = torch.cat(resized_mags, dim=0).numpy()

    # 利用 matplotlib 保存为带有 viridis 伪彩色的 RGB 图像
    plt.imsave(save_path, concat_matrix, cmap='viridis')


def main():
    print(f"========== 预处理启动 ==========")
    print(f"数据模型: {cfg.MODEL_TYPE} | 噪声模式: {cfg.DATA_MODE} | 转换算法: {cfg.TRANSFORM_METHOD.upper()}")

    # 动态读取对应模式的数据
    l1_data = np.load(cfg.L1_PATH, mmap_mode='r')
    l2_data = np.load(cfg.L2_PATH, mmap_mode='r')

    # 修改这里：兼容 UNL_PATHS 是列表的情况
    if isinstance(cfg.UNL_PATHS, (list, tuple)):
        print(f"检测到 UNL_PATHS 是一个列表，正在加载并合并 {len(cfg.UNL_PATHS)} 个文件...")
        unl_list = [np.load(p) for p in cfg.UNL_PATHS]
        # 在第 0 维度（样本数量维度）进行合并
        unl_data = np.concatenate(unl_list, axis=0)
    else:
        # 如果只是普通的字符串路径
        unl_data = np.load(cfg.UNL_PATHS, mmap_mode='r')

    # 创建隔离的分类文件夹
    out_dir = cfg.IMAGE_SAVE_DIR
    lensed_dir = os.path.join(out_dir, 'lensed')
    unlensed_dir = os.path.join(out_dir, 'unlensed')
    os.makedirs(lensed_dir, exist_ok=True)
    os.makedirs(unlensed_dir, exist_ok=True)

    sr = 4096 // cfg.STRIDE
    num_samples = len(l1_data)
    print(f"图片将被保存至: {out_dir}")

    # 1. 生成正样本 (Lensed Pairs) -> 对应透镜像1 + 透镜像2
    print("\n>>> 正在生成 Lensed 正样本...")
    for i in tqdm(range(num_samples)):
        c1 = pad_or_trim(l1_data[i], cfg.TARGET_LEN, cfg.STRIDE)
        c2 = pad_or_trim(l2_data[i], cfg.TARGET_LEN, cfg.STRIDE)
        save_path = os.path.join(lensed_dir, f"pos_{i:04d}.png")
        convert_to_2d_and_save(c1, c2, cfg.TRANSFORM_METHOD, sr, save_path)

    # 2. 生成负样本 (Unlensed Pairs) -> 按照 config 比例掺杂错配数据与非透镜噪声
    print("\n>>> 正在生成 Unlensed 负样本...")
    for i in tqdm(range(num_samples)):
        c1 = pad_or_trim(l1_data[i], cfg.TARGET_LEN, cfg.STRIDE)

        # 模仿你的 config_classifier 负采样逻辑
        if np.random.rand() < cfg.NEG_RATIO.get('diff_event', 0.7):
            rand_idx = np.random.randint(0, num_samples)
            while rand_idx == i: rand_idx = np.random.randint(0, num_samples)
            c2 = pad_or_trim(l2_data[rand_idx], cfg.TARGET_LEN, cfg.STRIDE)
        else:
            rand_unl = np.random.randint(0, len(unl_data))
            c2 = pad_or_trim(unl_data[rand_unl], cfg.TARGET_LEN, cfg.STRIDE)

        save_path = os.path.join(unlensed_dir, f"neg_{i:04d}.png")
        convert_to_2d_and_save(c1, c2, cfg.TRANSFORM_METHOD, sr, save_path)

    print("\n✅ 所有图片已生成完毕，可以开始全速训练！")


if __name__ == "__main__":
    main()