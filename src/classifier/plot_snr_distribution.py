# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== 配置区 ====================
# 【请修改这里】：填入你图片中显示的那个文件夹的绝对或相对路径
DATA_DIR = './your_dataset_path'  

# 定义你想读取的 SNR 文件
# 字典的 Key 是图例上的名字，Value 是对应的文件名
snr_files = {
    'Train Lensed (1)': 'SIS_optimal_SNR_1.npy',
    'Train Lensed (2)': 'SIS_optimal_SNR_2.npy',
    'Test Lensed (1)': 'test_SIS_optimal_SNR_1.npy',
    'Test Lensed (2)': 'test_SIS_optimal_SNR_2.npy'
}
# ================================================

snr_data = {}

# 1. 加载 .npy 数据
print("正在加载 SNR 数据...")
for label, filename in snr_files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        # 使用 np.load() 读取 .npy 文件
        data = np.load(filepath)
        # 将数据展平为一维数组（防止数据是二维或高维的）
        snr_data[label] = data.flatten()
        print(f"? 成功读取 {label}: 共 {len(snr_data[label])} 个样本")
    else:
        print(f"? 找不到文件: {filepath}")

if not snr_data:
    raise FileNotFoundError("没有找到任何 SNR 数据，请检查 DATA_DIR 路径是否正确！")

# 2. 开始绘制高颜值的学术图表
print("正在生成分布图...")
plt.figure(figsize=(10, 6))

# 设置论文常用字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

# 定义好看的颜色列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 遍历加载的数据并画图（使用核密度估计平滑曲线，加半透明填充）
for i, (label, data) in enumerate(snr_data.items()):
    sns.kdeplot(data, label=label, color=colors[i % len(colors)], 
                linewidth=2.5, fill=True, alpha=0.2)
    # 如果你更喜欢柱状图，可以把上面这句注释掉，换成下面这句：
    # plt.hist(data, bins=50, alpha=0.5, label=label, density=True)

# 设置图表细节
plt.title('Optimal SNR Distribution of the Dataset', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('Optimal SNR', fontsize=18, fontweight='bold')
plt.ylabel('Probability Density', fontsize=18, fontweight='bold')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14, loc='upper right')
plt.xlim(left=0) # 信噪比通常大于0，强制左侧从0开始（视你实际数据而定）

plt.tight_layout()

# 3. 保存图片
save_path = 'snr_distribution_plot.png'
plt.savefig(save_path, dpi=300)
plt.close()

print(f"?? 绘图完成！图片已保存为: {save_path}")