# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# ==================== 配置 ====================
OUTPUT_DIR = 'outputs/l-h-r'
LENSED_PREDS_FILE = os.path.join(OUTPUT_DIR, 'lensed_results.txt')
UNLENSED_PREDS_FILE = os.path.join(OUTPUT_DIR, 'unlensed_results.txt')

MODEL_NAME = "PM(Noisy)"
THRESHOLD = 0.5 # 置信度阈值 (0 到 1 之间)
# =======================================================

# 智能读取函数（自动忽略文本和乱码，只抓取概率数字）
def load_probabilities(filepath):
    probs = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            for part in reversed(parts):
                try:
                    probs.append(float(part))
                    break
                except ValueError:
                    continue
    return np.array(probs)

# 1. 加载您真实的预测概率
if not os.path.exists(LENSED_PREDS_FILE) or not os.path.exists(UNLENSED_PREDS_FILE):
    raise FileNotFoundError(f"在 {OUTPUT_DIR} 找不到预测文件。请先运行 predict_all.py。")

print("正在加载预测结果...")
preds_lensed = load_probabilities(LENSED_PREDS_FILE)
preds_unlensed = load_probabilities(UNLENSED_PREDS_FILE)

# 2. 生成真实标签和最终预测结果
y_true = np.concatenate([np.ones(len(preds_lensed)), np.zeros(len(preds_unlensed))])

all_preds_scores = np.concatenate([preds_lensed, preds_unlensed])
y_pred = (all_preds_scores >= THRESHOLD).astype(int)

# 3. 计算真实指标
cm = confusion_matrix(y_true, y_pred)
accuracy = np.mean(y_true == y_pred)
print(f"计算出的真实准确率: {accuracy*100:.2f}%")

# 4. 绘图：使用超大、加粗的字体
TITLE_SIZE = 30      # 顶部模型名称
SUBTITLE_SIZE = 30   # 准确率
LABEL_SIZE = 28      # 轴标签 
TICK_SIZE = 26       # 刻度标签 
ANNOT_SIZE = 30      # 单元格内的数字

print("正在生成自定义图表...")
# 稍微调整了画板比例，让正方形格子和右侧颜色条更协调
plt.figure(figsize=(11, 9)) 

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.weight'] = 'bold'

# 【重点修改】：加上了 square=True，强制让格子变成正方形
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
                 annot_kws={'size': ANNOT_SIZE, 'fontweight': 'bold'})

# 调整颜色条的字体大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=22)

tick_labels = ['Unlensed', 'Lensed']
plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, fontsize=TICK_SIZE, fontweight='bold')
plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, fontsize=TICK_SIZE, fontweight='bold', va='center')

plt.xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold', labelpad=20)
plt.ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold', labelpad=20)

main_title = f"{MODEL_NAME}\n"
subtitle = f"Acc: {accuracy*100:.2f}%"
plt.title(main_title + subtitle, fontsize=SUBTITLE_SIZE, fontweight='bold', pad=30)

plt.tight_layout(pad=3.0)

save_path = os.path.join(OUTPUT_DIR, 'custom_confusion_matrix.png')
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ 成功生成并保存至: {save_path}")