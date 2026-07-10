# test.py

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

from dataset import LensedDataset, get_val_transforms
from model import get_deit_tiny_distilled_enhanced
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on .png test images')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing lensed and unlensed folders')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='outputs/l-h-r',
                        help='Directory to save output plots')

    # ================= 图片文字修改接口 =================
    parser.add_argument('--cm_title', type=str, default="DeiT Model Eval",
                        help='混淆矩阵的主标题')
    parser.add_argument('--roc_title', type=str, default="ROC Curve - DeiT",
                        help='ROC曲线的主标题')
    parser.add_argument('--label_0', type=str, default='Unlensed',
                        help='类别0的名字')
    parser.add_argument('--label_1', type=str, default='Lensed',
                        help='类别1的名字')

    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Using device: {device}")
    print(f"Data Root: {args.data_root}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*50}")

    os.makedirs(args.out_dir, exist_ok=True)

    lensed_dir = os.path.join(args.data_root, 'lensed')
    unlensed_dir = os.path.join(args.data_root, 'unlensed')

    lensed_files = sorted(glob.glob(os.path.join(lensed_dir, "*.png")))
    unlensed_files = sorted(glob.glob(os.path.join(unlensed_dir, "*.png")))

    test_paths = lensed_files + unlensed_files
    # 设定标签：Lensed 是 1，Unlensed 是 0
    test_labels = [1] * len(lensed_files) + [0] * len(unlensed_files)

    print(f"Loaded {len(test_paths)} total images: {len(lensed_files)} lensed, {len(unlensed_files)} unlensed")

    val_tf = get_val_transforms()
    test_ds = LensedDataset(test_paths, test_labels, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_deit_tiny_distilled_enhanced(num_classes=2, pretrained=False).to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint from {args.checkpoint}...")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ================= 核心推理与概率计算 =================
    print("Evaluating on PNG test set...")
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # DeiT 输出 logits 形状为 [Batch, 2]
            logits = model(inputs)

            # 使用 Softmax 计算概率，并提取类别 1 (Lensed) 的概率用于画 ROC 和算 AUC
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算各项指标
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    best_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # ?? 在终端高亮输出 Accuracy 和 AUC
    print(f"\n{'='*50}")
    print(f"?? 模型评估完成！")
    print(f"?? Test Accuracy: {best_acc*100:.2f}%")
    print(f"?? Test AUC:      {roc_auc:.4f}")
    print(f"{'='*50}\n")

    # ================= 绘制 ROC 曲线 =================
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    plt.title(args.roc_title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    save_roc_path = os.path.join(args.out_dir, "ROC_curve_DeiT.png")
    plt.savefig(save_roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ================= 绘制精美学术风混淆矩阵图 =================
    print("Generating custom Confusion Matrix plot...")

    TITLE_SIZE = 34
    SUBTITLE_SIZE = 34
    LABEL_SIZE = 28
    TICK_SIZE = 26
    ANNOT_SIZE = 36

    plt.figure(figsize=(10, 8.5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
                     annot_kws={'size': ANNOT_SIZE, 'weight': 'bold'})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    for t in cbar.ax.get_yticklabels():
        t.set_weight("bold")

    tick_labels = [args.label_0, args.label_1]
    plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, fontsize=TICK_SIZE, fontweight='bold', rotation=0)
    plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, fontsize=TICK_SIZE, fontweight='bold', rotation=90, va='center')

    plt.xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold', labelpad=20)
    plt.ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold', labelpad=20)

    subtitle = f"Accuracy: {best_acc*100:.2f}%"
    plt.title(f"{args.cm_title}\n{subtitle}", fontsize=SUBTITLE_SIZE, fontweight='bold', pad=25)

    plt.tight_layout()

    cm_save_path = os.path.join(args.out_dir, "Confusion_Matrix_DeiT.png")
    plt.savefig(cm_save_path, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"? Figures successfully saved to: {args.out_dir}")

if __name__ == '__main__':
    main()