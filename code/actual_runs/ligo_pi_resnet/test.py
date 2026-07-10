# -*- coding: utf-8 -*-
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

import config_classifier as cfg
import data_classifier as data_lib
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation

def main():
    # ================= 1. 璁剧疆鍛戒护琛屽弬鏁� =================
    parser = argparse.ArgumentParser(description='Evaluate Model and Plot Custom Figures')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='妯″瀷鏉冮噸鏂囦欢璺�寰�')
    parser.add_argument('--dataset_type', type=str, default='SIS', choices=['SIS', 'PM'],
                        help='鏁版嵁闆嗙被鍨� (SIS 鎴� PM)')

    # 绾�鍑�/鍔犲櫔寮�鍏�
    parser.add_argument('--noise', type=str, default='noisy', choices=['noisy', 'pure'],
                        help='鏁版嵁璐ㄩ噺 (noisy 鎴� pure)')

    # 鍥剧墖鏂囧瓧淇�鏀规帴鍙�
    parser.add_argument('--cm_title', type=str, default="Model Eval",
                        help='娣锋穯鐭╅樀鐨勪富鏍囬��')
    parser.add_argument('--roc_title', type=str, default=None,
                        help='ROC鏇茬嚎鐨勪富鏍囬��')
    parser.add_argument('--label_0', type=str, default='Unlensed',
                        help='绫诲埆0鐨勫悕瀛�')
    parser.add_argument('--label_1', type=str, default='Lensed',
                        help='绫诲埆1鐨勫悕瀛�')

    args = parser.parse_args()

    dataset_type = args.dataset_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print(f" Evaluating Model: [{dataset_type}] on [{args.noise}] data")
    print(f" Using Checkpoint: {args.checkpoint}")
    print("="*60)

    # ================= 2. 鍔ㄦ�佺簿鍑嗛厤缃�鏁版嵁璺�寰� (缁堟瀬淇�澶�) =================
    # 鏂囦欢澶规案杩滄槸 SIS_data 鎴� PM_data锛岄潬鏂囦欢鍚嶅尯鍒� pure 鍜� noisy
    data_folder = f"{dataset_type}_data"

    if args.noise == 'pure':
        file_suffix = "h_strain"
        unl_filename = "unlensed_h_strain.npy"
    else:
        file_suffix = "data_strain"
        unl_filename = "unlensed_data_strain.npy"

    cfg.SOURCE_DIR = os.path.join(cfg.DATA_ROOT, data_folder)
    cfg.OUT_DIR = f"./runs/classifier_{dataset_type}_{args.noise}_eval_results"
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    l1_path = os.path.join(cfg.SOURCE_DIR, f"{dataset_type}_{file_suffix}_1.npy")
    l2_path = os.path.join(cfg.SOURCE_DIR, f"{dataset_type}_{file_suffix}_2.npy")
    unl_path = os.path.join(cfg.DATA_ROOT, "Unlensed_data", unl_filename)

    print("Loading data into memory...")
    l1 = data_lib.load_npy_data(l1_path)
    l2 = data_lib.load_npy_data(l2_path)
    unl = data_lib.load_npy_data(unl_path)

    n_samples = len(l1)
    n_tr = int(n_samples * 0.8)
    n_va = n_samples - n_tr
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    test_indices = indices[n_tr:n_tr+n_va]

    test_ds = data_lib.GWClassifierDataset(l1, l2, unl, test_indices, mode='val')
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # ================= 3. 缁堟瀬闆疯揪锛氳嚜鍔ㄦ帰娴嬮�氶亾鏁� =================
    sample_x, _ = test_ds[0]
    total_channels = sample_x.shape[0]
    actual_in_ch = total_channels // 2  # 瀛�鐢熺綉缁滃崟鏀�璺�閫氶亾鏁�
    print(f" [Auto-Detect] 鎺㈡祴鍒扮湡瀹炶緭鍏ラ�氶亾鏁�: {actual_in_ch} (鏃犵紳閫傞厤褰撳墠鏁版嵁)")

    model = BinaryPeriodicResNet1D_Ablation(
        in_channels=actual_in_ch,  # 浣跨敤鐪熷疄閫氶亾鏁帮紒
        d_model=256,
        width_scale=4.0,
        use_snake=False,
        use_se=True,
        use_physics_fusion=True
    ).to(device)

    model_path = args.checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Cannot find model weight at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"鉁� Successfully loaded model!")

    all_probs = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # 瀛�鐢熺綉缁滆嚜宸变細鍦ㄥ唴閮ㄥ妶寮�锛岃繖閲屼笉闇�瑕佷换浣曞垏鐗囦唬鐮�
            logits = model(inputs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    best_preds = (all_probs > best_threshold).astype(int)
    best_acc = accuracy_score(all_labels, best_preds)

    cm = confusion_matrix(all_labels, best_preds)

    # 鎵撳嵃 AUC 鍒扮粓绔�锛屾柟渚挎煡鐪�
    print(f"\n頎柬紵 妯″瀷鍦ㄥ綋鍓嶆祴璇曢泦涓婄殑 AUC 鍊间负: {roc_auc:.4f}\n")

    # ================= 4. 缁樺埗 ROC 鏇茬嚎 =================
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', s=100, label=f'Best Threshold = {best_threshold:.2f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    roc_title_text = args.roc_title if args.roc_title else f'ROC Curve - {dataset_type} ({args.noise})'
    plt.title(roc_title_text, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    save_roc_path = os.path.join(cfg.OUT_DIR, f"ROC_curve_{dataset_type}.png")
    plt.savefig(save_roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ================= 5. 缁樺埗绮剧編娣锋穯鐭╅樀鍥� =================
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

    cm_title_text = f"{args.cm_title}\n" if args.cm_title else f"{dataset_type} Model\n"
    subtitle = f"Accuracy: {best_acc*100:.2f}%"
    plt.title(cm_title_text + subtitle, fontsize=SUBTITLE_SIZE, fontweight='bold', pad=25)

    plt.tight_layout()

    cm_save_path = os.path.join(cfg.OUT_DIR, f"Confusion_Matrix_{dataset_type}.png")
    plt.savefig(cm_save_path, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"鉁� Figures successfully saved to: {cfg.OUT_DIR}")

if __name__ == '__main__':
    main()