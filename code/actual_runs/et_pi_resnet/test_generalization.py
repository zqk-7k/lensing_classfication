# -*- coding: utf-8 -*-
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

import config_classifier as cfg
import data_classifier as data_lib
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_classifier.pt")
    parser.add_argument('--train_dataset', type=str, required=True, choices=['SIS', 'PM'])
    parser.add_argument('--test_dataset', type=str, required=True, choices=['SIS', 'PM'])
    args = parser.parse_args()

    # Set test data directory
    if args.test_dataset == 'SIS':
        test_dir = os.path.join(cfg.DATA_ROOT, "SIS_data_0222")
    else:
        test_dir = os.path.join(cfg.DATA_ROOT, "PM_data_0222")

    print(f"\n{'='*60}")
    print(f" 🧪 [Generalization Test]")
    print(f" 📈 Trained on: {args.train_dataset}")
    print(f" 🔍 Testing on: {args.test_dataset}")
    print(f" 📂 Model: {args.model_path}")
    print(f"{'='*60}")

    # 1. Load Test Data
    l1 = data_lib.load_npy_data(os.path.join(test_dir, f"{args.test_dataset}_data_strain_1.npy"))
    l2 = data_lib.load_npy_data(os.path.join(test_dir, f"{args.test_dataset}_data_strain_2.npy"))

    # Use the common unlensed data path
    unl_path = "/root/autodl-tmp/qkzhang/Unlensed_data_0222/unlensed_data_strain.npy"
    if not os.path.exists(unl_path):
        unl_path = os.path.join(test_dir, "unlensed_data_strain.npy")
    unl = data_lib.load_npy_data(unl_path)

    # 2. Get the Validation Split (Consistency check)
    n_samples = len(l1)
    n_tr = int(n_samples * 0.8)
    n_va = n_samples - n_tr
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    val_indices = indices[n_tr:n_tr+n_va]

    te_ds = data_lib.GWClassifierDataset(l1, l2, unl, val_indices, mode='val')
    te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Initialize Model (Baseline Architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryPeriodicResNet1D_Ablation(
        d_model=256, width_scale=4.0,
        use_snake=False,   # Baseline is ReLU
        use_se=True,
        use_physics_fusion=True
    ).to(device)

    # 4. Load State Dict
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in te_loader:
            inputs = inputs.to(device)
            logits = model(inputs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n Results -> Acc: {acc*100:.2f}% | AUC: {auc:.4f}")

    # 6. Log to file
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/generalization_results.txt", "a") as f:
        f.write(f"Train: {args.train_dataset:3s} | Test: {args.test_dataset:3s} | Acc: {acc*100:.2f}% | AUC: {auc:.4f}\n")

if __name__ == '__main__':
    main()
