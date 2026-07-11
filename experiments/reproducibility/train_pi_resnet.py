#!/usr/bin/env python3
"""Reproducible PI-ResNet retraining on the ET 0222 catalog only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = ROOT / "src" / "classifier"
sys.path.insert(0, str(CLASSIFIER_DIR))

import config as cfg  # noqa: E402
import pair_dataset as data_lib  # noqa: E402
from pi_resnet import PIResNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=["SIS", "PM"], required=True)
    parser.add_argument("--data-root", default=os.environ.get("GW_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--output-root", default=str(ROOT / "artifacts" / "training"))
    parser.add_argument("--split-manifest", default=str(ROOT / "experiments" / "reproducibility" / "manifests" / "split_0222_seed42.npz"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class IndexedArrayView:
    """Memory-efficient indexed view over a memmapped array."""

    def __init__(self, array, indices):
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.array[self.indices[item]]


def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    losses, labels_all, scores_all = 0.0, [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            logits = model(inputs).squeeze(1)
            losses += criterion(logits, labels).item() * inputs.size(0)
            labels_all.extend(labels.cpu().numpy())
            scores_all.extend(torch.sigmoid(logits).cpu().numpy())
    labels_np = np.asarray(labels_all)
    scores_np = np.asarray(scores_all)
    preds = (scores_np >= 0.5).astype(np.int64)
    result = (losses / len(loader.dataset), accuracy_score(labels_np, preds), roc_auc_score(labels_np, scores_np))
    return result + (labels_np, scores_np) if return_predictions else result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    source_dir = Path(args.data_root) / f"{args.lens}_data_0222"
    unl_dir = Path(args.data_root) / "Unlensed_data_0222"
    l1_path = source_dir / f"{args.lens}_data_strain_1.npy"
    l2_path = source_dir / f"{args.lens}_data_strain_2.npy"
    unl_path = unl_dir / "unlensed_data_strain.npy"
    for path in (l1_path, l2_path, unl_path):
        if not path.exists():
            raise FileNotFoundError(path)

    run_dir = Path(args.output_root) / f"pi_resnet_{args.lens.lower()}_noisy_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.DATA_MODE = "noisy"
    cfg.MODEL_TYPE = args.lens
    cfg.TARGET_LEN = 8192
    cfg.STRIDE = 2
    cfg.AUG_PROB = 0.5
    cfg.AUG_FLIP = True
    cfg.AUG_INDEPENDENT_ROLL = True
    cfg.AUG_ROLL_MAX = 1024
    cfg.NEG_RATIO = {"diff_event": 0.7, "noise": 0.3}
    cfg.SEED = args.seed

    l1 = data_lib.load_npy_data(str(l1_path))
    l2 = data_lib.load_npy_data(str(l2_path))
    unl = data_lib.load_npy_data(str(unl_path))
    split_path = Path(args.split_manifest)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = np.load(split_path, allow_pickle=False)
    prefix = args.lens.lower()
    train_idx = split[f"{prefix}_train_source_ids"]
    val_idx = split[f"{prefix}_val_source_ids"]
    unl_train_idx = split["unlensed_train_source_ids"]
    unl_val_idx = split["unlensed_val_source_ids"]
    assert set(train_idx).isdisjoint(val_idx)
    assert set(unl_train_idx).isdisjoint(unl_val_idx)
    unl_train = IndexedArrayView(unl, unl_train_idx)
    unl_val = IndexedArrayView(unl, unl_val_idx)
    np.savez_compressed(run_dir / "split_manifest.npz", train_source_ids=train_idx,
                        val_source_ids=val_idx, unlensed_train_source_ids=unl_train_idx,
                        unlensed_val_source_ids=unl_val_idx)

    train_ds = data_lib.PairDataset(l1, l2, unl_train, train_idx, mode="train")
    val_ds = data_lib.PairDataset(l1, l2, unl_val, val_idx, mode="val")
    train_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, generator=train_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    model = PIResNet(
        in_channels=1, d_model=256, width_scale=4.0,
        use_snake=False, use_se=True, use_pairwise_fusion=True,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    metadata = vars(args) | {
        "git_commit": git_commit(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "train_catalog": "0222", "reserved_test_catalog": "0228",
        "l1_path": str(l1_path), "l2_path": str(l2_path), "unl_path": str(unl_path),
        "selection_metric": "validation_auc", "use_snake": False,
        "use_se": True, "use_pairwise_fusion": True,
        "protocol_status": "frozen_training_run", "shared_split_manifest": str(split_path),
        "shared_split_manifest_sha256": sha256(split_path),
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best.pt"
    best_auc = -np.inf
    start = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "lr"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, labels_all, preds_all = 0.0, [], []
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * inputs.size(0)
                labels_all.extend(labels.cpu().numpy())
                preds_all.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy())
            train_loss = loss_sum / len(train_loader.dataset)
            train_acc = accuracy_score(labels_all, preds_all)
            val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                             "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc, "lr": lr})
            handle.flush()
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_auc": val_auc, "metadata": metadata}, checkpoint_path)
            scheduler.step()
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} val_auc={val_auc:.6f}", flush=True)

    summary = {"best_val_auc": float(best_auc), "runtime_seconds": time.time() - start,
               "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path)}
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    _, _, _, val_labels, val_scores = evaluate(model, val_loader, criterion, device, True)
    np.savez_compressed(run_dir / "val_predictions.npz", labels=val_labels, scores=val_scores)
    deterministic_train_ds = data_lib.PairDataset(l1, l2, unl_train, train_idx, mode="val")
    deterministic_train_loader = DataLoader(deterministic_train_ds, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers, pin_memory=True)
    _, _, _, train_labels, train_scores = evaluate(model, deterministic_train_loader, criterion, device, True)
    np.savez_compressed(run_dir / "train_predictions.npz", labels=train_labels, scores=train_scores)
    (run_dir / "checkpoint.sha256").write_text(summary["checkpoint_sha256"] + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(metadata["git_commit"] + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({key: metadata[key] for key in ("python", "torch", "cuda")}, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
