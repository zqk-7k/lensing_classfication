#!/usr/bin/env python3
"""Reproducible SEMD-inspired baseline retraining on 0222 CQT images."""

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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SEMD_DIR = ROOT / "src" / "semd"
sys.path.insert(0, str(SEMD_DIR))

from dataset import LensedDataset, get_train_transforms, get_val_transforms  # noqa: E402
from model import get_deit_tiny_distilled_enhanced  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=["SIS", "PM"], required=True)
    parser.add_argument("--image-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--output-root", default=str(ROOT / "runs" / "apjs_resubmission"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, labels_all, scores_all = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * images.size(0)
            labels_all.extend(labels.cpu().numpy())
            scores_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    labels_np, scores_np = np.asarray(labels_all), np.asarray(scores_all)
    return (loss_sum / len(loader.dataset),
            accuracy_score(labels_np, scores_np >= 0.5),
            roc_auc_score(labels_np, scores_np))


def main():
    args = parse_args()
    seed_everything(args.seed)
    image_root = Path(args.image_root) / f"dataset_images_{args.lens}_noisy_cqt"
    paths, labels = [], []
    for label, folder in [(1, image_root / "lensed"), (0, image_root / "unlensed")]:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        paths.extend(map(str, files))
        labels.extend([label] * len(files))
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=args.seed, stratify=labels)

    run_dir = Path(args.output_root) / f"semd_{args.lens.lower()}_noisy_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "path"])
        writer.writerows(("train", y, p) for p, y in zip(train_paths, train_labels))
        writer.writerows(("val", y, p) for p, y in zip(val_paths, val_labels))

    train_ds = LensedDataset(train_paths, train_labels, transform=get_train_transforms())
    val_ds = LensedDataset(val_paths, val_labels, transform=get_val_transforms())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    model = get_deit_tiny_distilled_enhanced(num_classes=2, pretrained=True,
                                             hidden_dim=512, dropout_rate=0.5,
                                             freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    metadata = vars(args) | {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "train_catalog": "0222-derived CQT images", "reserved_test_catalog": "0228",
        "selection_metric": "validation_auc", "implementation": "SEMD-inspired",
        "image_root_resolved": str(image_root),
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    checkpoint_path, history_path = run_dir / "best.pth", run_dir / "history.csv"
    best_auc, start = -np.inf, time.time()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "lr"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, labels_all, preds_all = 0.0, [], []
            for images, labels_batch in train_loader:
                images = images.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * images.size(0)
                labels_all.extend(labels_batch.cpu().numpy())
                preds_all.extend(logits.argmax(dim=1).cpu().numpy())
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
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
