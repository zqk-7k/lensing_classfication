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
import re
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
SEMD_DIR = ROOT / "src" / "semd"
sys.path.insert(0, str(SEMD_DIR))

from dataset import LensedDataset, get_train_transforms, get_val_transforms  # noqa: E402
from model import get_deit_tiny_distilled_enhanced  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=["SIS", "PM"], required=True)
    parser.add_argument("--image-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--output-root", default=str(ROOT / "runs" / "apjs_resubmission_final_v1"))
    parser.add_argument("--split-manifest", default=str(ROOT / "experiments" / "apjs_resubmission" / "manifests" / "split_0222_seed42.npz"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pretrained-weights",
        default=str(ROOT / "runs" / "pretrained" /
                    "deit_tiny_distilled_patch16_224-b40b3cf7.pth"),
    )
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


def predict(model, loader, device):
    model.eval()
    labels_all, scores_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            labels_all.extend(labels.numpy())
            scores_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.asarray(labels_all), np.asarray(scores_all)


def source_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot extract source ID from {path}")
    return int(match.group(1))


def main():
    args = parse_args()
    seed_everything(args.seed)
    image_root = Path(args.image_root) / f"dataset_images_{args.lens}_noisy_cqt"
    records = []
    for label, folder in [(1, image_root / "lensed"), (0, image_root / "unlensed")]:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        records.extend((str(path), label, source_id(path)) for path in files)
    split_path = Path(args.split_manifest)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    shared = np.load(split_path, allow_pickle=False)
    prefix = args.lens.lower()
    train_ids = set(map(int, shared[f"{prefix}_train_source_ids"]))
    val_ids = set(map(int, shared[f"{prefix}_val_source_ids"]))
    assert train_ids.isdisjoint(val_ids)
    unknown = sorted({record[2] for record in records} - train_ids - val_ids)
    if unknown:
        raise ValueError(f"Image source IDs absent from shared split: {unknown[:10]}")
    train_records = [record for record in records if record[2] in train_ids]
    val_records = [record for record in records if record[2] in val_ids]
    train_paths = [record[0] for record in train_records]
    train_labels = [record[1] for record in train_records]
    val_paths = [record[0] for record in val_records]
    val_labels = [record[1] for record in val_records]
    assert {record[2] for record in train_records}.isdisjoint(record[2] for record in val_records)

    run_dir = Path(args.output_root) / f"cqt_deit_{args.lens.lower()}_noisy_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "source_id", "split", "pair_type"])
        writer.writerows((p, y, source, "train", "positive" if y else "negative")
                         for p, y, source in train_records)
        writer.writerows((p, y, source, "val", "positive" if y else "negative")
                         for p, y, source in val_records)

    train_ds = LensedDataset(train_paths, train_labels, transform=get_train_transforms())
    val_ds = LensedDataset(val_paths, val_labels, transform=get_val_transforms())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    model = get_deit_tiny_distilled_enhanced(num_classes=2, pretrained=False,
                                             hidden_dim=512, dropout_rate=0.5,
                                             freeze_backbone=False).to(device)
    pretrained_path = Path(args.pretrained_weights)
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Official DeiT pretrained weights are required but missing: {pretrained_path}"
        )
    raw_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    state = raw_state.get("model", raw_state)
    incompatible = model.load_state_dict(state, strict=False)
    non_head_missing = [key for key in incompatible.missing_keys
                        if not key.startswith(("head.", "head_dist."))]
    if non_head_missing:
        raise RuntimeError(f"Unexpected missing backbone keys: {non_head_missing}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    metadata = vars(args) | {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "train_catalog": "0222-derived CQT images", "reserved_test_catalog": "0228",
        "selection_metric": "validation_auc", "implementation": "SEMD-inspired",
        "image_root_resolved": str(image_root),
        "pretrained_weights_sha256": sha256(pretrained_path),
        "protocol_status": "final_v1_training", "shared_split_manifest": str(split_path),
        "shared_split_manifest_sha256": sha256(split_path),
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
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    train_eval_ds = LensedDataset(train_paths, train_labels, transform=get_val_transforms())
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)
    train_pred_labels, train_scores = predict(model, train_eval_loader, device)
    val_pred_labels, val_scores = predict(model, val_loader, device)
    np.savez_compressed(run_dir / "train_predictions.npz", labels=train_pred_labels, scores=train_scores)
    np.savez_compressed(run_dir / "val_predictions.npz", labels=val_pred_labels, scores=val_scores)
    (run_dir / "checkpoint.sha256").write_text(summary["checkpoint_sha256"] + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(metadata["git_commit"] + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({key: metadata[key] for key in ("python", "torch", "cuda")}, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
