#!/usr/bin/env python3
"""Run frozen CQT-DeiT inference on the locked shared 0228 pair manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
SEMD_DIR = ROOT / "src/semd"
sys.path.insert(0, str(SEMD_DIR))
from model import get_deit_tiny_distilled_enhanced  # noqa: E402


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CQTPairDataset(Dataset):
    def __init__(self, frame, image1, image2, unlensed):
        self.frame = frame.reset_index(drop=True)
        self.lensed = {1: image1, 2: image2}
        self.unlensed = unlensed
        self.cmap = matplotlib.colormaps["viridis"]

    def __len__(self):
        return len(self.frame)

    def event(self, row, side):
        kind, index = row[f"{side}_event_type"], int(row[f"{side}_event_index"])
        return self.unlensed[index] if kind == "unlensed" else self.lensed[int(kind[-1])][index]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        matrix = np.concatenate([self.event(row, "left"), self.event(row, "right")], axis=0)
        minimum, maximum = float(matrix.min()), float(matrix.max())
        normalized = np.zeros_like(matrix) if maximum == minimum else (matrix - minimum) / (maximum - minimum)
        rgb = self.cmap(normalized, bytes=True)[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0
        rgb = (rgb - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(rgb)), index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=["SIS", "PM"], required=True)
    parser.add_argument("--manifest-dir", default=str(ROOT / "experiments/apjs_resubmission/manifests/0228_pairs"))
    parser.add_argument("--cache-dir", default=str(ROOT / "runs/apjs_resubmission_final_v1/cqt_cache_0228"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/apjs_resubmission_final_v1/predictions_0228"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    lower = args.lens.lower()
    manifests = [Path(args.manifest_dir) / f"0228_{lower}_{part}_pairs.csv.gz"
                 for part in ("calibration", "evaluation")]
    frame = pd.concat([pd.read_csv(path) for path in manifests], ignore_index=True)
    cache = Path(args.cache_dir)
    image1 = np.load(cache / f"{lower}_img1_spectra.npy", mmap_mode="r")
    image2 = np.load(cache / f"{lower}_img2_spectra.npy", mmap_mode="r")
    unlensed = np.load(cache / "unlensed_spectra.npy", mmap_mode="r")
    dataset = CQTPairDataset(frame, image1, image2, unlensed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    checkpoint = ROOT / f"runs/apjs_resubmission_final_v1/cqt_deit_{lower}_noisy_seed42/best.pth"
    device = torch.device(args.device)
    model = get_deit_tiny_distilled_enhanced(num_classes=2, pretrained=False, hidden_dim=512,
                                             dropout_rate=0.5, freeze_backbone=False).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    logits = np.empty((len(frame), 2), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            values = model(images.to(device, non_blocking=True)).cpu().numpy()
            logits[indices.numpy()] = values
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    output = frame[["pair_id", "label", "source_block_id", "negative_type",
                    "calibration_or_evaluation"]].copy()
    output["cqt_deit_logit_0"] = logits[:, 0]
    output["cqt_deit_logit_1"] = logits[:, 1]
    output["cqt_deit_score"] = probabilities[:, 1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cqt_deit_predictions_0228_{lower}.csv.gz"
    output.to_csv(output_path, index=False, compression="gzip")
    cache_metadata = cache / "cache_metadata.json"
    metadata = {
        "lens": args.lens, "rows": len(output), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": {path.name: sha256(path) for path in manifests},
        "cache_metadata_sha256": sha256(cache_metadata),
        "prediction_path": str(output_path), "prediction_sha256": sha256(output_path),
        "score_inspected_during_inference": False,
    }
    (output_dir / f"cqt_deit_predictions_0228_{lower}.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
