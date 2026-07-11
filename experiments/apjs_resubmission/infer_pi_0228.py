#!/usr/bin/env python3
"""Run frozen PI-ResNet inference on the locked shared 0228 pair manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = ROOT / "src/classifier"
sys.path.insert(0, str(CLASSIFIER_DIR))
from data_classifier import pad_or_trim  # noqa: E402
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PairDataset(Dataset):
    def __init__(self, frame, lensed_1, lensed_2, unlensed):
        self.frame = frame.reset_index(drop=True)
        self.lensed = {1: lensed_1, 2: lensed_2}
        self.unlensed = unlensed

    def __len__(self):
        return len(self.frame)

    def event(self, row, side):
        kind = row[f"{side}_event_type"]
        index = int(row[f"{side}_event_index"])
        if kind == "unlensed":
            return self.unlensed[index]
        return self.lensed[int(kind[-1])][index]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        left = pad_or_trim(self.event(row, "left"), 8192, 2)
        right = pad_or_trim(self.event(row, "right"), 8192, 2)
        left = (left - left.mean(axis=-1, keepdims=True)) / (left.std(axis=-1, keepdims=True) + 1e-8)
        right = (right - right.mean(axis=-1, keepdims=True)) / (right.std(axis=-1, keepdims=True) + 1e-8)
        pair = np.concatenate([left, right], axis=0).astype(np.float32)
        return torch.from_numpy(pair), index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=["SIS", "PM"], required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--manifest-dir", default=str(ROOT / "experiments/apjs_resubmission/manifests/0228_pairs"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/apjs_resubmission_final_v1/predictions_0228"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    lens_lower = args.lens.lower()
    manifest_paths = [Path(args.manifest_dir) / f"0228_{lens_lower}_{part}_pairs.csv.gz"
                      for part in ("calibration", "evaluation")]
    frame = pd.concat([pd.read_csv(path) for path in manifest_paths], ignore_index=True)
    base = Path(args.data_root) / f"{args.lens}_data_0228"
    lensed_1 = np.load(base / f"{args.lens}_data_strain_1.npy", mmap_mode="r")
    lensed_2 = np.load(base / f"{args.lens}_data_strain_2.npy", mmap_mode="r")
    unlensed = np.load(Path(args.data_root) / "Unlensed_data_0228/unlensed_data_strain.npy", mmap_mode="r")
    dataset = PairDataset(frame, lensed_1, lensed_2, unlensed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    checkpoint = ROOT / f"runs/apjs_resubmission_final_v1/pi_resnet_{lens_lower}_noisy_seed42/best.pt"
    device = torch.device(args.device)
    model = BinaryPeriodicResNet1D_Ablation(in_channels=1, d_model=256, width_scale=4.0,
                                            use_snake=False, use_se=True,
                                            use_physics_fusion=True).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    logits = np.empty(len(frame), dtype=np.float32)
    with torch.no_grad():
        for inputs, indices in loader:
            values = model(inputs.to(device, non_blocking=True)).squeeze(1).cpu().numpy()
            logits[indices.numpy()] = values
    output = frame[["pair_id", "label", "source_block_id", "negative_type",
                    "calibration_or_evaluation"]].copy()
    output["pi_logit"] = logits
    output["pi_score"] = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pi_predictions_0228_{lens_lower}.csv.gz"
    output.to_csv(output_path, index=False, compression="gzip")
    metadata = {
        "lens": args.lens, "rows": len(output), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": {path.name: sha256(path) for path in manifest_paths},
        "prediction_path": str(output_path), "prediction_sha256": sha256(output_path),
        "score_inspected_during_inference": False,
    }
    (output_dir / f"pi_predictions_0228_{lens_lower}.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
