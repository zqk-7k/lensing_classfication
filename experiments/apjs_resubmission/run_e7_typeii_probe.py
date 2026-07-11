#!/usr/bin/env python3
"""Run the frozen PI-ResNet on paired physical/no-Morse E7 interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/classifier"))
from data_classifier import pad_or_trim  # noqa: E402
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocess(left, right):
    values = []
    for strain in (left, right):
        strain = pad_or_trim(strain, 8192, 2)
        strain = (strain - strain.mean(axis=-1, keepdims=True)) / (strain.std(axis=-1, keepdims=True) + 1e-8)
        values.append(strain)
    return np.concatenate(values, axis=0).astype(np.float32)


class E7Dataset(Dataset):
    def __init__(self, frame, data1, data2, signal2):
        self.frame, self.data1, self.data2, self.signal2 = frame.reset_index(drop=True), data1, data2, signal2

    def __len__(self): return len(self.frame)

    def __getitem__(self, item):
        index = int(self.frame.iloc[item].source_id)
        left = np.asarray(self.data1[index], dtype=np.float32)
        physical_right = np.asarray(self.data2[index], dtype=np.float32)
        physical_signal = np.asarray(self.signal2[index], dtype=np.float32)
        noise = physical_right - physical_signal
        # Existing generator uses exp[-i*pi*n] with n=1/2 for the saddle image.
        # For a real time series this is the Hilbert-transform multiplier -i sign(f).
        # Therefore no-Morse = -H(physical saddle), up to DC/Nyquist components.
        no_morse_signal = -np.imag(hilbert(physical_signal)).astype(np.float32)
        target = int(np.argmax(np.abs(physical_signal)))
        control_peak = int(np.argmax(np.abs(no_morse_signal)))
        no_morse_signal = np.roll(no_morse_signal, target - control_peak)
        control_right = no_morse_signal + noise
        physical = preprocess(left, physical_right)
        control = preprocess(left, control_right)
        # Round-trip diagnostic before the discrete peak realignment.
        reconstructed = np.imag(hilbert(-np.imag(hilbert(physical_signal)))).astype(np.float32)
        relative = np.linalg.norm(reconstructed - physical_signal) / (np.linalg.norm(physical_signal) + 1e-30)
        return torch.from_numpy(physical), torch.from_numpy(control), item, np.float32(relative), target - control_peak


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=["SIS", "PM"], required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--manifest", default=str(ROOT / "experiments/apjs_resubmission/manifests/e7_typeii/e7_source_manifest_seed20260711.csv.gz"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/apjs_resubmission_final_v1/e7_typeii"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); lower = args.lens.lower()
    frame = pd.read_csv(args.manifest).query("lens_family == @args.lens").reset_index(drop=True)
    base = Path(args.data_root) / f"{args.lens}_data_0228"
    data1 = np.load(base / f"{args.lens}_data_strain_1.npy", mmap_mode="r")
    data2 = np.load(base / f"{args.lens}_data_strain_2.npy", mmap_mode="r")
    signal2 = np.load(base / f"{args.lens}_h_strain_2.npy", mmap_mode="r")
    dataset = E7Dataset(frame, data1, data2, signal2)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=args.workers, pin_memory=True)
    checkpoint = ROOT / f"runs/apjs_resubmission_final_v1/pi_resnet_{lower}_noisy_seed42/best.pt"
    device = torch.device(args.device)
    model = BinaryPeriodicResNet1D_Ablation(in_channels=1, d_model=256, width_scale=4.0,
                                            use_snake=False, use_se=True, use_physics_fusion=True).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.eval(); physical_scores = np.empty(len(frame)); control_scores = np.empty(len(frame))
    roundtrip = np.empty(len(frame)); peak_shifts = np.empty(len(frame), dtype=int)
    with torch.no_grad():
        for physical, control, indices, errors, shifts in loader:
            physical_values = torch.sigmoid(model(physical.to(device, non_blocking=True)).squeeze(1)).cpu().numpy()
            control_values = torch.sigmoid(model(control.to(device, non_blocking=True)).squeeze(1)).cpu().numpy()
            indices = indices.numpy(); physical_scores[indices] = physical_values; control_scores[indices] = control_values
            roundtrip[indices] = errors.numpy(); peak_shifts[indices] = shifts.numpy()
    frame["physical_score"] = physical_scores; frame["no_morse_score"] = control_scores
    frame["score_difference_physical_minus_control"] = physical_scores - control_scores
    frame["hilbert_roundtrip_relative_l2"] = roundtrip; frame["control_peak_shift_samples"] = peak_shifts
    core = json.load(open(ROOT / "runs/apjs_resubmission_final_v1/core_analysis_0228/core_results.json"))
    threshold = core["results"][lower]["models"]["pi_resnet"]["operating_points"]["0.001"]["threshold"]
    physical_detected = physical_scores >= threshold; control_detected = control_scores >= threshold
    rng = np.random.RandomState(20260711); delta = physical_scores - control_scores
    mean_boot, eff_boot = np.empty(10000), np.empty(10000)
    for index in range(10000):
        selected = rng.randint(0, len(frame), len(frame))
        mean_boot[index] = delta[selected].mean()
        eff_boot[index] = physical_detected[selected].mean() - control_detected[selected].mean()
    test = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    scores_path = output / f"e7_scores_{lower}.csv.gz"; frame.to_csv(scores_path, index=False, compression="gzip")
    result = {
        "status": "complete", "lens": args.lens, "sources": len(frame), "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": sha256(args.manifest), "fpp_1e3_threshold": float(threshold),
        "mean_score_difference": float(delta.mean()),
        "mean_score_difference_ci": [float(np.quantile(mean_boot, .025)), float(np.quantile(mean_boot, .975))],
        "median_score_difference": float(np.median(delta)),
        "wilcoxon_statistic": float(test.statistic), "wilcoxon_pvalue": float(test.pvalue),
        "physical_efficiency": float(physical_detected.mean()), "no_morse_efficiency": float(control_detected.mean()),
        "efficiency_difference": float(physical_detected.mean() - control_detected.mean()),
        "efficiency_difference_ci": [float(np.quantile(eff_boot, .025)), float(np.quantile(eff_boot, .975))],
        "max_hilbert_roundtrip_relative_l2": float(roundtrip.max()),
        "score_file": str(scores_path), "score_file_sha256": sha256(scores_path),
        "interpretation": "Paired model-sensitivity diagnostic; not a discovery of new lensing physics."
    }
    result_path = output / f"e7_results_{lower}.json"; result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / f"e7_{lower}.sha256").write_text(f"{sha256(scores_path)}  {scores_path.name}\n{sha256(result_path)}  {result_path.name}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
