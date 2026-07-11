#!/usr/bin/env python3
"""Code-level sanity check of delay removal by peak alignment at fixed source and y."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def classifier_preprocess(x):
    x = np.asarray(x, dtype=np.float32)[-8192::2]
    return (x - x.mean()) / (x.std() + 1e-8)


def peak_align(signal, target_peak):
    peak = int(np.argmax(np.abs(signal)))
    return np.roll(signal, target_peak - peak)


def sha256(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--output", default=str(ROOT / "runs/apjs_resubmission_final_v1/zl_sanity/zl_invariance_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    shifts = [-16384, -4096, -1024, 1024, 4096, 16384]
    source_ids = [0, 127, 511, 1023, 2047]
    results = []
    for lens in ("SIS", "PM"):
        base = root / f"{lens}_data_0228"
        data = np.load(base / f"{lens}_data_strain_2.npy", mmap_mode="r")
        signal = np.load(base / f"{lens}_h_strain_2.npy", mmap_mode="r")
        for source in source_ids:
            h0 = np.asarray(signal[source], dtype=np.float32)
            d0 = np.asarray(data[source], dtype=np.float32)
            noise = d0 - h0
            target_peak = int(np.argmax(np.abs(h0)))
            reference = classifier_preprocess(d0)
            for shift in shifts:
                delayed = np.roll(h0, shift)
                realigned = peak_align(delayed, target_peak)
                candidate = classifier_preprocess(realigned + noise)
                delta = candidate.astype(np.float64) - reference.astype(np.float64)
                results.append({
                    "lens": lens, "source_id": source, "delay_shift_samples": shift,
                    "max_abs_difference": float(np.max(np.abs(delta))),
                    "mean_abs_difference": float(np.mean(np.abs(delta))),
                    "relative_l2_difference": float(np.linalg.norm(delta) / (np.linalg.norm(reference) + 1e-30)),
                })
    summary = {
        "status": "pass" if max(item["max_abs_difference"] for item in results) <= 1e-6 else "review_required",
        "interpretation": "Discrete code-level delay intervention at fixed waveform, magnification, and noise; peak alignment removes the intervention before classifier preprocessing.",
        "scope_limit": "This establishes invariance only for the present discrete peak-aligned pair-verification input, not general lens-redshift independence.",
        "source_ids": source_ids, "delay_shift_samples": shifts,
        "maximum_over_all_trials": {
            "max_abs_difference": max(item["max_abs_difference"] for item in results),
            "mean_abs_difference": max(item["mean_abs_difference"] for item in results),
            "relative_l2_difference": max(item["relative_l2_difference"] for item in results),
        },
        "trials": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output.parent / "README.md").write_text(
        "# Lens-redshift invariance sanity check\n\n"
        "This directory records a deterministic code-level test of the statement that, at fixed source and dimensionless impact parameter, lens redshift enters the present input only through relative delay. Signal-only image-2 arrays are shifted, realigned to the original peak, recombined with the identical stored noise realization, and passed through the classifier preprocessing. See `zl_invariance_results.json` for all trials and numerical differences.\n",
        encoding="utf-8")
    (output.parent / "results.sha256").write_text(sha256(output) + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps(summary["maximum_over_all_trials"], indent=2))
    print("status", summary["status"])


if __name__ == "__main__":
    main()
