#!/usr/bin/env python3
"""Freeze the source sample and intervention contract for the minimal E7 probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/root/autodl-tmp/qkzhang")
PAIR_DIR = ROOT / "experiments/apjs_resubmission/manifests/0228_pairs"
OUT = ROOT / "experiments/apjs_resubmission/manifests/e7_typeii"
SEED = 20260711


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = np.load(PAIR_DIR / "split_0228_calibration_evaluation_seed20260711.npz")
    rng = np.random.RandomState(SEED + 7000)
    rows = []
    for lens in ("SIS", "PM"):
        ids = split[f"{lens.lower()}_evaluation_source_ids"]
        selected = np.sort(rng.choice(ids, 250, replace=False))
        base = DATA / f"{lens}_data_0228"
        source = pd.read_csv(base / "source_samples.csv")
        params = pd.read_csv(base / "lens_params.csv")
        values = pd.read_csv(base / "lens.csv")
        snr1 = np.load(base / f"{lens}_optimal_SNR_1.npy")
        snr2 = np.load(base / f"{lens}_optimal_SNR_2.npy")
        for index in selected:
            row = {"probe_id": f"e7_{lens.lower()}_{int(index):05d}", "lens_family": lens,
                   "source_id": int(index), "seed": SEED, "physical_delta_phi": 0.5,
                   "control_delta_phi": 0.0, "same_source": True, "same_magnification": True,
                   "same_noise_realization": True, "same_alignment": True,
                   "rho_1": float(snr1[index]), "rho_2": float(snr2[index]),
                   "rho_min": float(min(snr1[index], snr2[index])), "rho_max": float(max(snr1[index], snr2[index]))}
            row.update({f"source_{key}": value for key, value in source.iloc[index].items()})
            row.update({f"lens_{key}": value for key, value in params.iloc[index].items()})
            row.update({f"image_{key}": value for key, value in values.iloc[index].items()})
            rows.append(row)
    manifest = pd.DataFrame(rows)
    assert len(manifest) == 500 and manifest.probe_id.is_unique
    path = OUT / "e7_source_manifest_seed20260711.csv.gz"; manifest.to_csv(path, index=False, compression="gzip")
    contract = {
        "status": "sources_frozen_waveform_generation_pending", "sources": 500,
        "per_lens_family": 250, "source_catalog": "0228 final-evaluation partition",
        "model": "frozen final-v1 PI-ResNet checkpoint for each lens family",
        "interventions": {"physical_minimum_saddle": "delta_phi_21=1/2",
                          "no_morse_control": "delta_phi_21=0"},
        "paired_controls": ["source parameters", "magnification", "noise realization", "arrival alignment"],
        "primary_statistics": ["paired score difference", "paired bootstrap CI", "Wilcoxon signed-rank",
                               "efficiency change at frozen 1e-3 threshold"],
        "scope": "physics-informed model diagnostic; not a new lensing-physics discovery",
        "manifest_sha256": sha256(path),
        "generation_gate": "Do not generate until the waveform-domain Morse intervention is verified against the existing generator."
    }
    meta = OUT / "e7_protocol.json"; meta.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (OUT / "SHA256SUMS").write_text(f"{sha256(path)}  {path.name}\n{sha256(meta)}  {meta.name}\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
