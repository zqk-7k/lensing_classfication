#!/usr/bin/env python3
"""Audit catalog-0222/0228 independence without evaluating any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()


def normalized_rows(path: Path) -> set[tuple[str, ...]]:
    frame = pd.read_csv(path)
    return {tuple("" if pd.isna(value) else format(float(value), ".15g") if isinstance(value, (float, np.floating)) else str(value)
                  for value in row) for row in frame.itertuples(index=False, name=None)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "experiments" / "reproducibility" / "manifests" / "0222_0228_independence_audit.json"))
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    relpaths = []
    for catalog in ("0222", "0228"):
        for lens in ("SIS", "PM"):
            base = root / f"{lens}_data_{catalog}"
            relpaths.extend([
                (catalog, lens, "strain_1", base / f"{lens}_data_strain_1.npy"),
                (catalog, lens, "strain_2", base / f"{lens}_data_strain_2.npy"),
                (catalog, lens, "snr_1", base / f"{lens}_optimal_SNR_1.npy"),
                (catalog, lens, "snr_2", base / f"{lens}_optimal_SNR_2.npy"),
                (catalog, lens, "source_parameters", base / "source_samples.csv"),
                (catalog, lens, "lens_parameters", base / "lens_params.csv"),
            ])
        unl = root / f"Unlensed_data_{catalog}"
        relpaths.extend([
            (catalog, "unlensed", "strain", unl / "unlensed_data_strain.npy"),
            (catalog, "unlensed", "source_parameters", unl / "source_samples.csv"),
        ])
    for _, _, _, path in relpaths:
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {f"{catalog}/{family}/{kind}": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
              for catalog, family, kind, path in relpaths}
    all_cross_file_hashes_different = all(
        hashes[f"0222/{family}/{kind}"]["sha256"] != hashes[f"0228/{family}/{kind}"]["sha256"]
        for family in ("SIS", "PM", "unlensed")
        for kind in (("strain_1", "strain_2", "snr_1", "snr_2", "source_parameters", "lens_parameters")
                     if family != "unlensed" else ("strain", "source_parameters"))
    )

    parameter_overlap = {}
    for family in ("SIS", "PM", "unlensed"):
        p22 = root / (f"{family}_data_0222" if family != "unlensed" else "Unlensed_data_0222") / "source_samples.csv"
        p28 = root / (f"{family}_data_0228" if family != "unlensed" else "Unlensed_data_0228") / "source_samples.csv"
        rows22, rows28 = normalized_rows(p22), normalized_rows(p28)
        parameter_overlap[family] = {"0222_rows": len(rows22), "0228_rows": len(rows28), "exact_row_overlap": len(rows22 & rows28)}

    rng = np.random.RandomState(args.seed)
    waveform_checks = {}
    for family, name22, name28 in (
        ("SIS_1", root / "SIS_data_0222/SIS_data_strain_1.npy", root / "SIS_data_0228/SIS_data_strain_1.npy"),
        ("SIS_2", root / "SIS_data_0222/SIS_data_strain_2.npy", root / "SIS_data_0228/SIS_data_strain_2.npy"),
        ("PM_1", root / "PM_data_0222/PM_data_strain_1.npy", root / "PM_data_0228/PM_data_strain_1.npy"),
        ("PM_2", root / "PM_data_0222/PM_data_strain_2.npy", root / "PM_data_0228/PM_data_strain_2.npy"),
        ("unlensed", root / "Unlensed_data_0222/unlensed_data_strain.npy", root / "Unlensed_data_0228/unlensed_data_strain.npy"),
    ):
        a, b = np.load(name22, mmap_mode="r"), np.load(name28, mmap_mode="r")
        ia = rng.choice(len(a), min(args.sample_count, len(a)), replace=False)
        ib = rng.choice(len(b), min(args.sample_count, len(b)), replace=False)
        da = {row_digest(a[index]) for index in ia}
        db = {row_digest(b[index]) for index in ib}
        waveform_checks[family] = {"sampled_0222": len(da), "sampled_0228": len(db), "exact_hash_overlap": len(da & db)}

    seed_files = sorted(str(path) for path in root.glob("**/*seed*") if "0222" in str(path) or "0228" in str(path))
    result = {
        "status": "pass" if all_cross_file_hashes_different and all(v["exact_row_overlap"] == 0 for v in parameter_overlap.values()) and all(v["exact_hash_overlap"] == 0 for v in waveform_checks.values()) else "review_required",
        "scope": "data-only audit; no model inference or score inspection",
        "catalog_interpretation": "independently generated IID holdout from the same simulation priors; not OOD",
        "all_corresponding_file_hashes_different": all_cross_file_hashes_different,
        "files": hashes,
        "source_parameter_overlap": parameter_overlap,
        "sampled_waveform_hash_checks": waveform_checks,
        "generation_seed_metadata_files": seed_files,
        "seed_metadata_note": "Seed metadata unavailable in catalog directories" if not seed_files else "See listed files; semantic verification remains required",
        "sample_seed": args.seed,
        "sample_count_per_array": args.sample_count,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
