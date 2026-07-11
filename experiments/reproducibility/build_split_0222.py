#!/usr/bin/env python3
"""Build the locked, shared source-level 0222 train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "reproducibility" / "manifests"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(size: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.random.RandomState(seed).permutation(size).astype(np.int64)
    boundary = int(fraction * size)
    return np.sort(ids[:boundary]), np.sort(ids[boundary:])


def count_rows(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    files = {
        "sis_image_1": data_root / "SIS_data_0222" / "SIS_data_strain_1.npy",
        "sis_image_2": data_root / "SIS_data_0222" / "SIS_data_strain_2.npy",
        "pm_image_1": data_root / "PM_data_0222" / "PM_data_strain_1.npy",
        "pm_image_2": data_root / "PM_data_0222" / "PM_data_strain_2.npy",
        "unlensed": data_root / "Unlensed_data_0222" / "unlensed_data_strain.npy",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    sizes = {
        "sis": count_rows(files["sis_image_1"]),
        "pm": count_rows(files["pm_image_1"]),
        "unlensed": count_rows(files["unlensed"]),
    }
    if count_rows(files["sis_image_2"]) != sizes["sis"]:
        raise ValueError("SIS image arrays have different source counts")
    if count_rows(files["pm_image_2"]) != sizes["pm"]:
        raise ValueError("PM image arrays have different source counts")

    # Separate deterministic streams avoid accidental identical permutations across pools.
    sis_train, sis_val = split_ids(sizes["sis"], args.train_fraction, args.seed + 101)
    pm_train, pm_val = split_ids(sizes["pm"], args.train_fraction, args.seed + 202)
    unl_train, unl_val = split_ids(sizes["unlensed"], args.train_fraction, args.seed + 303)
    split_map = {
        "sis_train_source_ids": sis_train,
        "sis_val_source_ids": sis_val,
        "pm_train_source_ids": pm_train,
        "pm_val_source_ids": pm_val,
        "unlensed_train_source_ids": unl_train,
        "unlensed_val_source_ids": unl_val,
    }
    checks = {
        "sis_train_val_disjoint": bool(set(sis_train).isdisjoint(sis_val)),
        "pm_train_val_disjoint": bool(set(pm_train).isdisjoint(pm_val)),
        "unlensed_train_val_disjoint": bool(set(unl_train).isdisjoint(unl_val)),
    }
    assert all(checks.values()), checks

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"split_0222_seed{args.seed}.npz"
    hashes = {name: sha256(path) for name, path in files.items()}
    np.savez_compressed(
        manifest_path,
        **split_map,
        random_seed=np.asarray(args.seed, dtype=np.int64),
        train_fraction=np.asarray(args.train_fraction, dtype=np.float64),
        catalog_hash_names=np.asarray(list(hashes)),
        catalog_hash_values=np.asarray(list(hashes.values())),
    )
    audit = {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "random_seed": args.seed,
        "train_fraction": args.train_fraction,
        "catalog_sizes": sizes,
        "split_sizes": {key: int(len(value)) for key, value in split_map.items()},
        "disjointness_checks": checks,
        "catalog_files": {name: str(path) for name, path in files.items()},
        "catalog_sha256": hashes,
    }
    (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
