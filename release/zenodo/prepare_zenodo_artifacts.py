#!/usr/bin/env python3
"""Materialize the canonical Zenodo artifact tree and verify registered hashes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_SOURCES = {
    "pi_resnet_sis_noisy": (
        "artifacts/training/pi_resnet_sis_noisy_seed42/best.pt",
        "runs/apjs_resubmission_final_v1/pi_resnet_sis_noisy_seed42/best.pt",
    ),
    "pi_resnet_pm_noisy": (
        "artifacts/training/pi_resnet_pm_noisy_seed42/best.pt",
        "runs/apjs_resubmission_final_v1/pi_resnet_pm_noisy_seed42/best.pt",
    ),
    "cqt_deit_sis_noisy": (
        "artifacts/training/cqt_deit_sis_noisy_seed42/best.pth",
        "runs/apjs_resubmission_final_v1/cqt_deit_sis_noisy_seed42/best.pth",
    ),
    "cqt_deit_pm_noisy": (
        "artifacts/training/cqt_deit_pm_noisy_seed42/best.pth",
        "runs/apjs_resubmission_final_v1/cqt_deit_pm_noisy_seed42/best.pth",
    ),
}

PRETRAINED_NAME = "deit_tiny_distilled_patch16_224-b40b3cf7.pth"
PRETRAINED_SOURCE = f"runs/pretrained/{PRETRAINED_NAME}"
PRETRAINED_TARGET = f"artifacts/pretrained/{PRETRAINED_NAME}"

CACHE_SOURCE_DIR = "runs/apjs_resubmission_final_v1/cqt_cache_0228"
CACHE_TARGET_DIR = "artifacts/preprocessing/cqt_cache_0228"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def materialize(source: Path, target: Path, expected_hash: str) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = digest(source)
    if source_hash != expected_hash:
        raise RuntimeError(f"Source hash mismatch: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    method = "existing"
    if target.exists():
        if not target.is_file() or digest(target) != expected_hash:
            raise RuntimeError(f"Existing target is not the registered artifact: {target}")
    else:
        try:
            os.link(source, target)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            method = "copy"

    return {
        "path": str(target.relative_to(ROOT)),
        "source": str(source.relative_to(ROOT)),
        "bytes": target.stat().st_size,
        "sha256": expected_hash,
        "method": method,
    }


def main() -> None:
    registry = json.loads(
        (ROOT / "experiments/reproducibility/manifests/checkpoint_registry.json").read_text()
    )
    cache_metadata = json.loads(
        (ROOT / "results/preprocessing/cqt_cache_metadata.json").read_text()
    )

    rows = []
    for key, (target_name, source_name) in CHECKPOINT_SOURCES.items():
        expected = registry["checkpoints"][key]["sha256"]
        rows.append(materialize(ROOT / source_name, ROOT / target_name, expected))

    pretrained_hashes = {
        row["pretrained_weights_sha256"]
        for row in registry["checkpoints"].values()
        if "pretrained_weights_sha256" in row
    }
    if len(pretrained_hashes) != 1:
        raise RuntimeError("Checkpoint registry does not define one pinned DeiT hash")
    rows.append(
        materialize(
            ROOT / PRETRAINED_SOURCE,
            ROOT / PRETRAINED_TARGET,
            pretrained_hashes.pop(),
        )
    )

    for name, metadata in sorted(cache_metadata["products"].items()):
        rows.append(
            materialize(
                ROOT / CACHE_SOURCE_DIR / name,
                ROOT / CACHE_TARGET_DIR / name,
                metadata["sha256"],
            )
        )

    print(
        json.dumps(
            {
                "artifact_count": len(rows),
                "total_bytes": sum(row["bytes"] for row in rows),
                "artifacts": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
