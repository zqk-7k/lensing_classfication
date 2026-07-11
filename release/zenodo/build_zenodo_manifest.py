#!/usr/bin/env python3
"""Build a SHA-256 inventory for the final Zenodo deposition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATTERNS = [
    "README.md",
    "README.zh-CN.md",
    "requirements.txt",
    "src/**/*.py",
    "experiments/reproducibility/*.py",
    "docs/*.md",
    "experiments/reproducibility/manifests/**/*",
    "results/predictions/*",
    "results/core/**/*",
    "results/transfer/*",
    "results/diagnostics/type_ii/*",
    "results/diagnostics/lens_redshift/*",
    "results/benchmarks/**/*",
    "results/preprocessing/*",
    "results/training/**/*",
    "artifacts/training/*_noisy_seed42/best.*",
    "artifacts/training/*_noisy_seed42/config.json",
    "artifacts/training/*_noisy_seed42/history.csv",
    "artifacts/training/*_noisy_seed42/summary.json",
    "artifacts/pretrained/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

files = {}
for pattern in PATTERNS:
    for path in ROOT.glob(pattern):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = path
rows = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sorted(files.items())]
output = Path(__file__).with_name("MANIFEST.json")
output.write_text(json.dumps({"files": rows, "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows)}, indent=2))
Path(__file__).with_name("MANIFEST.sha256").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in rows))
print(json.dumps({"file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows)}, indent=2))
