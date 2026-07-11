#!/usr/bin/env python3
"""Build the complete SHA-256 inventory for the frozen Zenodo deposition."""

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
    "results/**/*",
    "artifacts/training/*_noisy_seed42/best.*",
    "artifacts/pretrained/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
    "artifacts/preprocessing/cqt_cache_0228/*.npy",
]


def digest(path: Path) -> str:
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

rows = [
    {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
    for name, path in sorted(files.items())
]
payload = {
    "schema_version": 1,
    "files": rows,
    "file_count": len(rows),
    "total_bytes": sum(row["bytes"] for row in rows),
}
output = Path(__file__).with_name("MANIFEST.json")
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
Path(__file__).with_name("MANIFEST.sha256").write_text(
    "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
    encoding="utf-8",
)
print(json.dumps({"file_count": len(rows), "total_bytes": payload["total_bytes"]}, indent=2))
