#!/usr/bin/env python3
"""Verify the committed manifest and build a deterministic Zenodo tarball."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = Path(__file__).resolve().parent
CONTROL_FILES = [
    "release/zenodo/MANIFEST.json",
    "release/zenodo/MANIFEST.sha256",
    "release/zenodo/README.md",
    "release/zenodo/RELEASE_NOTES.md",
    "release/zenodo/metadata.example.json",
    "release/zenodo/prepare_zenodo_artifacts.py",
    "release/zenodo/build_zenodo_manifest.py",
    "release/zenodo/build_frozen_release.py",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pi-resnet-gw-lensing-frozen-v1")
    parser.add_argument("--compression-level", type=int, default=6, choices=range(1, 10))
    args = parser.parse_args()

    manifest_path = RELEASE_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["files"]
    if manifest["file_count"] != len(rows):
        raise RuntimeError("Manifest file_count does not match its rows")
    if manifest["total_bytes"] != sum(row["bytes"] for row in rows):
        raise RuntimeError("Manifest total_bytes does not match its rows")

    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Size mismatch: {row['path']}")
        if digest(path) != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {row['path']}")

    for name in CONTROL_FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)

    archive_paths = [row["path"] for row in rows] + CONTROL_FILES
    if len(archive_paths) != len(set(archive_paths)):
        raise RuntimeError("Archive path list contains duplicates")

    dist = RELEASE_DIR / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    file_list = dist / f"{args.name}.filelist.txt"
    file_list.write_text("".join(f"{name}\n" for name in archive_paths), encoding="utf-8")

    tar_path = dist / f"{args.name}.tar"
    archive_path = dist / f"{args.name}.tar.gz"
    subprocess.run(
        [
            "tar",
            "--create",
            f"--file={tar_path}",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            f"--transform=s,^,{args.name}/,",
            f"--files-from={file_list}",
        ],
        cwd=ROOT,
        check=True,
    )
    with archive_path.open("wb") as output:
        subprocess.run(
            ["gzip", "-n", f"-{args.compression_level}", "-c", str(tar_path)],
            stdout=output,
            check=True,
        )
    tar_path.unlink()

    archive_hash = digest(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")

    members = subprocess.check_output(["tar", "-tzf", str(archive_path)], text=True).splitlines()
    if len(members) != len(archive_paths):
        raise RuntimeError("Archive member count does not match the build list")

    info = {
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_hash,
        "archive_member_count": len(members),
        "manifest_file_count": manifest["file_count"],
        "manifest_total_bytes": manifest["total_bytes"],
        "root_directory": args.name,
    }
    (dist / f"{args.name}.archive_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
