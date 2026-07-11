#!/usr/bin/env python3
"""Validate CQT preprocessing and cache per-event 0228 CQT magnitudes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {}
OUTPUTS = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_or_trim(x, target_len=8192, stride=2):
    x = np.asarray(x, dtype=np.float32)
    x = x[..., -target_len:] if x.shape[-1] >= target_len else np.pad(x, (target_len - x.shape[-1], 0))
    return x[..., ::stride] if stride > 1 else x


def spectrum(x):
    x = pad_or_trim(x)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    values = librosa.cqt(x, sr=2048, fmin=20.0, n_bins=112, bins_per_octave=24, hop_length=16)
    mag = librosa.amplitude_to_db(np.abs(values), ref=np.max)
    tensor = torch.tensor(mag, dtype=torch.float32)[None, None]
    return F.interpolate(tensor, size=(112, 224), mode="bilinear", align_corners=False).squeeze().numpy()


def render_pair(left, right):
    matrix = np.concatenate([left, right], axis=0)
    buffer = io.BytesIO()
    plt.imsave(buffer, matrix, cmap="viridis", format="png")
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def validate(data_root: Path):
    results = {}
    for lens in ("SIS", "PM"):
        base = data_root / f"{lens}_data_0222"
        a = np.load(base / f"{lens}_data_strain_1.npy", mmap_mode="r")
        b = np.load(base / f"{lens}_data_strain_2.npy", mmap_mode="r")
        generated = render_pair(spectrum(a[0]), spectrum(b[0]))
        reference_path = data_root / f"dataset_images_{lens}_noisy_cqt/lensed/pos_0000.png"
        reference = np.asarray(Image.open(reference_path).convert("RGB"))
        diff = np.abs(generated.astype(np.int16) - reference.astype(np.int16))
        results[lens] = {
            "reference": str(reference_path), "shape_equal": generated.shape == reference.shape,
            "max_abs_pixel_difference": int(diff.max()), "mean_abs_pixel_difference": float(diff.mean()),
            "exact_pixel_fraction": float((diff == 0).mean()),
        }
    return results


def init_worker():
    global INPUTS, OUTPUTS
    INPUTS, OUTPUTS = {}, {}


def compute_task(task):
    input_path, index, output_path, output_shape = task
    if input_path not in INPUTS:
        INPUTS[input_path] = np.load(input_path, mmap_mode="r")
    if output_path not in OUTPUTS:
        OUTPUTS[output_path] = np.lib.format.open_memmap(output_path, mode="r+", dtype=np.float32, shape=output_shape)
    OUTPUTS[output_path][index] = spectrum(INPUTS[input_path][index])
    return output_path, index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--output-dir", default=str(ROOT / "runs/apjs_resubmission_final_v1/cqt_cache_0228"))
    parser.add_argument("--workers", type=int, default=48)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = validate(data_root)
    validation_path = output / "preprocessing_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    # Matplotlib/Pillow versions can differ by one quantization level; larger differences indicate protocol drift.
    if any(not item["shape_equal"] or item["max_abs_pixel_difference"] > 1 for item in validation.values()):
        raise RuntimeError(f"CQT preprocessing validation failed: {validation}")

    specs = []
    for lens in ("SIS", "PM"):
        base = data_root / f"{lens}_data_0228"
        for image in (1, 2):
            specs.append((str(base / f"{lens}_data_strain_{image}.npy"), 2500,
                          str(output / f"{lens.lower()}_img{image}_spectra.npy")))
    specs.append((str(data_root / "Unlensed_data_0228/unlensed_data_strain.npy"), 5000,
                  str(output / "unlensed_spectra.npy")))
    tasks = []
    shape_tail = (112, 224)
    for input_path, count, output_path in specs:
        shape = (count,) + shape_tail
        np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape).flush()
        tasks.extend((input_path, index, output_path, shape) for index in range(count))
    context = mp.get_context("spawn")
    completed = 0
    with context.Pool(args.workers, initializer=init_worker) as pool:
        for _, _ in pool.imap_unordered(compute_task, tasks, chunksize=4):
            completed += 1
            if completed % 250 == 0:
                print(f"completed={completed}/{len(tasks)}", flush=True)
    products = {Path(output_path).name: {"shape": [count, 112, 224], "sha256": sha256(Path(output_path))}
                for _, count, output_path in specs}
    metadata = {"status": "pass", "events_cached": len(tasks), "workers": args.workers,
                "preprocessing_validation": validation, "products": products}
    (output / "cache_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
