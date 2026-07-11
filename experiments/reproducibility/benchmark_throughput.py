#!/usr/bin/env python3
"""Measure preprocessing and GPU inference throughput for both models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "classifier"))
sys.path.insert(0, str(ROOT / "src" / "cqt_deit"))
sys.path.insert(0, str(ROOT / "experiments" / "reproducibility"))

from model import build_cqt_deit  # noqa: E402
from pair_dataset import pad_or_trim  # noqa: E402
from pi_resnet import PIResNet  # noqa: E402
from prepare_cqt_cache_0228 import spectrum  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=str(ROOT / "results" / "benchmarks" / "throughput" / "throughput.json"))
    return parser.parse_args()


def preprocess_pi(first, second, index):
    events = []
    for event in (first[index], second[index]):
        event = pad_or_trim(event, 8192, 2)
        event = (event - event.mean(axis=-1, keepdims=True)) / (event.std(axis=-1, keepdims=True) + 1e-8)
        events.append(event)
    return np.concatenate(events).astype(np.float32)


def preprocess_cqt(first, second, index):
    matrix = np.concatenate([spectrum(first[index]), spectrum(second[index])])
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    rgb = matplotlib.colormaps["viridis"](matrix, bytes=True)[..., :3]
    rgb = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (rgb - mean) / std


def timed_preprocessing(function, count, workers):
    start = time.perf_counter()
    if workers == 1:
        values = [function(index) for index in range(count)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(function, range(count)))
    return np.stack(values).astype(np.float32), time.perf_counter() - start


def timed_gpu(model, values, device, repetitions=100):
    inputs = torch.from_numpy(values).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            model(inputs)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / (repetitions * len(values))


def main():
    args = parse_args()
    data = Path(args.data_root) / "SIS_data_0228"
    first = np.load(data / "SIS_data_strain_1.npy", mmap_mode="r")
    second = np.load(data / "SIS_data_strain_2.npy", mmap_mode="r")
    pi_function = lambda index: preprocess_pi(first, second, index)
    cqt_function = lambda index: preprocess_cqt(first, second, index)
    pi_values, pi_serial = timed_preprocessing(pi_function, args.samples, 1)
    _, pi_parallel = timed_preprocessing(pi_function, args.samples, args.workers)
    cqt_values, cqt_serial = timed_preprocessing(cqt_function, args.samples, 1)
    _, cqt_parallel = timed_preprocessing(cqt_function, args.samples, args.workers)

    device = torch.device(args.device)
    pi_model = PIResNet(use_snake=False, use_se=True, use_pairwise_fusion=True).to(device).eval()
    pi_checkpoint = ROOT / "artifacts" / "training" / "pi_resnet_sis_noisy_seed42" / "best.pt"
    pi_model.load_state_dict(torch.load(pi_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    cqt_model = build_cqt_deit(pretrained=False).to(device).eval()
    cqt_checkpoint = ROOT / "artifacts" / "training" / "cqt_deit_sis_noisy_seed42" / "best.pth"
    cqt_model.load_state_dict(torch.load(cqt_checkpoint, map_location=device, weights_only=False)["model_state_dict"])

    pi_gpu = timed_gpu(pi_model, pi_values, device)
    cqt_gpu = timed_gpu(cqt_model, cqt_values, device)
    result = {
        "status": "complete",
        "hardware": {"gpu": torch.cuda.get_device_name(device), "cpu_threads": args.workers},
        "pairs": args.samples,
        "batch_size": args.batch_size,
        "pi_resnet": {
            "preprocess_ms_per_pair_serial": 1000 * pi_serial / args.samples,
            "preprocess_ms_per_pair_parallel_wall": 1000 * pi_parallel / args.samples,
            "gpu_inference_ms_per_pair": 1000 * pi_gpu,
            "gpu_pairs_per_second": 1 / pi_gpu,
        },
        "cqt_deit": {
            "preprocess_ms_per_pair_serial_including_cqt": 1000 * cqt_serial / args.samples,
            "preprocess_ms_per_pair_parallel_wall_including_cqt": 1000 * cqt_parallel / args.samples,
            "gpu_inference_ms_per_pair": 1000 * cqt_gpu,
            "gpu_pairs_per_second": 1 / cqt_gpu,
        },
        "note": "Parallel preprocessing values are pipeline wall time divided by pair count, not single-pair latency.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
