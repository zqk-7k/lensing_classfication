#!/usr/bin/env python3
"""Create the locked source split and shared 0228 event-pair manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LENS_OFFSETS = {"SIS": 1000, "PM": 2000}
PARTITION_OFFSETS = {"calibration": 10, "evaluation": 20}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(size: int, n_calibration: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.random.RandomState(seed).permutation(size).astype(np.int64)
    return np.sort(ids[:n_calibration]), np.sort(ids[n_calibration:])


def assign_blocks(ids: np.ndarray, n_blocks: int, seed: int) -> dict[int, int]:
    shuffled = np.random.RandomState(seed).permutation(ids)
    return {int(source): int(position % n_blocks) for position, source in enumerate(shuffled)}


def sample_unique_hard(ids, blocks, count, rng):
    pairs = set()
    by_block = {block: [int(x) for x in ids if blocks[int(x)] == block] for block in range(max(blocks.values()) + 1)}
    while len(pairs) < count:
        block = int(rng.randint(len(by_block)))
        left, right = rng.choice(by_block[block], 2, replace=False)
        left_image, right_image = int(rng.randint(1, 3)), int(rng.randint(1, 3))
        pairs.add((int(left), left_image, int(right), right_image, block))
    return sorted(pairs)


def sample_unique_easy(lensed_ids, unlensed_ids, lens_blocks, unl_blocks, count, rng):
    pairs = set()
    lens_by_block = {block: [int(x) for x in lensed_ids if lens_blocks[int(x)] == block] for block in range(max(lens_blocks.values()) + 1)}
    unl_by_block = {block: [int(x) for x in unlensed_ids if unl_blocks[int(x)] == block] for block in range(max(unl_blocks.values()) + 1)}
    while len(pairs) < count:
        block = int(rng.randint(len(lens_by_block)))
        source = int(rng.choice(lens_by_block[block]))
        image = int(rng.randint(1, 3))
        unlensed = int(rng.choice(unl_by_block[block]))
        pairs.add((source, image, unlensed, block))
    return sorted(pairs)


def event_id(lens: str, source: int, image: int) -> str:
    return f"{lens.lower()}_0228_lensed_{source:05d}_img{image}"


def build_manifest(lens, partition, source_ids, unlensed_ids, source_blocks, unl_blocks,
                   lens_params, lens_values, snr1, snr2, unl_snr, background_count, seed):
    rows = []
    for source in source_ids:
        source = int(source)
        mu0, mu1 = float(lens_values.iloc[source]["mu_0"]), float(lens_values.iloc[source]["mu_1"])
        rows.append({
            "pair_id": f"{lens.lower()}_{partition}_pos_{source:05d}", "label": 1,
            "lens_family": lens, "left_event_id": event_id(lens, source, 1),
            "right_event_id": event_id(lens, source, 2), "left_source_id": source,
            "right_source_id": source, "source_block_id": source_blocks[source],
            "negative_type": "positive", "left_event_type": "lensed_img1",
            "right_event_type": "lensed_img2", "left_event_index": source,
            "right_event_index": source, "y": float(lens_params.iloc[source]["y"]),
            "mu_plus": mu0, "mu_minus": mu1, "flux_ratio": abs(mu0 / mu1),
            "rho_1": float(snr1[source]), "rho_2": float(snr2[source]),
            "rho_min": float(min(snr1[source], snr2[source])),
            "rho_max": float(max(snr1[source], snr2[source])),
            "calibration_or_evaluation": partition, "pair_seed": seed,
        })
    rng = np.random.RandomState(seed)
    hard_count = int(background_count * 0.7)
    easy_count = background_count - hard_count
    hard = sample_unique_hard(source_ids, source_blocks, hard_count, rng)
    easy = sample_unique_easy(source_ids, unlensed_ids, source_blocks, unl_blocks, easy_count, rng)
    for index, (left, left_image, right, right_image, block) in enumerate(hard):
        left_rho = float((snr1 if left_image == 1 else snr2)[left])
        right_rho = float((snr1 if right_image == 1 else snr2)[right])
        rows.append({
            "pair_id": f"{lens.lower()}_{partition}_hard_{index:06d}", "label": 0,
            "lens_family": lens, "left_event_id": event_id(lens, left, left_image),
            "right_event_id": event_id(lens, right, right_image), "left_source_id": left,
            "right_source_id": right, "source_block_id": block, "negative_type": "hard",
            "left_event_type": f"lensed_img{left_image}", "right_event_type": f"lensed_img{right_image}",
            "left_event_index": left, "right_event_index": right, "y": np.nan,
            "mu_plus": np.nan, "mu_minus": np.nan, "flux_ratio": np.nan,
            "rho_1": left_rho, "rho_2": right_rho, "rho_min": min(left_rho, right_rho),
            "rho_max": max(left_rho, right_rho), "calibration_or_evaluation": partition,
            "pair_seed": seed,
        })
    for index, (left, left_image, right, block) in enumerate(easy):
        left_rho, right_rho = float((snr1 if left_image == 1 else snr2)[left]), float(unl_snr[right])
        rows.append({
            "pair_id": f"{lens.lower()}_{partition}_easy_{index:06d}", "label": 0,
            "lens_family": lens, "left_event_id": event_id(lens, left, left_image),
            "right_event_id": f"unlensed_0228_{right:05d}", "left_source_id": left,
            "right_source_id": right, "source_block_id": block, "negative_type": "easy",
            "left_event_type": f"lensed_img{left_image}", "right_event_type": "unlensed",
            "left_event_index": left, "right_event_index": right, "y": np.nan,
            "mu_plus": np.nan, "mu_minus": np.nan, "flux_ratio": np.nan,
            "rho_1": left_rho, "rho_2": right_rho, "rho_min": min(left_rho, right_rho),
            "rho_max": max(left_rho, right_rho), "calibration_or_evaluation": partition,
            "pair_seed": seed,
        })
    frame = pd.DataFrame(rows)
    assert frame["pair_id"].is_unique
    assert int((frame.label == 0).sum()) == background_count
    assert int((frame.negative_type == "hard").sum()) == hard_count
    assert int((frame.negative_type == "easy").sum()) == easy_count
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/qkzhang")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/apjs_resubmission/manifests/0228_pairs"))
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--background-count", type=int, default=100000)
    parser.add_argument("--blocks", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sis_cal, sis_eval = split_ids(2500, 750, args.seed + 1)
    pm_cal, pm_eval = split_ids(2500, 750, args.seed + 2)
    unl_cal, unl_eval = split_ids(5000, 1500, args.seed + 3)
    np.savez_compressed(output / "split_0228_calibration_evaluation_seed20260711.npz",
                        sis_calibration_source_ids=sis_cal, sis_evaluation_source_ids=sis_eval,
                        pm_calibration_source_ids=pm_cal, pm_evaluation_source_ids=pm_eval,
                        unlensed_calibration_source_ids=unl_cal, unlensed_evaluation_source_ids=unl_eval,
                        random_seed=np.asarray(args.seed))
    products = {}
    for lens, cal_ids, eval_ids in (("SIS", sis_cal, sis_eval), ("PM", pm_cal, pm_eval)):
        base = data_root / f"{lens}_data_0228"
        lens_params, lens_values = pd.read_csv(base / "lens_params.csv"), pd.read_csv(base / "lens.csv")
        snr1 = np.load(base / f"{lens}_optimal_SNR_1.npy")
        snr2 = np.load(base / f"{lens}_optimal_SNR_2.npy")
        unl_snr = np.load(data_root / "Unlensed_data_0228/unlensed_optimal_SNR.npy")
        for partition, source_ids, unl_ids in (("calibration", cal_ids, unl_cal), ("evaluation", eval_ids, unl_eval)):
            seed = args.seed + LENS_OFFSETS[lens] + PARTITION_OFFSETS[partition]
            source_blocks = assign_blocks(source_ids, args.blocks, seed + 1)
            unl_blocks = assign_blocks(unl_ids, args.blocks, seed + 2)
            frame = build_manifest(lens, partition, source_ids, unl_ids, source_blocks, unl_blocks,
                                   lens_params, lens_values, snr1, snr2, unl_snr,
                                   args.background_count, seed)
            path = output / f"0228_{lens.lower()}_{partition}_pairs.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            products[path.name] = {"rows": len(frame), "positives": int(frame.label.sum()),
                                   "background": int((frame.label == 0).sum()), "sha256": sha256(path)}
    split_path = output / "split_0228_calibration_evaluation_seed20260711.npz"
    audit = {"status": "pass", "seed": args.seed, "blocks": args.blocks,
             "background_per_lens_partition": args.background_count,
             "split_sha256": sha256(split_path), "products": products,
             "score_inspection": False}
    (output / "pair_manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
