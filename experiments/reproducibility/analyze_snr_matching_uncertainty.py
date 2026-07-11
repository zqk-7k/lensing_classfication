#!/usr/bin/env python3
"""Bootstrap the locked SNR/y-matched SIS-PM residual efficiency gap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "results/core"
MODELS = {"pi_resnet": "pi_score", "cqt_deit": "cqt_deit_score"}
SEED = 20260711
REPLICATES = 10000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data = {lens: pd.read_csv(CORE / f"unified_predictions_0228_{lens}.csv.gz") for lens in ("sis", "pm")}
    calibration = {lens: frame.query("calibration_or_evaluation == 'calibration' and label == 1").copy()
                   for lens, frame in data.items()}
    evaluation = {lens: frame.query("calibration_or_evaluation == 'evaluation' and label == 1").copy()
                  for lens, frame in data.items()}
    pooled = pd.concat(calibration.values())
    edges = {
        "log_rho_min": np.linspace(np.log(pooled.rho_min).min(), np.log(pooled.rho_min).max(), 7),
        "log_rho_max": np.linspace(np.log(pooled.rho_max).min(), np.log(pooled.rho_max).max(), 7),
        "y": np.linspace(0.01, 0.3, 6),
    }
    def cell_ids(frame):
        a = np.clip(np.digitize(np.log(frame.rho_min), edges["log_rho_min"]) - 1, 0, 5)
        b = np.clip(np.digitize(np.log(frame.rho_max), edges["log_rho_max"]) - 1, 0, 5)
        c = np.clip(np.digitize(frame.y, edges["y"]) - 1, 0, 4)
        return list(zip(a, b, c))
    counts = {lens: pd.Series(cell_ids(frame)).value_counts() for lens, frame in calibration.items()}
    common = sorted(set(counts["sis"].index) & set(counts["pm"].index))
    totals = {lens: sum(counts[lens][cell] for cell in common) for lens in counts}
    target = {cell: 0.5 * (counts["sis"][cell] / totals["sis"] + counts["pm"][cell] / totals["pm"])
              for cell in common}
    thresholds = json.load(open(CORE / "core_results.json"))["results"]
    rng = np.random.RandomState(SEED)
    rows, diagnostics = [], {}
    samples = {}
    for lens, frame in evaluation.items():
        frame = frame.copy(); frame["cell"] = cell_ids(frame); frame = frame[frame.cell.isin(common)].copy()
        frame["weight"] = frame.cell.map({cell: target[cell] / (counts[lens][cell] / totals[lens]) for cell in common})
        w = frame.weight.to_numpy()
        diagnostics[lens] = {
            "n": len(frame), "effective_sample_size": float(w.sum() ** 2 / np.square(w).sum()),
            "weight_min": float(w.min()), "weight_median": float(np.median(w)),
            "weight_p95": float(np.quantile(w, 0.95)), "weight_max": float(w.max()),
        }
        blocks = sorted(frame.source_block_id.unique())
        samples[lens] = {}
        for model, score in MODELS.items():
            threshold = thresholds[lens]["models"][model]["operating_points"]["0.001"]["threshold"]
            frame["detected"] = frame[score] >= threshold
            aggregates = {block: (float(part.loc[part.detected, "weight"].sum()), float(part.weight.sum()))
                          for block, part in frame.groupby("source_block_id")}
            values = np.empty(REPLICATES)
            for index in range(REPLICATES):
                chosen = rng.choice(blocks, len(blocks), replace=True)
                detected = sum(aggregates[int(block)][0] for block in chosen)
                weight = sum(aggregates[int(block)][1] for block in chosen)
                values[index] = detected / weight
            samples[lens][model] = values
            point = float(np.average(frame.detected, weights=frame.weight))
            rows.append({"lens": lens.upper(), "model": model, "matched_efficiency": point,
                         "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
                         "n": len(frame), "effective_sample_size": diagnostics[lens]["effective_sample_size"]})
    gaps = {}
    for model in MODELS:
        values = samples["sis"][model] - samples["pm"][model]
        gaps[model] = {"point": next(row["matched_efficiency"] for row in rows if row["lens"] == "SIS" and row["model"] == model) -
                               next(row["matched_efficiency"] for row in rows if row["lens"] == "PM" and row["model"] == model),
                       "ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
                       "fraction_bootstrap_gap_le_zero": float((values <= 0).mean())}
    table = CORE / "snr_matched_bootstrap.csv"; pd.DataFrame(rows).to_csv(table, index=False)
    result = {"status": "complete", "replicates": REPLICATES, "seed": SEED, "common_cells": len(common),
              "diagnostics": diagnostics, "matched_residual_gap_sis_minus_pm": gaps,
              "interpretation_rule": "SNR/y fully explains the gap only if the residual-gap interval includes zero."}
    path = CORE / "snr_matched_bootstrap.json"; path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (CORE / "snr_matched_bootstrap.sha256").write_text(f"{sha256(table)}  {table.name}\n{sha256(path)}  {path.name}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
