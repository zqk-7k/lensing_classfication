#!/usr/bin/env python3
"""Freeze unified predictions and run the locked 0228 core analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
TARGET_FPPS = (1e-2, 1e-3, 1e-4)
MODELS = {"pi_resnet": "pi_score", "cqt_deit": "cqt_deit_score"}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def freeze_predictions(lens, prediction_dir, manifest_dir, output_dir):
    pi = pd.read_csv(prediction_dir / f"pi_predictions_0228_{lens}.csv.gz")
    cqt = pd.read_csv(prediction_dir / f"cqt_deit_predictions_0228_{lens}.csv.gz")
    assert len(pi) == len(cqt) == 202500
    assert pi.pair_id.is_unique and cqt.pair_id.is_unique
    merged = pi.merge(cqt[["pair_id", "label", "source_block_id", "negative_type",
                           "calibration_or_evaluation", "cqt_deit_logit_0",
                           "cqt_deit_logit_1", "cqt_deit_score"]], on="pair_id", suffixes=("", "_cqt"),
                      validate="one_to_one")
    for column in ("label", "source_block_id", "negative_type", "calibration_or_evaluation"):
        assert (merged[column] == merged[f"{column}_cqt"]).all()
        merged.drop(columns=f"{column}_cqt", inplace=True)
    manifests = pd.concat([pd.read_csv(manifest_dir / f"0228_{lens}_{part}_pairs.csv.gz")
                           for part in ("calibration", "evaluation")], ignore_index=True)
    physical = ["pair_id", "lens_family", "left_event_id", "right_event_id", "left_source_id",
                "right_source_id", "y", "mu_plus", "mu_minus", "flux_ratio", "rho_1", "rho_2",
                "rho_min", "rho_max"]
    merged = merged.merge(manifests[physical], on="pair_id", validate="one_to_one")
    assert len(merged) == 202500 and merged.pair_id.is_unique
    path = output_dir / f"unified_predictions_0228_{lens}.csv.gz"
    merged.to_csv(path, index=False, compression="gzip")
    return merged, path


def block_bootstrap(frame, thresholds, replicates, seed):
    blocks = sorted(frame.source_block_id.unique())
    rng = np.random.RandomState(seed)
    metrics = {model: {fpp: {"fpp": [], "efficiency": []} for fpp in TARGET_FPPS} for model in MODELS}
    delta = {fpp: [] for fpp in TARGET_FPPS}
    block_rows = {block: frame[frame.source_block_id == block] for block in blocks}
    for _ in range(replicates):
        sampled = rng.choice(blocks, len(blocks), replace=True)
        for fpp in TARGET_FPPS:
            efficiencies = {}
            for model, score in MODELS.items():
                threshold = thresholds[model][fpp]
                false, negatives, detected, positives = 0, 0, 0, 0
                for block in sampled:
                    part = block_rows[int(block)]
                    neg, pos = part[part.label == 0], part[part.label == 1]
                    false += int((neg[score] >= threshold).sum()); negatives += len(neg)
                    detected += int((pos[score] >= threshold).sum()); positives += len(pos)
                value = detected / positives
                metrics[model][fpp]["fpp"].append(false / negatives)
                metrics[model][fpp]["efficiency"].append(value)
                efficiencies[model] = value
            delta[fpp].append(efficiencies["pi_resnet"] - efficiencies["cqt_deit"])
    return metrics, delta


def selection_tables(lens, calibration, evaluation, thresholds, output_dir):
    cal_pos, ev_pos = calibration[calibration.label == 1], evaluation[evaluation.label == 1].copy()
    definitions = {
        "y": np.linspace(0.01, 0.3, 6),
        "flux_ratio": np.unique(np.quantile(cal_pos.flux_ratio, np.linspace(0, 1, 6))),
        "rho_min": np.unique(np.quantile(cal_pos.rho_min, np.linspace(0, 1, 6))),
    }
    rows = []
    for variable, edges in definitions.items():
        bins = pd.cut(ev_pos[variable], edges, include_lowest=True, duplicates="drop")
        for model, score in MODELS.items():
            threshold = thresholds[model][1e-3]
            for category in bins.cat.categories:
                mask = bins == category
                total = int(mask.sum()); successes = int((ev_pos.loc[mask, score] >= threshold).sum())
                low, high = wilson(successes, total)
                rows.append({"lens": lens.upper(), "model": model, "variable": variable,
                             "bin_left": float(category.left), "bin_right": float(category.right),
                             "n": total, "detected": successes, "efficiency": successes / total if total else np.nan,
                             "ci_low": low, "ci_high": high, "target_fpp": 1e-3})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / f"selection_functions_0228_{lens}.csv", index=False)
    # Locked 5x5 two-SNR plane from calibration marginal quintiles.
    e1 = np.unique(np.quantile(cal_pos.rho_1, np.linspace(0, 1, 6)))
    e2 = np.unique(np.quantile(cal_pos.rho_2, np.linspace(0, 1, 6)))
    b1, b2 = pd.cut(ev_pos.rho_1, e1, include_lowest=True), pd.cut(ev_pos.rho_2, e2, include_lowest=True)
    grid = []
    for model, score in MODELS.items():
        threshold = thresholds[model][1e-3]
        for i in b1.cat.categories:
            for j in b2.cat.categories:
                mask = (b1 == i) & (b2 == j); total = int(mask.sum())
                detected = int((ev_pos.loc[mask, score] >= threshold).sum())
                low, high = wilson(detected, total)
                grid.append({"lens": lens.upper(), "model": model, "rho1_left": float(i.left),
                             "rho1_right": float(i.right), "rho2_left": float(j.left),
                             "rho2_right": float(j.right), "n": total, "detected": detected,
                             "efficiency": detected / total if total else np.nan, "ci_low": low, "ci_high": high})
    pd.DataFrame(grid).to_csv(output_dir / f"selection_rho1_rho2_0228_{lens}.csv", index=False)


def snr_matched(datasets, thresholds, output_dir):
    calibration = {lens: data[data.calibration_or_evaluation == "calibration"].query("label == 1").copy()
                   for lens, data in datasets.items()}
    evaluation = {lens: data[data.calibration_or_evaluation == "evaluation"].query("label == 1").copy()
                  for lens, data in datasets.items()}
    pooled = pd.concat(calibration.values())
    xedges = np.linspace(np.log(pooled.rho_min).min(), np.log(pooled.rho_min).max(), 7)
    yedges = np.linspace(np.log(pooled.rho_max).min(), np.log(pooled.rho_max).max(), 7)
    zedges = np.linspace(0.01, 0.3, 6)
    def cells(frame):
        x = np.clip(np.digitize(np.log(frame.rho_min), xedges) - 1, 0, 5)
        y = np.clip(np.digitize(np.log(frame.rho_max), yedges) - 1, 0, 5)
        z = np.clip(np.digitize(frame.y, zedges) - 1, 0, 4)
        return list(zip(x, y, z))
    cal_counts = {lens: pd.Series(cells(frame)).value_counts() for lens, frame in calibration.items()}
    common = set(cal_counts["sis"].index) & set(cal_counts["pm"].index)
    target = {cell: 0.5 * (cal_counts["sis"][cell] / sum(cal_counts["sis"][c] for c in common) +
                           cal_counts["pm"][cell] / sum(cal_counts["pm"][c] for c in common)) for cell in common}
    rows = []
    for lens, frame in evaluation.items():
        frame = frame.copy(); frame["cell"] = cells(frame); frame = frame[frame.cell.isin(common)]
        cal_total = sum(cal_counts[lens][cell] for cell in common)
        weights = {cell: target[cell] / (cal_counts[lens][cell] / cal_total) for cell in common}
        w = frame.cell.map(weights).to_numpy(dtype=float)
        for model, score in MODELS.items():
            detected = (frame[score].to_numpy() >= thresholds[lens][model][1e-3]).astype(float)
            rows.append({"lens": lens.upper(), "model": model, "n_common_support": len(frame),
                         "common_cells": len(common), "weighted_efficiency": float(np.average(detected, weights=w)),
                         "weighted_mean_score": float(np.average(frame[score], weights=w)),
                         "unweighted_efficiency_common_support": float(detected.mean())})
    pd.DataFrame(rows).to_csv(output_dir / "snr_matched_sis_pm.csv", index=False)
    return {"common_cells": len(common), "rows": rows}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default=str(ROOT / "results/predictions"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "experiments/reproducibility/manifests/0228_pairs"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/core"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main():
    args = parse_args(); prediction_dir, manifest_dir = Path(args.prediction_dir), Path(args.manifest_dir)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    datasets, unified_paths, thresholds_all, summary, primary_rows = {}, {}, {}, {}, []
    for lens in ("sis", "pm"):
        data, path = freeze_predictions(lens, prediction_dir, manifest_dir, output_dir)
        datasets[lens], unified_paths[lens] = data, path
        cal = data[data.calibration_or_evaluation == "calibration"]
        ev = data[data.calibration_or_evaluation == "evaluation"]
        thresholds = {model: {fpp: float(np.quantile(cal.loc[cal.label == 0, score], 1 - fpp, interpolation="higher"))
                              for fpp in TARGET_FPPS} for model, score in MODELS.items()}
        thresholds_all[lens] = thresholds
        boot, delta = block_bootstrap(ev, thresholds, args.bootstrap_replicates, args.seed + (1 if lens == "sis" else 2))
        lens_summary = {"thresholds": thresholds, "models": {}, "paired": {}}
        for model, score in MODELS.items():
            labels = ev.label.to_numpy(); scores = ev[score].to_numpy()
            model_result = {"auc": float(roc_auc_score(labels, scores)),
                            "average_precision": float(average_precision_score(labels, scores)),
                            "accuracy_at_0.5": float(accuracy_score(labels, scores >= 0.5)), "operating_points": {}}
            for fpp in TARGET_FPPS:
                threshold = thresholds[model][fpp]; neg, pos = ev[ev.label == 0], ev[ev.label == 1]
                achieved = float((neg[score] >= threshold).mean()); efficiency = float((pos[score] >= threshold).mean())
                hard = float((neg.loc[neg.negative_type == "hard", score] >= threshold).mean())
                easy = float((neg.loc[neg.negative_type == "easy", score] >= threshold).mean())
                result = {"threshold": threshold, "achieved_fpp": achieved, "hard_fpp": hard, "easy_fpp": easy,
                          "efficiency": efficiency, "fpp_ci": interval(boot[model][fpp]["fpp"]),
                          "efficiency_ci": interval(boot[model][fpp]["efficiency"])}
                model_result["operating_points"][str(fpp)] = result
                primary_rows.append({"lens": lens.upper(), "model": model, "target_fpp": fpp, **result})
            lens_summary["models"][model] = model_result
        pi_pred, cq_pred, labels = ev.pi_score.to_numpy() >= 0.5, ev.cqt_deit_score.to_numpy() >= 0.5, ev.label.to_numpy(bool)
        b = int((pi_pred == labels).astype(int)[(cq_pred != labels)].sum())
        c = int((cq_pred == labels).astype(int)[(pi_pred != labels)].sum())
        pvalue = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        block_aucs = {model: [] for model in MODELS}
        for block in sorted(ev.source_block_id.unique()):
            part = ev[ev.source_block_id == block]
            for model, score in MODELS.items(): block_aucs[model].append(roc_auc_score(part.label, part[score]))
        rng = np.random.RandomState(args.seed + 100)
        auc_delta = []
        for _ in range(args.bootstrap_replicates):
            indices = rng.randint(0, len(block_aucs["pi_resnet"]), len(block_aucs["pi_resnet"]))
            auc_delta.append(float(np.mean(np.asarray(block_aucs["pi_resnet"])[indices] - np.asarray(block_aucs["cqt_deit"])[indices])))
        lens_summary["paired"] = {"mcnemar_b_pi_only_correct": b, "mcnemar_c_cqt_only_correct": c,
                                   "mcnemar_exact_pvalue": pvalue,
                                   "auc_difference_pi_minus_cqt": float(roc_auc_score(ev.label, ev.pi_score) - roc_auc_score(ev.label, ev.cqt_deit_score)),
                                   "block_auc_difference_ci": interval(auc_delta),
                                   "efficiency_difference": {str(fpp): {"point": lens_summary["models"]["pi_resnet"]["operating_points"][str(fpp)]["efficiency"] - lens_summary["models"]["cqt_deit"]["operating_points"][str(fpp)]["efficiency"],
                                                                                 "ci": interval(delta[fpp])} for fpp in TARGET_FPPS}}
        summary[lens] = lens_summary
        selection_tables(lens, cal, ev, thresholds, output_dir)
    matched = snr_matched(datasets, thresholds_all, output_dir)
    pd.DataFrame(primary_rows).to_csv(output_dir / "fixed_fpp_primary_table.csv", index=False)
    result = {"status": "complete", "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
              "unified_predictions": {lens: {"path": str(path), "sha256": sha256(path)} for lens, path in unified_paths.items()},
              "results": summary, "snr_matched": matched}
    result_path = output_dir / "core_results.json"; result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    (output_dir / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({lens: summary[lens]["paired"] for lens in summary}, indent=2))


if __name__ == "__main__":
    main()
