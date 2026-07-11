#!/usr/bin/env python3
"""Analyze frozen 0228 cross-lens checkpoint transfer for both architectures."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results/predictions"
OUT = ROOT / "results/transfer"
REPLICATES, SEED = 10000, 20260711

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

rows, details = [], {}
for target, source in (("sis", "pm"), ("pm", "sis")):
    details[target] = {}
    for model, prefix, score in (("pi_resnet", "pi", "pi_score"), ("cqt_deit", "cqt_deit", "cqt_deit_score")):
        in_path = PRED / f"{prefix}_predictions_0228_{target}.csv.gz"
        cross_path = PRED / f"{prefix}_predictions_0228_{target}_checkpoint_{source}.csv.gz"
        within = pd.read_csv(in_path); cross = pd.read_csv(cross_path)
        assert np.array_equal(within.pair_id, cross.pair_id)
        cal_mask = within.calibration_or_evaluation == "calibration"; ev_mask = ~cal_mask
        labels = within.loc[ev_mask, "label"].to_numpy()
        in_scores = within.loc[ev_mask, score].to_numpy(); cross_scores = cross.loc[ev_mask, score].to_numpy()
        in_threshold = float(np.quantile(within.loc[cal_mask & (within.label == 0), score], .999, interpolation="higher"))
        cross_threshold = float(np.quantile(cross.loc[cal_mask & (cross.label == 0), score], .999, interpolation="higher"))
        positive = labels == 1; negative = ~positive
        in_detected, cross_detected = in_scores >= in_threshold, cross_scores >= cross_threshold
        point = {
            "target_lens": target.upper(), "checkpoint_lens": source.upper(), "model": model,
            "within_auc": float(roc_auc_score(labels, in_scores)), "cross_auc": float(roc_auc_score(labels, cross_scores)),
            "auc_change_cross_minus_within": float(roc_auc_score(labels, cross_scores)-roc_auc_score(labels, in_scores)),
            "within_ap": float(average_precision_score(labels, in_scores)), "cross_ap": float(average_precision_score(labels, cross_scores)),
            "within_threshold_1e3": in_threshold, "cross_threshold_1e3": cross_threshold,
            "within_achieved_fpp": float(in_detected[negative].mean()), "cross_achieved_fpp": float(cross_detected[negative].mean()),
            "within_efficiency": float(in_detected[positive].mean()), "cross_efficiency": float(cross_detected[positive].mean()),
            "efficiency_change_cross_minus_within": float(cross_detected[positive].mean()-in_detected[positive].mean()),
        }
        blocks = within.loc[ev_mask, "source_block_id"].to_numpy(); unique = np.unique(blocks)
        block_delta = {block: (int(cross_detected[(blocks==block)&positive].sum()-in_detected[(blocks==block)&positive].sum()),
                               int(((blocks==block)&positive).sum())) for block in unique}
        rng=np.random.RandomState(SEED + (1 if target=="sis" else 2) + (10 if model=="cqt_deit" else 0)); samples=[]
        for _ in range(REPLICATES):
            chosen=rng.choice(unique,len(unique),replace=True); num=sum(block_delta[b][0] for b in chosen); den=sum(block_delta[b][1] for b in chosen); samples.append(num/den)
        point["efficiency_change_ci"]=[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]
        rows.append(point); details[target][model]=point
OUT.mkdir(parents=True,exist_ok=True)
table=OUT/"cross_lens_transfer.csv"; pd.DataFrame(rows).to_csv(table,index=False)
result=OUT/"cross_lens_transfer.json"; result.write_text(json.dumps({"status":"complete","bootstrap_replicates":REPLICATES,"results":details},indent=2))
(OUT/"SHA256SUMS").write_text(f"{sha(table)}  {table.name}\n{sha(result)}  {result.name}\n")
print(json.dumps(details,indent=2))
