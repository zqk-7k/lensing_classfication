#!/usr/bin/env python3
"""Check whether logit-space calibration changes fixed-FPP tail selections."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "runs/apjs_resubmission_final_v1/core_analysis_0228"

results = {}
for lens in ("sis", "pm"):
    data = pd.read_csv(CORE / f"unified_predictions_0228_{lens}.csv.gz")
    cal = data[data.calibration_or_evaluation == "calibration"]
    ev = data[data.calibration_or_evaluation == "evaluation"]
    results[lens] = {}
    definitions = {
        "pi_resnet": ("pi_score", "pi_logit"),
        "cqt_deit": ("cqt_deit_score", None),
    }
    for model, (score, logit) in definitions.items():
        if logit is None:
            data["_ranking_logit"] = data.cqt_deit_logit_1 - data.cqt_deit_logit_0
            cal = data[data.calibration_or_evaluation == "calibration"]
            ev = data[data.calibration_or_evaluation == "evaluation"]
            logit = "_ranking_logit"
        model_rows = {}
        for fpp in (1e-3, 1e-4):
            score_t = float(np.quantile(cal.loc[cal.label == 0, score], 1-fpp, method="higher"))
            logit_t = float(np.quantile(cal.loc[cal.label == 0, logit], 1-fpp, method="higher"))
            score_selected = ev[score].to_numpy() >= score_t
            logit_selected = ev[logit].to_numpy() >= logit_t
            intersection = int((score_selected & logit_selected).sum()); union = int((score_selected | logit_selected).sum())
            model_rows[str(fpp)] = {
                "score_threshold": score_t, "logit_threshold": logit_t,
                "identical_selected_pairs": bool(np.array_equal(score_selected, logit_selected)),
                "selection_jaccard": intersection / union if union else 1.0,
                "score_selected": int(score_selected.sum()), "logit_selected": int(logit_selected.sum()),
                "score_unique_values": int(ev[score].nunique()), "logit_unique_values": int(ev[logit].nunique()),
            }
        results[lens][model] = model_rows
output = CORE / "logit_tail_check.json"
output.write_text(json.dumps({"status": "complete", "results": results,
                              "conclusion_rule": "A monotonic transform cannot change empirical rank selections unless stored-score ties/rounding alter the order."}, indent=2))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
(CORE / "logit_tail_check.sha256").write_text(f"{digest}  {output.name}\n")
print(json.dumps(results, indent=2))
