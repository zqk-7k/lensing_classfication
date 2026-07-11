#!/usr/bin/env python3
"""Post-hoc secondary diagnostics from released per-pair predictions.

Computes, at the FROZEN calibration thresholds (no re-selection, protocol unchanged):
  (a) sanity reproduction of published evaluation efficiencies;
  (b) detected-pair subset (rho_1 >= 8 and rho_2 >= 8) efficiencies + paired differences;
  (c) paired source-block bootstrap accuracy@0.5 difference (replaces exact McNemar p);
  (d) delete-one-block jackknife of the primary Delta-efficiency at FPP 1e-3.
All uncertainty via 10,000-replicate source-block bootstrap, seed 20260711.
Inputs: results/core/{unified_predictions_0228_*.csv.gz, core_results.json}
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "core"
SEED, REPS, RHO_DET = 20260711, 10000, 8.0
with (BASE / "core_results.json").open(encoding="utf-8") as handle:
    core = json.load(handle)
out = {"seed": SEED, "replicates": REPS, "rho_detected": RHO_DET, "note":
       "Post-hoc secondary diagnostics at frozen thresholds; primary locked analysis unchanged.",
       "families": {}}

for fam in ["sis", "pm"]:
    df = pd.read_csv(BASE / f"unified_predictions_0228_{fam}.csv.gz")
    ev = df[df.calibration_or_evaluation == "evaluation"].copy()
    pos = ev[ev.label == 1]
    thr = {m: {f: core["results"][fam]["thresholds"][m][f] for f in ["0.01", "0.001", "0.0001"]}
           for m in ["pi_resnet", "cqt_deit"]}
    scol = {"pi_resnet": "pi_score", "cqt_deit": "cqt_deit_score"}
    blocks = np.sort(ev.source_block_id.unique())
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(blocks), size=(REPS, len(blocks)))  # shared across all paired stats

    def block_counts(rows, indicator):
        num = rows.groupby("source_block_id")[indicator].sum().reindex(blocks, fill_value=0).to_numpy(float)
        den = rows.groupby("source_block_id")[indicator].size().reindex(blocks, fill_value=0).to_numpy(float)
        return num, den

    def boot_ratio(num, den):
        n, d = num[draws].sum(1), den[draws].sum(1)
        r = np.where(d > 0, n / np.maximum(d, 1), np.nan)
        return r

    fam_out = {"n_eval_pos": int(len(pos))}
    # (a)+(b): full and detected-subset efficiencies
    det_mask = (pos.rho_1 >= RHO_DET) & (pos.rho_2 >= RHO_DET)
    sub = pos[det_mask]
    fam_out["detected_subset"] = {"n": int(len(sub)), "fraction": float(det_mask.mean())}
    for label, rows in [("all_positives", pos), ("detected_subset", sub)]:
        for m in ["pi_resnet", "cqt_deit"]:
            for f in ["0.01", "0.001", "0.0001"]:
                key = f"{label}.{m}.eff@{f}"
                ind = (rows[scol[m]] >= thr[m][f]).astype(float)
                rows2 = rows.assign(_i=ind)
                num, den = block_counts(rows2, "_i")
                point = float(num.sum() / den.sum())
                bs = boot_ratio(num, den)
                fam_out[key] = {"point": point, "ci": [float(np.nanpercentile(bs, 2.5)),
                                                       float(np.nanpercentile(bs, 97.5))]}
        # paired delta at each FPP on this row set
        for f in ["0.01", "0.001", "0.0001"]:
            ip = (rows[scol["pi_resnet"]] >= thr["pi_resnet"][f]).astype(float)
            ic = (rows[scol["cqt_deit"]] >= thr["cqt_deit"][f]).astype(float)
            r2 = rows.assign(_p=ip, _c=ic)
            npi, den = block_counts(r2, "_p"); ncq, _ = block_counts(r2, "_c")
            d_point = float((npi.sum() - ncq.sum()) / den.sum())
            dbs = boot_ratio(npi, den) - boot_ratio(ncq, den)
            fam_out[f"{label}.delta_eff@{f}"] = {"point": d_point,
                "ci": [float(np.nanpercentile(dbs, 2.5)), float(np.nanpercentile(dbs, 97.5))]}
    # (c) paired block-bootstrap accuracy@0.5 difference over ALL evaluation pairs
    for m in ["pi_resnet", "cqt_deit"]:
        ev[f"_corr_{m}"] = ((ev[scol[m]] >= 0.5).astype(int) == ev.label).astype(float)
    npi, den = block_counts(ev, "_corr_pi_resnet"); ncq, _ = block_counts(ev, "_corr_cqt_deit")
    d_point = float((npi.sum() - ncq.sum()) / den.sum())
    dbs = boot_ratio(npi, den) - boot_ratio(ncq, den)
    b = int(((ev._corr_pi_resnet == 1) & (ev._corr_cqt_deit == 0)).sum())
    c = int(((ev._corr_pi_resnet == 0) & (ev._corr_cqt_deit == 1)).sum())
    fam_out["paired_block_accuracy_difference"] = {"point": d_point,
        "ci": [float(np.nanpercentile(dbs, 2.5)), float(np.nanpercentile(dbs, 97.5))],
        "discordant_b_pi_only": b, "discordant_c_cqt_only": c}
    # (d) delete-one-block jackknife of Delta-eff at 1e-3, all positives
    ip = (pos[scol["pi_resnet"]] >= thr["pi_resnet"]["0.001"]).astype(float)
    ic = (pos[scol["cqt_deit"]] >= thr["cqt_deit"]["0.001"]).astype(float)
    r2 = pos.assign(_p=ip, _c=ic)
    npi, den = block_counts(r2, "_p"); ncq, _ = block_counts(r2, "_c")
    jk = [float(((npi.sum()-npi[i]) - (ncq.sum()-ncq[i])) / (den.sum()-den[i])) for i in range(len(blocks))]
    fam_out["jackknife_delta_eff_1e3"] = {"values": jk, "min": min(jk), "max": max(jk),
                                          "all_same_sign": bool(all(v > 0 for v in jk) or all(v < 0 for v in jk))}
    out["families"][fam] = fam_out

with (BASE / "posthoc_diagnostics.json").open("w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
# concise printout
for fam, fo in out["families"].items():
    print(f"===== {fam.upper()} =====  n_pos={fo['n_eval_pos']}  detected n={fo['detected_subset']['n']} ({fo['detected_subset']['fraction']:.1%})")
    for lab in ["all_positives", "detected_subset"]:
        for f in ["0.01", "0.001", "0.0001"]:
            p = fo[f"{lab}.pi_resnet.eff@{f}"]; c = fo[f"{lab}.cqt_deit.eff@{f}"]; d = fo[f"{lab}.delta_eff@{f}"]
            print(f"  {lab:16s} FPP {f:>6s}: PI {p['point']:.4f} [{p['ci'][0]:.4f},{p['ci'][1]:.4f}]"
                  f" | CQT {c['point']:.4f} [{c['ci'][0]:.4f},{c['ci'][1]:.4f}]"
                  f" | dEff {d['point']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]")
    a = fo["paired_block_accuracy_difference"]
    print(f"  dAcc@0.5 {a['point']:+.5f} [{a['ci'][0]:+.5f},{a['ci'][1]:+.5f}]  (b={a['discordant_b_pi_only']}, c={a['discordant_c_cqt_only']})")
    j = fo["jackknife_delta_eff_1e3"]
    print(f"  jackknife dEff@1e-3: min {j['min']:+.4f} max {j['max']:+.4f} same-sign={j['all_same_sign']}")
