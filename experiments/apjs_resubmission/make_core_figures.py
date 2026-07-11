#!/usr/bin/env python3
"""Generate publication-oriented figures exclusively from frozen core tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "runs/apjs_resubmission_final_v1/core_analysis_0228"
FIG = CORE / "figures"
COLORS = {"pi_resnet": "#1764ab", "cqt_deit": "#d1495b"}
LABELS = {"pi_resnet": "PI-ResNet", "cqt_deit": "CQT-DeiT"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    png, pdf = FIG / f"{name}.png", FIG / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(CORE / "fixed_fpp_primary_table.csv")
    fixed["ci_low"] = fixed.efficiency_ci.str.extract(r"\[([^,]+)")[0].astype(float)
    fixed["ci_high"] = fixed.efficiency_ci.str.extract(r",\s*([^\]]+)")[0].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, lens in zip(axes, ("SIS", "PM")):
        for model in COLORS:
            part = fixed[(fixed.lens == lens) & (fixed.model == model)].sort_values("target_fpp")
            axis.plot(part.target_fpp, part.efficiency, marker="o", color=COLORS[model], label=LABELS[model])
            axis.fill_between(part.target_fpp, part.ci_low, part.ci_high, color=COLORS[model], alpha=.18)
        axis.set_xscale("log"); axis.set_title(lens); axis.set_xlabel("Target per-pair FPP")
        axis.grid(alpha=.25); axis.invert_xaxis()
    axes[0].set_ylabel("Detection efficiency"); axes[1].legend(frameon=False)
    save(fig, "fixed_fpp_efficiency")

    variables = [("y", "Impact parameter y"), ("flux_ratio", "Absolute magnification ratio"),
                 ("rho_min", "Weaker-image SNR")]
    for lens in ("sis", "pm"):
        data = pd.read_csv(CORE / f"selection_functions_0228_{lens}.csv")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for axis, (variable, label) in zip(axes, variables):
            for model in COLORS:
                part = data[(data.variable == variable) & (data.model == model)].copy()
                x = (part.bin_left + part.bin_right) / 2
                axis.errorbar(x, part.efficiency, yerr=[part.efficiency - part.ci_low, part.ci_high - part.efficiency],
                              marker="o", capsize=2, color=COLORS[model], label=LABELS[model])
            axis.set_xlabel(label); axis.grid(alpha=.25); axis.set_ylim(0, 1.03)
        axes[0].set_ylabel("Efficiency at target FPP=1e-3"); axes[-1].legend(frameon=False)
        fig.suptitle(lens.upper())
        save(fig, f"selection_functions_{lens}")

    matched = pd.read_csv(CORE / "snr_matched_sis_pm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True)
    for axis, model in zip(axes, COLORS):
        part = matched[matched.model == model]
        x = np.arange(2); width = .34
        axis.bar(x - width/2, part.unweighted_efficiency_common_support, width, label="Unmatched", color="#a9a9a9")
        axis.bar(x + width/2, part.weighted_efficiency, width, label="SNR/y matched", color=COLORS[model])
        axis.set_xticks(x, part.lens); axis.set_title(LABELS[model]); axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Efficiency at lens-specific FPP=1e-3"); axes[1].legend(frameon=False)
    save(fig, "snr_matched_sis_pm")

    files = sorted(FIG.glob("*"))
    (FIG / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(f"generated {len(files)} figure files")


if __name__ == "__main__":
    main()
