#!/usr/bin/env python3
"""Plot the paired E7 score shifts from frozen result files."""

from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/apjs_resubmission_final_v1/e7_typeii"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for axis, lens in zip(axes, ("sis", "pm")):
    data = pd.read_csv(BASE / f"e7_scores_{lens}.csv.gz")
    delta = data.score_difference_physical_minus_control
    axis.hist(delta, bins=30, color="#1764ab", alpha=.8)
    axis.axvline(0, color="black", linewidth=1)
    axis.axvline(delta.mean(), color="#d1495b", linestyle="--", label=f"mean={delta.mean():.3f}")
    axis.set_title(lens.upper()); axis.set_xlabel("Physical minus no-Morse score"); axis.legend(frameon=False)
axes[0].set_ylabel("Sources")
png, pdf = BASE / "e7_score_shift.png", BASE / "e7_score_shift.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
(BASE / "e7_figure.sha256").write_text(f"{sha(png)}  {png.name}\n{sha(pdf)}  {pdf.name}\n")
