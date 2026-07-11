# Calibrated Pair Verification for Strongly Lensed Gravitational Waves

This repository provides the code, frozen evaluation manifests, prediction scores,
and derived results for a controlled comparison of two gravitational-wave
event-pair ranking methods:

- **PI-ResNet**, a one-dimensional Siamese residual network operating directly on
  peak-aligned strain segments; and
- **CQT--DeiT**, a two-dimensional baseline inspired by the published SEMD
  framework and operating on constant-Q representations.

The project addresses *pair verification*: given two already identified event
segments, it assigns a ranking score for consistency with strong lensing. It is
not a complete catalog search, false-alarm-rate pipeline, or real-noise analysis.

[中文说明](README.zh-CN.md)

## Main results

All headline values use an independently generated held-out catalog, a calibration
partition that is separate from final evaluation, and source-block uncertainty
estimation. At the preregistered primary operating point of per-pair false-positive
probability (FPP) $10^{-3}$, PI-ResNet reaches efficiencies of 53.5% (SIS) and
22.7% (point-mass), exceeding CQT--DeiT by 15.8 and 13.5 percentage points,
respectively. The complete tables, confidence intervals, tail behavior, selection
functions, transfer tests, and diagnostic null results are documented in
[docs/RESULTS.md](docs/RESULTS.md).

The $10^{-4}$ tail is reported without concealment: PI-ResNet is worse than the
baseline for SIS at that operating point. The repository therefore treats
$10^{-3}$ as primary and the more extreme tail as a resolution/calibration
diagnostic.

## Repository layout

```text
src/
  classifier/          PI-ResNet model and pair dataset
  cqt_deit/            CQT--DeiT comparison model
  generation/          simulated catalog generation programs
experiments/
  reproducibility/     locked split, inference, statistics, and figure scripts
results/
  core/                primary tables, figures, and unified predictions
  predictions/         model-specific per-pair scores
  transfer/            cross-lens-family transfer results
  diagnostics/         controlled physical/model diagnostics
  benchmarks/          measured throughput
  training/            compact training records (not model weights)
docs/                   methods, result index, limitations, and audit trail
release/zenodo/         archival-release manifest tools
```

Large input catalogs, CQT caches, pretrained weights, and trained checkpoints are
not stored in Git. Their expected locations and integrity records are described in
[docs/ARTIFACTS.md](docs/ARTIFACTS.md). Compact prediction files and all data needed
to reproduce the reported statistical analysis are retained in `results/`.

## Environment

Python 3.10 or later is recommended. Install the declared dependencies in a clean
environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place catalogs under `data/`, or pass an explicit `--data-root`. GPU inference and
training require a CUDA-enabled PyTorch installation.

## Reproduce the analysis

The public result tables and figures can be regenerated from the frozen prediction
files without model checkpoints:

```bash
python experiments/reproducibility/analyze_0228_core.py
python experiments/reproducibility/analyze_snr_matching_uncertainty.py
python experiments/reproducibility/make_core_figures.py
python experiments/reproducibility/analyze_cross_lens_transfer.py
```

For the full path from catalog audit through inference, see
[experiments/reproducibility/README.md](experiments/reproducibility/README.md).
The statistical rules were frozen before held-out scores were inspected; the
protocol is preserved in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

## Scientific scope

The current evidence is limited to simulated, stationary Gaussian noise; the
training prior uses $y\in[0.01,0.3]$; inputs are peak aligned; and the study does
not validate glitches or a catalog-level false-alarm rate. See
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) before reusing the scores operationally.

## Citation and archival release

The archival package is prepared under `release/zenodo/`. A DOI and formal citation
metadata will be added when the associated manuscript record is finalized. Until
then, cite the repository commit used in your analysis.
