# Frozen evaluation protocol (version 1.1)

Status: locked before any model inference or score inspection on catalog 0228.

## Scope

The final experiment evaluates four frozen ET noisy checkpoints trained on catalog 0222: PI-ResNet and a CQT-DeiT (SEMD-inspired) baseline for SIS and PM lenses. Catalog 0228 is an independently generated IID holdout sampled from the same simulation priors. It is not external or out-of-distribution validation.

Checkpoint identities and hashes are fixed in `experiments/reproducibility/manifests/checkpoint_registry.json`. No checkpoint, preprocessing choice, or model hyperparameter may be changed after 0228 scores are inspected. Any software defect discovered later must be documented as a protocol deviation and all affected results regenerated.

## Held-out source split

- Split unit: source ID.
- Seed: 20260711.
- Per lens family: 750 calibration sources and 1750 final-evaluation sources.
- The 5000-event 0228 unlensed pool is split independently with the same deterministic protocol into 1500 calibration and 3500 final-evaluation events.
- Calibration and evaluation source/event IDs must be disjoint.

## Shared pair manifests

PI-ResNet and CQT-DeiT must consume the same event pairs and `pair_id` values.

For each lens family and each calibration/evaluation partition:

- One positive physical image pair per lensed source.
- 100,000 background pairs.
- Main background mixture: 70% hard negatives and 30% easy negatives.
- Hard negative: image 1 from one lensed source paired with image 2 from a different lensed source.
- Easy negative: a lensed image-1 event paired with an unlensed event.
- Background construction seed: 20260711 plus a fixed, documented lens/partition offset.
- Sources are assigned to 10 disjoint source blocks. Background pairs may only use events from the same block. Bootstrap resamples blocks, never individual pair rows. This was revised from 50 to 10 blocks before any 0228 inference because 50 calibration blocks cannot supply 70,000 unique within-block hard-negative event pairs; the revision prevents duplicated pairs.
- Hard-negative and easy-negative FPP are reported separately in addition to the fixed 70/30 mixture.

No additional background pairs will be added after score inspection. A future larger background study must be labeled separately and cannot replace the locked primary analysis.

## Calibration and primary operating points

For each model and lens family independently, thresholds are empirical upper-tail quantiles of calibration-background scores at target per-pair FPP values:

- 1e-2
- 1e-3 (primary operating point)
- 1e-4

Thresholds are frozen after calibration. Final evaluation reports achieved FPP and positive-pair efficiency at those thresholds. Zero observed false positives are reported with a binomial upper confidence bound, not as exact zero risk.

## Uncertainty and model comparison

- Primary confidence interval: 95% source-block bootstrap interval with 10,000 replicates and seed 20260711.
- Accuracy comparison: exact McNemar test on identical evaluation pairs, supplemented by paired block-bootstrap accuracy difference.
- AUC comparison: paired source-block bootstrap difference.
- Primary architecture comparison: paired efficiency difference at fixed target FPP, especially 1e-3.
- Conventional AUC, average precision, threshold-0.5 accuracy, and confusion matrices are secondary metrics.

## Selection functions

Selection functions use final-evaluation positive pairs and calibration-derived thresholds only. The primary presentation uses the PI-ResNet threshold at FPP 1e-3; other thresholds/models are supplementary.

- `y`: five equal-width bins over the simulated interval [0.01, 0.3].
- absolute magnification/flux ratio: five bins defined by fixed calibration-positive quintiles, then applied unchanged to evaluation.
- `rho_min`: five bins defined by fixed calibration-positive quintiles.
- `(rho_1, rho_2)`: a 5 by 5 grid using calibration-positive marginal quintiles.

Every bin reports efficiency, 95% interval, and sample count. Interpretation is restricted to the present BBH prior, Gaussian stationary noise, peak-aligned pair-verification setup, and simulated y interval.

## SNR-matched SIS-PM analysis

The locked matching variables are `log(rho_min)`, `log(rho_max)`, and `y`. The primary method is transparent three-dimensional histogram reweighting:

- six equal-width bins in each log-SNR dimension over pooled calibration common support;
- five equal-width y bins over [0.01, 0.3];
- retain only cells populated by both lens families;
- derive cell weights using calibration positives and apply the fixed map to evaluation positives.

Report weighted score distribution, AUC where a valid common background comparison exists, and fixed-FPP efficiency. Claims distinguish "fully explained" from "substantially explained" according to whether the paired/bootstrap interval for the residual gap includes zero.

## Primary outputs

1. Fixed-FPP efficiency table with achieved FPP and 95% intervals.
2. Efficiency-versus-FPP curve.
3. Paired PI-ResNet minus CQT-DeiT comparison table.
4. Selection functions in y, flux ratio, rho_min, and the two-SNR plane.
5. SNR-matched SIS-PM comparison.

The later type-II counterfactual probe and lens-redshift preprocessing sanity check are separate, pre-specified diagnostic experiments and do not alter the four frozen checkpoints or the primary 0228 analysis.
