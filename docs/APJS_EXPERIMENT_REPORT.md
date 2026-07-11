# APJS resubmission experiment report

## Status

The clean 0222 training protocol, independent 0228 calibration/evaluation protocol, four-model inference, core fixed-FPP analysis, selection functions, SNR/y reweighting, lens-redshift sanity check, and minimal E7 type-II diagnostic are complete. Final Zenodo deposition remains pending.

## Data roles

- Catalog 0222: training and checkpoint-selection validation only.
- Catalog 0228 calibration partition: operating-threshold and bin-edge calibration only.
- Catalog 0228 evaluation partition: final metrics, paired comparisons, and selection functions only.
- Catalog 0228 is an independently generated IID holdout from the same simulation priors, not an OOD or external dataset.

## Frozen models

Final-v1 contains PI-ResNet and CQT-DeiT (SEMD-inspired) checkpoints for ET SIS-Noisy and PM-Noisy. Exact paths, best epochs, validation AUC values, and SHA-256 sums are recorded in the checkpoint registry.

## Independent-test headline results

PI-ResNet evaluation AUC is 0.98877 for SIS and 0.98522 for PM, compared with 0.97670 and 0.96124 for CQT-DeiT. The paired block-AUC differences are 0.01207 (95% CI 0.00948--0.01421) and 0.02398 (0.02050--0.02713).

At the primary target FPP of 1e-3, PI-ResNet efficiency is 0.5354 (0.5097--0.5629) for SIS and 0.2274 (0.2109--0.2457) for PM. CQT-DeiT reaches 0.3771 (0.3589--0.3949) and 0.0926 (0.0817--0.1040). The paired PI-minus-CQT efficiency gains are 0.1583 (0.1366--0.1811) and 0.1349 (0.1171--0.1514).

The ordering is not uniform at target FPP 1e-4. For SIS, PI-ResNet is lower by 0.0400 (CI -0.0606 to -0.0166). For PM, the 0.0091 difference is not significant (CI -0.0017 to 0.0211). The manuscript must therefore use 1e-3 as the primary operating point and treat 1e-4 as a tail diagnostic.

## Selection effects

Scores and efficiencies depend strongly on weaker-image SNR. Efficiency decreases with increasing impact parameter and magnification imbalance. In the present lens models, impact parameter and flux ratio are tightly linked and must not be presented as independent discoveries.

## SNR/y matching

Common-support reweighting reduces, but does not remove, the SIS-PM efficiency gap. For PI-ResNet the common-support gap changes from about 0.310 before weighting to 0.220 after weighting. The matched residual gap is 0.2201 with 95% CI [0.1922, 0.2489]. For CQT-DeiT it is 0.2294 [0.2064, 0.2520]. Effective sample sizes are 1527 for SIS and 1468 for PM, and maximum weights are below 4.6. SNR/y therefore explains a substantial fraction, not all, of the lens-family gap.

## E7 controlled type-II diagnostic

The pre-specified 500-source physical/no-Morse comparison is a null result. SIS has mean physical-minus-control score -0.0137 (CI -0.0456 to 0.0180; Wilcoxon p=0.453), while PM has -0.00384 (-0.0284 to 0.0214; p=0.848). Fixed-1e-3-threshold efficiency differences also include zero. The present network therefore shows no measurable sensitivity to the controlled intervention, and no higher-mode/inclination subdivision is pursued.

## Lens-redshift sanity check

Across 60 deterministic delay interventions, the maximum classifier-input difference after peak realignment is 2.38e-7 and the maximum relative L2 difference is 2.14e-8. This establishes numerical invariance only within the present discrete peak-aligned verification setup at fixed source and y.

## Supported conclusion

The evidence supports PI-ResNet as a calibrated time-domain pair-ranking statistic with materially higher efficiency than the CQT-DeiT baseline at the primary 1e-3 per-pair FPP. It does not support uniform superiority at 1e-4, real-noise robustness, catalog-level FAR claims, complete SNR explanation of the SIS-PM gap, or measurable Morse-phase sensitivity in the minimal E7 probe.

## Reproducibility

All compact code, manifests, per-pair predictions, statistical tables, and hashes are stored in GitHub. Large checkpoints and CQT caches are hash-registered for the Zenodo release.
