# ApJS resubmission experiment status

Last updated: 2026-07-11 UTC

## Scope

This ledger tracks only experiments required to establish an independently tested, calibrated pair-ranking result. Paper editing and claims are out of scope here.

## Audit conclusions

1. Historical PI-ResNet and SEMD checkpoint files are unavailable. Git retains paths and SHA-256 values, but not the weight files. A filesystem-wide search and trash/open-file check found no recoverable copies.
2. Historical headline ET experiments used `SIS_data_0222`, `PM_data_0222`, and `Unlensed_data_0222`. The archived experiment code contains no evaluation reference to the corresponding `0228` arrays.
3. `SIS_data_0228` and `PM_data_0228` each contain 2,500 sources. Exact rounded-row comparison against the 0222 source manifests found zero overlapping sources for both lens families.
4. Consequently, the least expensive valid protocol is minimal retraining on 0222 followed by frozen evaluation on 0228. The 0228 catalog must never be used for checkpoint, threshold, hyperparameter, or preprocessing selection.
5. PI-ResNet historical runs used an 80/20 train-validation split and selected checkpoints on validation performance. No independent test was used for the headline values.
6. The available SEMD implementation is a publication-description implementation with custom CQT preprocessing. It must be reported as an SEMD-inspired baseline unless equivalence to the official implementation is later demonstrated.

## Experiment matrix

| ID | Task | Status | Required outputs |
| --- | --- | --- | --- |
| A0 | Code/data/checkpoint audit | complete | this ledger, source-overlap evidence |
| A1 | PI-ResNet SIS-noisy retrain on 0222 | pending | checkpoint, history, split manifest, config |
| A2 | PI-ResNet PM-noisy retrain on 0222 | pending | checkpoint, history, split manifest, config |
| A3 | SEMD-inspired SIS-noisy retrain on 0222 | pending | checkpoint, history, image split manifest, config |
| A4 | SEMD-inspired PM-noisy retrain on 0222 | pending | checkpoint, history, image split manifest, config |
| B1 | Frozen paired inference on 0228 | pending | per-pair logits/scores for both models |
| C1 | FPP calibration and source-block bootstrap | pending | thresholds and efficiency at 1e-2, 1e-3, 1e-4 |
| C2 | McNemar and paired bootstrap comparison | pending | paired tests and confidence intervals |
| D1 | Selection functions | pending | y, flux ratio, rho_min, rho1-rho2 outputs |
| D2 | SNR-matched SIS/PM | pending | matched/reweighted comparison |
| E1 | Type-II counterfactual probe | pending | physical/no-Morse controls and no-fusion comparison |
| E2 | Lens-redshift preprocessing sanity check | pending | tensor maximum absolute differences |

## Data policy

- Training/validation catalog: `/root/autodl-tmp/qkzhang/*_data_0222`.
- Independent test catalog: `/root/autodl-tmp/qkzhang/*_data_0228`.
- Independent test data are read-only for model development.
- Every run records Git commit, input paths, seed, split indices, command, package versions, checkpoint hash, and per-example scores.

## Important limitation

Because the historical checkpoints were deleted before this experiment request and were never committed upstream, the new results cannot be described as a frozen re-evaluation of the old checkpoints. They are minimal reproducibility retrains followed by strictly independent evaluation.
