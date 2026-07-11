# APJS experiment execution log

## Protocol and training

- Archived four original runs as Protocol-v0 pilots after identifying shared-unlensed-pool and image-level SEMD split issues.
- Built the shared source-level 0222 split with seed 42: SIS 2000/500, PM 2000/500, and unlensed 4000/1000 train/validation sources.
- Retrained four final-v1 checkpoints for 300 epochs and selected checkpoints by validation AUC only.
- Froze checkpoint hashes in `final_v1_checkpoint_registry.json`.

## Evaluation lock and holdout audit

- Locked the evaluation protocol before 0228 score inspection and tagged `apjs-eval-protocol-v1`.
- Revised 50 source blocks to 10 before inference to permit 100,000 unique within-block background pairs; tagged `apjs-eval-protocol-v1.1`.
- Audited 0222/0228 file hashes, exact source-row overlap, and sampled waveform hashes. The audit passed; seed metadata was unavailable and is not claimed.

## Shared pairs and inference

- Split 0228 at source level into 30% calibration and 70% evaluation using seed 20260711.
- Generated four shared manifests, each with one positive per source and 100,000 unique background pairs (70% hard, 30% easy).
- Ran PI-ResNet and CQT-DeiT on exactly the same `pair_id` values.
- Validated CQT preprocessing against stored 0222 PNGs before caching 15,000 event spectra.
- Froze unified SIS and PM prediction files and recorded SHA-256 sums.

## Core statistics

- Calibrated thresholds at FPP 1e-2, 1e-3, and 1e-4 on calibration background only.
- Evaluated achieved FPP and positive efficiency on the final evaluation partition.
- Completed 10,000 source-block bootstrap replicates.
- Completed exact McNemar, paired block-AUC, paired fixed-FPP efficiency, selection-function, and SNR/y matching analyses.
- Completed a 60-trial discrete peak-alignment lens-redshift sanity check.

## Pending

- Freeze and generate the minimal E7 physical/no-Morse counterfactual dataset.
- Add uncertainty and effective-sample-size diagnostics to SNR/y matching.
- Finalize figures, result index, Zenodo bundle, and E7 report.
