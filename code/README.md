# Code Archive

This directory contains snapshots copied from the experiment server on July 10, 2026.

## `actual_runs/et_pi_resnet`

ET PI-ResNet architecture, dataset, ablation, generalization, and diagnostic plotting code from `repro_old_data_v4/`. The exact standalone driver that produced all four Table 3 main-run logs was not retained; the architecture and dataset modules are the matching historical versions, and `main_ablation.py` is the exact Table 5 driver.

The historical `run_ablation_final.py` was transcoded from GB18030 to UTF-8 during publication so that the archived Python source can be parsed on a standard UTF-8 environment. CRLF line endings and trailing whitespace were also normalized across the code archive; executable statements were otherwise left unchanged.

## `actual_runs/ligo_pi_resnet`

Final multi-detector LIGO code from `ET_vs_LIGO_bf/`. In particular, `data_classifier.py` preserves the H1/L1 detector axis, and `main2.py` auto-detects the number of detector channels. This supersedes the older flattened-channel implementation.

## `actual_runs/semd`

Historical CQT/DeiT SEMD code, including preprocessing, training, sweeping, prediction, and evaluation scripts. Configuration files retain their last server-side task selection and absolute paths.

## `data_generation/et`

Latest retained ET generation scripts. Despite the historical parent-directory name, these scripts currently write to `*_0228`; the arrays used by the reported ET classifiers are the `*_0222` arrays documented in `data/README.md`. The exact source snapshot that originally generated the `0222` arrays was not found.

The LIGO SIS, PM, and unlensed generation scripts are stored in `actual_runs/ligo_pi_resnet/` because they are part of the final LIGO experiment snapshot.
