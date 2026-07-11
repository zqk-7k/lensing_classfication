# Dataset contract

The analysis uses two independently generated simulated catalog versions drawn
from the same priors:

| Identifier | Role | Lens families | Sources per family |
| --- | --- | --- | ---: |
| `0222` | training and checkpoint-selection validation | SIS, point mass | 2,500 |
| `0228` | calibration and final evaluation | SIS, point mass | 2,500 |

The held-out `0228` catalog is an IID holdout, not an external or
out-of-distribution dataset. Its sources, waveform arrays, and noise realizations
were audited against `0222`; the recorded audit is
`experiments/reproducibility/manifests/0222_0228_independence_audit.json`.

Expected layout under `data/` (or an explicit `--data-root`) is:

```text
SIS_data_0222/       PM_data_0222/       Unlensed_data_0222/
SIS_data_0228/       PM_data_0228/       Unlensed_data_0228/
dataset_images_SIS_noisy_cqt/
dataset_images_PM_noisy_cqt/
```

Each lensed directory contains two image strain arrays, optimal-SNR arrays, source
metadata, and lens metadata. The unlensed directory contains the corresponding
event pool. CSV metadata are part of the dataset and must remain aligned with the
array row indices.

The `0228` sources are split at source level into 30% calibration and 70% final
evaluation using seed `20260711`. Thresholds are determined only from calibration
background scores. Positive-pair efficiency and achieved FPP are measured only on
the final-evaluation partition.

Raw waveform catalogs are too large for Git and are distributed as external
release artifacts. Their SHA-256 checksums are retained in the independence audit.
