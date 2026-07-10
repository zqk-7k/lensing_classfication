# PI-ResNet for Strongly Lensed Gravitational-Wave Pair Classification

This repository is the archival code-and-results package for the draft paper:

> **Time-Domain Deep Learning for Pairwise Identification of Strongly Lensed Gravitational-Wave Candidates**
> Fan Zhang, Qikai Zhang, Qiyuan Yang, Jiaqing Huang, Yong Yuan, and Xilong Fan
> Draft dated July 10, 2026

The paper PDF is available at
[`paper/PI_ResNet_GW_Lensing_APJ_draft_2026-07-10.pdf`](paper/PI_ResNet_GW_Lensing_APJ_draft_2026-07-10.pdf).

## Contents

- `paper/`: the exact PDF supplied with this release.
- `figures/`: the final figure files collected for the paper draft.
- `results/paper_metrics.csv`: machine-readable transcription of Tables 3-6.
- `results/`: training logs, evaluation figures, ablation summaries, and provenance notes.
- `code/actual_runs/`: historical code snapshots used for the ET, LIGO, and SEMD experiments.
- `code/data_generation/`: ET data-generation snapshot; LIGO generation code is beside the LIGO run code.
- `data/README.md`: dataset locations, array shapes, and publication status.

## Reported Results

| Detector setting | Model | SIS noisy | PM noisy |
|---|---|---:|---:|
| ET design noise | PI-ResNet | 95.60% / 0.9910 AUC | 93.80% / 0.9897 AUC |
| ET design noise | SEMD | 89.80% / 0.9726 AUC | 87.70% / 0.9500 AUC |
| Simulated LIGO H1-L1 noise | PI-ResNet | 84.03% / 0.9168 AUC | 78.25% / 0.8685 AUC |
| Simulated LIGO H1-L1 noise | SEMD | 78.43% / 0.8762 AUC | 74.37% / 0.8346 AUC |

Each cell is `accuracy / AUC`. See [`results/README.md`](results/README.md) before using these values: this archive distinguishes values directly recoverable from logs from values available only in final evaluation figures or the paper table.

## Environment

```bash
conda env create -f environment.yml
conda activate gw-lensing-pair-classifier
```

The historical scripts retain their original configuration style and some absolute server paths. Update the data roots in the corresponding `config*.py` before running them elsewhere.

## Data and Checkpoints

Generated strain arrays are hundreds of gigabytes and the PI-ResNet checkpoints are about 125 MiB each, so neither is stored in normal Git history. `data/README.md` records the arrays used on the experiment server, while `results/checkpoint_manifest.tsv` records checkpoint sizes and SHA-256 hashes.

No public dataset DOI was available at the time of this archival snapshot. The included logs and figures are sufficient to audit the reported tables, but a full clean-room rerun also requires publishing the arrays and checkpoints separately.

## Reproducibility Scope

`code/actual_runs/` is intentionally an archival snapshot rather than a cleaned rewrite. It preserves the code associated with the stored artifacts. Known protocol and artifact differences are documented in `results/README.md` instead of being silently normalized.
