# Artifact storage policy

All source code, locked manifests, audit reports, compact training records, and per-pair 0228 prediction scores are versioned in GitHub. Large binary artifacts are intentionally excluded from ordinary Git history and are identified by cryptographic hashes in the committed registries and metadata.

## Stored in GitHub

- Experiment and analysis source code.
- Evaluation protocol locks and Git tags.
- 0222 source split and 0228 shared pair manifests.
- Independence audits.
- Model configs, histories, split manifests, summaries, environment records, and train/validation predictions.
- PI-ResNet and CQT-DeiT per-pair 0228 prediction scores and metadata.
- CQT preprocessing validation and cache metadata.
- Statistical tables, figures, reports, and result hashes after completion.

## Reserved for Zenodo release

- PI-ResNet checkpoints (about 131 MB each; above GitHub's ordinary per-file limit).
- CQT-DeiT checkpoints.
- Official DeiT initialization weights.
- CQT event caches (250--500 MB per array).

The frozen checkpoint identities are recorded in `experiments/reproducibility/manifests/checkpoint_registry.json`. CQT cache hashes and shapes are recorded in `results/preprocessing/cqt_cache_metadata.json`. The Zenodo bundle must be verified against these hashes before release.
