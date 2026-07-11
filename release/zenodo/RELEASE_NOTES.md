# Frozen experiment release v1

This release freezes the complete reproducibility payload for the calibrated 0222-training/0228-evaluation experiments.

Included large artifacts:

- Two PI-ResNet checkpoints: SIS noisy and PM noisy.
- Two CQT-DeiT checkpoints: SIS noisy and PM noisy.
- The pinned `deit_tiny_distilled_patch16_224-b40b3cf7.pth` initialization.
- Five float32 CQT event caches covering SIS image 1/2, PM image 1/2, and the shared unlensed pool.
- All source manifests, per-pair predictions, training records, derived statistics, reports, and figures selected by `build_zenodo_manifest.py`.

The CQT caches are derived intermediates for the 0228 holdout catalog. Raw 0222/0228 waveform catalogs are not included in this bundle.

Every manifest payload file is verified by size and SHA-256 before archive creation. The final archive has a separate SHA-256 sidecar and an `archive_info.json` record in `release/zenodo/dist/`.

## Measured release size

- Manifest payload: 159 files, 1,906,391,230 bytes (1.775 GiB).
- Archive members: 167 files, including eight release-control files.
- Deterministic gzip archive: approximately 1.69 GB.

The manuscript statement should use these measured values rather than the earlier 89-file/400-MB estimate. Suggested wording:

> A frozen release containing the trained checkpoints, the pinned DeiT initialization weights, the CQT caches, and all derived results (159 manifest payload files; 1.91 GB uncompressed and approximately 1.69 GB as a gzip-compressed archive; hash-verified against the committed manifest) is deposited on Zenodo under DOI `10.5281/zenodo.XXXXXXX`.

The Zenodo DOI must not be cited until `metadata.example.json` placeholders have been replaced and the deposition has been published.
