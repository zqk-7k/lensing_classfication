# Zenodo frozen release bundle

This directory defines and builds the frozen reproducibility archive.

## Build on the experiment server

```bash
python release/zenodo/prepare_zenodo_artifacts.py
python release/zenodo/build_zenodo_manifest.py
python release/zenodo/build_frozen_release.py
```

The first command maps the four final-v1 checkpoints, pinned DeiT initialization, and five CQT arrays into canonical `artifacts/` paths. It uses hard links when possible and verifies each source against the checkpoint registry or CQT metadata.

The second command hashes the complete payload and updates `MANIFEST.json` and `MANIFEST.sha256`. Commit those two files before describing the archive as hash-verified against the committed manifest.

The final command verifies every manifest row again and writes a deterministic archive, archive checksum, file list, and archive information record under `release/zenodo/dist/`. The `dist/` directory is intentionally excluded from Git.

Before publishing a Zenodo deposition, the authors must provide:

1. Final author names and ORCID identifiers.
2. A repository/software license.
3. Zenodo account authorization or an access token.
4. Final title, author order, affiliations, and related paper identifier.

Do not publish or cite a DOI while `metadata.example.json` contains `REQUIRED_*` placeholders.
