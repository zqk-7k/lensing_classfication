# Zenodo release bundle

This directory defines the reproducible experiment archive. Run `build_zenodo_manifest.py` on the experiment server after all paths in the script are present. The script hashes source manifests, predictions, checkpoints, model records, and final reports without copying multi-gigabyte CQT caches into Git.

Before creating a deposition, the authors must provide:

1. Final author names and ORCID identifiers.
2. A repository/software license (none is currently present).
3. Zenodo account authorization or an access token.
4. Final title, author order, affiliations, and related paper identifier.

Do not publish a DOI while `metadata.example.json` still contains `REQUIRED_*` placeholders.

Large files reserved for Zenodo include the four frozen checkpoints and official DeiT initialization weights. CQT caches are reproducible intermediates and are hash-registered; include them only if archive quota permits.
