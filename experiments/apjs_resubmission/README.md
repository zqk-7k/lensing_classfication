# ApJS resubmission experiments

This directory contains reproducible drivers for the independent ET noisy-catalog evaluation. Outputs are written under `runs/apjs_resubmission/` and are intentionally excluded from Git when they are large. Compact manifests and reports are copied to `results/apjs_resubmission/` after validation.

Training uses only the 0222 catalogs. The 0228 catalogs are reserved for frozen evaluation.

```bash
python experiments/apjs_resubmission/train_pi_resnet.py --lens SIS --device cuda:0
python experiments/apjs_resubmission/train_pi_resnet.py --lens PM --device cuda:1
```
