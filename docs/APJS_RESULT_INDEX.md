# APJS result index

| Result | Primary artifact | Producing code |
|---|---|---|
| 0222 shared source split | `experiments/apjs_resubmission/manifests/split_0222_seed42.npz` | `build_split_0222.py` |
| 0222/0228 independence | `experiments/apjs_resubmission/manifests/0222_0228_independence_audit.json` | `audit_0222_0228.py` |
| Evaluation protocol | `docs/APJS_EVALUATION_PROTOCOL_LOCK.md` | protocol tags v1/v1.1 |
| Shared 0228 pairs | `experiments/apjs_resubmission/manifests/0228_pairs/` | `build_0228_pair_manifests.py` |
| Frozen checkpoints | `experiments/apjs_resubmission/manifests/final_v1_checkpoint_registry.json` | final-v1 training drivers |
| PI 0228 scores | `runs/apjs_resubmission_final_v1/predictions_0228/pi_predictions_*.csv.gz` | `infer_pi_0228.py` |
| CQT 0228 scores | `runs/apjs_resubmission_final_v1/predictions_0228/cqt_deit_predictions_*.csv.gz` | `prepare_cqt_cache_0228.py`, `infer_cqt_0228.py` |
| Unified predictions | `runs/apjs_resubmission_final_v1/core_analysis_0228/unified_predictions_*.csv.gz` | `analyze_0228_core.py` |
| Fixed-FPP metrics | `fixed_fpp_primary_table.csv`, `core_results.json` | `analyze_0228_core.py` |
| Paired tests | `core_results.json` | `analyze_0228_core.py` |
| Selection functions | `selection_functions_*.csv`, `selection_rho1_rho2_*.csv` | `analyze_0228_core.py` |
| SNR/y matching | `snr_matched_sis_pm.csv` | `analyze_0228_core.py` |
| Lens-redshift check | `runs/apjs_resubmission_final_v1/zl_sanity/` | `zl_invariance_sanity.py` |
| E7 source contract | `experiments/apjs_resubmission/manifests/e7_typeii/` | `build_e7_manifest.py` |
| E7 paired results | `runs/apjs_resubmission_final_v1/e7_typeii/` | `run_e7_typeii_probe.py` |
| Cross-lens transfer | `runs/apjs_resubmission_final_v1/cross_lens_0228/` | `infer_*_0228.py`, `analyze_cross_lens_transfer.py` |
| Throughput | `runs/apjs_resubmission_final_v1/throughput/throughput.json` | `benchmark_throughput.py` |
| Logit tail check | `core_analysis_0228/logit_tail_check.json` | `analyze_logit_tail.py` |
| LIGO scope decision | `docs/APJS_LIGO_DECISION.md` | protocol audit |
| Artifact storage | `docs/APJS_ARTIFACT_STORAGE.md` | checkpoint/cache registries |
