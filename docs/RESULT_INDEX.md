# Result index

| Result | Primary artifact | Producing code |
|---|---|---|
| 0222 shared source split | `experiments/reproducibility/manifests/split_0222_seed42.npz` | `build_split_0222.py` |
| 0222/0228 independence | `experiments/reproducibility/manifests/0222_0228_independence_audit.json` | `audit_0222_0228.py` |
| Evaluation protocol | `docs/EVALUATION_PROTOCOL.md` | protocol tags v1/v1.1 |
| Shared 0228 pairs | `experiments/reproducibility/manifests/0228_pairs/` | `build_0228_pair_manifests.py` |
| Frozen checkpoints | `experiments/reproducibility/manifests/checkpoint_registry.json` | frozen v1 training drivers |
| PI 0228 scores | `results/predictions/pi_predictions_*.csv.gz` | `infer_pi_0228.py` |
| CQT 0228 scores | `results/predictions/cqt_deit_predictions_*.csv.gz` | `prepare_cqt_cache_0228.py`, `infer_cqt_0228.py` |
| Unified predictions | `results/core/unified_predictions_*.csv.gz` | `analyze_0228_core.py` |
| Fixed-FPP metrics | `fixed_fpp_primary_table.csv`, `core_results.json` | `analyze_0228_core.py` |
| Paired tests | `core_results.json` | `analyze_0228_core.py` |
| Selection functions | `selection_functions_*.csv`, `selection_rho1_rho2_*.csv` | `analyze_0228_core.py` |
| SNR/y matching | `snr_matched_sis_pm.csv` | `analyze_0228_core.py` |
| Lens-redshift check | `results/diagnostics/lens_redshift/` | `zl_invariance_sanity.py` |
| E7 source contract | `experiments/reproducibility/manifests/e7_typeii/` | `build_e7_manifest.py` |
| E7 paired results | `results/diagnostics/type_ii/` | `run_e7_typeii_probe.py` |
| Cross-lens transfer | `results/transfer/` | `infer_*_0228.py`, `analyze_cross_lens_transfer.py` |
| Throughput | `results/benchmarks/throughput/throughput.json` | `benchmark_throughput.py` |
| Logit tail check | `results/core/logit_tail_check.json` | `analyze_logit_tail.py` |
| LIGO scope decision | `docs/LIMITATIONS.md` | protocol audit |
| Artifact storage | `docs/ARTIFACTS.md` | checkpoint/cache registries |
