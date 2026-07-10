# Result Provenance

`paper_metrics.csv` is a direct transcription of Tables 3-6 in the July 10, 2026 paper draft. The directories below contain the closest retained experiment artifacts. They are preserved without rewriting historical values.

## Table 3: ET comparison

PI-ResNet logs are in `table3_et_pi_resnet/logs/`. The four corresponding checkpoint hashes are listed in `checkpoint_manifest.tsv`.

For the noisy runs, the paper reports the best accuracy and best AUC observed over training, even when they occur at different epochs:

- SIS: accuracy `0.9560` at epoch 54 had AUC `0.9874`; AUC `0.9910` at epoch 253 had accuracy `0.8800`.
- PM: accuracy `0.9380` at epoch 83 had AUC `0.9874`; AUC `0.9897` at epoch 103 had accuracy `0.9220`.

SEMD optimization logs are in `table3_et_semd/logs/`. The exact noisy SEMD pairs printed in Table 3 are not jointly recoverable from the retained logs or per-sample predictions. They are present in the final paper table and Figure 6 composite.

## Table 4: simulated LIGO H1-L1 comparison

PI-ResNet training logs and evaluation figures are in `table4_ligo_pi_resnet/`. The retained `test.py` uses a 20% source split, but selects the confusion-matrix threshold using Youden's J statistic. The paper text states a fixed threshold of 0.5, so the evaluation implementation and prose are not identical.

SEMD evaluation figures are in `table4_ligo_semd/evaluation/`. `PM_noisy_on_pure_eval` is a historical directory name; shell history shows that it was subsequently overwritten by the PM-noisy evaluation.

The SEMD `test.py` evaluated all 20,000 generated images in each LIGO task rather than selecting only the held-out image split. The final Figure 6 composite scales those confusion-matrix counts to a 4,000-sample presentation. The exact AUC and accuracy values in Table 4 can be read from the retained ROC and confusion-matrix figures.

## Table 5: ablation

`table5_ablation/ablation_suite_final.log` contains the eight full training runs. The paper accuracies agree with the retained run maxima. Several paper AUC values do not agree with `SIS_final_ablation_results.txt` and `PM_final_ablation_results.txt`; for example, the retained full-model summaries contain SIS `0.9920` and PM `0.9881`, while the paper reports `0.9926` and `0.9918`.

## Table 6: cross-regime generalization

`table6_generalization/generalization_results.txt` directly contains the selected cross-regime results:

- SIS-trained to PM: `0.9240` accuracy, `0.9812` AUC.
- PM-trained to SIS: `0.9410` accuracy, `0.9884` AUC.

The same-regime diagonal entries are taken from Table 3.

## Figure provenance

- Figure 5: `figures/merged_performance_noisy_large.png`. The retained `code/actual_runs/et_pi_resnet/merged.py` constructs display curves analytically from hard-coded summary metrics rather than loading prediction probabilities.
- Figure 6: `figures/confusion_matrices_recreated_aligned_apj.png`. This is the final recreated composite; no script that exactly regenerates the composite was retained.
- Figures 7-8: `figures/fig11_*.png` and `figures/fig12_efficiency_sweep_new.png`. A 20% diagnostic rerun was recorded, but the currently retained plotting scripts have since diverged from the exact versions that wrote all final files.
- Figures 9-10: `figures/false_positive_local_peak.png` and `figures/fig_generalization_heatmap.png`. The numerical source for Figure 10 is Table 6.
