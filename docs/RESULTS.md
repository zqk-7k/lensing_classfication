# Independent evaluation results

## Status

The clean 0222 training protocol, independent 0228 calibration/evaluation protocol, four-model inference, core fixed-FPP analysis, selection functions, SNR/y reweighting, lens-redshift sanity check, and minimal E7 type-II diagnostic are complete. Final Zenodo deposition remains pending.

## Data roles

- Catalog 0222: training and checkpoint-selection validation only.
- Catalog 0228 calibration partition: operating-threshold and bin-edge calibration only.
- Catalog 0228 evaluation partition: final metrics, paired comparisons, and selection functions only.
- Catalog 0228 is an independently generated IID holdout from the same simulation priors, not an OOD or external dataset.

## Frozen models

The frozen release contains PI-ResNet and CQT-DeiT (SEMD-inspired) checkpoints for ET SIS-Noisy and PM-Noisy. Exact paths, best epochs, validation AUC values, and SHA-256 sums are recorded in the checkpoint registry.

## Primary independent-test results

PI-ResNet evaluation AUC is 0.98877 for SIS and 0.98522 for PM, compared with 0.97670 and 0.96124 for CQT-DeiT. The paired block-AUC differences are 0.01207 (95% CI 0.00948--0.01421) and 0.02398 (0.02050--0.02713).

At the primary target FPP of 1e-3, PI-ResNet efficiency is 0.5354 (0.5097--0.5629) for SIS and 0.2274 (0.2109--0.2457) for PM. CQT-DeiT reaches 0.3771 (0.3589--0.3949) and 0.0926 (0.0817--0.1040). The paired PI-minus-CQT efficiency gains are 0.1583 (0.1366--0.1811) and 0.1349 (0.1171--0.1514).

The ordering is not uniform at target FPP 1e-4. For SIS, PI-ResNet is lower by 0.0400 (CI -0.0606 to -0.0166). For PM, the 0.0091 difference is not significant (CI -0.0017 to 0.0211). Accordingly, 1e-3 is the preregistered primary operating point and 1e-4 is interpreted as a tail diagnostic.

## Selection effects

Scores and efficiencies depend strongly on weaker-image SNR. Efficiency decreases with increasing impact parameter and magnification imbalance. In the present lens models, impact parameter and flux ratio are tightly linked and must not be presented as independent discoveries.

## SNR/y matching

Common-support reweighting reduces, but does not remove, the SIS-PM efficiency gap. For PI-ResNet the common-support gap changes from about 0.310 before weighting to 0.220 after weighting. The matched residual gap is 0.2201 with 95% CI [0.1922, 0.2489]. For CQT-DeiT it is 0.2294 [0.2064, 0.2520]. Effective sample sizes are approximately 1,156 for SIS and 1,468 for PM; maximum weights are 7.35 and 4.59, respectively. SNR/y therefore explains a substantial fraction, not all, of the lens-family gap.

## E7 controlled type-II diagnostic

The pre-specified 500-source physical/no-Morse comparison is a null result. SIS has mean physical-minus-control score -0.0137 (CI -0.0456 to 0.0180; Wilcoxon p=0.453), while PM has -0.00384 (-0.0284 to 0.0214; p=0.848). Fixed-1e-3-threshold efficiency differences also include zero. The present network therefore shows no measurable sensitivity to the controlled intervention, and no higher-mode/inclination subdivision is pursued.

## Cross-lens transfer

Transfer is strongly asymmetric. On SIS evaluation pairs, the PM-trained checkpoint reduces efficiency at target FPP 1e-3 from 0.535 to 0.221 for PI-ResNet (change -0.315, CI -0.345 to -0.287) and from 0.377 to 0.170 for CQT-DeiT (-0.207, CI -0.223 to -0.191). On PM evaluation pairs, the SIS-trained checkpoint increases efficiency from 0.227 to 0.398 for PI-ResNet (+0.170, CI 0.155 to 0.187) and from 0.093 to 0.314 for CQT-DeiT (+0.221, CI 0.202 to 0.240). This does not support a symmetric robustness claim; it suggests that the SIS training distribution produces a more transferable low-FPP rule under the present priors.

## Throughput

On an NVIDIA RTX 5000 Ada Generation with batch size 256, PI-ResNet GPU inference takes 0.728 ms per pair and CQT-DeiT model inference takes 0.218 ms per pair. However, direct PI preprocessing costs only 0.514 ms per pair, whereas serial CQT construction costs 66.0 ms per pair. Approximate serial-stage end-to-end costs are therefore 1.24 ms per pair for PI-ResNet and 66.3 ms for CQT-DeiT, a roughly 53-fold advantage for the direct time-domain pipeline. A 16-thread CQT trial was slower because of library-level thread oversubscription and is excluded from the primary comparison.

## Logit tail check

Recalibration in pre-sigmoid/logit space selects exactly the same evaluation pairs as probability-space calibration at target FPP 1e-3 and 1e-4 for both models and both lens families (Jaccard 1.0). A monotonic logit transform therefore does not remove the SIS 1e-4 reversal. Probability saturation may be described as a display/numerical-conditioning limitation, but it is not an empirically established mechanism for the reversal in these stored predictions.

## Lens-redshift sanity check

Across 60 deterministic delay interventions, the maximum classifier-input difference after peak realignment is 2.38e-7 and the maximum relative L2 difference is 2.14e-8. This establishes numerical invariance only within the present discrete peak-aligned verification setup at fixed source and y.

## Supported conclusion

The evidence supports PI-ResNet as a calibrated time-domain pair-ranking statistic with materially higher efficiency than the CQT-DeiT baseline at the primary 1e-3 per-pair FPP. It does not support uniform superiority at 1e-4, real-noise robustness, catalog-level FAR claims, complete SNR explanation of the SIS-PM gap, or measurable Morse-phase sensitivity in the minimal E7 probe.

## Reproducibility

All compact code, manifests, per-pair predictions, statistical tables, and hashes are stored in GitHub. Large checkpoints and CQT caches are hash-registered for the Zenodo release.
