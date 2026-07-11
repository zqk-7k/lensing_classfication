# E7 controlled type-II sensitivity report

## Question

Does the frozen frozen v1 PI-ResNet score respond measurably to the type-II geometric-optics phase intervention when source parameters, magnification, stored noise realization, and peak-aligned arrival are held fixed?

## Frozen sample

The manifest contains 500 untouched 0228 evaluation sources: 250 SIS and 250 PM. The source manifest was frozen before E7 score calculation and has SHA-256 `bbb9f2d99aa53e338548782cd32a25ac1fa7d33c3814161398d110d7ea13ee3b`.

## Intervention

The stored second image is the physical saddle image. The existing generator applies `exp[-i*pi*n]` with `n=1/2`, corresponding for a real time series to the Hilbert-transform multiplier `-i sign(f)`. The no-Morse counterfactual is therefore obtained as `-H(h_physical)`, followed by the same discrete peak-alignment rule and recombination with the identical stored noise realization. Hilbert round-trip relative errors are below 3.2e-4.

## Results

### SIS

- Mean physical-minus-control score: -0.01375, 95% paired bootstrap CI [-0.04562, 0.01803].
- Median score shift: approximately 0.
- Wilcoxon p-value: 0.453.
- Efficiency at the frozen 1e-3 threshold: 0.560 physical versus 0.568 control.
- Efficiency difference: -0.008, CI [-0.056, 0.040].

### PM

- Mean physical-minus-control score: -0.00384, 95% paired bootstrap CI [-0.02837, 0.02141].
- Median score shift: 0.
- Wilcoxon p-value: 0.848.
- Efficiency at the frozen 1e-3 threshold: 0.232 physical versus 0.244 control.
- Efficiency difference: -0.012, CI [-0.060, 0.036].

## Conclusion

The minimal E7 probe is a null result. Under the present simulation, preprocessing, model, intervention, and sample size, there is no measurable evidence that the network ranking score uses the controlled type-II phase distortion. This is a model-sensitivity diagnostic, not a test of whether the physical distortion exists. No higher-mode/inclination subdivision is triggered because the pre-specified first-stage signal criterion was not met.
