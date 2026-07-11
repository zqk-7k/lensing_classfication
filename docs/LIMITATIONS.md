# Scope and limitations

The reported quantitative conclusions apply to simulated ET observations in
stationary Gaussian noise. They do not establish robustness to nonstationarity,
glitches, calibration uncertainty, or real detector data.

The models verify pairs of preselected, peak-aligned event segments. They do not
perform event detection, search an entire catalog, estimate a catalog-level false
alarm rate, or replace joint Bayesian lensing inference.

The simulated lensing prior is restricted to dimensionless impact parameter
`y in [0.01, 0.3]`. Selection functions in this repository are conditional on that
range, the adopted binary-black-hole population, waveform assumptions, signal
duration, preprocessing, and noise model. They must not be extrapolated as a
general astrophysical selection function.

No clean independent LIGO comparison is released. Historical exploratory results
used a different evaluation protocol and are deliberately excluded from the
quantitative evidence in this repository.

The controlled type-II intervention is a model-sensitivity diagnostic. Its null
result does not test whether the physical lensing phase shift exists. Likewise,
the lens-redshift test demonstrates input invariance only at fixed `y` within the
present peak-aligned representation.
