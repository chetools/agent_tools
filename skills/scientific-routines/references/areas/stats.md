# Area guide: stats

The guidance here is general practice unless marked otherwise. Each routine's catalog entry records what it has actually been tested on; where the two disagree, the catalog wins.

## Scope

Regression and curve fitting with uncertainties, surrogate models, distributions, design of experiments, and residence-time analysis. Random-number generators belong in `random`; optimizers used inside a fit belong in `optimize`; linear-algebra kernels (Cholesky, LU) in `linalg`.

## Choosing a routine

| Situation | Method / routine | Established equivalent (e.g. SciPy, NumPy) | Notes |
|---|---|---|---|
| Smooth response from few (tens to hundreds) expensive runs, need predictions with variances | `gp_matern52` | scikit-learn `GaussianProcessRegressor` with `Matern(nu=2.5)` ARD | Scale inputs to [0, 1] or pass length-scale bounds in input units. Noise fitted or fixed. Posterior cross-covariance available. |
| Space-filling initial design in [0,1)^d, reproducible and identical in Rust and Python | `lhs_maximin` | `scipy.stats.qmc.LatinHypercube` | Best of k random Latin hypercubes, maximin distance criterion. Random position inside each slice. Weak in high dimension (d = 10, n = 50: min distance about 0.63 at k = 50); pass a `SplitMix64`. |
| Choose the next batch of runs to improve a GP surrogate everywhere | `imse_batch` | none in SciPy | Greedy IMSE on posterior covariances; candidates and reference points may differ; no refit between picks. |
| Mean residence time, variance, Pe or N from a sampled tracer response C(t), possibly cut before the tail | `rtd_analysis`: `rtd_moments` (tail=True), then `fit_moments` | `scipy.integrate.trapezoid` + algebra | Exact for exponential tails; about 4 % high in variance for tanks-like records cut at 2 tau. |
| Pe or N from a noisy tracer record | `rtd_analysis`: `fit_lsq` | `scipy.optimize.curve_fit` | Moments of noisy tails are biased high; least squares agreed with `curve_fit` to 1e-8 on the tested records. |
| Dispersion coefficient from two detectors in a bed | `rtd_analysis`: `fit_dispersion_planes` | none | Exact for advection-diffusion between the planes, any end conditions. |

## Pitfalls

- GP length-scale bounds are in input units: with unscaled inputs (for example temperature in K) the defaults (0.02 to 5) are meaningless.
- A fitted length scale or noise on its bound means the data do not determine it; the likelihood is flat there and two runs (or Rust vs Python) can stop at different points of equal likelihood.
- Near-duplicate inputs with tiny noise make the covariance ill-conditioned (cond ~ 1e8 in the tests); the fit still works through the 1e-8 jitter but BFGS may report "line search failed" at the optimum.
- A Latin hypercube guarantees one point per slice in each single input, not good coverage of the joint space: check the minimum distance (or discrepancy) of the design you get.
- Scale unit-cube designs to physical ranges yourself; if inputs have very different scales, space-filling distance in the unit cube is the meaningful measure, not distance in physical units.
- Closed-closed and open-open dispersion differ in mean (1 vs 1 + 2/Pe) and variance; use the same boundary condition for the moment fit and any later model curve. Only open-open has a time-domain curve (`e_dispersion`) in `rtd_analysis`.
- Variance is dominated by the tail: a record that stops while C is still above a few percent of its peak underestimates it. Check `tail_decay_time` is not None, or fit with `fit_lsq` instead.
- `fit_lsq` ignores a finite input pulse width; use it with a short pulse or deconvolve first.
