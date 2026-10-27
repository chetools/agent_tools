# Area guide: stats

The guidance here is general practice unless marked otherwise. Each routine's catalog entry records what it has actually been tested on; where the two disagree, the catalog wins.

## Scope

Regression and curve fitting with uncertainties, surrogate models, distributions, and design of experiments. Random-number generators belong in `random`; optimizers used inside a fit belong in `optimize`.

## Choosing a routine

| Situation | Method / routine | Established equivalent (e.g. SciPy, NumPy) | Notes |
|---|---|---|---|
| Smooth response from few (tens to hundreds) expensive runs, need predictions with variances | `gp_matern52` | scikit-learn `GaussianProcessRegressor` with `Matern(nu=2.5)` ARD | Scale inputs to [0, 1] or pass length-scale bounds in input units. Noise fitted or fixed. Posterior cross-covariance available. |
| Choose the next batch of runs to improve a GP surrogate everywhere | `imse_batch` | none in SciPy | Greedy IMSE on posterior covariances; candidates and reference points may differ; no refit between picks. |

## Pitfalls

- GP length-scale bounds are in input units: with unscaled inputs (for example temperature in K) the defaults (0.02 to 5) are meaningless.
- A fitted length scale or noise on its bound means the data do not determine it; the likelihood is flat there and two runs (or Rust vs Python) can stop at different points of equal likelihood.
- Near-duplicate inputs with tiny noise make the covariance ill-conditioned (cond ~ 1e8 in the tests); the fit still works through the 1e-8 jitter but BFGS may report "line search failed" at the optimum.
