# Area guide: stats

Regression and curve fitting, distributions and design of experiments, and domain statistics such as residence-time analysis. The guidance is general practice; the catalog records what each routine has been tested on and wins on disagreement.

## Scope

Fitting data to models, moments and distributions of measured signals, experimental design. Seeded generators belong in `random`; general optimisers in the solver guides.

## Choosing a routine

| Situation | Method / routine | Established equivalent (e.g. SciPy, NumPy) | Notes |
|---|---|---|---|
| Mean residence time, variance, Pe or N from a sampled tracer response C(t), possibly cut before the tail | `rtd_analysis`: `rtd_moments` (tail=True), then `fit_moments` | `scipy.integrate.trapezoid` + algebra | Exact for exponential tails; about 4 % high in variance for tanks-like records cut at 2 tau. |
| Pe or N from a noisy tracer record | `rtd_analysis`: `fit_lsq` | `scipy.optimize.curve_fit` | Moments of noisy tails are biased high; least squares agreed with `curve_fit` to 1e-8 on the tested records. |
| Dispersion coefficient from two detectors in a bed | `rtd_analysis`: `fit_dispersion_planes` | none | Exact for advection-diffusion between the planes, any end conditions. |

## Pitfalls

- Closed-closed and open-open dispersion differ in mean (1 vs 1 + 2/Pe) and variance; use the same boundary condition for the moment fit and any later model curve. Only open-open has a time-domain curve (`e_dispersion`) in `rtd_analysis`.
- Variance is dominated by the tail: a record that stops while C is still above a few percent of its peak underestimates it. Check `tail_decay_time` is not None, or fit with `fit_lsq` instead.
- `fit_lsq` ignores a finite input pulse width; use it with a short pulse or deconvolve first.
