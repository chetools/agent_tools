# Area guide: interpolate

Interpolation of tabulated data. The guidance is general practice; the catalog records what each routine has been tested on and wins on disagreement.

## Scope

Interpolants built from samples (splines, Hermite pieces, monotone interpolants, gridded data) and quantities read off them, such as their extrema. Fitting noisy data belongs in `stats`; dense output of a particular integrator belongs in `ode`.

## Choosing a routine

| Situation | Method / routine | Established equivalent | Notes |
|---|---|---|---|
| Values and slopes known at the knots (for example accepted ODE steps with their right-hand sides): evaluate between them | `cubic_hermite` | `scipy.interpolate.CubicHermiteSpline` | C1 piecewise cubic, 4th-order accurate for smooth data. Extrapolates with the end cubics. |
| Peak (or trough) of a profile known at knots with slopes | `hermite_extremum` (`Extremum::Max` / `Min`, Python `"max"` / `"min"`) | `CubicHermiteSpline` + dense grid + `minimize_scalar` | Exact, O(n), no search grid. Returns location, value and interval index. Peak value error is O(h^4) (observed order 3.99 on a narrow Gaussian); location error is O(h^3) (observed 3.0). |

## Pitfalls

- A sample maximum is off by up to a knot spacing, and a parabola through three samples fails at a sharp peak: use `hermite_extremum` when slopes are available.
- The interpolant only knows the data. If the knots are coarser than the feature (spacing comparable to the width of a peak), the result is a poor estimate of the true peak; check by halving the spacing, the error of the peak value should drop by about 16.
- Expectation, not measured: the extremum location of a flat peak is ill-conditioned (error up to about `sqrt(eps)` times the scale), though its value is accurate to rounding.
- Ties: on equal values `hermite_extremum` returns the smallest `t`, and a knot beats an interior point of equal value.
- `t` must be strictly increasing. `hermite_extremum` reports `success = false` for repeated or decreasing `t`; `cubic_hermite` does not check.
