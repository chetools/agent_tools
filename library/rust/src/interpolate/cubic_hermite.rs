//! Piecewise cubic Hermite interpolation through samples `(t_i, y_i, dy/dt_i)`.
//!
//! Source: Burden & Faires, "Numerical Analysis", 9th ed., section 3.4 (Hermite interpolation, cubic
//! case); the four Hermite basis polynomials h00, h10, h01, h11 are the standard ones (also used by
//! `scipy.interpolate.CubicHermiteSpline`, the reference for the tests). Developed in chetools/claude
//! `packed-bed-runaway/solver/src/lib.rs` (`hermite_peak`; same authors).
//!
//! Units: `t` and `x` share one unit; `y` and the result have the unit of `y`; `dy` has unit of `y` per
//! unit of `t`. Generic over [`Real`], so `y` and `dy` may be [`Dual`](crate::autodiff::dual::Dual)
//! numbers (the knots `t` and the query `x` are always plain `f64`).
//! Python twin: `library/python/agent_tools/interpolate/cubic_hermite.py`.

use crate::autodiff::dual::Real;

/// Value of the cubic Hermite polynomial on one interval of width `h` at the fraction `s = (x - t0)/h`.
///
/// `y0`, `y1` are the end values, `m0`, `m1` the end slopes (`dy/dt`, per unit of `t`), `h > 0` the
/// interval width (unit of `t`). `s` is dimensionless: 0 gives `y0`, 1 gives `y1`; values outside
/// `[0, 1]` extrapolate the same cubic. No checks are made.
pub fn hermite_piece<T: Real>(y0: T, y1: T, m0: T, m1: T, h: f64, s: f64) -> T {
    let (s2, s3) = (s * s, s * s * s);
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    y0 * T::cst(h00) + m0 * T::cst(h * h10) + y1 * T::cst(h01) + m1 * T::cst(h * h11)
}

/// Evaluate the piecewise cubic Hermite interpolant through `(t[i], y[i], dy[i])` at `x`.
///
/// `t` must be strictly increasing (not checked, for speed) with at least 2 points. `x` outside
/// `[t[0], t[n-1]]` is extrapolated with the first or last cubic (as SciPy does by default); a `x`
/// exactly equal to a knot uses the interval to its right (the last knot uses the last interval). At
/// a knot both neighbouring cubics return the same value `y[i]`, so the interpolant is continuous
/// and has the continuous slope `dy[i]`. NaN `x` gives NaN.
///
/// Panics if the three slices differ in length or have fewer than 2 points.
pub fn cubic_hermite<T: Real>(t: &[f64], y: &[T], dy: &[T], x: f64) -> T {
    let n = t.len();
    assert!(n >= 2 && y.len() == n && dy.len() == n, "cubic_hermite: t, y, dy need equal length >= 2");
    // interval j with t[j] <= x < t[j+1], clamped to [0, n-2]
    let j = t.partition_point(|&ti| ti <= x).saturating_sub(1).min(n - 2);
    let h = t[j + 1] - t[j];
    hermite_piece(y[j], y[j + 1], dy[j], dy[j + 1], h, (x - t[j]) / h)
}
