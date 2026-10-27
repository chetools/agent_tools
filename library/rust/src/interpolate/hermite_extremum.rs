//! Global maximum or minimum of the piecewise cubic Hermite interpolant through `(t_i, y_i, dy/dt_i)`.
//!
//! Typical use: the peak of a profile that an adaptive ODE integrator returned, using the accepted
//! steps and their right-hand sides (values and slopes are both known there), where taking the
//! largest sample is off by up to a step and a parabola through three samples fails at a sharp peak.
//!
//! Method: on each interval the derivative of the Hermite cubic is a quadratic in the scaled
//! coordinate `s in [0, 1]`, `A s^2 + B s + C` (dimension of `y`, slopes scaled by the interval width
//! `h`). Its roots in the open interval (0, 1) are the stationary points; the extremum is the best of
//! all knots and all stationary points. The quadratic is solved with the cancellation-free form
//! (`q = -(B + sign(B) sqrt(disc))/2`, roots `q/A` and `C/q`). Source: Burden & Faires, "Numerical
//! Analysis", 9th ed., section 3.4 for the cubic; the root formula is the standard stable one (Press
//! et al., "Numerical Recipes", 3rd ed., eq. 5.6.4). Developed in chetools/claude
//! `packed-bed-runaway/solver/src/lib.rs` (`hermite_peak`; same authors).
//!
//! Generic over [`Real`]. The stationary point is located with plain values and then frozen; the
//! value is evaluated in `T`. At a stationary point `dy/dt = 0`, so moving the location changes the
//! value only to second order (envelope theorem) and the derivative parts of a `Dual` result are
//! exact for the discrete interpolant (the location does not depend on the seeded input to first
//! order). At a knot the value is just `y_i` and its derivative parts pass through.
//!
//! Units: `t` and the returned `t` share a unit; `y` and the returned `value` share a unit; `dy` is
//! the unit of `y` per unit of `t`.
//! Python twin: `library/python/agent_tools/interpolate/hermite_extremum.py`.

use super::cubic_hermite::hermite_piece;
use crate::autodiff::dual::Real;

/// Which extremum to find. Default is [`Extremum::Max`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Extremum {
    /// Largest value of the interpolant.
    #[default]
    Max,
    /// Smallest value of the interpolant.
    Min,
}

/// Result of [`hermite_extremum`].
#[derive(Clone, Debug)]
pub struct HermiteExtremum<T> {
    /// Location of the extremum (unit of `t`). NaN if `success` is false.
    pub t: f64,
    /// Interpolant value at the extremum (unit of `y`). NaN if `success` is false.
    pub value: T,
    /// Index `j` of the interval `[t[j], t[j+1]]` that holds the extremum. For an extremum exactly at a
    /// knot `t[i]` it is `i`, except at the last knot where it is `n - 2`. 0 if `success` is false.
    pub interval: usize,
    /// True when the input was valid.
    pub success: bool,
    /// `"success"`, or the reason the input was rejected.
    pub message: String,
}

fn fail<T: Real>(msg: &str) -> HermiteExtremum<T> {
    HermiteExtremum { t: f64::NAN, value: T::cst(f64::NAN), interval: 0, success: false, message: msg.to_string() }
}

/// Global maximum or minimum of the piecewise cubic Hermite interpolant through `(t[i], y[i], dy[i])`.
///
/// Inputs: `t` strictly increasing (unit of `t`), `y` values, `dy` slopes `dy/dt`; all of equal length
/// `n >= 2`; every entry finite. Otherwise `success = false` with a message (no panic, no search).
/// `kind` selects maximum or minimum.
///
/// Behaviour on special cases (the same in the Python twin):
/// - Candidates are scanned in increasing `t`: knot `t[0]`, the stationary points inside interval 0 in
///   increasing order, knot `t[1]`, and so on. A candidate replaces the current best only if it is
///   **strictly** better, so on exact ties the **smallest `t`** wins, and a knot beats an interior
///   point of equal value (a flat cubic, or a tied plateau, returns its left end).
/// - Monotone pieces have no interior stationary point; the extremum is then a knot (an endpoint of the
///   data if the whole interpolant is monotone).
/// - Only genuine sign changes of the derivative count (discriminant strictly positive), so an
///   inflection with zero slope is never reported as an extremum.
/// - If `|A| < 1e-14 (|B| + |C|)` the derivative is treated as linear (root `-C/B`, none if `B = 0`),
///   which also covers a quadratic Hermite piece and a constant-slope piece.
/// - Cost: O(n), no iteration, no function evaluations beyond one cubic per stationary point.
///
/// The returned `value` is evaluated in `T`; the location uses only `val()` of the inputs.
pub fn hermite_extremum<T: Real>(t: &[f64], y: &[T], dy: &[T], kind: Extremum) -> HermiteExtremum<T> {
    let n = t.len();
    if n < 2 || y.len() != n || dy.len() != n {
        return fail("t, y and dy must have equal length >= 2");
    }
    for i in 0..n {
        if !(t[i].is_finite() && y[i].val().is_finite() && dy[i].val().is_finite()) {
            return fail("t, y and dy must be finite");
        }
        if i > 0 && !(t[i] > t[i - 1]) {
            return fail("t must be strictly increasing");
        }
    }
    let sg = if kind == Extremum::Max { 1.0 } else { -1.0 };
    let mut best = (y[0], t[0], 0usize);
    for j in 0..n - 1 {
        let h = t[j + 1] - t[j];
        let (y0, y1) = (y[j].val(), y[j + 1].val());
        let (m0, m1) = (h * dy[j].val(), h * dy[j + 1].val());
        // d p / d s = A s^2 + B s + C for the cubic Hermite p(s), s in [0, 1]
        let a = 6.0 * y0 + 3.0 * m0 - 6.0 * y1 + 3.0 * m1;
        let b = -6.0 * y0 - 4.0 * m0 + 6.0 * y1 - 2.0 * m1;
        let c = m0;
        let mut roots = [f64::NAN; 2];
        if a.abs() < 1e-14 * (b.abs() + c.abs()) {
            if b != 0.0 {
                roots[0] = -c / b;
            }
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc > 0.0 {
                let sq = disc.sqrt();
                let qq = -0.5 * (b + b.signum() * sq);
                let (r1, r2) = (qq / a, if qq != 0.0 { c / qq } else { f64::NAN });
                roots = if r2.is_nan() || r1 <= r2 { [r1, r2] } else { [r2, r1] };
            }
        }
        for &s in roots.iter() {
            if !(s > 0.0 && s < 1.0) {
                continue;
            }
            let p = hermite_piece(y[j], y[j + 1], dy[j], dy[j + 1], h, s);
            if sg * p.val() > sg * best.0.val() {
                best = (p, t[j] + s * h, j);
            }
        }
        // the right knot closes interval j; its index is j+1, or n-2 for the last knot
        if sg * y1 > sg * best.0.val() {
            best = (y[j + 1], t[j + 1], if j + 1 < n - 1 { j + 1 } else { j });
        }
    }
    HermiteExtremum { t: best.1, value: best.0, interval: best.2, success: true, message: "success".to_string() }
}
