//! Numerically safe log-mean temperature difference with partial derivatives.
//!
//! Source: standard definition LMTD = (dT1 - dT2) / ln(dT1 / dT2) (e.g. Kern, Process Heat
//! Transfer, 1950; Incropera et al., Fundamentals of Heat and Mass Transfer, counter-current
//! exchanger); series branch and derivative formulas derived here and checked in the tests.
//! Ported from chetools/claude heat-exchanger-network (hen.py `lmtd`, solver/src/lib.rs `lmtd`).
//!
//! With m = (dT1 + dT2)/2 and d = dT1 - dT2:
//! * if |d| < [`LMTD_SERIES_THRESHOLD`] * m use the series
//!   L ~= m - d^2/(12 m), dL/ddT1 ~= 1/2 - d/(6 m) + d^2/(24 m^2),
//!   dL/ddT2 ~= 1/2 + d/(6 m) + d^2/(24 m^2);
//! * otherwise the closed form with lr = ln(dT1/dT2): L = d/lr,
//!   dL/ddT1 = (lr - d/dT1)/lr^2, dL/ddT2 = (d/dT2 - lr)/lr^2.
//!
//! Accuracy of both branches is measured by tests/generators/lmtd.py against 50-digit mpmath
//! values (see the catalog entry).

/// Relative switch threshold `|dT1 - dT2| < threshold * mean(dT1, dT2)` (dimensionless).
pub const LMTD_SERIES_THRESHOLD: f64 = 1e-4;

/// Log-mean temperature difference and its partial derivatives.
///
/// Inputs: `dt1`, `dt2` are the two terminal temperature differences [K], both > 0 (any
/// consistent temperature-difference unit works; derivatives are then dimensionless).
/// Returns `(L, dL/ddt1, dL/ddt2)`: `L` in the unit of the inputs, derivatives dimensionless.
/// Failure mode: if either input is <= 0 or not finite, returns `(NaN, NaN, NaN)`; the caller
/// must treat a temperature cross as infeasible. Symmetric: `L(a,b) = L(b,a)`.
pub fn lmtd(dt1: f64, dt2: f64) -> (f64, f64, f64) {
    if !(dt1.is_finite() && dt2.is_finite()) || dt1 <= 0.0 || dt2 <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let m = 0.5 * (dt1 + dt2);
    let d = dt1 - dt2;
    if d.abs() < LMTD_SERIES_THRESHOLD * m {
        let l = m - d * d / (12.0 * m);
        return (
            l,
            0.5 - d / (6.0 * m) + d * d / (24.0 * m * m),
            0.5 + d / (6.0 * m) + d * d / (24.0 * m * m),
        );
    }
    let lr = (dt1 / dt2).ln();
    (d / lr, (lr - d / dt1) / (lr * lr), (d / dt2 - lr) / (lr * lr))
}
