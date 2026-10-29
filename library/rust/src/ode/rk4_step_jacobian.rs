//! Classical fixed-step Runge-Kutta (RK4) step with its exact step Jacobian `d y1 / d y0`, generic
//! over [`Real`].
//!
//! Source: classical RK4 (Hairer, Norsett and Wanner, "Solving Ordinary Differential Equations I",
//! 2nd ed. (Springer 1993), section II.1); the step Jacobian is the derivative of the
//! discrete map, obtained by the chain rule through the stage equations (the variational equation
//! discretised by the same four stages; "internal differentiation").
//! Ported from chetools/muse `twin-cstr/solver/src/model.rs` (`rk4_step`) and its Python twin
//! `twin-cstr/python/twin_mhe.py` (`_rk4_batch`), where it linearised a 4-state CSTR model for an
//! extended Kalman filter / moving-horizon estimator. Here the system is any `n`, the right-hand
//! side and its Jacobian are caller closures, and the Jacobian is dense row-major.
//! Python twin: `library/python/agent_tools/ode/rk4_step_jacobian.py`.
//!
//! For `y' = f(t, y)` and a step `h` the classical stages are
//!
//! ```text
//! k1 = f(t, y)            k2 = f(t + h/2, y + h/2 k1)
//! k3 = f(t + h/2, y + h/2 k2)    k4 = f(t + h, y + h k3)        y1 = y + h/6 (k1 + 2 k2 + 2 k3 + k4)
//! ```
//!
//! With `Gi = df/dy` at stage `i` and stage sensitivities `A1 = I`, `A2 = I + h/2 G1`,
//! `A3 = I + h/2 G2 A2`, `A4 = I + h G3 A3`, the step Jacobian is
//!
//! ```text
//! Phi = I + h/6 (G1 + 2 G2 A2 + 2 G3 A3 + G4 A4)
//! ```
//!
//! which is exactly `d y1 / d y` of the discrete step (not an approximation of the flow's
//! sensitivity matrix; the two differ by the O(h^5) local truncation error of the step). Over
//! several substeps the factors multiply: `Phi = Phi_m ... Phi_1`.
//!
//! **Generic over `Real`: chosen** (the cost is one `Real` bound and `T::cst` for constants). A
//! `Dual` state through [`rk4_step`] then gives the same derivative as `Phi` to rounding, an
//! independent check that the hand-coded variational recursion is the derivative of the step.
//! Time `t` and step `h` are plain `f64`.

use crate::autodiff::Real;

/// Result of [`rk4_step_jacobian`].
#[derive(Clone, Debug)]
pub struct Rk4StepResult<T> {
    /// State after the step, length `n`, units of `y`. On failure: a copy of `y0`.
    pub y1: Vec<T>,
    /// Step Jacobian `d y1_i / d y0_k` at `phi[i*n + k]`, row-major `n*n`, dimensionless when the
    /// components of `y` share units (otherwise units of `y_i / y_k`). On failure: the identity.
    pub phi: Vec<T>,
    /// Number of components.
    pub n: usize,
    /// True unless an argument was invalid.
    pub success: bool,
    /// `"success"` or the reason the arguments were rejected.
    pub message: String,
    /// Calls of `f` (`4 n_sub`; 0 on failure).
    pub nfev: usize,
    /// Calls of `jac` (`4 n_sub`; 0 on failure).
    pub njac: usize,
}

/// Result of [`rk4_step`].
#[derive(Clone, Debug)]
pub struct Rk4Result<T> {
    /// State after the step, length `n`, units of `y`. On failure: a copy of `y0`.
    pub y1: Vec<T>,
    /// True unless an argument was invalid.
    pub success: bool,
    /// `"success"` or the reason the arguments were rejected.
    pub message: String,
    /// Calls of `f` (`4 n_sub`; 0 on failure).
    pub nfev: usize,
}

fn check(n: usize, t0: f64, h: f64, n_sub: usize) -> Option<&'static str> {
    if n == 0 {
        Some("empty initial state")
    } else if n_sub == 0 {
        Some("n_sub must be >= 1")
    } else if !t0.is_finite() || !h.is_finite() {
        Some("t0 and h must be finite")
    } else {
        None
    }
}

/// `out = a * b` for row-major `n x n` matrices.
fn matmul<T: Real>(a: &[T], b: &[T], n: usize, out: &mut [T]) {
    for i in 0..n {
        for k in 0..n {
            let mut s = a[i * n] * b[k];
            for j in 1..n {
                s = s + a[i * n + j] * b[j * n + k];
            }
            out[i * n + k] = s;
        }
    }
}

/// One classical RK4 step of size `h` split into `n_sub` equal substeps (state only, no Jacobian).
///
/// * `f(t, y, dydt)` writes `dy/dt` (length `n`, units of `y` per unit of `t`).
/// * `t0` start time and `h` the step over the whole interval, in the units of `t` (`h` may be
///   negative to step backwards); `y0` the initial state, length `n >= 1`.
/// * `n_sub >= 1` is the number of substeps (no default; 1 = a single RK4 step). The substep
///   is `h / n_sub`; the global error scales as `(h / n_sub)^4`.
///
/// Invalid arguments (`n = 0`, `n_sub = 0`, non-finite `t0` or `h`) are reported in the result
/// (never a panic). Non-finite values produced by `f` propagate; nothing is checked.
/// This is the same arithmetic as the state part of [`rk4_step_jacobian`] (bit-identical).
pub fn rk4_step<T: Real>(
    mut f: impl FnMut(f64, &[T], &mut [T]),
    t0: f64,
    y0: &[T],
    h: f64,
    n_sub: usize,
) -> Rk4Result<T> {
    let n = y0.len();
    if let Some(m) = check(n, t0, h, n_sub) {
        return Rk4Result { y1: y0.to_vec(), success: false, message: m.to_string(), nfev: 0 };
    }
    let hs = h / n_sub as f64;
    let z = T::cst(0.0);
    let mut y = y0.to_vec();
    let (mut k1, mut k2, mut k3, mut k4, mut x) = (vec![z; n], vec![z; n], vec![z; n], vec![z; n], vec![z; n]);
    for s in 0..n_sub {
        let t = t0 + s as f64 * hs;
        f(t, &y, &mut k1);
        for i in 0..n {
            x[i] = y[i] + T::cst(0.5 * hs) * k1[i];
        }
        f(t + 0.5 * hs, &x, &mut k2);
        for i in 0..n {
            x[i] = y[i] + T::cst(0.5 * hs) * k2[i];
        }
        f(t + 0.5 * hs, &x, &mut k3);
        for i in 0..n {
            x[i] = y[i] + T::cst(hs) * k3[i];
        }
        f(t + hs, &x, &mut k4);
        for i in 0..n {
            y[i] = y[i] + T::cst(hs / 6.0) * (k1[i] + T::cst(2.0) * k2[i] + T::cst(2.0) * k3[i] + k4[i]);
        }
    }
    Rk4Result { y1: y, success: true, message: "success".to_string(), nfev: 4 * n_sub }
}

/// One classical RK4 step of size `h` (split into `n_sub` equal substeps) and its exact step
/// Jacobian `Phi = d y1 / d y0`, propagated through the stages (see the file header).
///
/// * `f(t, y, dydt)` writes `dy/dt` (length `n`).
/// * `jac(t, y, j)` writes the dense Jacobian `j[i*n + k] = d f_i / d y_k` (row-major `n*n`,
///   units of 1 / unit of `t` when the components share units). It is called at the four stage
///   states, so it should be cheap. For a `Dual` state, `jac` must be Dual too (same generic code
///   as `f`), else the derivative parts of the result are wrong. A Jacobian that disagrees with
///   `f` gives a `Phi` that is not the derivative of the step; check `jac` against finite
///   differences of `f` first.
/// * `t0`, `h`, `y0`, `n_sub`: as in [`rk4_step`]. The returned `phi` is the product of the
///   substep Jacobians.
///
/// Failures (`n = 0`, `n_sub = 0`, non-finite `t0` or `h`) are reported in the result, never a
/// panic. Cost per substep: 4 calls each of `f` and `jac`, 3 dense `n x n` products for the
/// stage factors and 1 for the accumulation (about 8 n^3 flops).
/// The state part is bit-identical to [`rk4_step`].
pub fn rk4_step_jacobian<T: Real>(
    mut f: impl FnMut(f64, &[T], &mut [T]),
    mut jac: impl FnMut(f64, &[T], &mut [T]),
    t0: f64,
    y0: &[T],
    h: f64,
    n_sub: usize,
) -> Rk4StepResult<T> {
    let n = y0.len();
    let z = T::cst(0.0);
    let one = T::cst(1.0);
    let mut eye = vec![z; n * n];
    for i in 0..n {
        eye[i * n + i] = one;
    }
    if let Some(m) = check(n, t0, h, n_sub) {
        return Rk4StepResult { y1: y0.to_vec(), phi: eye, n, success: false, message: m.to_string(), nfev: 0, njac: 0 };
    }
    let hs = h / n_sub as f64;
    let nn = n * n;
    let mut y = y0.to_vec();
    let mut phi = eye.clone();
    let v = || vec![z; n];
    let m = || vec![z; nn];
    let (mut k1, mut k2, mut k3, mut k4, mut x) = (v(), v(), v(), v(), v());
    let (mut g1, mut g2, mut g3, mut g4) = (m(), m(), m(), m());
    let (mut a2, mut a3, mut a4) = (m(), m(), m());
    let (mut g2a2, mut g3a3, mut g4a4) = (m(), m(), m());
    let (mut astep, mut tmp) = (m(), m());
    for s in 0..n_sub {
        let t = t0 + s as f64 * hs;
        f(t, &y, &mut k1);
        jac(t, &y, &mut g1);
        for i in 0..nn {
            a2[i] = eye[i] + T::cst(0.5 * hs) * g1[i];
        }
        for i in 0..n {
            x[i] = y[i] + T::cst(0.5 * hs) * k1[i];
        }
        f(t + 0.5 * hs, &x, &mut k2);
        jac(t + 0.5 * hs, &x, &mut g2);
        matmul(&g2, &a2, n, &mut g2a2);
        for i in 0..nn {
            a3[i] = eye[i] + T::cst(0.5 * hs) * g2a2[i];
        }
        for i in 0..n {
            x[i] = y[i] + T::cst(0.5 * hs) * k2[i];
        }
        f(t + 0.5 * hs, &x, &mut k3);
        jac(t + 0.5 * hs, &x, &mut g3);
        matmul(&g3, &a3, n, &mut g3a3);
        for i in 0..nn {
            a4[i] = eye[i] + T::cst(hs) * g3a3[i];
        }
        for i in 0..n {
            x[i] = y[i] + T::cst(hs) * k3[i];
        }
        f(t + hs, &x, &mut k4);
        jac(t + hs, &x, &mut g4);
        matmul(&g4, &a4, n, &mut g4a4);
        for i in 0..n {
            y[i] = y[i] + T::cst(hs / 6.0) * (k1[i] + T::cst(2.0) * k2[i] + T::cst(2.0) * k3[i] + k4[i]);
        }
        for i in 0..nn {
            astep[i] = eye[i] + T::cst(hs / 6.0) * (g1[i] + T::cst(2.0) * g2a2[i] + T::cst(2.0) * g3a3[i] + g4a4[i]);
        }
        matmul(&astep, &phi, n, &mut tmp);
        phi.copy_from_slice(&tmp);
    }
    Rk4StepResult { y1: y, phi, n, success: true, message: "success".to_string(), nfev: 4 * n_sub, njac: 4 * n_sub }
}
