//! Adaptive Rosenbrock 2(3) integrator for stiff initial value problems (the scheme of MATLAB `ode23s`),
//! generic over [`Real`] so that dual numbers differentiate through the adaptive steps.
//!
//! Source: L. F. Shampine and M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput. 18(1),
//! 1-22 (1997), section 2 (the modified Rosenbrock formula with `d = 1/(2+sqrt 2)` and
//! `e32 = 6+sqrt 2`, a third-order error estimate, and first-same-as-last reuse of the final
//! derivative). Ported from chetools/claude `packed-bed-runaway/solver/src/lib.rs` (`integrate`),
//! where it marched a reactor model with a banded Jacobian; here the Jacobian is dense and supplied
//! by the caller, the linear algebra is [`crate::linalg::lu_dense`].
//! Reference for tests: SciPy `solve_ivp(method="Radau")` at rtol 1e-12, atol 1e-14.
//!
//! One step from `(t, y)` with step `h`, `W = I - h d J`, `J = df/dy`, `T = df/dt` (all stages
//! reuse one LU factorization, so there is no Newton iteration):
//!
//! ```text
//! k1 = W^-1 (f(t,y) + h d T)
//! k2 = W^-1 (f(t+h/2, y + h/2 k1) - k1) + k1          y_new = y + h k2     (2nd order)
//! k3 = W^-1 (f(t+h, y_new) - e32 (k2 - f1) - 2 (k1 - f0) + h d T)
//! err = h/6 (k1 - 2 k2 + k3)                                               (3rd-order estimate)
//! ```
//!
//! Step acceptance and sizing: the error is scaled componentwise by `atol_i + rtol max(|y_i|,
//! |y_new_i|)` and its max-norm must be <= 1; the next step is `h * clamp(0.8 err^(-1/3), 0.2, 5)`
//! (`5` if `err == 0`).
//!
//! **Derivative parts in the error test.** The step sizes are plain `f64`, so for dual numbers the
//! derivative part of the result is the exact derivative of the *discrete* solution
//! ("internal differentiation"). Its accuracy is not controlled by the value-based test, so the
//! same test is applied to the derivative parts of the error vector (`deriv()`), scaled by
//! `atol_i + rtol max_j(|dy_j|, |dy_new_j|)`, the largest derivative component of the state:
//! tiny derivative components carry only round-off and must not stall the step size. For `f64`
//! `deriv()` is 0 and this part vanishes, so a plain solve is unaffected.
//!
//! Python twin: `library/python/agent_tools/ode/rosenbrock23.py`.

use crate::autodiff::Real;
use crate::linalg::lu_dense::{lu_factor, lu_solve};

/// Options of [`rosenbrock23`]. `Default` gives the source defaults (see each field).
#[derive(Clone, Debug)]
pub struct Rosenbrock23Options {
    /// Relative tolerance, dimensionless. Default 1e-5. Must be > 0.
    pub rtol: f64,
    /// Absolute tolerance, in the units of `y` (used for every component unless `atol_vec` is
    /// non-empty). Default 1e-8. Must be >= 0.
    pub atol: f64,
    /// Per-component absolute tolerances, units of `y`. Empty (default) means `atol` for all;
    /// otherwise its length must equal the number of components.
    pub atol_vec: Vec<f64>,
    /// First step, in the units of `t`. Default `None` = 1e-4 * (t1 - t0). Too large a first step
    /// only costs a few rejections.
    pub h0: Option<f64>,
    /// Largest step, in the units of `t`. Default `None` = no limit (t1 - t0). The source capped
    /// the step at 0.05 of its unit interval to resolve a profile maximum; that cap was
    /// problem-specific and is not kept.
    pub hmax: Option<f64>,
    /// Maximum number of attempted steps (accepted + rejected). Default 200 000.
    pub max_steps: usize,
    /// When no `dfdt` closure is given: `false` (default) treats the system as autonomous
    /// (`df/dt = 0`, the source behaviour, exact for autonomous systems); `true` estimates
    /// `df/dt` by a forward difference of `f` in `t` (one extra `f` call per Jacobian
    /// evaluation, accuracy about sqrt(machine epsilon)). The source had no `df/dt`.
    pub fd_dfdt: bool,
}

impl Default for Rosenbrock23Options {
    fn default() -> Self {
        Rosenbrock23Options {
            rtol: 1e-5,
            atol: 1e-8,
            atol_vec: Vec::new(),
            h0: None,
            hmax: None,
            max_steps: 200_000,
            fd_dfdt: false,
        }
    }
}

/// Result of [`rosenbrock23`]. The trajectory holds the initial point and every accepted step.
#[derive(Clone, Debug)]
pub struct Rosenbrock23Result<T> {
    /// Times of the stored points, in the units of `t` (`t[0] = t0`; ends at `t1` on success).
    pub t: Vec<f64>,
    /// States at those times, row-major `t.len() x n`, units of `y`.
    pub y: Vec<T>,
    /// `dy/dt = f(t, y)` at those times, same layout as `y`, units of `y` per unit of `t`
    /// (the data for cubic Hermite dense output).
    pub dydt: Vec<T>,
    /// Number of components.
    pub n: usize,
    /// True if the integration reached `t1`.
    pub success: bool,
    /// Human-readable status (`"success"` or the reason for stopping early).
    pub message: String,
    /// Calls of `f` (including those for the finite-difference `df/dt`, and rejected steps).
    pub nfev: usize,
    /// Evaluations of the Jacobian (one per accepted step; reused after a rejected step).
    pub njac: usize,
    /// LU factorizations of `W` (one per attempted step).
    pub nlu: usize,
    /// Accepted steps.
    pub nsteps: usize,
    /// Rejected steps.
    pub nrejected: usize,
}

impl<T: Real> Rosenbrock23Result<T> {
    /// State at the last stored time (at `t1` if `success`).
    pub fn y_final(&self) -> &[T] {
        &self.y[self.y.len() - self.n..]
    }
    /// State at stored point `k`.
    pub fn y_at(&self, k: usize) -> &[T] {
        &self.y[k * self.n..(k + 1) * self.n]
    }
}

/// Integrate `dy/dt = f(t, y)`, `y(t0) = y0`, from `t0` to `t1 > t0` (forward in time only).
///
/// * `f(t, y, dydt)` writes the right-hand side into `dydt` (all length `n`).
/// * `jac(t, y, j)` writes the dense Jacobian `j[i*n + k] = d f_i / d y_k` (row-major `n*n`,
///   units of 1/time for the entries when `y` components share units). For `Dual` states the
///   Jacobian must be Dual too (build it with the same generic code as `f`), else the derivative
///   parts of the solution are wrong. An inexact Jacobian only costs accuracy per step, not
///   correctness of the error control.
/// * `dfdt`: `Some(&mut closure)` writing `df/dt` (length `n`), or `None` for an autonomous system
///   (or the finite-difference estimate if `opts.fd_dfdt`). A time-dependent `f` with
///   `dfdt = None` and `fd_dfdt = false` is integrated with a wrong `df/dt = 0`, which lowers the
///   order and the accuracy of the step.
/// * `t_span = (t0, t1)`, `y0` the initial state (length `n`, at least 1).
///
/// Failures are reported in the result (never a panic) for: `t1 <= t0`, empty `y0`, bad tolerances
/// or `atol_vec` length, a singular `W` (zero pivot), step size underflow
/// (`h < 1e-15 max(|t|, 1e-3 (t1 - t0))`, the source's rule scaled to the span) and `max_steps`;
/// the partial trajectory is returned. A non-finite error estimate rejects the step.
/// Cost per attempted step: one LU of `n x n` (2n^3/3 flops), three solves, two `f` calls.
pub fn rosenbrock23<T: Real>(
    mut f: impl FnMut(f64, &[T], &mut [T]),
    mut jac: impl FnMut(f64, &[T], &mut [T]),
    mut dfdt: Option<&mut dyn FnMut(f64, &[T], &mut [T])>,
    t_span: (f64, f64),
    y0: &[T],
    opts: &Rosenbrock23Options,
) -> Rosenbrock23Result<T> {
    let n = y0.len();
    let (t0, t1) = t_span;
    let z = T::cst(0.0);
    let mut res = Rosenbrock23Result {
        t: vec![t0],
        y: y0.to_vec(),
        dydt: Vec::new(),
        n,
        success: false,
        message: String::new(),
        nfev: 0,
        njac: 0,
        nlu: 0,
        nsteps: 0,
        nrejected: 0,
    };
    let bad = |m: &str, mut r: Rosenbrock23Result<T>| {
        r.message = m.to_string();
        r
    };
    if n == 0 {
        return bad("empty initial state", res);
    }
    if !(t1 > t0) || !t0.is_finite() || !t1.is_finite() {
        return bad("t_span must satisfy t0 < t1 (both finite)", res);
    }
    if !(opts.rtol > 0.0) || !(opts.atol >= 0.0) {
        return bad("need rtol > 0 and atol >= 0", res);
    }
    if !opts.atol_vec.is_empty() && opts.atol_vec.len() != n {
        return bad("atol_vec must be empty or have one entry per component", res);
    }
    let atol = |i: usize| if opts.atol_vec.is_empty() { opts.atol } else { opts.atol_vec[i] };
    let span = t1 - t0;
    // ode23s constants (Shampine & Reichelt 1997)
    let d = 1.0 / (2.0 + 2f64.sqrt());
    let e32 = 6.0 + 2f64.sqrt();

    let mut y = y0.to_vec();
    let mut f0 = vec![z; n];
    let mut f1 = vec![z; n];
    let mut f2 = vec![z; n];
    let mut tv = vec![z; n]; // df/dt
    let mut k1 = vec![z; n];
    let mut k2 = vec![z; n];
    let mut yt = vec![z; n];
    let mut jm = vec![z; n * n];
    let mut w = vec![z; n * n];

    f(t0, &y, &mut f0);
    res.nfev = 1;
    res.dydt.extend_from_slice(&f0);

    let mut t = t0;
    let mut h = opts.h0.unwrap_or(1e-4 * span);
    let hmax = opts.hmax.unwrap_or(span);
    let mut have_jac = false; // J and df/dt at the current (t, y); kept across a rejected step
    let mut attempts = 0usize;

    while t < t1 {
        if attempts >= opts.max_steps {
            res.message = format!("max_steps ({}) reached at t = {t}", opts.max_steps);
            return res;
        }
        attempts += 1;
        if !(h > 0.0) {
            res.message = format!("step size {h} is not positive at t = {t}");
            return res;
        }
        h = h.min(hmax);
        let last = h >= t1 - t;
        if last {
            h = t1 - t;
        }
        let ht = T::cst(h);
        let hd = T::cst(h * d);
        if !have_jac {
            jac(t, &y, &mut jm);
            res.njac += 1;
            if let Some(g) = dfdt.as_mut() {
                g(t, &y, &mut tv);
            } else if opts.fd_dfdt {
                let dt = f64::EPSILON.sqrt() * t.abs().max(1e-3 * span);
                f(t + dt, &y, &mut tv);
                res.nfev += 1;
                for i in 0..n {
                    tv[i] = (tv[i] - f0[i]) / T::cst(dt);
                }
            }
            have_jac = true;
        }
        for i in 0..n {
            for k in 0..n {
                let id = if i == k { T::cst(1.0) } else { z };
                w[i * n + k] = id - hd * jm[i * n + k];
            }
        }
        let fac = lu_factor(&w, n);
        res.nlu += 1;
        if !fac.success {
            res.message = format!("W = I - h d J is singular at t = {t}, h = {h}: {}", fac.message);
            return res;
        }
        let solve = |rhs: &[T]| lu_solve(&fac, rhs).x;

        // stage 1: W k1 = f0 + h d T
        let rhs: Vec<T> = (0..n).map(|i| f0[i] + hd * tv[i]).collect();
        k1.copy_from_slice(&solve(&rhs));
        // stage 2: W (k2 - k1) = f(t + h/2, y + h/2 k1) - k1
        for i in 0..n {
            yt[i] = y[i] + T::cst(0.5) * ht * k1[i];
        }
        f(t + 0.5 * h, &yt, &mut f1);
        let rhs: Vec<T> = (0..n).map(|i| f1[i] - k1[i]).collect();
        let s = solve(&rhs);
        for i in 0..n {
            k2[i] = s[i] + k1[i];
            yt[i] = y[i] + ht * k2[i]; // candidate y_new (2nd order)
        }
        let t_new = if last { t1 } else { t + h };
        f(t_new, &yt, &mut f2);
        res.nfev += 2;
        // stage 3, only for the error estimate
        let rhs: Vec<T> = (0..n)
            .map(|i| f2[i] - T::cst(e32) * (k2[i] - f1[i]) - T::cst(2.0) * (k1[i] - f0[i]) + hd * tv[i])
            .collect();
        let k3 = solve(&rhs);

        // error = h/6 (k1 - 2 k2 + k3), scaled by atol + rtol max(|y|, |y_new|); the derivative
        // parts get the same test, scaled by their largest component (see module docs)
        let mut dmax: f64 = 0.0;
        for i in 0..n {
            dmax = dmax.max(y[i].deriv().abs()).max(yt[i].deriv().abs());
        }
        let mut err: f64 = 0.0;
        for i in 0..n {
            let e = (h / 6.0) * (k1[i].val() - 2.0 * k2[i].val() + k3[i].val());
            let sc = atol(i) + opts.rtol * y[i].val().abs().max(yt[i].val().abs());
            err = err.max((e / sc).abs());
            let es = (h / 6.0) * (k1[i].deriv() - 2.0 * k2[i].deriv() + k3[i].deriv());
            let ss = atol(i) + opts.rtol * dmax;
            err = err.max((es / ss).abs());
        }
        if !err.is_finite() {
            err = 1e10;
        }
        if err <= 1.0 {
            t = t_new;
            y.copy_from_slice(&yt);
            f0.copy_from_slice(&f2); // first-same-as-last
            have_jac = false;
            res.nsteps += 1;
            res.t.push(t);
            res.y.extend_from_slice(&y);
            res.dydt.extend_from_slice(&f0);
        } else {
            res.nrejected += 1;
        }
        // new step: 3rd-order error estimate -> exponent 1/3, safety 0.8
        let fac_h = if err == 0.0 { 5.0 } else { (0.8 * err.powf(-1.0 / 3.0)).clamp(0.2, 5.0) };
        h *= fac_h;
        if t < t1 && h < 1e-15 * t.abs().max(1e-3 * span) {
            res.message = format!("step size underflow at t = {t}");
            return res;
        }
    }
    res.success = true;
    res.message = "success".to_string();
    res
}
