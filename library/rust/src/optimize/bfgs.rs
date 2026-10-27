//! BFGS quasi-Newton minimization with an Armijo backtracking line search (unconstrained).
//!
//! Source: Nocedal & Wright, "Numerical Optimization", 2nd ed. (2006): Algorithm 6.1 (BFGS),
//! eq. (6.17) (inverse-Hessian update `H <- (I - r s y^T) H (I - r y s^T) + r s s^T`, `r = 1/(y^T s)`)
//! and Algorithm 3.1 (backtracking with the sufficient-decrease condition, eq. 3.4, `c1 = 1e-4`).
//! Ported from chetools/claude `sequential-design/solver/src/lib.rs` (`bfgs`; same authors) with the
//! Gaussian-process likelihood left out. Reference for tests: `scipy.optimize.minimize(method="BFGS")`.
//! Python twin: `library/python/agent_tools/optimize/bfgs.py`.
//!
//! The source has no bound handling (the Gaussian-process code that called it maps its bounded
//! hyperparameters to unbounded ones with a tanh transform before calling `bfgs`; that transform
//! is problem-specific and is not part of this routine). To respect bounds, minimize over
//! transformed unbounded variables and multiply the gradient by the chain-rule factor.
//!
//! Algorithm, per iteration (exactly the steps of the Python twin):
//! 1. Stop with success if `max|g| < gtol` (strict).
//! 2. Direction `p = -H g`; if `g.p >= 0` (not a descent direction), reset `H = I` and use `p = -g`.
//! 3. Backtrack `a = 1, 1/2, 1/4, ...` until `f(x + a p) <= f(x) + C1 a g.p`; fail if `a < MIN_STEP`.
//! 4. Update `H` unless the curvature `s.y <= SY_MIN`; accept the step.
//!
//! The inverse Hessian starts as the identity (no scaling), so the first step is a steepest-descent
//! step of length `|g|` and the line search may need several halvings when the objective is badly
//! scaled. Cost: `O(n^2)` memory and `O(n^3)` work per iteration (the update is written as plain
//! matrix products); intended for small to moderate `n` (tens), not for large-scale problems.

/// Sufficient-decrease constant `c1` of the Armijo condition (dimensionless).
pub const C1: f64 = 1e-4;
/// Smallest accepted line-search step length (dimensionless, relative to the direction `p`).
pub const MIN_STEP: f64 = 1e-10;
/// The inverse-Hessian update is skipped when `s.y <= SY_MIN`. Absolute threshold, in
/// (units of x) * (units of g): rescale the problem if its curvature is of this size.
pub const SY_MIN: f64 = 1e-12;

/// Options of [`bfgs`].
#[derive(Clone, Debug)]
pub struct BfgsOptions {
    /// Maximum number of accepted steps (iterations). Default 200.
    pub max_iter: usize,
    /// Gradient tolerance: stop when the max-norm of the gradient is below this value, in the
    /// units of the gradient (units of `f` per unit of `x`). Default 1e-5. Must be positive.
    pub gtol: f64,
}

impl Default for BfgsOptions {
    fn default() -> Self {
        BfgsOptions { max_iter: 200, gtol: 1e-5 }
    }
}

/// Result of [`bfgs`].
#[derive(Clone, Debug)]
pub struct BfgsResult {
    /// Last accepted point (units of `x`), also when `success` is false.
    pub x: Vec<f64>,
    /// Objective value at `x` (units of `f`).
    pub f: f64,
    /// Max-norm of the gradient at `x` (units of `f` per unit of `x`); this is the quantity
    /// compared with `gtol`.
    pub grad_norm: f64,
    /// True if `grad_norm < gtol` was reached.
    pub success: bool,
    /// Status text: `"converged: ..."`, or the reason for failure (iteration limit, line-search
    /// failure, non-finite objective or gradient).
    pub message: String,
    /// Number of accepted steps.
    pub nit: usize,
    /// Number of calls of `fg` (each evaluates `f` and the gradient together), including the
    /// call at `x0` and the rejected line-search trial points.
    pub nfev: usize,
}

fn max_norm(g: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &v in g {
        if !v.is_finite() {
            return f64::NAN;
        }
        m = m.max(v.abs());
    }
    m
}

/// Minimize `f(x)` over `x` in R^n (unconstrained) with BFGS and Armijo backtracking.
///
/// * `fg(x, grad)` must return `f(x)` and write the gradient `df/dx` (same length as `x`) into
///   `grad`. It may return `+inf` or NaN for points where `f` is undefined: such trial points are
///   rejected and the step is halved (this is how a failed Cholesky factorization is handled in the
///   Gaussian-process use case); the gradient written for a rejected point is ignored.
/// * `x0` is the starting point (units of `x`); `x0.len()` is `n >= 1`.
/// * `options`: see [`BfgsOptions`].
///
/// Failure modes (`success = false`, best point so far returned): non-finite `f` or gradient at
/// `x0` or at an accepted point; line search shrinks the step below [`MIN_STEP`] (typical for a
/// wrong gradient sign or a discontinuous objective); `max_iter` steps without convergence.
pub fn bfgs<F: FnMut(&[f64], &mut [f64]) -> f64>(mut fg: F, x0: &[f64], options: &BfgsOptions) -> BfgsResult {
    let p = x0.len();
    let mut z = x0.to_vec();
    let mut g = vec![0.0; p];
    let mut f = fg(&z, &mut g);
    let mut nfev = 1usize;
    let mut nit = 0usize;
    let mut gn = max_norm(&g);
    let fail = |z: Vec<f64>, f: f64, gn: f64, msg: String, nit: usize, nfev: usize| BfgsResult {
        x: z,
        f,
        grad_norm: gn,
        success: false,
        message: msg,
        nit,
        nfev,
    };
    if !f.is_finite() || !gn.is_finite() {
        return fail(z, f, gn, "non-finite objective or gradient at x0".to_string(), nit, nfev);
    }
    let eye = |h: &mut Vec<f64>| {
        h.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..p {
            h[i * p + i] = 1.0;
        }
    };
    let mut h = vec![0.0; p * p];
    eye(&mut h);
    let mut z1 = vec![0.0; p];
    let mut g1 = vec![0.0; p];
    loop {
        if gn < options.gtol {
            return BfgsResult {
                x: z,
                f,
                grad_norm: gn,
                success: true,
                message: format!("converged: max|grad| = {gn:e} < gtol = {:e}", options.gtol),
                nit,
                nfev,
            };
        }
        if nit >= options.max_iter {
            return fail(z, f, gn, format!("max_iter = {} reached, max|grad| = {gn:e}", options.max_iter), nit, nfev);
        }
        let mut dir: Vec<f64> = (0..p).map(|i| -(0..p).map(|j| h[i * p + j] * g[j]).sum::<f64>()).collect();
        let mut gp: f64 = (0..p).map(|i| g[i] * dir[i]).sum();
        if gp >= 0.0 {
            eye(&mut h);
            dir = g.iter().map(|v| -v).collect();
            gp = (0..p).map(|i| g[i] * dir[i]).sum();
        }
        let mut a = 1.0;
        let f1 = loop {
            for i in 0..p {
                z1[i] = z[i] + a * dir[i];
            }
            let f1 = fg(&z1, &mut g1);
            nfev += 1;
            if f1 <= f + C1 * a * gp {
                break f1;
            }
            a *= 0.5;
            if a < MIN_STEP {
                return fail(z, f, gn, format!("line search failed: step below {MIN_STEP:e}"), nit, nfev);
            }
        };
        let s: Vec<f64> = (0..p).map(|i| z1[i] - z[i]).collect();
        let yv: Vec<f64> = (0..p).map(|i| g1[i] - g[i]).collect();
        let sy: f64 = (0..p).map(|i| s[i] * yv[i]).sum();
        if sy > SY_MIN {
            let r = 1.0 / sy;
            // H <- (I - r s y^T) H (I - r y s^T) + r s s^T
            let mut a1 = vec![0.0; p * p];
            for i in 0..p {
                for j in 0..p {
                    a1[i * p + j] = if i == j { 1.0 } else { 0.0 } - r * s[i] * yv[j];
                }
            }
            let mut t = vec![0.0; p * p];
            for i in 0..p {
                for j in 0..p {
                    t[i * p + j] = (0..p).map(|k| a1[i * p + k] * h[k * p + j]).sum();
                }
            }
            for i in 0..p {
                for j in 0..p {
                    // (I - r y s^T)[k][j] = a1[j][k]
                    h[i * p + j] = (0..p).map(|k| t[i * p + k] * a1[j * p + k]).sum::<f64>() + r * s[i] * s[j];
                }
            }
        }
        z.copy_from_slice(&z1);
        g.copy_from_slice(&g1);
        f = f1;
        gn = max_norm(&g);
        nit += 1;
        if !gn.is_finite() {
            return fail(z, f, gn, "non-finite gradient at an accepted point".to_string(), nit, nfev);
        }
    }
}
