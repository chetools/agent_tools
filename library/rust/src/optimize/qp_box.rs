//! Box-constrained convex quadratic program, primal active-set method.
//!
//! Problem: minimize `f(x) = 0.5 x^T H x + g^T x` subject to `lo <= x <= hi`, `H` symmetric positive definite.
//!
//! Source: Nocedal & Wright, "Numerical Optimization", 2nd ed. (2006), Algorithm 16.3 (Active-Set Method for
//! Convex QP, Section 16.5), specialised to bound constraints: the working set is a set of variables held
//! at a bound, so the equality-constrained subproblem is an unconstrained solve on the free variables, the
//! blocking-constraint step length is the book's `min(1, ...)` ratio test, and the multiplier test is the
//! book's sign test (a bound is optimal if its multiplier is >= 0; the multiplier of a bound is the
//! sign-adjusted gradient component `q_i = (Hx + g)_i`). Equation numbers were not checked against the book.
//! Ported from chetools/muse `twin-cstr/solver/src/qp.rs` (same authors); the Cholesky factorization is the
//! library routine `linalg::cholesky`. Reference for tests: `scipy.optimize.minimize(method="L-BFGS-B")` and exact
//! enumeration of active sets. Python twin: `library/python/agent_tools/optimize/qp_box.py`.
//!
//! Algorithm (one pass of the loop = one iteration `nit`; exactly the steps of the Python twin):
//! 0. Factor `H` once (probe); fail with "not positive definite" if that fails. Start at
//!    `x = clamp(0, lo, hi)` with every variable at a bound (`x_i <= lo_i` or `x_i >= hi_i`) in the
//!    working set. A variable with `lo_i == hi_i` is fixed: it stays in the working set for good.
//! 1. `q = H x + g`. Free set `F` = variables not in the working set. Solve `H_FF p_F = -q_F` by
//!    Cholesky (one factorization per pass, none right after a full step), `p = 0` elsewhere.
//! 2. If `max|p| <= tol_step`, `x` is the minimizer on the free set. The multiplier of a variable at
//!    its lower bound is `q_i`, at its upper bound `-q_i`. If no non-fixed bound has a multiplier
//!    below `-tol_dual`, stop (success); otherwise release the bound with the most negative multiplier
//!    and go to 1.
//! 3. Else step `x += alpha p`, `alpha = min(1, ratios to the bounds of the free variables)`; the first
//!    bound reached (lowest index on ties) joins the working set and `x_i` is set exactly to it. After a
//!    full step (`alpha = 1`, nothing blocks) `x` is the free-set minimizer by construction, so the next
//!    pass takes `p = 0` without a new solve and goes straight to the multiplier test of step 2.
//!
//! In exact arithmetic the method terminates finitely when no blocking step has length zero; the pass count is
//! capped at `max_iter` (default `max(50, 20 n)`) as a safety net against rounding and unforeseen cycling. The cap
//! is not a theoretical bound: releases count as passes, and well-conditioned nondegenerate problems were found
//! that need about `5 n` passes (see `default_cap_reproducers`); the random test problems needed at most 85 at n = 124.
//!
//! Changes from the muse source (each is a test in `tests/qp_box.rs`):
//! - the diagonal shift `1e-10 max(1, max H_ii)` applied when `H` fails Cholesky is specific to its caller and is
//!   NOT ported: `H` not positive definite is a reported failure;
//! - after a full step the next pass takes `p = 0` instead of re-solving. The source re-solves and needs
//!   `max|p| <= tol_step`, which rounding denies when `cond(H) |x| eps > tol_step`; it then runs into its cap
//!   and fails (n = 2, cond 6e6: `ill_conditioned_free_set_does_not_stall`);
//! - a fixed variable (`lo == hi`) is never released (the source releases one whose multiplier has the "wrong"
//!   sign and re-adds it later: 6 iterations instead of 2 on `fixed_variable_n5`);
//! - the blocking variable is set exactly to its bound and free variables are clamped to `[lo, hi]` after each
//!   step (feasibility holds exactly);
//! - the result carries the active set, multipliers, objective and counts instead of `Option<x>`; only the lower
//!   triangle of `H` is read, everywhere (as in `cholesky`); the cap and tolerances are options.
//!
//! Units: `x`, `lo`, `hi` in units of x; `H` in f/x^2; `g` in f/x; `f` in units of f; multipliers in f/x
//! (>= 0). `tol_step` is in units of x (absolute), `tol_dual` in f/x (absolute): rescale the problem if
//! its natural size is far from 1.
//!
//! Semidefinite `H` is rejected (`success = false`, `x = clamp(0)`): the up-front factorization fails;
//! add a small diagonal term yourself. Infinite bounds (+-inf, any mix) are allowed: a variable with
//! both bounds infinite is always free, and with `H` positive definite the problem is bounded below.
//! NaN bounds, `lo > hi`, non-finite `H` or `g` are invalid input (`success = false`, `nfact = 0`).

use crate::linalg::cholesky::{cholesky, cholesky_solve};

/// Default `tol_step` (units of x, absolute).
pub const TOL_STEP: f64 = 1e-10;
/// Default `tol_dual` (units of f/x, absolute).
pub const TOL_DUAL: f64 = 1e-9;

/// Options of [`qp_box`].
#[derive(Clone, Debug)]
pub struct QpBoxOptions {
    /// Cap on active-set iterations (passes); `None` means `max(50, 20 n)` (the source's `4 n` was too small). Must be >= 1 if given.
    pub max_iter: Option<usize>,
    /// The free-set subproblem counts as solved when `max|p| <= tol_step` (units of x, absolute).
    /// Default 1e-10. Finite and >= 0.
    pub tol_step: f64,
    /// A bound is released only if its multiplier is below `-tol_dual` (units of f/x, absolute).
    /// Default 1e-9. Finite and >= 0.
    pub tol_dual: f64,
}

impl Default for QpBoxOptions {
    fn default() -> Self {
        QpBoxOptions { max_iter: None, tol_step: TOL_STEP, tol_dual: TOL_DUAL }
    }
}

/// Result of [`qp_box`].
#[derive(Clone, Debug)]
pub struct QpBoxResult {
    /// Solution (units of x); the last iterate (feasible) when `success` is false.
    pub x: Vec<f64>,
    /// Objective `0.5 x^T H x + g^T x` at `x` (units of f); NaN for invalid input.
    pub f: f64,
    /// `-1` variable at its lower bound, `+1` at its upper bound, `0` free. A fixed variable
    /// (`lo == hi`) is `-1` if its multiplier is >= 0, else `+1`.
    pub active: Vec<i8>,
    /// Multipliers (units of f/x), >= 0 for a bound-active variable, 0 for a free one. The multiplier
    /// of a lower bound is `q_i = (Hx + g)_i`, of an upper bound `-q_i`; a fixed variable reports
    /// `|q_i|`. An active variable with multiplier 0 is a degenerate (weakly active) bound.
    pub multipliers: Vec<f64>,
    /// True if the KKT conditions hold to the tolerances.
    pub success: bool,
    /// `"converged"` or the reason for failure (invalid input or option, not positive definite,
    /// `max_iter` reached).
    pub message: String,
    /// Active-set iterations (passes of the loop, including the last, certifying one).
    pub nit: usize,
    /// Cholesky factorizations (the up-front probe of `H` plus one per pass with a non-empty free set, except the
    /// pass right after a full step).
    pub nfact: usize,
}

/// `H x` reading only the lower triangle of the row-major `h`, accumulating left to right.
fn symmetric_matvec(h: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += (if j <= i { h[i * n + j] } else { h[j * n + i] }) * x[j];
        }
        y[i] = s;
    }
    y
}

fn objective(h: &[f64], g: &[f64], x: &[f64]) -> f64 {
    let hx = symmetric_matvec(h, g.len(), x);
    let mut f = 0.0;
    for i in 0..x.len() {
        f += 0.5 * x[i] * hx[i] + g[i] * x[i];
    }
    f
}

fn active_of(at_lo: &[bool], at_hi: &[bool]) -> Vec<i8> {
    (0..at_lo.len()).map(|i| if at_lo[i] { -1 } else if at_hi[i] { 1 } else { 0 }).collect()
}

/// Minimize `0.5 x^T H x + g^T x` subject to `lo <= x <= hi` by the primal active-set method.
///
/// `h`: row-major `n x n`, symmetric positive definite, units f/x^2 (lower triangle read), `n = g.len()`.
/// `g`: units f/x. `lo`, `hi`: units of x, may be `-inf` / `+inf`, `lo <= hi`. Never panics; invalid
/// input is reported in the result. See the module docs for the algorithm, defaults and failure modes.
pub fn qp_box(h: &[f64], g: &[f64], lo: &[f64], hi: &[f64], opt: &QpBoxOptions) -> QpBoxResult {
    let n = g.len();
    let max_iter = opt.max_iter.unwrap_or(50.max(20 * n));
    let fail = |msg: String, x: Vec<f64>, nit: usize, nfact: usize, active: Vec<i8>| -> QpBoxResult {
        let f = if nfact == 0 && nit == 0 { f64::NAN } else { objective(h, g, &x) };
        QpBoxResult { x, f, active, multipliers: vec![0.0; n], success: false, message: msg, nit, nfact }
    };
    let bad = |msg: &str| -> QpBoxResult {
        QpBoxResult {
            x: vec![0.0; n], f: f64::NAN, active: vec![0; n], multipliers: vec![0.0; n],
            success: false, message: msg.to_string(), nit: 0, nfact: 0,
        }
    };
    if n == 0 || h.len() != n * n || lo.len() != n || hi.len() != n {
        return bad("invalid input: need n >= 1, h n x n, g, lo, hi of length n");
    }
    if !(h.iter().all(|v| v.is_finite()) && g.iter().all(|v| v.is_finite())) {
        return bad("invalid input: h or g not finite");
    }
    if (0..n).any(|i| lo[i].is_nan() || hi[i].is_nan() || lo[i] > hi[i] || lo[i] == f64::INFINITY || hi[i] == f64::NEG_INFINITY) {
        return bad("invalid input: bounds must satisfy lo <= hi, be finite or infinite of the right sign, not NaN");
    }
    if !(max_iter >= 1 && opt.tol_step.is_finite() && opt.tol_step >= 0.0 && opt.tol_dual.is_finite() && opt.tol_dual >= 0.0) {
        return bad("invalid option: max_iter >= 1, tol_step and tol_dual finite and >= 0");
    }

    let mut x: Vec<f64> = (0..n).map(|i| 0.0f64.max(lo[i]).min(hi[i])).collect();
    let probe = cholesky(h, n);
    let mut nfact = 1;
    if !probe.success {
        return fail(format!("h is not positive definite ({})", probe.message), x, 0, nfact, vec![0; n]);
    }
    let fixed: Vec<bool> = (0..n).map(|i| lo[i] == hi[i]).collect();
    let mut at_lo: Vec<bool> = (0..n).map(|i| x[i] <= lo[i]).collect();
    let mut at_hi: Vec<bool> = (0..n).map(|i| x[i] >= hi[i]).collect();
    let mut nit = 0;
    let mut solved = false; // true right after a full step: x is then the free-set minimizer, p = 0 without a new solve
    while nit < max_iter {
        nit += 1;
        let mut q = symmetric_matvec(h, n, &x);
        for i in 0..n {
            q[i] += g[i];
        }
        let free: Vec<usize> = (0..n).filter(|&i| !at_lo[i] && !at_hi[i]).collect();
        let mut p = vec![0.0; n];
        if !free.is_empty() && !solved {
            let nf = free.len();
            let mut hff = vec![0.0; nf * nf];
            for a in 0..nf {
                for b in 0..nf {
                    let (i, j) = (free[a], free[b]);
                    hff[a * nf + b] = if j <= i { h[i * n + j] } else { h[j * n + i] };
                }
            }
            let c = cholesky(&hff, nf);
            nfact += 1;
            if !c.success {
                // cannot happen for H positive definite (principal submatrix)
                return fail(format!("free-set submatrix not positive definite ({})", c.message), x, nit, nfact, active_of(&at_lo, &at_hi));
            }
            let mut pf: Vec<f64> = free.iter().map(|&i| -q[i]).collect();
            cholesky_solve(&c.l, nf, &mut pf, 1);
            for (a, &i) in free.iter().enumerate() {
                p[i] = pf[a];
            }
        }
        solved = false;
        let mut pmax = 0.0f64;
        for v in &p {
            pmax = pmax.max(v.abs());
        }
        if pmax <= opt.tol_step {
            let mut drop: Option<usize> = None;
            let mut worst = opt.tol_dual;
            for i in 0..n {
                if fixed[i] {
                    continue;
                }
                if at_lo[i] && -q[i] > worst {
                    worst = -q[i];
                    drop = Some(i);
                }
                if at_hi[i] && q[i] > worst {
                    worst = q[i];
                    drop = Some(i);
                }
            }
            match drop {
                None => {
                    let mut active = active_of(&at_lo, &at_hi);
                    let mut mult = vec![0.0; n];
                    for i in 0..n {
                        if at_lo[i] || at_hi[i] {
                            if fixed[i] {
                                active[i] = if q[i] >= 0.0 { -1 } else { 1 };
                                mult[i] = q[i].abs();
                            } else {
                                let m = if at_lo[i] { q[i] } else { -q[i] };
                                mult[i] = m.max(0.0); // in [-tol_dual, 0) roundoff: reported as 0
                            }
                        }
                    }
                    let f = objective(h, g, &x);
                    return QpBoxResult { x, f, active, multipliers: mult, success: true, message: "converged".to_string(), nit, nfact };
                }
                Some(i) => {
                    at_lo[i] = false;
                    at_hi[i] = false;
                    continue;
                }
            }
        }
        let mut alpha = 1.0;
        let mut blocker: Option<(usize, bool)> = None;
        for &i in &free {
            if p[i] > 0.0 {
                let a = (hi[i] - x[i]) / p[i];
                if a < alpha {
                    alpha = a;
                    blocker = Some((i, true));
                }
            } else if p[i] < 0.0 {
                let a = (lo[i] - x[i]) / p[i];
                if a < alpha {
                    alpha = a;
                    blocker = Some((i, false));
                }
            }
        }
        for &i in &free {
            x[i] = (x[i] + alpha * p[i]).max(lo[i]).min(hi[i]);
        }
        solved = blocker.is_none();
        if let Some((i, upper)) = blocker {
            if upper {
                at_hi[i] = true;
                x[i] = hi[i];
            } else {
                at_lo[i] = true;
                x[i] = lo[i];
            }
        }
    }
    fail("max_iter reached".to_string(), x, nit, nfact, active_of(&at_lo, &at_hi))
}
