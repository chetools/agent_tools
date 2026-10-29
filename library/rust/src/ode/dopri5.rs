//! Adaptive Dormand-Prince 5(4) explicit Runge-Kutta integrator for non-stiff initial value problems
//! of any dimension `n` (the scheme of SciPy `solve_ivp(method="RK45")` and MATLAB `ode45`).
//!
//! Sources: J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae", J. Comput.
//! Appl. Math. 6(1), 19-26 (1980) (the 7-stage FSAL tableau, pair 5(4)); E. Hairer, S. P. Norsett and
//! G. Wanner, "Solving Ordinary Differential Equations I", 2nd ed., Springer (1993), Sec. II.4
//! (starting step size, step size control). Ported from chetools/muse `twin-cstr` (`solver/src/model.rs`
//! `dopri5`, hard-wired to a 4-vector, and `python/cstr_model.py` `_solve_ivp`); generalised to any `n`
//! with an `f(t, y, dydt)` closure and reporting failures in the result instead of panicking.
//! Reference for tests: SciPy `solve_ivp(method="RK45")` (same tolerances, compared step by step) and
//! SciPy `DOP853` at rtol 1e-13, atol 1e-14.
//!
//! One step from `(t, y)` with step `h` and `k1 = f(t, y)` (stored from the previous accepted step,
//! "first same as last"):
//!
//! ```text
//! k_s = f(t + c_s h, y + h sum_{j<s} a_sj k_j),  s = 2..6      (c = 1/5, 3/10, 4/5, 8/9, 1)
//! y_new = y + h (35/384 k1 + 500/1113 k3 + 125/192 k4 - 2187/6784 k5 + 11/84 k6)   (5th order)
//! k7 = f(t + h, y_new)
//! err = h (71/57600 k1 - 71/16695 k3 + 71/1920 k4 - 17253/339200 k5 + 22/525 k6 - 1/40 k7)  (4th-order estimate)
//! ```
//!
//! Step acceptance and sizing (SciPy `RK45`): `E = rms_i(err_i / (atol_i + rtol max(|y_i|, |y_new_i|)))`,
//! accepted if `E < 1`. With `fac = 0.9 E^(-1/5)`: after an accepted step `h <- h min(10, fac)` (`10` if
//! `E == 0`), and `fac` is at most 1 if the step before was rejected; after a rejected step
//! `h <- h max(0.2, fac)`. The initial step is SciPy's `select_initial_step` (Hairer et al. II.4): see
//! [`dopri5`].
//!
//! **Changes from the source** (`model.rs` / `_solve_ivp`), made to follow SciPy `RK45` exactly: growth
//! cap 10 (source 5), shrink floor 0.2 (source 0.1), no growth right after a rejection (source: none),
//! acceptance `E < 1` (source `E <= 1`); starting step as SciPy (source: `1e-6` only if both `d0` and
//! `d1` are small, and `max(d0, d1, d2)` in `h1`, which divides by zero for `f(t0, y0) = 0`). Also new:
//! failures are returned instead of panicking/raising, non-finite error estimates reject the step,
//! per-component `atol`, `h0`/`hmax` options, and step-size underflow detection.
//!
//! Python twin: `library/python/agent_tools/ode/dopri5.py`.

/// Options of [`dopri5`]. `Default` gives the source defaults (see each field).
#[derive(Clone, Debug)]
pub struct Dopri5Options {
    /// Relative tolerance, dimensionless. Default 1e-6 (the source default). Must be > 0.
    pub rtol: f64,
    /// Absolute tolerance, in the units of `y` (used for every component unless `atol_vec` is
    /// non-empty). Default 1e-9 (the source default). Must be >= 0.
    pub atol: f64,
    /// Per-component absolute tolerances, units of `y`. Empty (default) means `atol` for all;
    /// otherwise its length must equal the number of components.
    pub atol_vec: Vec<f64>,
    /// First step (SciPy `first_step`), in the units of `t`. Default `None` = the automatic
    /// starting-step heuristic (costs one extra `f` call). Must be > 0 if given.
    pub h0: Option<f64>,
    /// Largest step (SciPy `max_step`), in the units of `t`. Default `None` = no limit (t1 - t0).
    pub hmax: Option<f64>,
    /// Maximum number of attempted steps (accepted + rejected). Default 100 000 (the source budget).
    pub max_steps: usize,
}

impl Default for Dopri5Options {
    fn default() -> Self {
        Dopri5Options { rtol: 1e-6, atol: 1e-9, atol_vec: Vec::new(), h0: None, hmax: None, max_steps: 100_000 }
    }
}

/// Result of [`dopri5`]. The trajectory holds the initial point and every accepted step.
#[derive(Clone, Debug)]
pub struct Dopri5Result {
    /// Times of the stored points, in the units of `t` (`t[0] = t0`; ends exactly at `t1` on success).
    pub t: Vec<f64>,
    /// States at those times, row-major `t.len() x n`, units of `y`.
    pub y: Vec<f64>,
    /// Number of components.
    pub n: usize,
    /// True if the integration reached `t1`.
    pub success: bool,
    /// Human-readable status (`"success"` or the reason for stopping early).
    pub message: String,
    /// Calls of `f` (1 at `t0`, 1 for the automatic first step, 6 per attempted step).
    pub nfev: usize,
    /// Accepted steps.
    pub nsteps: usize,
    /// Rejected steps.
    pub nrejected: usize,
}

impl Dopri5Result {
    /// State at the last stored time (at `t1` if `success`).
    pub fn y_final(&self) -> &[f64] {
        &self.y[self.y.len() - self.n..]
    }
    /// State at stored point `k`.
    pub fn y_at(&self, k: usize) -> &[f64] {
        &self.y[k * self.n..(k + 1) * self.n]
    }
}

/// Root-mean-square norm, `NaN` if any entry is not finite (so a bad value is never hidden by `max`).
fn rms(v: impl Iterator<Item = f64>, n: usize) -> f64 {
    let s: f64 = v.map(|x| x * x).sum();
    if s.is_finite() {
        (s / n as f64).sqrt()
    } else {
        f64::NAN
    }
}

/// Integrate `dy/dt = f(t, y)`, `y(t0) = y0`, from `t0` to `t1 > t0` (forward in time only).
///
/// * `f(t, y, dydt)` writes the right-hand side into `dydt` (all length `n`; `t` in the units of the
///   independent variable, `y` and `dydt` in the units of the state and state per unit of `t`).
///   It is called at the stage times, so time-dependent systems need nothing extra.
/// * `t_span = (t0, t1)`, `y0` the initial state (length `n`, at least 1).
///
/// Starting step (if `opts.h0` is `None`), SciPy's heuristic: with `scale_i = atol_i + rtol |y0_i|`,
/// `d0 = rms(y0/scale)`, `d1 = rms(f0/scale)`, `h0 = 1e-6` if `d0 < 1e-5` or `d1 < 1e-5` else
/// `0.01 d0/d1`; one explicit Euler probe gives `d2 = rms((f(t0+h0, y0+h0 f0) - f0)/scale)/h0`;
/// `h1 = (0.01/max(d1, d2))^(1/5)` (`max(1e-6, 1e-3 h0)` if both are <= 1e-15); the start is
/// `min(100 h0, h1, t1 - t0, hmax)`.
///
/// Failures are reported in the result (never a panic) for: `t1 <= t0`, empty `y0`, bad tolerances,
/// `atol_vec` length or `h0`/`hmax`, an automatic first step that is not a positive finite number
/// (zero `scale` or non-finite `f`: pass `h0`), step size underflow
/// (`h < 1e-15 max(|t|, 1e-3 (t1 - t0))`, which is what a persistent NaN or a finite-time blow-up ends in)
/// and `max_steps`; the partial trajectory is returned. A non-finite error estimate rejects the step
/// (`h` shrinks by the floor factor 0.2).
/// Cost per attempted step: 6 calls of `f` and about `n * 40` flops. Not suited to stiff problems:
/// stability then limits the step size, and `nsteps` grows with the stiffness, not with the accuracy.
pub fn dopri5(
    mut f: impl FnMut(f64, &[f64], &mut [f64]),
    t_span: (f64, f64),
    y0: &[f64],
    opts: &Dopri5Options,
) -> Dopri5Result {
    // Dormand-Prince tableau (Dormand & Prince 1980)
    const C: [f64; 5] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];
    const A: [[f64; 5]; 5] = [
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    ];
    const B: [f64; 6] = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0];
    const E: [f64; 7] =
        [71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0];

    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;

    let n = y0.len();
    let (t0, t1) = t_span;
    let mut res = Dopri5Result {
        t: vec![t0],
        y: y0.to_vec(),
        n,
        success: false,
        message: String::new(),
        nfev: 0,
        nsteps: 0,
        nrejected: 0,
    };
    let bad = |m: &str, mut r: Dopri5Result| {
        r.message = m.to_string();
        r
    };
    if n == 0 {
        return bad("empty initial state", res);
    }
    if !(t1 > t0) || !t0.is_finite() || !t1.is_finite() {
        return bad("t_span must satisfy t0 < t1 (both finite)", res);
    }
    if !(opts.rtol > 0.0) || !(opts.atol >= 0.0) || !opts.rtol.is_finite() || !opts.atol.is_finite() {
        return bad("need finite rtol > 0 and atol >= 0", res);
    }
    if !opts.atol_vec.is_empty() && (opts.atol_vec.len() != n || opts.atol_vec.iter().any(|a| !(*a >= 0.0))) {
        return bad("atol_vec must be empty or have one entry >= 0 per component", res);
    }
    if opts.h0.map_or(false, |h| !(h > 0.0)) || opts.hmax.map_or(false, |h| !(h > 0.0)) {
        return bad("h0 and hmax must be > 0", res);
    }
    let atol = |i: usize| if opts.atol_vec.is_empty() { opts.atol } else { opts.atol_vec[i] };
    let span = t1 - t0;
    let hmax = opts.hmax.unwrap_or(span);

    let mut y = y0.to_vec();
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; 7];
    let mut yt = vec![0.0; n];
    let mut ynew = vec![0.0; n];

    f(t0, &y, &mut k[0]);
    res.nfev = 1;

    let mut h = match opts.h0 {
        Some(h) => h,
        None => {
            // SciPy select_initial_step (Hairer, Norsett, Wanner II.4)
            let sc: Vec<f64> = (0..n).map(|i| atol(i) + opts.rtol * y[i].abs()).collect();
            let d0 = rms((0..n).map(|i| y[i] / sc[i]), n);
            let d1 = rms((0..n).map(|i| k[0][i] / sc[i]), n);
            let h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 };
            for i in 0..n {
                yt[i] = y[i] + h0 * k[0][i];
            }
            let mut f1 = vec![0.0; n];
            f(t0 + h0, &yt, &mut f1);
            res.nfev += 1;
            let d2 = rms((0..n).map(|i| (f1[i] - k[0][i]) / sc[i]), n) / h0;
            let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
                (1e-6f64).max(h0 * 1e-3)
            } else {
                (0.01 / d1.max(d2)).powf(0.2)
            };
            (100.0 * h0).min(h1)
        }
    };
    h = h.min(span).min(hmax);
    if !(h > 0.0) || !h.is_finite() {
        return bad(
            "automatic first step is not a positive finite number (zero scale or non-finite f); pass h0",
            res,
        );
    }

    let mut t = t0;
    let mut attempts = 0usize;
    let mut rejected = false; // the previous attempt was rejected
    while t < t1 {
        if attempts >= opts.max_steps {
            res.message = format!("max_steps ({}) reached at t = {t}", opts.max_steps);
            return res;
        }
        attempts += 1;
        h = h.min(hmax);
        let last = h >= t1 - t;
        if last {
            h = t1 - t;
        }
        for s in 0..5 {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..=s {
                    acc += A[s][j] * k[j][i];
                }
                yt[i] = y[i] + h * acc;
            }
            let (_, hi) = k.split_at_mut(s + 1);
            f(t + C[s] * h, &yt, &mut hi[0]);
        }
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..6 {
                acc += B[j] * k[j][i];
            }
            ynew[i] = y[i] + h * acc;
        }
        let t_new = if last { t1 } else { t + h };
        {
            let (_, hi) = k.split_at_mut(6);
            f(t_new, &ynew, &mut hi[0]);
        }
        res.nfev += 6;
        let en = rms(
            (0..n).map(|i| {
                let mut acc = 0.0;
                for j in 0..7 {
                    acc += E[j] * k[j][i];
                }
                h * acc / (atol(i) + opts.rtol * y[i].abs().max(ynew[i].abs()))
            }),
            n,
        );
        let en = if en.is_finite() { en } else { 1e10 };
        if en < 1.0 {
            t = t_new;
            y.copy_from_slice(&ynew);
            k.swap(0, 6); // first-same-as-last
            res.nsteps += 1;
            res.t.push(t);
            res.y.extend_from_slice(&y);
            let mut fac = if en == 0.0 { MAX_FACTOR } else { (SAFETY * en.powf(-0.2)).min(MAX_FACTOR) };
            if rejected {
                fac = fac.min(1.0); // no growth right after a rejection
            }
            h *= fac;
            rejected = false;
        } else {
            res.nrejected += 1;
            h *= (SAFETY * en.powf(-0.2)).max(MIN_FACTOR);
            rejected = true;
        }
        if t < t1 && h < 1e-15 * t.abs().max(1e-3 * span) {
            res.message = format!("step size underflow at t = {t}");
            return res;
        }
    }
    res.success = true;
    res.message = "success".to_string();
    res
}
