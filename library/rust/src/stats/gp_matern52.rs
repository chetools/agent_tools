//! Gaussian-process regression with an ARD Matern-5/2 kernel and type-II maximum-likelihood
//! hyperparameters.
//!
//! Source: Rasmussen & Williams, "Gaussian Processes for Machine Learning" (MIT Press, 2006):
//! eq. (4.17) (Matern class, nu = 5/2), Algorithm 2.1 (Cholesky-based predictions, eqs. 2.25-2.26),
//! eq. (5.8) (log marginal likelihood) and eq. (5.9) (its gradient, `0.5 tr((alpha alpha^T - K^-1) dK/dtheta)`).
//! Ported from chetools/claude `sequential-design/solver/src/lib.rs` (`matern52`, `nll_grad`,
//! `gp_fit`, `gp_build`, `gp_predict`, `posterior_cov`; same authors) using the library's
//! `linalg::cholesky` and `optimize::bfgs`. Reference for tests: an independent NumPy/SciPy
//! implementation of the same marginal likelihood and `scipy.optimize.minimize(method="L-BFGS-B")`.
//! Python twin: `library/python/agent_tools/stats/gp_matern52.py`.
//!
//! # Model
//! `y = f(x) + e`, `f ~ GP(ym, ys^2 k)`, `e ~ N(0, noise_sd^2)`, with the kernel
//! `k(a, b) = signal_sd^2 (1 + s + s^2/3) exp(-s)`, `s = sqrt(5) sqrt(sum_i ((a_i - b_i)/ell_i)^2)`.
//! `y` is standardised before fitting, `yz = (y - ym) / ys` (`ym` the mean, `ys` the population
//! standard deviation, or 1 if `y` is constant); the GP is fitted to `yz`, and every output below is
//! converted back to the units of `y`. A jitter [`JITTER`] = 1e-8 (variance of the standardised
//! `y`) is added to the diagonal of the covariance matrix in every factorization, so the effective
//! noise variance is `noise_sd^2 + 1e-8 ys^2`; it is part of the model definition.
//!
//! # Inputs and their scaling
//! `X` is `n x d` row-major, in any units. The length scales are in the units of `X`. The default
//! bounds on the length scales ([`GpOptions::log_length_bounds`]) and the default starting length
//! scales assume that **inputs are scaled to a unit domain (about [0, 1])**: `0.02 .. 5` times the
//! domain width. If `X` is in other units, scale it to [0, 1] first, or pass bounds and starts in
//! the units of `X`.
//!
//! # Hyperparameter fitting
//! The fitted parameters are `theta = [ln ell_1 .. ln ell_d, ln sf, ln sn]` with `sf = signal_sd/ys`
//! and `sn = noise_sd/ys` (standardised units; `sn` is absent when the noise is fixed). `bfgs` is
//! unconstrained, so each parameter is boxed with the source's transform
//! `theta_k = lo_k + (hi_k - lo_k) (1 + tanh(z_k/2))/2` and BFGS runs on the unbounded `z`, with the
//! chain rule `d theta/d z = (theta - lo)(hi - theta)/(hi - lo)`. Consequence: when the optimum lies
//! on a bound the likelihood is flat in `z`, the optimizer stops when the `z`-gradient is below
//! `gtol` and the parameter sits close to (not exactly at) the bound. Each start is a BFGS run; the
//! start with the lowest finite negative log marginal likelihood is kept. Starts whose
//! covariance matrix fails the Cholesky factorization return `+inf` and are rejected by the line
//! search. Starts: one per entry of [`GpOptions::length_starts`] (isotropic, `sf = 1`, `sn = 0.1`),
//! then [`GpOptions::n_random_starts`] with log length scales uniform in the box, drawn from
//! `SplitMix64::from_seed_stream(seed, 0)`. Start values are moved inside the box by at least 1 % of
//! its width.
//!
//! Cost: `O(n^3)` per likelihood evaluation, `O(n^2 d)` memory; intended for `n` up to a few hundred.

use std::f64::consts::PI;

use crate::linalg::cholesky::{cholesky, solve_lower, solve_upper_t};
use crate::optimize::bfgs::{bfgs, BfgsOptions};
use crate::random::splitmix64::SplitMix64;

/// `sqrt(5)`.
pub const SQRT5: f64 = 2.23606797749979;
/// Diagonal jitter added to the covariance matrix, in variance units of the standardised `y`.
pub const JITTER: f64 = 1e-8;
/// Start values are kept this fraction of the box width away from its edges (dimensionless).
pub const START_MARGIN: f64 = 0.01;

/// Matern-5/2 covariance between the points `a` and `b` (length `d` each, units of `x`) with ARD
/// length scales `ell` (units of `x`, positive) and signal variance `sf2` (units of `y^2`).
pub fn matern52(a: &[f64], b: &[f64], ell: &[f64], sf2: f64) -> f64 {
    let mut r2 = 0.0;
    for i in 0..ell.len() {
        let t = (a[i] - b[i]) / ell[i];
        r2 += t * t;
    }
    let s = SQRT5 * r2.sqrt();
    sf2 * (1.0 + s + s * s / 3.0) * (-s).exp()
}

/// Covariance matrix between the point sets `a` (`na x d`) and `b` (`nb x d`), row-major `na x nb`
/// (units of `y^2`). Arguments as in [`matern52`].
pub fn matern52_matrix(a: &[f64], b: &[f64], d: usize, ell: &[f64], sf2: f64) -> Vec<f64> {
    let (na, nb) = (a.len() / d, b.len() / d);
    let mut k = vec![0.0; na * nb];
    for i in 0..na {
        for j in 0..nb {
            k[i * nb + j] = matern52(&a[i * d..i * d + d], &b[j * d..j * d + d], ell, sf2);
        }
    }
    k
}

/// Population mean and standard deviation of `v` (standard deviation replaced by 1 if it is 0).
fn standardization(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|t| (t - m) * (t - m)).sum::<f64>() / v.len() as f64;
    let s = var.sqrt();
    (m, if s == 0.0 { 1.0 } else { s })
}

/// Map the unbounded `z` into the box `lo < theta < hi` (`theta = lo + (hi-lo)(1+tanh(z/2))/2`).
/// Returns `theta` and `d theta/d z` (both in the units of `theta`, here logarithms).
pub fn to_theta(z: &[f64], lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut th = vec![0.0; z.len()];
    let mut dth = vec![0.0; z.len()];
    for k in 0..z.len() {
        th[k] = lo[k] + (hi[k] - lo[k]) * 0.5 * (1.0 + (0.5 * z[k]).tanh());
        dth[k] = (th[k] - lo[k]) * (hi[k] - th[k]) / (hi[k] - lo[k]);
    }
    (th, dth)
}

/// Inverse of [`to_theta`]: `z = ln(p/(1-p))`, `p = (theta-lo)/(hi-lo)`. `theta` must lie strictly
/// inside the box (`z` is infinite on the edge).
pub fn to_z(theta: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
    (0..theta.len())
        .map(|k| {
            let p = (theta[k] - lo[k]) / (hi[k] - lo[k]);
            (p / (1.0 - p)).ln()
        })
        .collect()
}

/// Negative log marginal likelihood of the standardised data `y_std` and its gradient with respect
/// to `theta` (written into `grad`).
///
/// * `x`: `n x d` row-major inputs (units of `x`); `y_std`: length `n`, dimensionless.
/// * `theta = [ln ell_1 .. ln ell_d, ln sf, ln sn]`, length `d + 2` (`ln ell` in ln(units of `x`),
///   `sf`, `sn` relative to the standard deviation of `y`); with `fixed_noise_var = Some(v)` the
///   last entry is absent (length `d + 1`) and the noise variance is `v` (standardised units).
/// * Returns `+inf` if the covariance matrix is not numerically positive definite (the gradient is
///   then not written). Includes the constant `n/2 ln(2 pi)`; the jitter [`JITTER`] is added.
pub fn nll_grad(x: &[f64], d: usize, y_std: &[f64], theta: &[f64], fixed_noise_var: Option<f64>, grad: &mut [f64]) -> f64 {
    let n = y_std.len();
    assert_eq!(x.len(), n * d, "nll_grad: x must be n*d");
    let fit_noise = fixed_noise_var.is_none();
    let p = d + 1 + fit_noise as usize;
    assert_eq!(theta.len(), p, "nll_grad: theta must have d+2 entries (d+1 with fixed noise)");
    assert_eq!(grad.len(), p, "nll_grad: grad must match theta");
    let ell: Vec<f64> = theta[..d].iter().map(|t| t.exp()).collect();
    let sf2 = (2.0 * theta[d]).exp();
    let sn2 = match fixed_noise_var {
        Some(v) => v,
        None => (2.0 * theta[d + 1]).exp(),
    };
    // pairwise scaled squared distances per dimension, kernel and the common derivative factor
    let mut d2 = vec![0.0; n * n * d];
    let mut kf = vec![0.0; n * n];
    let mut common = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut r2 = 0.0;
            for c in 0..d {
                let t = (x[i * d + c] - x[j * d + c]) / ell[c];
                d2[(i * n + j) * d + c] = t * t;
                r2 += t * t;
            }
            let s = SQRT5 * r2.sqrt();
            let e = (-s).exp();
            kf[i * n + j] = sf2 * (1.0 + s + s * s / 3.0) * e;
            common[i * n + j] = sf2 * (5.0 / 3.0) * (1.0 + s) * e;
        }
    }
    let mut k = kf.clone();
    for i in 0..n {
        k[i * n + i] += sn2 + JITTER;
    }
    let ch = cholesky(&k, n);
    if !ch.success {
        return f64::INFINITY;
    }
    let l = ch.l;
    let mut alpha = y_std.to_vec();
    solve_lower(&l, n, &mut alpha, 1);
    solve_upper_t(&l, n, &mut alpha, 1);
    let mut nll = 0.5 * n as f64 * (2.0 * PI).ln();
    for i in 0..n {
        nll += 0.5 * y_std[i] * alpha[i] + l[i * n + i].ln();
    }
    // W = K^-1 - alpha alpha^T; d nll/d theta_k = 0.5 sum(W * dK/d theta_k)
    let mut w = vec![0.0; n * n];
    for i in 0..n {
        w[i * n + i] = 1.0;
    }
    solve_lower(&l, n, &mut w, n);
    solve_upper_t(&l, n, &mut w, n);
    for i in 0..n {
        for j in 0..n {
            w[i * n + j] -= alpha[i] * alpha[j];
        }
    }
    let mut g = vec![0.0; p];
    let mut tr = 0.0;
    for i in 0..n {
        tr += w[i * n + i];
        for j in 0..n {
            let wij = w[i * n + j];
            for c in 0..d {
                g[c] += wij * common[i * n + j] * d2[(i * n + j) * d + c];
            }
            g[d] += wij * 2.0 * kf[i * n + j];
        }
    }
    if fit_noise {
        g[d + 1] = tr * 2.0 * sn2;
    }
    for k in 0..p {
        grad[k] = 0.5 * g[k];
    }
    nll
}

/// Options of [`gp_fit`].
#[derive(Clone, Debug)]
pub struct GpOptions {
    /// `None` (default): fit the noise standard deviation. `Some(s)`: fix it to `s >= 0`
    /// (units of `y`); it is not fitted.
    pub noise_sd: Option<f64>,
    /// Box for `ln ell_i`, (lower, upper), in ln(units of `x`), the same for every input.
    /// Default `(ln 0.02, ln 5)`, meant for inputs scaled to [0, 1] (see the module docs).
    pub log_length_bounds: (f64, f64),
    /// Box for `ln(signal_sd / ys)`, `ys` the standard deviation of `y`. Default `(ln 0.1, ln 10)`.
    pub log_signal_bounds: (f64, f64),
    /// Box for `ln(noise_sd / ys)` (used only when the noise is fitted). Default `(ln 1e-3, 0)`.
    pub log_noise_bounds: (f64, f64),
    /// Isotropic starting length scales, one BFGS run each (units of `x`, positive).
    /// Default `[0.1, 0.3, 1.0]`.
    pub length_starts: Vec<f64>,
    /// Number of extra random starts (log length scales uniform in the box). Default 0.
    pub n_random_starts: usize,
    /// Seed of the random starts (`SplitMix64::from_seed_stream(seed, 0)`). Default 0.
    pub seed: u64,
    /// BFGS options on the unbounded variables `z`. Default `max_iter = 200`, `gtol = 1e-5`
    /// (the library defaults; `gtol` applies to the gradient with respect to `z`).
    pub bfgs: BfgsOptions,
}

impl Default for GpOptions {
    fn default() -> Self {
        GpOptions {
            noise_sd: None,
            log_length_bounds: (0.02f64.ln(), 5.0f64.ln()),
            log_signal_bounds: (0.1f64.ln(), 10.0f64.ln()),
            log_noise_bounds: (1e-3f64.ln(), 0.0),
            length_starts: vec![0.1, 0.3, 1.0],
            n_random_starts: 0,
            seed: 0,
            bfgs: BfgsOptions::default(),
        }
    }
}

/// A Gaussian process conditioned on data with given hyperparameters.
#[derive(Clone, Debug)]
pub struct GpModel {
    /// Training inputs, `n x d` row-major (units of `x`).
    pub x: Vec<f64>,
    /// Number of inputs.
    pub d: usize,
    /// Number of training points.
    pub n: usize,
    /// Mean used to standardise `y` (units of `y`).
    pub y_mean: f64,
    /// Scale used to standardise `y` (units of `y`).
    pub y_scale: f64,
    /// ARD length scales (units of `x`).
    pub length_scales: Vec<f64>,
    /// Signal standard deviation (units of `y`).
    pub signal_sd: f64,
    /// Observation-noise standard deviation (units of `y`), without the jitter.
    pub noise_sd: f64,
    /// Lower Cholesky factor of `K + (sn2 + jitter) I`, standardised units, `n x n` row-major.
    pub l: Vec<f64>,
    /// `K^-1 (y - y_mean)/y_scale` (dimensionless).
    pub alpha: Vec<f64>,
    /// Log marginal likelihood of `y` at these hyperparameters, in the units of `y` (includes the
    /// `-n ln(y_scale)` Jacobian of the standardisation and the constant `-n/2 ln(2 pi)`).
    pub log_marginal_likelihood: f64,
}

/// Result of [`gp_fit`].
#[derive(Clone, Debug)]
pub struct GpFit {
    /// The GP at the fitted hyperparameters.
    pub model: GpModel,
    /// Negative log marginal likelihood of the standardised data that was minimised (the value
    /// BFGS reached for the best start; `= -model.log_marginal_likelihood - n ln(y_scale)`).
    pub nll_std: f64,
    /// Whether the best start's BFGS run met its gradient tolerance.
    pub success: bool,
    /// Message of the best start's BFGS run.
    pub message: String,
    /// Index of the best start (order: `length_starts`, then the random starts).
    pub best_start: usize,
    /// Final negative log marginal likelihood of every start (`+inf` if a start never produced a
    /// finite value).
    pub start_nll: Vec<f64>,
    /// Total number of likelihood-and-gradient evaluations over all starts.
    pub nfev: usize,
    /// Total number of accepted BFGS steps over all starts.
    pub nit: usize,
}

/// Fit the hyperparameters of a Matern-5/2 ARD GP to `(x, y)` by maximum marginal likelihood.
///
/// `x`: `n x d` row-major (units of `x`, see the module docs on scaling), `y`: length `n` (units of
/// `y`). Panics if the shapes disagree, `n = 0`, any value is not finite, or an option is invalid
/// (an empty or inverted box, a non-positive start, no start at all, a negative fixed noise).
/// Returns `Err` if no start reached a finite likelihood, or if the final factorization fails.
/// `success = false` in the result means the best BFGS run did not meet its tolerance (iteration
/// limit, line-search failure); the model is still the best point found.
pub fn gp_fit(x: &[f64], d: usize, y: &[f64], options: &GpOptions) -> Result<GpFit, String> {
    let n = y.len();
    assert!(d >= 1 && n >= 1 && x.len() == n * d, "gp_fit: x must be n*d with n, d >= 1");
    assert!(x.iter().chain(y).all(|v| v.is_finite()), "gp_fit: x and y must be finite");
    let o = options;
    let (llo, lhi) = o.log_length_bounds;
    let (slo, shi) = o.log_signal_bounds;
    let (nlo, nhi) = o.log_noise_bounds;
    assert!(llo < lhi && slo < shi && (o.noise_sd.is_some() || nlo < nhi), "gp_fit: empty bound box");
    assert!(o.length_starts.iter().all(|&v| v > 0.0), "gp_fit: length starts must be positive");
    assert!(o.length_starts.len() + o.n_random_starts >= 1, "gp_fit: no starting point");
    if let Some(s) = o.noise_sd {
        assert!(s >= 0.0 && s.is_finite(), "gp_fit: fixed noise_sd must be >= 0");
    }
    let (ym, ys) = standardization(y);
    let yz: Vec<f64> = y.iter().map(|v| (v - ym) / ys).collect();
    let fixed_var = o.noise_sd.map(|s| (s / ys) * (s / ys));
    let fit_noise = fixed_var.is_none();
    let mut lo = vec![llo; d];
    let mut hi = vec![lhi; d];
    lo.push(slo);
    hi.push(shi);
    if fit_noise {
        lo.push(nlo);
        hi.push(nhi);
    }
    let p = lo.len();
    // starting points (theta), moved inside the box
    let mut starts: Vec<Vec<f64>> = Vec::new();
    for &l0 in &o.length_starts {
        let mut t = vec![l0.ln(); d];
        t.push(0.0);
        if fit_noise {
            t.push(0.1f64.ln());
        }
        starts.push(t);
    }
    let mut rng = SplitMix64::from_seed_stream(o.seed, 0);
    for _ in 0..o.n_random_starts {
        let mut t: Vec<f64> = (0..d).map(|_| llo + (lhi - llo) * rng.uniform()).collect();
        t.push(0.0);
        if fit_noise {
            t.push(0.1f64.ln());
        }
        starts.push(t);
    }
    let mut best: Option<(usize, Vec<f64>, f64, bool, String)> = None;
    let mut start_nll = Vec::new();
    let (mut nfev, mut nit) = (0usize, 0usize);
    let mut gth = vec![0.0; p];
    for (si, t0) in starts.iter().enumerate() {
        let t0: Vec<f64> = (0..p)
            .map(|k| {
                let m = START_MARGIN * (hi[k] - lo[k]);
                t0[k].max(lo[k] + m).min(hi[k] - m)
            })
            .collect();
        let z0 = to_z(&t0, &lo, &hi);
        let r = bfgs(
            |z, g| {
                let (th, dth) = to_theta(z, &lo, &hi);
                let f = nll_grad(x, d, &yz, &th, fixed_var, &mut gth);
                if f.is_finite() {
                    for k in 0..p {
                        g[k] = gth[k] * dth[k];
                    }
                }
                f
            },
            &z0,
            &o.bfgs,
        );
        nfev += r.nfev;
        nit += r.nit;
        start_nll.push(if r.f.is_finite() { r.f } else { f64::INFINITY });
        if r.f.is_finite() && best.as_ref().map_or(true, |b| r.f < b.2) {
            best = Some((si, r.x.clone(), r.f, r.success, r.message.clone()));
        }
    }
    let (best_start, z, nll_std, success, message) = best.ok_or("gp_fit: no start gave a finite likelihood")?;
    let (th, _) = to_theta(&z, &lo, &hi);
    let ell: Vec<f64> = th[..d].iter().map(|t| t.exp()).collect();
    let noise_sd = match o.noise_sd {
        Some(s) => s,
        None => th[d + 1].exp() * ys,
    };
    let model = gp_build(x, d, y, &ell, th[d].exp() * ys, noise_sd, None)?;
    Ok(GpFit { model, nll_std, success, message, best_start, start_nll, nfev, nit })
}

/// Condition a GP on `(x, y)` with given hyperparameters.
///
/// `ell`: length scales (units of `x`, length `d`, positive); `signal_sd`, `noise_sd`: standard
/// deviations in the units of `y` (`noise_sd >= 0`). `standardization_override`: `Some((mean, scale))` to
/// standardise `y` with a given mean and scale (units of `y`, scale `> 0`) instead of the mean and
/// population standard deviation of `y`; use it to keep the hyperparameters' meaning when data
/// points are added. Returns `Err` if the covariance matrix is not numerically positive definite.
pub fn gp_build(
    x: &[f64],
    d: usize,
    y: &[f64],
    ell: &[f64],
    signal_sd: f64,
    noise_sd: f64,
    standardization_override: Option<(f64, f64)>,
) -> Result<GpModel, String> {
    let n = y.len();
    assert!(d >= 1 && n >= 1 && x.len() == n * d && ell.len() == d, "gp_build: inconsistent shapes");
    assert!(ell.iter().all(|&v| v > 0.0) && signal_sd > 0.0 && noise_sd >= 0.0, "gp_build: invalid hyperparameters");
    let (ym, ys) = standardization_override.unwrap_or_else(|| standardization(y));
    assert!(ys > 0.0, "gp_build: scale must be positive");
    let sf2 = (signal_sd / ys) * (signal_sd / ys);
    let sn2 = (noise_sd / ys) * (noise_sd / ys);
    let mut k = matern52_matrix(x, x, d, ell, sf2);
    for i in 0..n {
        k[i * n + i] += sn2 + JITTER;
    }
    let ch = cholesky(&k, n);
    if !ch.success {
        return Err(format!("gp_build: {}", ch.message));
    }
    let l = ch.l;
    let mut alpha: Vec<f64> = y.iter().map(|v| (v - ym) / ys).collect();
    let mut lml = -0.5 * n as f64 * (2.0 * PI).ln() - n as f64 * ys.ln();
    solve_lower(&l, n, &mut alpha, 1);
    // after the forward solve alpha holds L^-1 yz: yz^T K^-1 yz = |L^-1 yz|^2
    let quad: f64 = alpha.iter().map(|v| v * v).sum();
    lml -= 0.5 * quad;
    for i in 0..n {
        lml -= l[i * n + i].ln();
    }
    solve_upper_t(&l, n, &mut alpha, 1);
    Ok(GpModel {
        x: x.to_vec(),
        d,
        n,
        y_mean: ym,
        y_scale: ys,
        length_scales: ell.to_vec(),
        signal_sd,
        noise_sd,
        l,
        alpha,
        log_marginal_likelihood: lml,
    })
}

impl GpModel {
    fn sf2_std(&self) -> f64 {
        (self.signal_sd / self.y_scale) * (self.signal_sd / self.y_scale)
    }

    /// Posterior mean and variance of the latent function `f` (noise-free) at the `m x d` points
    /// `g` (units of `x`): mean in units of `y`, variance in units of `y^2`, clipped at 0.
    /// Add `noise_sd^2` to get the variance of a new observation.
    pub fn predict(&self, g: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = g.len() / self.d;
        let sf2 = self.sf2_std();
        let mut v = matern52_matrix(&self.x, g, self.d, &self.length_scales, sf2); // n x m
        let mut mean = vec![self.y_mean; m];
        for k in 0..self.n {
            let a = self.y_scale * self.alpha[k];
            for j in 0..m {
                mean[j] += a * v[k * m + j];
            }
        }
        solve_lower(&self.l, self.n, &mut v, m);
        let mut var = vec![sf2; m];
        for k in 0..self.n {
            for j in 0..m {
                var[j] -= v[k * m + j] * v[k * m + j];
            }
        }
        let ys2 = self.y_scale * self.y_scale;
        for t in &mut var {
            *t = t.max(0.0) * ys2;
        }
        (mean, var)
    }

    /// Posterior covariance of the latent function between the `na x d` points `a` and the
    /// `nb x d` points `b` (units of `x`): row-major `na x nb`, units of `y^2`
    /// (`K(a,b) - K(a,X) (K + noise)^-1 K(X,b)`). With `a == b` it is symmetric with the posterior
    /// variance (unclipped) on the diagonal.
    pub fn posterior_cov(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let d = self.d;
        let (na, nb) = (a.len() / d, b.len() / d);
        let sf2 = self.sf2_std();
        let mut va = matern52_matrix(&self.x, a, d, &self.length_scales, sf2); // n x na
        solve_lower(&self.l, self.n, &mut va, na);
        let mut vb = matern52_matrix(&self.x, b, d, &self.length_scales, sf2); // n x nb
        solve_lower(&self.l, self.n, &mut vb, nb);
        let mut c = matern52_matrix(a, b, d, &self.length_scales, sf2);
        for k in 0..self.n {
            for i in 0..na {
                let t = va[k * na + i];
                for j in 0..nb {
                    c[i * nb + j] -= t * vb[k * nb + j];
                }
            }
        }
        let ys2 = self.y_scale * self.y_scale;
        for t in &mut c {
            *t *= ys2;
        }
        c
    }
}
