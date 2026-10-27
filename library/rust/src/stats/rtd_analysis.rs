//! Residence-time distribution (RTD) analysis from a sampled tracer response.
//!
//! Sources:
//! * O. Levenspiel, "Chemical Reaction Engineering", 3rd ed., Wiley 1999, ch. 11 (RTD, moments,
//!   tanks in series) and ch. 13 (dispersion model; closed-closed variance
//!   `sigma_theta^2 = 2/Pe - 2/Pe^2 (1 - exp(-Pe))`); also H. S. Fogler, "Elements of Chemical
//!   Reaction Engineering", ch. 18-19. Equation numbers were NOT checked against the books; the
//!   formulas used here are verified numerically in the tests.
//! * Ported from chetools/claude `porous-media-lbm/python/lbm.py` (`tail_fit`, `rtd_moments`,
//!   `fit_dispersion`, `E_dispersion`, `E_tanks`); the moment inversions and least-squares fits
//!   are new.
//!
//! Conventions: time `t` in s, tracer response `C(t)` in any consistent unit (only its shape
//! matters), `E(t) = C / integral(C dt)` in 1/s, `theta = t / tau` (dimensionless), `tau` in s,
//! `E_theta = tau E(t)`.
//!
//! * Moments: trapezoid rule on the (possibly non-uniform) samples; negative `C` clipped to 0.
//!   Optional exponential tail `C_T exp(-(t - T)/lam)` beyond the last sample, `lam` from a
//!   least-squares line through `ln C` over the last 25 % of the samples.
//! * Dispersion model, `Pe = u L / D_L` (dimensionless). [`BoundaryCondition::OpenOpen`]: the
//!   unbounded-medium Green's function seen as concentration at distance L,
//!   `E_theta = sqrt(Pe/(4 pi theta)) exp(-Pe (1-theta)^2/(4 theta))`, exactly normalised, mean
//!   `theta = 1 + 2/Pe`, variance `2/Pe + 8/Pe^2`. These are the open-open moments of Levenspiel
//!   ch. 13; here they are checked by quadrature of the formula (generator), not against the
//!   book's text. [`BoundaryCondition::ClosedClosed`]
//!   (Danckwerts conditions): mean `theta = 1`; only its moments are provided.
//! * Tanks in series (gamma): `E_theta = N^N theta^(N-1) exp(-N theta)/Gamma(N)`, mean 1,
//!   variance `1/N`.
//! * Least squares: minimise `sum (E_data - E_model)^2` over the samples with a nested
//!   golden-section search in `ln(parameter)` (no derivatives, no external crates).
//!
//! Python twin: `library/python/agent_tools/stats/rtd_analysis.py`.

/// Fraction of samples used for the tail fit.
pub const TAIL_FRAC: f64 = 0.25;
/// Minimum number of samples used for the tail fit.
pub const TAIL_MIN_POINTS: usize = 5;
/// `theta` is clipped below to this value in `E_theta`.
pub const THETA_FLOOR: f64 = 1e-12;
/// Golden-section tolerance in `ln(parameter)`.
pub const GOLDEN_TOL: f64 = 1e-9;
/// Golden-section iteration limit.
pub const GOLDEN_MAXIT: usize = 200;
/// Search range of `N` or `Pe` in [`fit_lsq`] (dimensionless).
pub const LSQ_SHAPE_RANGE: (f64, f64) = (0.05, 1e4);
/// Search range of `tau` in [`fit_lsq`], as factors times the data mean.
pub const LSQ_TAU_RANGE: (f64, f64) = (0.2, 5.0);
/// Bisection steps in the closed-closed moment inversion.
pub const BISECT_ITER: usize = 200;

/// Boundary conditions of the axial-dispersion model (see the module header).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// Unbounded-medium Green's function seen as concentration at distance L.
    OpenOpen,
    /// Danckwerts conditions at both ends; mean `theta` is exactly 1.
    ClosedClosed,
}

/// Which residence-time model to fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RtdModel {
    /// Tanks in series, parameter `N` (dimensionless).
    Tanks,
    /// Open-open axial dispersion, parameter `Pe` (dimensionless).
    Dispersion,
}

fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..x.len() {
        s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    s
}

/// Decay time `lam` (s) of `C ~ exp(-t/lam)` fitted to the last `frac` (default [`TAIL_FRAC`] = 0.25)
/// of the samples.
///
/// Least-squares line through `ln C` over the last `max(5, int(frac n))` samples. Returns `None`
/// if any of them is `<= 0` (or NaN), if fewer samples exist, or if the slope is not negative.
/// Panics if `t.len() != c.len()`.
pub fn tail_fit(t: &[f64], c: &[f64], frac: f64) -> Option<f64> {
    assert_eq!(t.len(), c.len(), "tail_fit: t and c must have equal length");
    let n = TAIL_MIN_POINTS.max((frac * t.len() as f64) as usize);
    if n > t.len() {
        return None;
    }
    let (tt, cc) = (&t[t.len() - n..], &c[t.len() - n..]);
    if cc.iter().any(|&v| !(v > 0.0)) {
        return None;
    }
    let nf = n as f64;
    let mt = tt.iter().sum::<f64>() / nf;
    let ml = cc.iter().map(|v| v.ln()).sum::<f64>() / nf;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for i in 0..n {
        let dx = tt[i] - mt;
        sxy += dx * (cc[i].ln() - ml);
        sxx += dx * dx;
    }
    let slope = sxy / sxx;
    if slope < 0.0 {
        Some(-1.0 / slope)
    } else {
        None
    }
}

/// Result of [`rtd_moments`].
#[derive(Clone, Debug)]
pub struct RtdMoments {
    /// Normalised `E(t)` at the samples, 1/s (C divided by the zeroth moment, tail included).
    pub e: Vec<f64>,
    /// Mean residence time, s (pulse-width corrected).
    pub mean: f64,
    /// Variance of the residence time, s^2 (pulse-width corrected).
    pub variance: f64,
    /// Decay time `lam` (s) used for the tail extrapolation, `None` if none was applied.
    pub tail_decay_time: Option<f64>,
}

/// Normalised `E(t)`, mean and variance of a tracer response `c(t)` sampled at times `t`.
///
/// `t`: increasing sample times, s (at least 2). `c`: response, any unit; negative values are
/// clipped to 0. `pulse`: width of a square input pulse, s (default 0 = delta input); it adds
/// `pulse/2` to the mean and `pulse^2/12` to the variance, both removed here. `tail`: if true
/// (default) and [`tail_fit`] finds a decaying tail, the signal beyond the last sample is
/// extended as `C_T exp(-(t - T)/lam)` (analytic moments). The input pulse is assumed to start
/// at `t = 0` and the samples to cover the whole peak. Zero total area gives NaN. Panics if
/// the lengths differ.
pub fn rtd_moments(t: &[f64], c: &[f64], pulse: f64, tail: bool) -> RtdMoments {
    assert_eq!(t.len(), c.len(), "rtd_moments: t and c must have equal length");
    let c: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();
    let tc: Vec<f64> = t.iter().zip(&c).map(|(a, b)| a * b).collect();
    let ttc: Vec<f64> = t.iter().zip(&tc).map(|(a, b)| a * b).collect();
    let (mut m0, mut m1, mut m2) = (trapz(&c, t), trapz(&tc, t), trapz(&ttc, t));
    let lam = if tail { tail_fit(t, &c, TAIL_FRAC) } else { None };
    if let Some(lam) = lam {
        let tt = t[t.len() - 1];
        let ct = c[c.len() - 1];
        m0 += ct * lam;
        m1 += ct * lam * (tt + lam);
        m2 += ct * lam * (tt * tt + 2.0 * tt * lam + 2.0 * lam * lam);
    }
    let mean = m1 / m0;
    let var = m2 / m0 - mean * mean;
    RtdMoments {
        e: c.iter().map(|v| v / m0).collect(),
        mean: mean - 0.5 * pulse,
        variance: var - pulse * pulse / 12.0,
        tail_decay_time: lam,
    }
}

/// Result of [`fit_dispersion_planes`].
#[derive(Clone, Debug)]
pub struct PlaneFit {
    /// Mean velocity `dx/dt`, m/s.
    pub u: f64,
    /// Peclet number `u dx / D_L` (dimensionless, based on the plane spacing).
    pub pe: f64,
    /// Axial dispersion coefficient, m^2/s.
    pub d_l: f64,
    /// Equivalent number of tanks in series between the planes.
    pub n: f64,
    /// Mean residence time between the planes, s.
    pub tau: f64,
    /// Variance difference between the planes, s^2.
    pub var: f64,
}

/// Two-detector (moment-difference) analysis of the axial dispersion model.
///
/// Between two planes a distance `dx` (m) apart, `d(var)/d(mean)^2 = 2/Pe` exactly, independent
/// of the boundary conditions at the ends. Inputs: mean residence times `mean1`, `mean2` (s) and
/// variances `var1`, `var2` (s^2) at the upstream and downstream plane (from [`rtd_moments`] with
/// the same pulse). Needs `mean2 > mean1` and `var2 > var1`; otherwise `u`, `pe` or `d_l` are
/// negative or infinite (no check is made).
pub fn fit_dispersion_planes(mean1: f64, var1: f64, mean2: f64, var2: f64, dx: f64) -> PlaneFit {
    let (dt, dv) = (mean2 - mean1, var2 - var1);
    let u = dx / dt;
    let pe = 2.0 * dt * dt / dv;
    PlaneFit { u, pe, d_l: u * dx / pe, n: dt * dt / dv, tau: dt, var: dv }
}

/// Open-open axial-dispersion response to a delta input, `E_theta` (dimensionless).
///
/// `theta = t u / L`, `pe = u L / D_L > 0`. Unbounded-medium solution seen as concentration at
/// distance L (module header). `theta` is clipped below at 1e-12. As a vessel model it is
/// accurate for `Pe` of about 5 and above.
pub fn e_dispersion(theta: f64, pe: f64) -> f64 {
    let th = theta.max(THETA_FLOOR);
    (pe / (4.0 * std::f64::consts::PI * th)).sqrt() * (-pe * (1.0 - th) * (1.0 - th) / (4.0 * th)).exp()
}

const LANCZOS: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

/// `ln Gamma(x)` for `x > 0` (Lanczos approximation, g = 7, n = 9; relative accuracy about 1e-15).
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    let t = x + 7.5;
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Tanks-in-series response (gamma distribution), `E_theta` (dimensionless).
///
/// `theta = t / tau` with `tau` the total mean residence time; `n > 0` tanks (non-integer
/// allowed). `theta` is clipped below at 1e-12.
pub fn e_tanks(theta: f64, n: f64) -> f64 {
    let th = theta.max(THETA_FLOOR);
    (n * n.ln() + (n - 1.0) * th.ln() - n * th - ln_gamma(n)).exp()
}

/// Dimensionless variance of `E_theta` for the dispersion model, `pe = u L / D_L > 0`, relative
/// to `tau^2` with `tau = L/u`.
///
/// Open-open: `2/Pe + 8/Pe^2` (about the mean `1 + 2/Pe`). Closed-closed:
/// `2/Pe - 2/Pe^2 (1 - exp(-Pe))` (Levenspiel ch. 13).
pub fn sigma2_dispersion(pe: f64, bc: BoundaryCondition) -> f64 {
    match bc {
        BoundaryCondition::OpenOpen => 2.0 / pe + 8.0 / (pe * pe),
        BoundaryCondition::ClosedClosed => 2.0 / pe - 2.0 / (pe * pe) * (1.0 - (-pe).exp()),
    }
}

/// Result of [`fit_moments`].
#[derive(Clone, Debug)]
pub struct MomentFit {
    /// `N` (tanks) or `Pe` (dispersion), dimensionless; NaN if not `success`.
    pub param: f64,
    /// Model time scale, s (tanks: the mean; open-open: `mean/(1+2/Pe) = L/u`; closed-closed: the mean).
    pub tau: f64,
    /// False if the variance is not positive or outside the model's range.
    pub success: bool,
    /// `"success"` or the reason.
    pub message: String,
}

/// Fit `N` or `Pe` from the mean (s) and variance (s^2) of a measured RTD.
///
/// Tanks: `N = mean^2/variance`. Dispersion: `r = variance/mean^2` is matched to the model;
/// open-open has the closed form `Pe = (1 - 2 r + sqrt(1 + 4 r))/r`; closed-closed has
/// `mean = tau` and `r = 2/Pe - 2/Pe^2 (1 - exp(-Pe))`, solved by bisection in `ln Pe` on
/// `[1e-6, 1e8]` (`r` must lie in (0, 1)). `bc` is ignored for tanks. Fails (`success` false)
/// for non-positive or NaN mean or variance, or `r` outside the model's range. The variance of a
/// record cut before the tail is biased low: use [`rtd_moments`] with `tail = true`.
pub fn fit_moments(mean: f64, variance: f64, model: RtdModel, bc: BoundaryCondition) -> MomentFit {
    let fail = |m: &str| MomentFit { param: f64::NAN, tau: f64::NAN, success: false, message: m.to_string() };
    if !(mean > 0.0 && variance > 0.0) {
        return fail("mean and variance must be positive");
    }
    let r = variance / (mean * mean);
    let ok = |param: f64, tau: f64| MomentFit { param, tau, success: true, message: "success".to_string() };
    match (model, bc) {
        (RtdModel::Tanks, _) => ok(1.0 / r, mean),
        (RtdModel::Dispersion, BoundaryCondition::OpenOpen) => {
            let pe = (1.0 - 2.0 * r + (1.0 + 4.0 * r).sqrt()) / r;
            ok(pe, mean / (1.0 + 2.0 / pe))
        }
        (RtdModel::Dispersion, BoundaryCondition::ClosedClosed) => {
            if !(r < 1.0) {
                return fail("closed-closed variance/mean^2 must be < 1");
            }
            let (mut lo, mut hi) = (1e-6f64.ln(), 1e8f64.ln());
            for _ in 0..BISECT_ITER {
                let mid = 0.5 * (lo + hi);
                if sigma2_dispersion(mid.exp(), BoundaryCondition::ClosedClosed) > r {
                    lo = mid; // variance decreases with Pe: need larger Pe
                } else {
                    hi = mid;
                }
            }
            ok((0.5 * (lo + hi)).exp(), mean)
        }
    }
}

/// Minimise a unimodal `f` on `[a, b]` by golden-section search.
///
/// Stops when the bracket is shorter than `tol` (absolute, in the units of x) or after `maxit`
/// iterations. Returns `(x, f(x), nfev)`. Finds a local minimum only if `f` is unimodal on the
/// bracket.
pub fn golden_section<F: FnMut(f64) -> f64>(mut f: F, mut a: f64, mut b: f64, tol: f64, maxit: usize) -> (f64, f64, usize) {
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    let mut nfev = 2;
    let mut it = 0;
    while b - a > tol && it < maxit {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
        nfev += 1;
        it += 1;
    }
    let x = 0.5 * (a + b);
    let fx = f(x);
    (x, fx, nfev + 1)
}

/// Result of [`fit_lsq`].
#[derive(Clone, Debug)]
pub struct LsqFit {
    /// `N` or `Pe` (dimensionless).
    pub param: f64,
    /// Fitted (or given) model time scale, s (for dispersion `L/u`, not the mean).
    pub tau: f64,
    /// Sum of squared residuals of `E`, 1/s^2.
    pub sse: f64,
    /// Number of model evaluations (objective calls).
    pub nfev: usize,
    /// False if a parameter ended on the edge of its search range or the data are unusable.
    pub success: bool,
    /// `"success"` or the reason.
    pub message: String,
}

/// Least-squares fit of `E(t) = E_theta(t/tau; p)/tau` for the tanks or open-open dispersion model.
///
/// `t`: sample times, s. `e`: measured `E(t)`, 1/s (normalised, e.g. [`rtd_moments`]`.e`).
/// `tau`: if `Some` (s) only `p` is fitted (1-D search); if `None`, `tau` and `p` are both
/// fitted by a nested golden-section search in `ln p` over `[0.05, 1e4]` and `ln tau` over
/// `[0.2, 5]` times the data mean (trapezoid first moment of `e`). Unweighted residuals at all
/// samples; a finite input pulse width is ignored. The nested search assumes the objective is
/// unimodal in each variable (local-minimum caveat). Panics if the lengths differ.
pub fn fit_lsq(t: &[f64], e: &[f64], model: RtdModel, tau: Option<f64>) -> LsqFit {
    assert_eq!(t.len(), e.len(), "fit_lsq: t and e must have equal length");
    let shape = |th: f64, p: f64| match model {
        RtdModel::Tanks => e_tanks(th, p),
        RtdModel::Dispersion => e_dispersion(th, p),
    };
    let m0 = trapz(e, t);
    if !(m0 > 0.0) {
        return LsqFit { param: f64::NAN, tau: f64::NAN, sse: f64::NAN, nfev: 0, success: false, message: "E has no positive area".to_string() };
    }
    let te: Vec<f64> = t.iter().zip(e).map(|(a, b)| a * b).collect();
    let mean = trapz(&te, t) / m0;
    let count = std::cell::Cell::new(0usize);
    let sse = |tau_: f64, p: f64| -> f64 {
        count.set(count.get() + 1);
        t.iter().zip(e).map(|(&ti, &ei)| { let r = ei - shape(ti / tau_, p) / tau_; r * r }).sum()
    };
    let lp = (LSQ_SHAPE_RANGE.0.ln(), LSQ_SHAPE_RANGE.1.ln());
    let (u, fx, tau_fit, lt_edge);
    match tau {
        Some(tv) => {
            let r = golden_section(|x| sse(tv, x.exp()), lp.0, lp.1, GOLDEN_TOL, GOLDEN_MAXIT);
            u = r.0;
            fx = r.1;
            tau_fit = tv;
            lt_edge = false;
        }
        None => {
            let lt = ((LSQ_TAU_RANGE.0 * mean).ln(), (LSQ_TAU_RANGE.1 * mean).ln());
            let inner = |x: f64| golden_section(|y| sse(y.exp(), x.exp()), lt.0, lt.1, GOLDEN_TOL, GOLDEN_MAXIT);
            let (uu, _, _) = golden_section(|x| inner(x).1, lp.0, lp.1, GOLDEN_TOL, GOLDEN_MAXIT);
            let (y, f2, _) = inner(uu);
            u = uu;
            fx = f2;
            tau_fit = y.exp();
            lt_edge = (y - lt.0).min(lt.1 - y) < 1e-6;
        }
    }
    let edge = lt_edge || (u - lp.0).min(lp.1 - u) < 1e-6;
    LsqFit {
        param: u.exp(),
        tau: tau_fit,
        sse: fx,
        nfev: count.get(),
        success: !edge,
        message: if edge { "parameter at the edge of its search range".to_string() } else { "success".to_string() },
    }
}
