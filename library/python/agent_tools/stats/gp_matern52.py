"""Gaussian-process regression with an ARD Matern-5/2 kernel and type-II maximum-likelihood
hyperparameters (Python reference of the Rust `stats::gp_matern52`).

Source: Rasmussen & Williams, "Gaussian Processes for Machine Learning" (MIT Press, 2006):
eq. (4.17) (Matern class, nu = 5/2), Algorithm 2.1 (Cholesky-based predictions, eqs. 2.25-2.26),
eq. (5.8) (log marginal likelihood) and eq. (5.9) (its gradient, 0.5 tr((alpha alpha^T - K^-1) dK/dtheta)).
Ported from chetools/claude `sequential-design/python/sdl.py` (same authors) using the library's
`linalg.cholesky` and `optimize.bfgs`. Rust twin: library/rust/src/stats/gp_matern52.rs.
Reference for tests: an independent NumPy/SciPy implementation of the same marginal likelihood
and scipy.optimize.minimize(method="L-BFGS-B").

Model: y = f(x) + e, f ~ GP(ym, ys^2 k), e ~ N(0, noise_sd^2), with the kernel
k(a, b) = signal_sd^2 (1 + s + s^2/3) exp(-s), s = sqrt(5) sqrt(sum_i ((a_i - b_i)/ell_i)^2).
y is standardised before fitting, yz = (y - ym)/ys (ym the mean, ys the population standard
deviation, or 1 if y is constant); the GP is fitted to yz and every output is converted back to the
units of y. A jitter JITTER = 1e-8 (variance of the standardised y) is added to the diagonal in every
factorization, so the effective noise variance is noise_sd^2 + 1e-8 ys^2; it is part of the model.

Inputs and their scaling: X is (n, d), in any units; the length scales are in the units of X. The
default bounds on the length scales (`GpOptions.log_length_bounds`) and the default starting length
scales assume that inputs are scaled to a unit domain (about [0, 1]): 0.02 .. 5 times the domain
width. If X is in other units, scale it to [0, 1] first, or pass bounds and starts in the units of X.

Hyperparameter fitting: theta = [ln ell_1 .. ln ell_d, ln sf, ln sn] with sf = signal_sd/ys and
sn = noise_sd/ys (standardised units; sn is absent when the noise is fixed). `bfgs` is
unconstrained, so each parameter is boxed with the source's transform
theta_k = lo_k + (hi_k - lo_k)(1 + tanh(z_k/2))/2 and BFGS runs on the unbounded z, with the chain
rule d theta/d z = (theta - lo)(hi - theta)/(hi - lo). Consequence: when the optimum lies on a bound
the likelihood is flat in z, the optimizer stops when the z-gradient is below gtol, and the parameter
sits close to (not exactly at) the bound. Each start is a BFGS run; the start with the lowest finite
negative log marginal likelihood is kept. Starts whose covariance matrix fails the Cholesky
factorization return +inf and are rejected by the line search. Starts: one per entry of
`length_starts` (isotropic, sf = 1, sn = 0.1), then `n_random_starts` with log length scales uniform
in the box, drawn from SplitMix64.from_seed_stream(seed, 0). Start values are moved inside the box by
at least 1 % of its width.

Cost: O(n^3) per likelihood evaluation, O(n^2 d) memory; intended for n up to a few hundred.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from agent_tools.linalg.cholesky import cholesky, solve_lower, solve_upper_t
from agent_tools.optimize.bfgs import BfgsOptions, bfgs
from agent_tools.random.splitmix64 import SplitMix64

SQRT5 = math.sqrt(5.0)
JITTER = 1e-8        # added to the covariance diagonal, variance units of the standardised y
START_MARGIN = 0.01  # start values stay this fraction of the box width away from its edges


def matern52(A, B, ell, sf2):
    """Matern-5/2 covariance between the point sets A (na, d) and B (nb, d) -> (na, nb), units y^2.

    ell: ARD length scales (units of x, positive, length d); sf2: signal variance (units of y^2).
    """
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    D = (A[:, None, :] - B[None, :, :]) / np.asarray(ell, float)
    s = SQRT5 * np.sqrt(np.sum(D ** 2, axis=-1))
    return sf2 * (1 + s + s ** 2 / 3) * np.exp(-s)


def _standardization(y):
    m = float(np.mean(y))
    s = float(np.std(y))
    return m, (s if s != 0.0 else 1.0)


def to_theta(z, lo, hi):
    """Map the unbounded z into the box lo < theta < hi (theta = lo + (hi-lo)(1+tanh(z/2))/2).

    Returns theta and d theta/d z (both in the units of theta, here logarithms)."""
    z, lo, hi = (np.asarray(v, float) for v in (z, lo, hi))
    theta = lo + (hi - lo) * 0.5 * (1.0 + np.tanh(0.5 * z))
    return theta, (theta - lo) * (hi - theta) / (hi - lo)


def to_z(theta, lo, hi):
    """Inverse of `to_theta`: z = ln(p/(1-p)), p = (theta-lo)/(hi-lo). theta must lie strictly
    inside the box (z is infinite on the edge)."""
    theta, lo, hi = (np.asarray(v, float) for v in (theta, lo, hi))
    p = (theta - lo) / (hi - lo)
    return np.log(p / (1.0 - p))


def nll_grad(X, y_std, theta, fixed_noise_var=None):
    """Negative log marginal likelihood of the standardised data y_std and its gradient wrt theta.

    X: (n, d) inputs (units of x); y_std: (n,), dimensionless. theta = [ln ell_1 .. ln ell_d,
    ln sf, ln sn] of length d + 2 (ln ell in ln(units of x); sf, sn relative to the standard
    deviation of y); with fixed_noise_var given (standardised units) the last entry is absent
    (length d + 1) and the noise variance is that value. Returns (nll, grad); (inf, None) if the
    covariance matrix is not numerically positive definite. Includes n/2 ln(2 pi); JITTER is added.
    """
    X = np.asarray(X, float)
    y_std = np.asarray(y_std, float)
    theta = np.asarray(theta, float)
    n, d = X.shape
    fit_noise = fixed_noise_var is None
    if len(theta) != d + 1 + int(fit_noise):
        raise ValueError("theta must have d+2 entries (d+1 with fixed noise)")
    ell = np.exp(theta[:d])
    sf2 = math.exp(2 * theta[d])
    sn2 = math.exp(2 * theta[d + 1]) if fit_noise else float(fixed_noise_var)
    D2 = ((X[:, None, :] - X[None, :, :]) / ell) ** 2          # (n, n, d)
    s = SQRT5 * np.sqrt(D2.sum(-1))
    e = np.exp(-s)
    Kf = sf2 * (1 + s + s ** 2 / 3) * e
    common = sf2 * (5 / 3) * (1 + s) * e                       # dK/d ln ell_i = common * D2_i
    K = Kf + (sn2 + JITTER) * np.eye(n)
    ch = cholesky(K)
    if not ch.success:
        return math.inf, None
    L = ch.l
    alpha = solve_upper_t(L, solve_lower(L, y_std))
    nll = 0.5 * n * math.log(2 * math.pi)
    nll += float(np.sum(0.5 * y_std * alpha + np.log(np.diag(L))))
    # W = K^-1 - alpha alpha^T; d nll / d theta_k = 0.5 sum(W * dK/d theta_k)
    W = solve_upper_t(L, solve_lower(L, np.eye(n))) - np.outer(alpha, alpha)
    g = np.empty(d + 1 + int(fit_noise))
    for i in range(d):
        g[i] = np.sum(W * common * D2[:, :, i])
    g[d] = np.sum(W * 2 * Kf)
    if fit_noise:
        g[d + 1] = np.trace(W) * 2 * sn2
    return nll, 0.5 * g


@dataclass
class GpOptions:
    """Options of `gp_fit`.

    noise_sd: None (default) fits the noise standard deviation; a number >= 0 fixes it (units of y).
    log_length_bounds: (lower, upper) box for ln ell_i, in ln(units of x), the same for every input.
        Default (ln 0.02, ln 5), meant for inputs scaled to [0, 1] (see the module docstring).
    log_signal_bounds: box for ln(signal_sd/ys), ys the standard deviation of y. Default (ln 0.1, ln 10).
    log_noise_bounds: box for ln(noise_sd/ys), used only when the noise is fitted. Default (ln 1e-3, 0).
    length_starts: isotropic starting length scales, one BFGS run each (units of x, positive).
        Default (0.1, 0.3, 1.0).
    n_random_starts: number of extra random starts (log length scales uniform in the box). Default 0.
    seed: seed of the random starts (SplitMix64.from_seed_stream(seed, 0)). Default 0.
    bfgs: BFGS options on the unbounded variables z (default max_iter = 200, gtol = 1e-5, the
        library defaults; gtol applies to the gradient with respect to z).
    """
    noise_sd: float = None
    log_length_bounds: tuple = (math.log(0.02), math.log(5.0))
    log_signal_bounds: tuple = (math.log(0.1), math.log(10.0))
    log_noise_bounds: tuple = (math.log(1e-3), 0.0)
    length_starts: tuple = (0.1, 0.3, 1.0)
    n_random_starts: int = 0
    seed: int = 0
    bfgs: BfgsOptions = field(default_factory=BfgsOptions)


@dataclass
class GpModel:
    """A Gaussian process conditioned on data with given hyperparameters (see `gp_build`).

    x (n, d): training inputs (units of x). y_mean, y_scale: standardisation of y (units of y).
    length_scales (d,): units of x. signal_sd, noise_sd: units of y (noise without the jitter).
    l (n, n): lower Cholesky factor of K + (sn2 + jitter) I, standardised units.
    alpha (n,): K^-1 (y - y_mean)/y_scale, dimensionless.
    log_marginal_likelihood: of y at these hyperparameters, in the units of y (includes the
        -n ln(y_scale) Jacobian of the standardisation and the constant -n/2 ln(2 pi)).
    """
    x: np.ndarray
    y_mean: float
    y_scale: float
    length_scales: np.ndarray
    signal_sd: float
    noise_sd: float
    l: np.ndarray
    alpha: np.ndarray
    log_marginal_likelihood: float

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def d(self):
        return self.x.shape[1]

    def _sf2_std(self):
        return (self.signal_sd / self.y_scale) ** 2

    def predict(self, G):
        """Posterior mean and variance of the latent function f (noise-free) at the (m, d) points G
        (units of x): mean in units of y, variance in units of y^2, clipped at 0. Add noise_sd^2
        to get the variance of a new observation."""
        G = np.atleast_2d(G)
        sf2 = self._sf2_std()
        Ks = matern52(self.x, G, self.length_scales, sf2)          # (n, m)
        mean = self.y_mean + self.y_scale * (Ks.T @ self.alpha)
        V = solve_lower(self.l, Ks)
        var = np.maximum(sf2 - np.sum(V ** 2, axis=0), 0.0)
        return mean, var * self.y_scale ** 2

    def posterior_cov(self, A, B):
        """Posterior covariance of the latent function between the (na, d) points A and the (nb, d)
        points B (units of x): array (na, nb), units of y^2
        (K(A,B) - K(A,X) (K + noise)^-1 K(X,B)). With A == B it is symmetric with the posterior
        variance (unclipped) on the diagonal."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        sf2 = self._sf2_std()
        Va = solve_lower(self.l, matern52(self.x, A, self.length_scales, sf2))
        Vb = solve_lower(self.l, matern52(self.x, B, self.length_scales, sf2))
        C = matern52(A, B, self.length_scales, sf2) - Va.T @ Vb
        return C * self.y_scale ** 2


@dataclass
class GpFit:
    """Result of `gp_fit`.

    model: the `GpModel` at the fitted hyperparameters.
    nll_std: negative log marginal likelihood of the standardised data that was minimised (the value
        BFGS reached for the best start; = -model.log_marginal_likelihood - n ln(y_scale)).
    success: whether the best start's BFGS run met its gradient tolerance.
    message: message of the best start's BFGS run.
    best_start: index of the best start (order: length_starts, then the random starts).
    start_nll: final negative log marginal likelihood of every start (inf if never finite).
    nfev: total number of likelihood-and-gradient evaluations over all starts.
    nit: total number of accepted BFGS steps over all starts.
    """
    model: GpModel
    nll_std: float
    success: bool
    message: str
    best_start: int
    start_nll: list
    nfev: int
    nit: int


def gp_fit(X, y, options=None):
    """Fit the hyperparameters of a Matern-5/2 ARD GP to (X, y) by maximum marginal likelihood.

    X: (n, d) (units of x, see the module docstring on scaling), y: (n,) (units of y). Raises
    ValueError if the shapes disagree, n = 0, any value is not finite, or an option is invalid (an
    empty or inverted box, a non-positive start, no start at all, a negative fixed noise); raises
    RuntimeError if no start reached a finite likelihood or the final factorization fails.
    success False in the result means the best BFGS run did not meet its tolerance (iteration
    limit, line-search failure); the model is still the best point found.
    """
    o = options if options is not None else GpOptions()
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or y.shape[0] < 1:
        raise ValueError("gp_fit: X must be (n, d) and y (n,) with n >= 1")
    n, d = X.shape
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("gp_fit: X and y must be finite")
    llo, lhi = o.log_length_bounds
    slo, shi = o.log_signal_bounds
    nlo, nhi = o.log_noise_bounds
    if not (llo < lhi and slo < shi and (o.noise_sd is not None or nlo < nhi)):
        raise ValueError("gp_fit: empty bound box")
    if not all(v > 0 for v in o.length_starts):
        raise ValueError("gp_fit: length starts must be positive")
    if len(o.length_starts) + o.n_random_starts < 1:
        raise ValueError("gp_fit: no starting point")
    if o.noise_sd is not None and not (o.noise_sd >= 0 and math.isfinite(o.noise_sd)):
        raise ValueError("gp_fit: fixed noise_sd must be >= 0")
    ym, ys = _standardization(y)
    yz = (y - ym) / ys
    fixed_var = None if o.noise_sd is None else (o.noise_sd / ys) * (o.noise_sd / ys)
    fit_noise = fixed_var is None
    lo = np.array([llo] * d + [slo] + ([nlo] if fit_noise else []))
    hi = np.array([lhi] * d + [shi] + ([nhi] if fit_noise else []))
    p = len(lo)
    starts = []
    for l0 in o.length_starts:
        starts.append([math.log(l0)] * d + [0.0] + ([math.log(0.1)] if fit_noise else []))
    rng = SplitMix64.from_seed_stream(o.seed, 0)
    for _ in range(o.n_random_starts):
        t = [llo + (lhi - llo) * rng.uniform() for _ in range(d)]
        starts.append(t + [0.0] + ([math.log(0.1)] if fit_noise else []))

    def fg(z, g):
        th, dth = to_theta(z, lo, hi)
        f, gth = nll_grad(X, yz, th, fixed_var)
        if math.isfinite(f):
            g[:] = gth * dth
        return f

    best = None
    start_nll = []
    nfev = nit = 0
    for si, t0 in enumerate(starts):
        m = START_MARGIN * (hi - lo)
        t0 = np.minimum(np.maximum(np.array(t0), lo + m), hi - m)
        r = bfgs(fg, to_z(t0, lo, hi), o.bfgs)
        nfev += r.nfev
        nit += r.nit
        start_nll.append(r.f if math.isfinite(r.f) else math.inf)
        if math.isfinite(r.f) and (best is None or r.f < best[2]):
            best = (si, r.x.copy(), r.f, r.success, r.message)
    if best is None:
        raise RuntimeError("gp_fit: no start gave a finite likelihood")
    best_start, z, nll_std, success, message = best
    th, _ = to_theta(z, lo, hi)
    noise_sd = o.noise_sd if o.noise_sd is not None else math.exp(th[d + 1]) * ys
    model = gp_build(X, y, np.exp(th[:d]), math.exp(th[d]) * ys, noise_sd)
    return GpFit(model, nll_std, success, message, best_start, start_nll, nfev, nit)


def gp_build(X, y, ell, signal_sd, noise_sd, standardization=None):
    """Condition a GP on (X, y) with given hyperparameters.

    ell: length scales (units of x, length d, positive); signal_sd, noise_sd: standard deviations in
    the units of y (noise_sd >= 0). standardization: (mean, scale) to standardise y with a given mean
    and scale (units of y, scale > 0) instead of the mean and population standard deviation of y;
    use it to keep the hyperparameters' meaning when data points are added. Raises RuntimeError if
    the covariance matrix is not numerically positive definite (the Rust twin returns Err).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ell = np.asarray(ell, float)
    n, d = X.shape
    if y.shape != (n,) or ell.shape != (d,) or n < 1:
        raise ValueError("gp_build: inconsistent shapes")
    if not (np.all(ell > 0) and signal_sd > 0 and noise_sd >= 0):
        raise ValueError("gp_build: invalid hyperparameters")
    ym, ys = standardization if standardization is not None else _standardization(y)
    if not ys > 0:
        raise ValueError("gp_build: scale must be positive")
    sf2 = (signal_sd / ys) ** 2
    sn2 = (noise_sd / ys) ** 2
    K = matern52(X, X, ell, sf2) + (sn2 + JITTER) * np.eye(n)
    ch = cholesky(K)
    if not ch.success:
        raise RuntimeError("gp_build: " + ch.message)
    L = ch.l
    w = solve_lower(L, (y - ym) / ys)                          # L^-1 yz
    lml = -0.5 * n * math.log(2 * math.pi) - n * math.log(ys) - 0.5 * float(np.sum(w * w))
    lml -= float(np.sum(np.log(np.diag(L))))
    alpha = solve_upper_t(L, w)
    return GpModel(X, float(ym), float(ys), ell, float(signal_sd), float(noise_sd), L, alpha, lml)
