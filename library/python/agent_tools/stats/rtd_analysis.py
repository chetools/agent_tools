"""Residence-time distribution (RTD) analysis from a sampled tracer response (Python reference
of the Rust `rtd_analysis`).

Sources:
  * O. Levenspiel, "Chemical Reaction Engineering", 3rd ed., Wiley 1999, ch. 11 (RTD, moments,
    tanks in series) and ch. 13 (dispersion model; closed-closed variance
    sigma_theta^2 = 2/Pe - 2/Pe^2 (1 - exp(-Pe))); also H. S. Fogler, "Elements of Chemical
    Reaction Engineering", ch. 18 (RTD) and ch. 18-19 (dispersion, tanks in series). Equation
    numbers were NOT checked against the books; the formulas used here are verified numerically
    in the tests (see library/tests/python/test_rtd_analysis.py).
  * Ported from chetools/claude `porous-media-lbm/python/lbm.py` (`tail_fit`, `rtd_moments`,
    `fit_dispersion`, `E_dispersion`, `E_tanks`); the moment inversions and the least-squares
    fits are new.
Rust twin: library/rust/src/stats/rtd_analysis.rs.

Conventions: time t in s, tracer response C(t) in any consistent unit (only its shape matters),
E(t) = C / integral(C dt) in 1/s, theta = t / tau (dimensionless), tau in s, E_theta = tau E(t).

Algorithms:
  * Moments by the trapezoid rule on the (possibly non-uniform) samples; negative C is clipped
    to 0. Optional exponential tail beyond the last sample, C_T exp(-(t - T)/lam), with the decay
    time lam from a least-squares line through ln C over the last 25 % of the samples.
  * Axial-dispersion models, parameter Pe = u L / D_L (dimensionless):
      "open-open": the unbounded-medium Green's function seen as concentration at distance L,
        E_theta = sqrt(Pe/(4 pi theta)) exp(-Pe (1 - theta)^2 / (4 theta)); exactly normalised,
        mean theta = 1 + 2/Pe, variance 2/Pe + 8/Pe^2. These moments are the open-open values
        of Levenspiel ch. 13; here they are checked by quadrature of the formula (generator
        and tests), not against the book's text.
      "closed-closed": Danckwerts boundary conditions, mean theta = 1, variance as above. Only
        its moments are provided (no time-domain curve).
  * Tanks in series (gamma distribution): E_theta = N^N theta^(N-1) exp(-N theta) / Gamma(N),
    mean theta = 1, variance 1/N.
  * Least-squares fits minimise sum (E_data - E_model)^2 over the samples with a nested
    golden-section search in log(parameter) (no derivatives, no SciPy).
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

OPEN_OPEN = "open-open"
CLOSED_CLOSED = "closed-closed"
TANKS = "tanks"
DISPERSION = "dispersion"

TAIL_FRAC = 0.25            # fraction of samples used for the tail fit
TAIL_MIN_POINTS = 5         # minimum number of samples used for the tail fit
THETA_FLOOR = 1e-12         # theta is clipped below to this value in E_theta
GOLDEN_TOL = 1e-9           # golden-section tolerance in log(parameter)
GOLDEN_MAXIT = 200          # golden-section iteration limit
LSQ_SHAPE_RANGE = (0.05, 1e4)   # search range of N or Pe (dimensionless)
LSQ_TAU_RANGE = (0.2, 5.0)      # search range of tau, as a factor times the data mean
BRACKET_EDGE_TOL = 1e-6   # ln Pe distance from a bracket end that counts as "at the edge"
SIGMA2_SERIES_PE = 0.1     # closed-closed sigma2 uses a Taylor series below this Pe (cancellation)
BISECT_ITER = 200           # bisection steps in the closed-closed inversion


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def tail_fit(t, C, frac=TAIL_FRAC):
    """Decay time lam (s) of C ~ exp(-t/lam) fitted to the last `frac` of the samples.

    Least-squares line through ln C over the last max(5, int(frac*len(t))) samples. Returns None
    if fewer samples exist, any of them is <= 0 (or NaN), or the fitted slope is not negative (signal not decaying there).
    """
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)
    n = max(TAIL_MIN_POINTS, int(frac * len(t)))
    if n > len(t):
        return None
    tt, cc = t[-n:], C[-n:]
    if not np.all(cc > 0):
        return None
    slope = np.polyfit(tt, np.log(cc), 1)[0]
    return -1.0 / slope if slope < 0 else None


@dataclass
class RtdMoments:
    """Result of `rtd_moments`.

    e: normalised E(t) at the samples, 1/s (C divided by the zeroth moment, tail included).
    mean: mean residence time, s (pulse-width corrected).
    variance: variance of the residence time, s^2 (pulse-width corrected).
    tail_decay_time: lam in s used for the tail extrapolation, or None if none was applied.
    """
    e: np.ndarray
    mean: float
    variance: float
    tail_decay_time: Optional[float]


def rtd_moments(t, C, pulse=0.0, tail=True):
    """Normalised E(t), mean and variance of a tracer response C(t) sampled at times t.

    t: increasing sample times, s (needs at least 2). C: response, any unit; negative values are
    clipped to 0. pulse: width of a square input pulse, s (default 0 = delta input); it adds
    pulse/2 to the mean and pulse^2/12 to the variance, both removed here. tail: if True
    (default) and `tail_fit` finds a decaying tail, the signal beyond the last sample is extended
    as C_T exp(-(t - T)/lam) (analytic moments). Stagnant regions give long tails that dominate
    the variance, so a truncated record without tail=True underestimates it.
    The input pulse is assumed to start at t = 0 and the samples to cover the whole peak.
    Zero total area gives NaN.
    """
    t = np.asarray(t, dtype=float)
    C = np.clip(np.asarray(C, dtype=float), 0.0, None)
    m0, m1, m2 = _trapz(C, t), _trapz(t * C, t), _trapz(t * t * C, t)
    lam = tail_fit(t, C) if tail else None
    if lam is not None:
        T, CT = t[-1], C[-1]
        m0 += CT * lam
        m1 += CT * lam * (T + lam)
        m2 += CT * lam * (T * T + 2 * T * lam + 2 * lam * lam)
    if not m0 > 0.0:     # zero (or NaN) total area: NaN, as in Rust
        nan = float("nan")
        return RtdMoments(np.full(C.shape, nan), nan, nan, lam)
    mean = m1 / m0
    var = m2 / m0 - mean * mean
    return RtdMoments(C / m0, mean - 0.5 * pulse, var - pulse * pulse / 12.0, lam)


def fit_dispersion_planes(mean1, var1, mean2, var2, dx):
    """Two-detector (moment-difference) analysis of the axial dispersion model.

    Between two planes a distance dx (m) apart inside the bed, d(var)/d(mean)^2 = 2/Pe exactly,
    independent of the boundary conditions at the ends. Inputs: mean residence times mean1,
    mean2 (s) and variances var1, var2 (s^2) at the upstream and downstream plane (from
    `rtd_moments` with the same pulse). Returns a dict: u = dx/dt mean velocity (m/s),
    Pe = u dx / D_L (dimensionless, based on dx), D_L = u dx / Pe (m^2/s), N = equivalent number
    of tanks in series between the planes, tau = dt (s), var = dv (s^2). Needs mean2 > mean1 and
    var2 > var1; otherwise u, Pe or D_L are negative or infinite.
    """
    dt, dv = mean2 - mean1, var2 - var1
    u = dx / dt
    Pe = 2.0 * dt * dt / dv
    return dict(u=u, Pe=Pe, D_L=u * dx / Pe, N=dt * dt / dv, tau=dt, var=dv)


def e_dispersion(theta, Pe):
    """Open-open axial-dispersion response to a delta input, E_theta (dimensionless).

    theta = t u / L, Pe = u L / D_L > 0. Unbounded-medium solution seen as concentration at
    distance L (see the module header for the boundary-condition statement). theta is clipped
    below at 1e-12. Accurate as a model of a vessel for Pe >~ 5; for small Pe the real
    (closed) vessel differs.
    """
    th = np.maximum(np.asarray(theta, dtype=float), THETA_FLOOR)
    return np.sqrt(Pe / (4 * np.pi * th)) * np.exp(-Pe * (1 - th) ** 2 / (4 * th))


# Lanczos approximation (g = 7, n = 9; Godfrey), relative accuracy about 1e-15 for x > 0.
_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7)


def ln_gamma(x):
    """ln Gamma(x) for x > 0 (Lanczos approximation, identical in Rust)."""
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - ln_gamma(1.0 - x)
    x -= 1.0
    a = _LANCZOS[0]
    t = x + 7.5
    for i in range(1, 9):
        a += _LANCZOS[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def e_tanks(theta, N):
    """Tanks-in-series response (gamma distribution), E_theta (dimensionless).

    theta = t / tau with tau = total mean residence time; N > 0 tanks (non-integer allowed).
    theta is clipped below at 1e-12.
    """
    th = np.maximum(np.asarray(theta, dtype=float), THETA_FLOOR)
    return np.exp(N * math.log(N) + (N - 1) * np.log(th) - N * th - ln_gamma(N))


def sigma2_dispersion(Pe, bc):
    """Dimensionless variance of E_theta for the dispersion model, Pe = u L / D_L > 0.

    bc = "open-open": 2/Pe + 8/Pe^2 (about the mean 1 + 2/Pe; relative to the mean squared it is
    (2 Pe + 8)/(Pe + 2)^2). bc = "closed-closed": 2/Pe - 2/Pe^2 (1 - exp(-Pe)) (Levenspiel ch. 13).
    The value returned is relative to tau^2 (tau = L/u).
    """
    if bc == OPEN_OPEN:
        return 2.0 / Pe + 8.0 / (Pe * Pe)
    if bc == CLOSED_CLOSED:
        if Pe < SIGMA2_SERIES_PE:   # the closed form loses all digits as Pe -> 0: use its Taylor series
            return 2.0 * sum((-1.0) ** k * Pe ** (k - 2) / math.factorial(k) for k in range(2, 12))
        return 2.0 / Pe - 2.0 / (Pe * Pe) * (1.0 - math.exp(-Pe))
    raise ValueError(f"unknown boundary condition {bc!r}")


@dataclass
class MomentFit:
    """Result of `fit_moments`.

    param: N (tanks) or Pe (dispersion), dimensionless; NaN if not success.
    tau: time scale of the model, s (tanks: the mean; open-open: mean/(1+2/Pe) = L/u;
         closed-closed: the mean).
    success: False if the variance is not positive or outside the model's range.
    message: "success" or the reason.
    """
    param: float
    tau: float
    success: bool
    message: str


def fit_moments(mean, variance, model, bc=OPEN_OPEN):
    """Fit N or Pe from the mean (s) and variance (s^2) of a measured RTD.

    model = "tanks": N = mean^2/variance. model = "dispersion": r = variance/mean^2 is matched
    to the model; open-open has the closed form Pe = (1 - 2 r + sqrt(1 + 4 r))/r; closed-closed
    has mean = tau and r = 2/Pe - 2/Pe^2 (1 - exp(-Pe)), solved by bisection in log Pe on
    [1e-6, 1e8] (r must lie in (0, 1); a solution within 1e-6 in ln Pe of either bracket end is
    reported as failure, not clamped). Fails (success False) for mean <= 0, variance <= 0, NaN,
    or r outside the model's range. The variance of a record cut before the tail is biased low:
    use `rtd_moments` with tail=True.
    """
    nan = float("nan")
    if not (mean > 0 and variance > 0):
        return MomentFit(nan, nan, False, "mean and variance must be positive")
    r = variance / (mean * mean)
    if model == TANKS:
        return MomentFit(1.0 / r, mean, True, "success")
    if model != DISPERSION:
        raise ValueError(f"unknown model {model!r}")
    if bc == OPEN_OPEN:
        Pe = (1.0 - 2.0 * r + math.sqrt(1.0 + 4.0 * r)) / r
        return MomentFit(Pe, mean / (1.0 + 2.0 / Pe), True, "success")
    if bc == CLOSED_CLOSED:
        if not r < 1.0:
            return MomentFit(nan, nan, False, "closed-closed variance/mean^2 must be < 1")
        lo, hi = math.log(1e-6), math.log(1e8)
        for _ in range(BISECT_ITER):
            mid = 0.5 * (lo + hi)
            if sigma2_dispersion(math.exp(mid), CLOSED_CLOSED) > r:
                lo = mid          # variance decreases with Pe: need larger Pe
            else:
                hi = mid
        x = 0.5 * (lo + hi)
        if min(x - math.log(1e-6), math.log(1e8) - x) < BRACKET_EDGE_TOL:
            return MomentFit(nan, nan, False, "closed-closed Pe at the edge of the [1e-6, 1e8] bracket")
        return MomentFit(math.exp(x), mean, True, "success")
    raise ValueError(f"unknown boundary condition {bc!r}")


def golden_section(f, a, b, tol=GOLDEN_TOL, maxit=GOLDEN_MAXIT):
    """Minimise a unimodal f on [a, b] by golden-section search.

    Stops when the bracket is shorter than tol (absolute, in the units of x). Returns
    (x, f(x), nfev). Finds a local minimum only if f is unimodal on the bracket.
    """
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    nfev = 2
    it = 0
    while b - a > tol and it < maxit:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        nfev += 1
        it += 1
    x = 0.5 * (a + b)
    fx = f(x)
    return x, fx, nfev + 1


@dataclass
class LsqFit:
    """Result of `fit_lsq`.

    param: N or Pe (dimensionless). tau: time scale, s (as in `MomentFit`, but fitted:
    for "dispersion" it is L/u, the model time scale, not the mean). sse: sum of squared
    residuals of E, 1/s^2. nfev: number of model evaluations. success/message: False if a
    parameter ended on the edge of its search range (the minimum is outside) or the data are
    unusable.
    """
    param: float
    tau: float
    sse: float
    nfev: int
    success: bool
    message: str


def fit_lsq(t, E, model, tau=None):
    """Least-squares fit of E(t) = E_theta(t/tau; p)/tau for the tanks or open-open dispersion model.

    t: sample times, s. E: measured E(t), 1/s (normalised, e.g. `rtd_moments(...).e`). model =
    "tanks" (p = N) or "dispersion" (p = Pe, open-open `e_dispersion`, tau = L/u). tau: if
    given (s) only p is fitted (1-D search); if None, tau and p are both fitted by a nested
    golden-section search in log p over [0.05, 1e4] and log tau over [0.2, 5] times the data
    mean (trapezoid first moment of E over the samples). Unweighted residuals at all samples;
    a finite input pulse width is ignored (deconvolve first if it matters). Local-minimum
    caveat: the nested search assumes the objective is unimodal in each variable.
    """
    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    if model not in (TANKS, DISPERSION):
        raise ValueError(f"unknown model {model!r}")
    shape = e_tanks if model == TANKS else e_dispersion
    m0 = _trapz(E, t)
    if not (m0 > 0):
        return LsqFit(float("nan"), float("nan"), float("nan"), 0, False, "E has no positive area")
    mean = _trapz(t * E, t) / m0
    count = [0]

    def sse(tau_, p):
        count[0] += 1
        return float(np.sum((E - shape(t / tau_, p) / tau_) ** 2))

    lp = (math.log(LSQ_SHAPE_RANGE[0]), math.log(LSQ_SHAPE_RANGE[1]))
    if tau is not None:
        u, fx, _ = golden_section(lambda x: sse(tau, math.exp(x)), *lp)
        tau_fit = tau
        lt_edge = False
    else:
        lt = (math.log(LSQ_TAU_RANGE[0] * mean), math.log(LSQ_TAU_RANGE[1] * mean))

        def inner(x):
            return golden_section(lambda y: sse(math.exp(y), math.exp(x)), *lt)

        u, fx, _ = golden_section(lambda x: inner(x)[1], *lp)
        y, fx, _ = inner(u)
        tau_fit = math.exp(y)
        lt_edge = min(y - lt[0], lt[1] - y) < 1e-6
    p = math.exp(u)
    edge = lt_edge or min(u - lp[0], lp[1] - u) < 1e-6
    msg = "success" if not edge else "parameter at the edge of its search range"
    return LsqFit(p, tau_fit, fx, count[0], not edge, msg)
