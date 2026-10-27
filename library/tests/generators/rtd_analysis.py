"""Generator for reference_data/rtd_analysis.json.

Synthetic tracer responses from exact residence-time distributions, truncated early, and
independent reference numbers:
  * tanks in series: scipy.stats.gamma (shape N, scale tau/N);
  * open-open dispersion: the closed form E_theta = sqrt(Pe/(4 pi theta)) exp(-Pe (1-theta)^2/(4 theta));
    its exact moments by scipy.integrate.quad (compared in this script with 1 + 2/Pe and
    2/Pe + 8/Pe^2);
  * closed-closed dispersion (Danckwerts): time-domain series from the poles of the Laplace transform
    E(s) = 4 q exp(Pe/2) / [(1+q)^2 exp(Pe q/2) - (1-q)^2 exp(-Pe q/2)], q = sqrt(1 + 4 s/Pe)
    (Levenspiel ch. 13 / Wehner & Wilhelm 1956 form). Poles s_k = -Pe (1 + mu_k^2)/4 with mu_k the
    roots of Pe mu/2 + 2 arctan(mu) = k pi (scipy.optimize.brentq); residues from the analytic
    derivative of the denominator. The script checks that the series integrates to 1 and has mean 1
    and variance 2/Pe - 2/Pe^2 (1 - exp(-Pe)) by quad, which is an independent test of that formula.
  * moments of each truncated record: scipy.integrate.trapezoid on the samples; the exponential tail
    from scipy.stats.linregress on ln C over the last 25 % of the samples and the closed-form tail
    integrals by scipy.integrate.quad;
  * least-squares fits: scipy.optimize.curve_fit (method lm, tight tolerances), both with tau free
    and with tau fixed.
Noisy records add Gaussian noise (numpy RandomState seed below) to E.
Run: python library/tests/generators/rtd_analysis.py
"""
import warnings

import numpy as np
import scipy.integrate as si
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st

from _common import write

warnings.filterwarnings("ignore", category=si.IntegrationWarning)  # quad tolerance 1e-11 is near its limit
SEED = 20261030
rng = np.random.RandomState(SEED)
SCALE = 5.0   # C = SCALE * E (arbitrary concentration unit)


def e_oo(theta, Pe):
    th = np.maximum(theta, 1e-12)
    return np.sqrt(Pe / (4 * np.pi * th)) * np.exp(-Pe * (1 - th) ** 2 / (4 * th))


_POLES = {}


def _poles(Pe, K):
    if (Pe, K) not in _POLES:
        mu = np.empty(K)
        for i in range(K):
            kk = i + 1
            g = lambda m, kk=kk: Pe * m / 2 + 2 * np.arctan(m) - kk * np.pi
            mu[i] = so.brentq(g, (kk - 1) * np.pi * 2 / Pe, kk * np.pi * 2 / Pe, xtol=1e-14, rtol=1e-14)
        q = 1j * mu
        s = -Pe * (1 + mu ** 2) / 4
        ep, em = np.exp(Pe * q / 2), np.exp(-Pe * q / 2)
        dD_dq = (2 * (1 + q) + Pe / 2 * (1 + q) ** 2) * ep + (2 * (1 - q) + Pe / 2 * (1 - q) ** 2) * em
        res = 4 * q * np.exp(Pe / 2) / (dD_dq * 2 / (Pe * q))   # residue of E(s) at s_k
        _POLES[(Pe, K)] = (s, res)
    return _POLES[(Pe, K)]


def e_cc(theta, Pe, K=800):
    """Closed-closed E_theta from the Laplace-transform poles (see header). Valid for theta >= 0.02."""
    s, res = _poles(Pe, K)
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.real(np.exp(np.outer(th, s)) @ res)


def exact_moments(f, upper, lower=0.0):
    m0 = si.quad(f, lower, upper, limit=500, epsabs=1e-11, epsrel=1e-11)[0]
    m1 = si.quad(lambda x: x * f(x), lower, upper, limit=500, epsabs=1e-11, epsrel=1e-11)[0]
    m2 = si.quad(lambda x: x * x * f(x), lower, upper, limit=500, epsabs=1e-11, epsrel=1e-11)[0]
    return m0, m1 / m0, m2 / m0 - (m1 / m0) ** 2


def record_moments(t, C):
    """Independent moments of a record: trapezoid + exponential tail (scipy)."""
    C = np.clip(C, 0, None)
    out = {}
    m = [si.trapezoid(t ** k * C, t) for k in range(3)]
    out["mean_notail"] = m[1] / m[0]
    out["var_notail"] = m[2] / m[0] - (m[1] / m[0]) ** 2
    n = max(5, int(0.25 * len(t)))
    lam = None
    if np.all(C[-n:] > 0):
        slope = st.linregress(t[-n:], np.log(C[-n:])).slope
        if slope < 0:
            lam = -1.0 / slope
    out["lam"] = lam
    if lam is None:
        out["mean_tail"], out["var_tail"] = out["mean_notail"], out["var_notail"]
    else:
        T, CT = t[-1], C[-1]
        tail = [si.quad(lambda x, k=k: x ** k * CT * np.exp(-(x - T) / lam), T, np.inf,
                        epsabs=1e-13, epsrel=1e-13)[0] for k in range(3)]
        m = [a + b for a, b in zip(m, tail)]
        out["mean_tail"] = m[1] / m[0]
        out["var_tail"] = m[2] / m[0] - (m[1] / m[0]) ** 2
    return out


def fits(t, E, model, tau0, p0, tau_fixed):
    shape = (lambda th, N: st.gamma.pdf(th, N, scale=1.0 / N)) if model == "tanks" else e_oo
    f2 = lambda t_, tau, p: shape(t_ / tau, p) / tau
    popt, _ = so.curve_fit(f2, t, E, p0=[tau0, p0], method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    f1 = lambda t_, p: shape(t_ / tau_fixed, p) / tau_fixed
    p1, _ = so.curve_fit(f1, t, E, p0=[p0], method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    sse = lambda tau, p: float(np.sum((E - f2(t, tau, p)) ** 2))
    return {"tau": popt[0], "p": popt[1], "sse": sse(*popt), "p_fixed_tau": p1[0], "tau_fixed": tau_fixed,
            "sse_fixed_tau": sse(tau_fixed, p1[0])}


datasets = []


def add(name, kind, tau, p, tmax_theta, n, noise=0.0, fit_p0=None):
    """kind: tanks | oo | cc. Samples at t_i = linspace(t0, tmax, n)."""
    t0 = 0.02 * tau if kind == "cc" else 0.0
    t = np.linspace(t0, tmax_theta * tau, n)
    th = t / tau
    if kind == "tanks":
        E = st.gamma.pdf(th, p, scale=1.0 / p) / tau
        ex_mean, ex_var = tau, tau ** 2 / p
    elif kind == "oo":
        E = e_oo(th, p) / tau
        m0, mm, vv = exact_moments(lambda x: e_oo(x, p), 60.0)
        assert abs(m0 - 1) < 1e-8 and abs(mm - (1 + 2 / p)) < 1e-8 and abs(vv - (2 / p + 8 / p ** 2)) < 1e-7, (m0, mm, vv)
        ex_mean, ex_var = tau * mm, tau ** 2 * vv  # tau = L/u
    else:
        E = e_cc(th, p) / tau
        m0, mm, vv = exact_moments(lambda x: float(e_cc(x, p)[0]), 12.0, 0.02)
        print("cc", p, m0 - 1, mm - 1, vv - (2 / p - 2 / p ** 2 * (1 - np.exp(-p))))
        assert abs(m0 - 1) < 1e-6 and abs(mm - 1) < 1e-6, (m0, mm)
        assert abs(vv - (2 / p - 2 / p ** 2 * (1 - np.exp(-p)))) < 1e-5, (vv, p)
        ex_mean, ex_var = tau, tau ** 2 * vv
    if noise:
        E = E + noise * E.max() * rng.randn(n)
    C = SCALE * E
    d = {"name": name, "kind": kind, "tau": tau, "param": p, "noise_rel_peak": noise,
         "exact_mean": ex_mean, "exact_variance": ex_var, "t": t.tolist(), "E": E.tolist(),
         "C": C.tolist(), "ref_moments": record_moments(t, C)}
    if kind != "cc":
        model = "tanks" if kind == "tanks" else "dispersion"
        d["curve_fit"] = fits(t, E, model, tau * 1.2, p * 1.3, tau)
    datasets.append(d)


add("cstr_trunc1p5", "tanks", 4.0, 1.0, 1.5, 3001)          # exact exponential: tail is exact
add("tanks_N3", "tanks", 100.0, 3.0, 2.5, 1001)
add("tanks_N8", "tanks", 100.0, 8.0, 2.0, 1001)              # narrow peak, early cut
add("tanks_N4p3", "tanks", 50.0, 4.3, 3.0, 601)              # non-integer N
add("oo_Pe10", "oo", 100.0, 10.0, 4.0, 2001)
add("oo_Pe40", "oo", 100.0, 40.0, 2.0, 2001)
add("cc_Pe10", "cc", 100.0, 10.0, 5.0, 2001)
add("cc_Pe40", "cc", 100.0, 40.0, 3.0, 2001)
add("noisy_tanks_N6", "tanks", 100.0, 6.0, 4.0, 400, noise=0.03)
add("noisy_oo_Pe8", "oo", 100.0, 8.0, 4.0, 400, noise=0.03)

lg = np.array([0.1, 0.5, 0.9, 1.0, 1.5, 2.0, 3.3, 7.0, 10.5, 25.0, 100.0, 171.5])
write("rtd_analysis", "library/tests/generators/rtd_analysis.py",
      {"seed": SEED, "rng": "numpy.random.RandomState", "C_scale": SCALE,
       "tanks": "scipy.stats.gamma", "closed_closed": "Laplace pole series, brentq roots, 800 terms",
       "curve_fit": "scipy.optimize.curve_fit method=lm, xtol=ftol=gtol=1e-14, p0 = 1.2 tau, 1.3 p",
       "tail": "scipy.stats.linregress on ln C over the last 25 % of samples; quad for the tail integrals"},
      "t in s, E in 1/s, C = C_scale * E (arbitrary unit). exact_mean/exact_variance are those of the untruncated "
      "distribution (open-open: mean tau(1+2/Pe), var tau^2(2/Pe+8/Pe^2), confirmed by quad; closed-closed "
      "by quad of the series). ref_moments are computed on the truncated sampled record. cc records start at 0.02 tau.",
      {"datasets": datasets, "ln_gamma": {"x": lg.tolist(), "ref": sp.gammaln(lg).tolist()}})
