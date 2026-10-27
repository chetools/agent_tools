"""Generator for reference_data/gp_matern52.json.

Everything here is independent of the library: dense NumPy/SciPy only.

Model (the one in the library docs): standardised y, yz = (y - ym)/ys, kernel
k(a,b) = sf2 * matern_{5/2}(r), r^2 = sum_i ((a_i - b_i)/ell_i)^2, noise variance sn2 plus a diagonal
jitter 1e-8 (variance units of yz), all in standardised units; results converted back to y units.

Kernel reference: the general Matern formula with the modified Bessel function,
k = sf2 2^(1-nu)/Gamma(nu) (sqrt(2 nu) r)^nu K_nu(sqrt(2 nu) r), nu = 5/2 (Rasmussen & Williams 2006,
eq. 4.14), via scipy.special.kv; the polynomial-times-exponential closed form (eq. 4.17) is asserted to
agree with it.

Cases of gp_matern52.json:
  kernel        pairs of points with ARD length scales; Bessel-function Matern values.
  smooth2d      d = 2, n = 14: Branin function on [0,1]^2 (scaled inputs) at a random design plus noise
                (5 % of the std). NLL values and fourth-order central-difference gradients (step 1e-3) of the
                dense NLL at three thetas (two with fitted noise, one with fixed noise); L-BFGS-B fits
                (scipy.optimize.minimize, bounds on theta) from the three isotropic starts of the library
                defaults; predictions (mean, latent variance, cross covariance) at the best fitted
                hyperparameters by a direct np.linalg.solve.
  interp1d      d = 1, n = 8, Forrester function, noise_sd = 1e-4 (almost noise free): dense predictions.
  hard5d        d = 5, n = 40, 34 random points plus 6 points 1e-3 away from others (scaled domain),
                fixed noise_sd = 1e-5 (units of y): condition number of K up to ~1e8. NLL values and gradients at
                two thetas, L-BFGS-B fits from the three default starts plus 5 random starts.
Run: python library/tests/generators/gp_matern52.py
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

from _common import write

JITTER = 1e-8
SEED = 20261030
LB = (math.log(0.02), math.log(5.0))      # library defaults, scaled inputs
SB = (math.log(0.1), math.log(10.0))
NB = (math.log(1e-3), 0.0)


def matern_bessel(r, sf2):
    """General Matern nu = 5/2 via the Bessel function (r = scaled distance)."""
    nu = 2.5
    a = math.sqrt(2 * nu) * np.asarray(r, float)
    out = np.full(a.shape, sf2, float)
    m = a > 0
    out[m] = sf2 * 2 ** (1 - nu) / gamma(nu) * a[m] ** nu * kv(nu, a[m])
    return out


def kmat(A, B, ell, sf2):
    r = cdist(np.asarray(A) / ell, np.asarray(B) / ell)
    k = matern_bessel(r, sf2)
    s = math.sqrt(5) * r
    closed = sf2 * (1 + s + s * s / 3) * np.exp(-s)
    assert np.max(np.abs(k - closed)) < 1e-13 * sf2
    return k


def std(y):
    s = float(np.std(y))
    return float(np.mean(y)), (s if s != 0 else 1.0)


def nll_std(X, yz, ell, sf2, sn2):
    n = len(yz)
    K = kmat(X, X, ell, sf2) + (sn2 + JITTER) * np.eye(n)
    c = cho_factor(K, lower=True)
    a = cho_solve(c, yz)
    return float(0.5 * yz @ a + np.sum(np.log(np.diag(c[0]))) + 0.5 * n * math.log(2 * math.pi))


def nll_theta(theta, X, yz, fixed_var=None):
    d = X.shape[1]
    ell = np.exp(theta[:d])
    sf2 = math.exp(2 * theta[d])
    sn2 = fixed_var if fixed_var is not None else math.exp(2 * theta[d + 1])
    try:
        return nll_std(X, yz, ell, sf2, sn2)
    except np.linalg.LinAlgError:
        return math.inf


def nll_and_grad_theta(theta, X, yz, fixed_var=None):
    """Dense NLL and its analytic gradient (R&W eq. 5.9, 0.5 tr((K^-1 - a a^T) dK/dtheta)), written
    independently with np.linalg.inv; used as `jac` for L-BFGS-B and checked against fd_grad."""
    n, d = X.shape
    ell = np.exp(theta[:d])
    sf2 = math.exp(2 * theta[d])
    sn2 = fixed_var if fixed_var is not None else math.exp(2 * theta[d + 1])
    f = nll_theta(theta, X, yz, fixed_var)
    if not math.isfinite(f):
        return math.inf, np.zeros(len(theta))
    r = cdist(X / ell, X / ell)
    s = math.sqrt(5) * r
    Kf = sf2 * (1 + s + s * s / 3) * np.exp(-s)
    Ki = np.linalg.inv(Kf + (sn2 + JITTER) * np.eye(n))
    a = Ki @ yz
    W = Ki - np.outer(a, a)
    g = []
    for i in range(d):
        D2 = ((X[:, None, i] - X[None, :, i]) / ell[i]) ** 2
        dK = sf2 * (5 / 3) * (1 + s) * np.exp(-s) * D2       # d k / d ln ell_i
        g.append(0.5 * np.sum(W * dK))
    g.append(0.5 * np.sum(W * 2 * Kf))
    if fixed_var is None:
        g.append(0.5 * np.trace(W) * 2 * sn2)
    return f, np.array(g)


def fd_grad(f, theta, h=1e-3):
    """Fourth-order central differences."""
    g = np.empty(len(theta))
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = h
        g[i] = (-f(theta + 2 * e) + 8 * f(theta + e) - 8 * f(theta - e) + f(theta - 2 * e)) / (12 * h)
    return g


def lbfgsb_fit(X, yz, starts, fixed_var=None):
    d = X.shape[1]
    lo = [LB[0]] * d + [SB[0]] + ([] if fixed_var is not None else [NB[0]])
    hi = [LB[1]] * d + [SB[1]] + ([] if fixed_var is not None else [NB[1]])
    out = []
    for t0 in starts:
        r = minimize(nll_and_grad_theta, t0, args=(X, yz, fixed_var), jac=True, method="L-BFGS-B",
                     bounds=list(zip(lo, hi)),
                     options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 20000})
        out.append({"start": list(map(float, t0)), "theta": r.x.tolist(), "nll_std": float(r.fun),
                    "nit": int(r.nit), "nfev": int(r.nfev)})
    return out


def dense_post(X, y, ell, signal_sd, noise_sd, A, B):
    """Posterior (mean at A, latent covariance A x B) in y units, direct np.linalg.solve."""
    ym, ys = std(y)
    sf2, sn2 = (signal_sd / ys) ** 2, (noise_sd / ys) ** 2
    K = kmat(X, X, ell, sf2) + (sn2 + JITTER) * np.eye(len(y))
    yz = (y - ym) / ys
    mean = ym + ys * (kmat(A, X, ell, sf2) @ np.linalg.solve(K, yz))
    cov = kmat(A, B, ell, sf2) - kmat(A, X, ell, sf2) @ np.linalg.solve(K, kmat(X, B, ell, sf2))
    return mean, ys * ys * cov


def branin(x):
    x1, x2 = -5 + 15 * x[:, 0], 15 * x[:, 1]
    return ((x2 - 5.1 / (4 * math.pi ** 2) * x1 ** 2 + 5 / math.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * math.pi)) * np.cos(x1) + 10)


def forrester(u):
    return (6 * u - 2) ** 2 * np.sin(12 * u - 4)


def theta_cases(X, yz, thetas, fixed_var=None):
    for t in thetas:                       # analytic gradient (used as jac) vs finite differences
        ga = nll_and_grad_theta(np.array(t, float), X, yz, fixed_var)[1]
        gf = fd_grad(lambda th: nll_theta(th, X, yz, fixed_var), np.array(t, float))
        assert np.max(np.abs(ga - gf)) < 1e-5 * max(1.0, np.max(np.abs(gf))), (ga, gf)
    return [{"theta": list(map(float, t)), "fixed_noise_var": fixed_var, "nll_std": nll_theta(np.array(t), X, yz, fixed_var),
             "grad_fd": fd_grad(lambda th: nll_theta(th, X, yz, fixed_var), np.array(t, float)).tolist()}
            for t in thetas]


rng = np.random.RandomState(SEED)
cases = {}

# ---- kernel
P = rng.rand(6, 3)
Q = rng.rand(6, 3)
ell_k = np.array([0.3, 0.7, 1.9])
cases["kernel"] = {"a": P.tolist(), "b": Q.tolist(), "ell": ell_k.tolist(), "sf2": 2.5,
                   "k": [float(kmat(P[i:i + 1], Q[i:i + 1], ell_k, 2.5)[0, 0]) for i in range(6)],
                   "k_zero_distance": float(kmat(P[:1], P[:1], ell_k, 2.5)[0, 0])}

# ---- smooth2d
X = rng.rand(14, 2)
yb = branin(X)
y = yb + 0.05 * np.std(yb) * rng.randn(14)
ym, ys = std(y)
yz = (y - ym) / ys
thetas = [[0.3, -0.2, 0.1, -0.5], [-1.0, 1.0, 0.7, -1.5]]
starts = [[math.log(l0)] * 2 + [0.0, math.log(0.1)] for l0 in (0.1, 0.3, 1.0)]
fits = lbfgsb_fit(X, yz, starts)
best = min(fits, key=lambda f: f["nll_std"])
th = np.array(best["theta"])
ell_b, sd_b, nsd_b = np.exp(th[:2]), math.exp(th[2]) * ys, math.exp(th[3]) * ys
G = rng.rand(7, 2)
mean, cov = dense_post(X, y, ell_b, sd_b, nsd_b, G, G)
mean_ab, cov_ab = dense_post(X, y, ell_b, sd_b, nsd_b, G[:4], G[3:])
cases["smooth2d"] = {
    "d": 2, "x": X.ravel().tolist(), "y": y.tolist(), "y_mean": ym, "y_scale": ys,
    "theta_checks": theta_cases(X, yz, thetas) + theta_cases(X, yz, [[0.3, -0.2, 0.1]], 0.04),
    "lbfgsb_fits": fits, "best": best,
    "fitted": {"ell": ell_b.tolist(), "signal_sd": sd_b, "noise_sd": nsd_b,
               "log_marginal_likelihood": -best["nll_std"] - len(y) * math.log(ys)},
    "test_points": G.ravel().tolist(), "pred_mean": mean.tolist(), "pred_var": np.diag(cov).tolist(),
    "cov_a_rows": 4, "cov_b_rows": 4, "cov_ab": cov_ab.ravel().tolist()}
# fixed-noise fit (noise_sd = 0.3 in y units) for the fixed-noise path
fvar = (0.3 / ys) ** 2
ffits = lbfgsb_fit(X, yz, [t[:3] for t in starts], fvar)
cases["smooth2d"]["fixed_noise"] = {"noise_sd": 0.3, "lbfgsb_fits": ffits, "best": min(ffits, key=lambda f: f["nll_std"])}

# ---- interp1d
X1 = np.sort(rng.rand(8, 1), axis=0)
y1 = forrester(X1[:, 0])
G1 = np.linspace(0.02, 0.98, 6).reshape(-1, 1)
m1, c1 = dense_post(X1, y1, np.array([0.2]), 1.0 * std(y1)[1], 1e-4, G1, G1)
cases["interp1d"] = {"d": 1, "x": X1.ravel().tolist(), "y": y1.tolist(), "ell": [0.2], "signal_sd": std(y1)[1],
                     "noise_sd": 1e-4, "test_points": G1.ravel().tolist(), "pred_mean": m1.tolist(),
                     "pred_var": np.diag(c1).tolist()}

# ---- hard5d
base = rng.rand(34, 5)
pick = rng.permutation(34)[:6]
close = base[pick] + 1e-3 * rng.randn(6, 5)
Xh = np.vstack([base, np.clip(close, 0, 1)])
yh = (np.sin(3 * Xh[:, 0]) + 0.5 * np.cos(5 * Xh[:, 1]) * Xh[:, 2] + Xh[:, 3] ** 2 - 0.3 * Xh[:, 4]
      + 0.2 * np.sin(8 * Xh[:, 0] * Xh[:, 4]))
ymh, ysh = std(yh)
yzh = (yh - ymh) / ysh
fvh = (1e-5 / ysh) ** 2
Kh = kmat(Xh, Xh, np.full(5, 1.0), 1.0) + (fvh + JITTER) * np.eye(40)
thh = [[math.log(0.5)] * 5 + [0.0], [math.log(0.2), math.log(0.4), math.log(0.8), math.log(1.5), math.log(3.0), 0.3]]
startsh = [[math.log(l0)] * 5 + [0.0] for l0 in (0.1, 0.3, 1.0)]
rs = np.random.RandomState(SEED + 1)
startsh += [list(LB[0] + (LB[1] - LB[0]) * rs.rand(5)) + [0.0] for _ in range(5)]
fitsh = lbfgsb_fit(Xh, yzh, startsh, fvh)
besth = min(fitsh, key=lambda f: f["nll_std"])
cases["hard5d"] = {"d": 5, "x": Xh.ravel().tolist(), "y": yh.tolist(), "y_mean": ymh, "y_scale": ysh, "noise_sd": 1e-5,
                   "cond_K_at_ell1": float(np.linalg.cond(Kh)),
                   "min_pair_distance": float(np.min(cdist(Xh, Xh) + 1e9 * np.eye(40))),
                   "theta_checks": theta_cases(Xh, yzh, thh, fvh), "lbfgsb_fits": fitsh, "best": besth,
                   "n_default_starts": 3}

write("gp_matern52", "library/tests/generators/gp_matern52.py",
      {"seed": SEED, "rng": "numpy.random.RandomState", "kernel_reference": "scipy.special.kv (general Matern nu=5/2)",
       "optimizer_reference": "scipy.optimize.minimize L-BFGS-B on theta with bounds and an independent analytic gradient (checked against finite differences), ftol 1e-14, gtol 1e-9",
       "bounds_ln_length": list(LB), "bounds_ln_signal": list(SB), "bounds_ln_noise": list(NB), "jitter": JITTER,
       "gradient_reference": "4th-order central differences, step 1e-3, of the dense NLL"},
      "Standardised-y marginal likelihood with the 1e-8 diagonal jitter. theta = [ln ell_1..ln ell_d, ln sf, (ln sn)] "
      "with sf, sn relative to the std of y. nll_std includes n/2 ln 2pi. Inputs are in the scaled unit domain. "
      "Posterior values are in y units. See the generator docstring for the cases.",
      cases)

print("smooth2d best", best["nll_std"], [f["nll_std"] for f in fits])
print("hard5d", cases["hard5d"]["cond_K_at_ell1"], cases["hard5d"]["min_pair_distance"], [round(f["nll_std"], 6) for f in fitsh])
