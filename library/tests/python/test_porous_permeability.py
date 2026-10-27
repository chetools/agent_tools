"""Tests for properties.porous_permeability: formulas vs independently re-implemented reference data
(no published table values; see the generator), asymptotic limits, Ergun limits, the Kozeny-constant
inversion vs scipy curve_fit, NaN policy, and Rust vs Python."""
import math

import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.properties import porous_permeability as pp

RTOL_FORMULA = 1e-12      # float64, same formula in another algebraic form
RTOL_FIT = 1e-8           # inversion vs curve_fit (curve_fit converges to ~1e-10)
RTOL_PY_RS = 1e-12
TOL_ASYMPTOTE = 1e-3      # relative departure from the limiting form at the stated distance


def _relerr(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a)


def test_formulas_vs_reference():
    c = load_reference("porous_permeability")["cases"]
    worst = {}
    for r in c["kozeny_carman"]:
        worst["kozeny_carman"] = max(worst.get("kozeny_carman", 0), _relerr(pp.kozeny_carman(r["eps"], r["a_s"], r["K"]), r["k"]))
    for r in c["ergun"]:
        g = pp.ergun_gradient(r["u"], r["eps"], r["d"], r["mu"], r["rho"], 150.0, 1.75)
        worst["ergun"] = max(worst.get("ergun", 0), _relerr(g, r["grad"]))
    for name in ("gebart_square", "gebart_hexagonal", "sangani_acrivos_square"):
        for r in c[name]:
            worst[name] = max(worst.get(name, 0), _relerr(getattr(pp, name)(r["eps"], r["d"]), r["k"]))
    for name, w in worst.items():
        print(f"VERIFY porous_permeability {name} py_max_rel_err_vs_reference={w:e}")
        assert w < RTOL_FORMULA


def test_nan_outside_domain():
    for name, rows in load_reference("porous_permeability")["cases"]["nan_inputs"].items():
        for args in rows:
            assert math.isnan(float(getattr(pp, name)(*args))), (name, args)
    # arrays broadcast, NaN elementwise
    out = pp.gebart_square(np.array([0.1, 0.5]), 1e-3)
    assert math.isnan(out[0]) and out[1] > 0
    assert pp.gebart_square(1 - pp.PHI_MAX_SQUARE, 1e-3) == 0.0   # k = 0 exactly at close packing
    print("VERIFY porous_permeability nan_policy py_cases_checked=ok")


def test_dilute_sangani_acrivos_vs_hasimoto():
    """SA minus the two-term Hasimoto form, scaled by a^2/(8 phi) phi^2, equals -1.774 + 4.076 phi exactly (series terms)."""
    out = []
    for phi in (1e-3, 1e-2):
        a = 0.5
        hasimoto = a * a / (8 * phi) * (-math.log(phi) - 1.476 + 2 * phi)
        s = (pp.sangani_acrivos_square(1 - phi, 1.0) - hasimoto) / (a * a / (8 * phi) * phi ** 2)
        out.append(s)
    assert abs(out[0] - (-1.774 + 4.076e-3)) < 1e-6 and abs(out[1] - (-1.774 + 4.076e-2)) < 1e-6
    print(f"VERIFY porous_permeability dilute_sa_vs_hasimoto coeff_phi2_at_1e-3={out[0]:.5f} at_1e-2={out[1]:.5f} (series: -1.774+4.076 phi)")
    # leading log behaviour: k phi / (a^2/8) = -ln phi - 1.476 + O(phi)
    phi = 1e-8
    lead = pp.sangani_acrivos_square(1 - phi, 1.0) * phi / (0.25 / 8)
    assert abs(lead - (-math.log(phi) - 1.476)) < 1e-6


def test_gebart_near_close_packing():
    """k ~ C1 R^2 ((phi_max - phi)/(2 phi_max))^(5/2) as phi -> phi_max; relative departure is O(delta)."""
    worst = {}
    for name, c1, pm in (("gebart_square", pp.C1_SQUARE, pp.PHI_MAX_SQUARE), ("gebart_hexagonal", pp.C1_HEX, pp.PHI_MAX_HEX)):
        for delta in (1e-3, 1e-5):
            phi = pm - delta
            k = getattr(pp, name)(1 - phi, 2.0)       # R = 1
            lim = c1 * (delta / (2 * pm)) ** 2.5
            worst[name, delta] = k / lim - 1
        assert abs(worst[name, 1e-5]) < TOL_ASYMPTOTE * 0.1 and abs(worst[name, 1e-3]) < 10 * TOL_ASYMPTOTE * 10
        assert abs(worst[name, 1e-5]) < abs(worst[name, 1e-3]) / 50   # converges linearly in delta
    print("VERIFY porous_permeability gebart_asymptote " + " ".join(f"{n}@{d:g}={v:.3e}" for (n, d), v in worst.items()))
    # constants
    assert abs(pp.C1_SQUARE - 0.4001) < 1e-4 and abs(pp.C1_HEX - 0.2310) < 1e-4 and abs(pp.C1_SQUARE / pp.C1_HEX - math.sqrt(3)) < 1e-14


def test_ergun_limits_and_blake_kozeny():
    eps, d, mu, rho = 0.4, 2e-3, 1e-3, 1000.0
    k = eps ** 3 * d * d / (150 * (1 - eps) ** 2)
    # laminar: -dp/dx -> mu U / k  (Re_p = 1e-4)
    u = 1e-4 * mu * (1 - eps) / (rho * d)
    lam = pp.ergun_gradient(u, eps, d, mu, rho) / (mu * u / k) - 1
    # turbulent: -> Burke-Plummer 1.75 rho U^2 (1-eps)/(eps^3 d)  (Re_p = 1e6)
    u = 1e6 * mu * (1 - eps) / (rho * d)
    bp = 1.75 * rho * u * u * (1 - eps) / (eps ** 3 * d)
    tur = pp.ergun_gradient(u, eps, d, mu, rho) / bp - 1
    # crossover: terms equal at Re_p = 150/1.75
    u = (150 / 1.75) * mu * (1 - eps) / (rho * d)
    visc = pp.ergun_gradient(u, eps, d, mu, rho, 150.0, 0.0)
    inert = pp.ergun_gradient(u, eps, d, mu, rho, 0.0, 1.75)
    cross = visc / inert - 1
    print(f"VERIFY porous_permeability ergun_limits laminar_rel={lam:e} turbulent_rel={tur:e} crossover_rel={cross:e}")
    # the departures are the first corrections: (1.75/150) Re_p and (150/1.75)/Re_p
    assert abs(lam - 1.75 / 150 * 1e-4) < 1e-9 and abs(tur - 150 / 1.75 / 1e6) < 1e-9 and abs(cross) < 1e-12
    # Kozeny-Carman with a_s = 6/d and K = 150/36 is the Blake-Kozeny viscous term of Ergun
    kc = pp.kozeny_carman(eps, 6 / d, 150 / 36)
    print(f"VERIFY porous_permeability kc_vs_ergun_viscous rel={abs(kc / k - 1):e}")
    assert abs(kc / k - 1) < 1e-14


def test_fit_kozeny_constant_vs_curve_fit():
    from scipy.optimize import curve_fit
    f = load_reference("porous_permeability")["cases"]["fit"]
    eps, a_s = np.array(f["eps"]), f["a_s"]
    k = np.array(f["exact"]["k"])
    K = pp.fit_kozeny_constant(k, eps, a_s)
    # recompute curve_fit here too (the stored value is a second witness)
    popt, _ = curve_fit(lambda e, K: pp.kozeny_carman(e, a_s, K), eps, k, p0=[1.0], xtol=1e-14, ftol=1e-14)
    e1 = np.max(np.abs(K / f["K_true"] - 1))
    e2 = abs(popt[0] / K[0] - 1)
    e3 = abs(f["exact"]["curve_fit_K"] / K[0] - 1)
    print(f"VERIFY porous_permeability fit_exact max_rel_err_vs_true={e1:e} vs_curve_fit={e2:e} vs_stored={e3:e}")
    assert e1 < 1e-12 and e2 < RTOL_FIT and e3 < RTOL_FIT
    # noisy data: each inversion reproduces its own measurement (round trip); the mean is not asserted equal to curve_fit
    kn = np.array(f["noisy"]["k"])
    Kn = pp.fit_kozeny_constant(kn, eps, a_s)
    assert np.max(np.abs(pp.kozeny_carman(eps, a_s, Kn) / kn - 1)) < 1e-12
    print(f"VERIFY porous_permeability fit_noisy mean_K={Kn.mean():.4f} curve_fit_K={f['noisy']['curve_fit_K']:.4f} (not equal: elementwise inversion is not a least-squares fit)")


def test_python_vs_rust_example():
    py, rs = python_example("porous_permeability"), rust_example("porous_permeability")
    worst = 0.0
    for key, v in py.items():
        if isinstance(v, bool):
            assert v == rs[key]
        else:
            worst = max(worst, _relerr(rs[key], v))
    print(f"VERIFY porous_permeability python_vs_rust max_rel_diff={worst:e}")
    assert worst < RTOL_PY_RS
