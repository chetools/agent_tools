"""Generator for reference_data/porous_permeability.json.

Independent of library/python (does not import it).  Every value is computed here from the formulas
as printed in the sources, written in a DIFFERENT but algebraically equivalent form than the library
(dimensionless K/d^2 forms, Ergun in its Re_p form), so a transcription error in the library shows up:

* Sangani & Acrivos (1982) / Hasimoto (1959), Int. J. Multiphase Flow 8:193: K/d^2 =
  1/(32 phi) (ln(1/phi) - 1.476 + 2 phi - 1.774 phi^2 + 4.076 phi^3), as printed in Table 1 of Yazdchi,
  Srivastava & Luding (2011), Int. J. Multiphase Flow "Microstructural effects on the permeability of
  periodic fibrous porous media" (secondary source).
* Gebart (1992), J. Compos. Mater. 26:1100: K/R^2 = C (sqrt(phi_max/phi)-1)^(5/2), square C = 16/(9 pi sqrt2),
  phi_max = pi/4; hexagonal C = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3) (same Table 1; there given through
  the critical porosity eps_c = 1 - phi_max).
* Ergun (1952), Chem. Eng. Prog. 48(2):89: dp/L = (mu U/d^2) (1-eps)^2/eps^3 [150 + 1.75 Re_p],
  Re_p = rho U d/(mu (1-eps)).
* Kozeny-Carman: k = eps^3 / (K a_s^2 (1-eps)^2), K = 5 (Carman 1937).  Blake-Kozeny link: a_s = 6/d,
  K = 150/36 gives k = eps^3 d^2/(150 (1-eps)^2).
No numeric table values from the papers are used: none could be verified (the papers were not available;
only the formulas, as quoted in the secondary source).  The fit case uses scipy.optimize.curve_fit.

Run: python library/tests/generators/porous_permeability.py
"""
import numpy as np
from scipy.optimize import curve_fit

from _common import write

pi = np.pi
EPS = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.9999]


def kc(eps, a_s, K):
    return 1.0 / (K * a_s ** 2 * ((1 - eps) / eps ** 1.5) ** 2)


def ergun(u, eps, d, mu, rho):
    re_p = rho * u * d / (mu * (1 - eps))
    return mu * u / d ** 2 * (1 - eps) ** 2 / eps ** 3 * (150.0 + 1.75 * re_p)


def sa(eps, d):
    phi = 1 - eps
    return d * d / (32 * phi) * (np.log(1 / phi) - 1.476 + 2 * phi - 1.774 * phi ** 2 + 4.076 * phi ** 3)


def gebart(eps, d, c, phi_max):
    phi = 1 - eps
    return (d / 2) ** 2 * c * (np.sqrt(phi_max / phi) - 1) ** 2.5


cases = {}
cases["kozeny_carman"] = [dict(eps=e, a_s=a, K=K, k=kc(e, a, K))
                          for e in EPS for a, K in [(6 / 1e-3, 5.0), (4 / 2e-4, 4.0 + 1 / 6)]]
cases["ergun"] = [dict(u=u, eps=e, d=d, mu=mu, rho=rho, grad=ergun(u, e, d, mu, rho))
                  for e in [0.1, 0.36, 0.5, 0.9] for u in [0.0, 1e-6, 1e-3, 0.1, 5.0]
                  for d, mu, rho in [(3e-3, 1e-3, 1000.0), (1e-4, 1.8e-5, 1.2)]]
phi_sq, phi_hex = pi / 4, pi / (2 * np.sqrt(3))
c_sq, c_hex = 16 / (9 * pi * np.sqrt(2)), 16 / (9 * pi * np.sqrt(6))
# eps within 1e-9 .. of the close-packing limit is the hard case for the power 5/2
eps_sq = [1 - phi_sq + x for x in (1e-9, 1e-6, 1e-3, 0.02, 0.1, 0.3, 0.5, 0.7)]
eps_hex = [1 - phi_hex + x for x in (1e-9, 1e-6, 1e-3, 0.02, 0.1, 0.3, 0.5, 0.7)]
cases["gebart_square"] = [dict(eps=e, d=d, k=gebart(e, d, c_sq, phi_sq)) for e in eps_sq for d in (1e-5, 2.0)]
cases["gebart_hexagonal"] = [dict(eps=e, d=d, k=gebart(e, d, c_hex, phi_hex)) for e in eps_hex for d in (1e-5, 2.0)]
cases["sangani_acrivos_square"] = [dict(eps=e, d=d, k=sa(e, d)) for e in [1 - phi_sq + 1e-9, 0.25, 0.4, 0.6, 0.8, 0.95, 0.999, 1 - 1e-6]
                                   for d in (1e-5, 2.0)]
# outside the domain: NaN
cases["nan_inputs"] = {
    "kozeny_carman": [[0.0, 1e3, 5.0], [1.0, 1e3, 5.0], [0.5, 0.0, 5.0], [0.5, 1e3, 0.0], [-0.1, 1e3, 5.0]],
    "fit_kozeny_constant": [[0.0, 0.5, 1e3], [1e-9, 1.0, 1e3], [1e-9, 0.5, -1.0]],
    "ergun_gradient": [[-1.0, 0.4, 1e-3, 1e-3, 1e3], [1.0, 1.0, 1e-3, 1e-3, 1e3], [1.0, 0.4, 0.0, 1e-3, 1e3],
                       [1.0, 0.4, 1e-3, -1.0, 1e3]],
    "gebart_square": [[0.1, 1e-3], [0.0, 1e-3], [1.0, 1e-3], [0.5, 0.0]],
    "gebart_hexagonal": [[0.05, 1e-3], [0.5, -1.0]],
    "sangani_acrivos_square": [[0.1, 1e-3], [1.0, 1e-3], [0.5, 0.0]],
}
# fit: synthetic exact and noisy data; scipy curve_fit of k(eps; K) at fixed a_s
rng = np.random.default_rng(12345)
eps_f = np.linspace(0.3, 0.7, 9)
a_s = 6 / 5e-4
K_true = 4.7
k_exact = kc(eps_f, a_s, K_true)
k_noisy = k_exact * (1 + 0.03 * rng.standard_normal(eps_f.size))
fit = {}
for name, kk in (("exact", k_exact), ("noisy", k_noisy)):
    popt, _ = curve_fit(lambda e, K: kc(e, a_s, K), eps_f, kk, p0=[1.0], xtol=1e-14, ftol=1e-14)
    fit[name] = dict(k=list(kk), curve_fit_K=float(popt[0]))
cases["fit"] = dict(eps=list(eps_f), a_s=a_s, K_true=K_true, **fit)

write("porous_permeability", "library/tests/generators/porous_permeability.py",
      {"kozeny_carman_default_K": 5.0, "ergun": [150.0, 1.75], "noise": "3 % multiplicative, default_rng(12345)",
       "curve_fit": "p0=1, xtol=ftol=1e-14"},
      "SI units. Values computed from the published formulas in a different algebraic form than the library "
      "(see the generator docstring); no paper table values used. NaN policy cases are inputs only.",
      cases)
