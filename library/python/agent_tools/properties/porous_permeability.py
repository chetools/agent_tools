"""Permeability and pressure-drop correlations for porous media.

Python reference implementation (NumPy only); Rust twin:
library/rust/src/properties/porous_permeability.rs.  Ported from chetools/claude
`porous-media-lbm/python/lbm.py`.

Units (SI) -- eps dimensionless; d, a_s^-1, R in m; k in m^2; U in m/s; mu in Pa s; rho in kg/m^3;
pressure gradient in Pa/m; a_s in 1/m; K, A, B dimensionless.  Functions broadcast over NumPy
arrays and return NumPy floats/arrays.

Out-of-domain policy (identical in Rust): an input outside the mathematical/physical domain gives NaN,
never an exception and never a clipped value.  The domain is stated per function.  The
*recommended* range of a correlation (where it is accurate) is only documented, not enforced.

Sources
-------
* Kozeny, J. (1927), Sitzungsber. Akad. Wiss. Wien, Abt. IIa 136, 271-306; Carman, P. C. (1937),
  Trans. Inst. Chem. Eng. 15, 150-166.  k = eps^3 / (K a_s^2 (1-eps)^2), a_s = solid surface per
  solid volume.  Carman's constant K = 5 (default here); for spheres a_s = 6/d (Blake-Kozeny),
  and K = 150/36 = 4.1667 reproduces the viscous term of Ergun's equation.
* Ergun, S. (1952), Chem. Eng. Prog. 48(2), 89-94: -dp/dx = A mu U (1-eps)^2/(eps^3 d^2)
  + B rho U^2 (1-eps)/(eps^3 d), A = 150, B = 1.75, spherical/granular particles of diameter d,
  U the superficial velocity.  The viscous and inertial terms are equal at Re_p = A/B = 85.7,
  Re_p = rho U d / (mu (1-eps)).  The range of Ergun's own data is NOT verified here.
* Gebart, B. R. (1992), J. Compos. Mater. 26, 1100-1133 (lubrication model, transverse flow
  through ordered cylinder arrays): k = C1 (sqrt(phi_max/phi) - 1)^(5/2) R^2, with
  square: C1 = 16/(9 pi sqrt2), phi_max = pi/4; hexagonal: C1 = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3).
  Intended for high solid fractions (close packing); phi_max is the geometric close-packing limit.
  The constants were read from Table 1 of Yazdchi, Srivastava & Luding (2011), Int. J. Multiphase Flow,
  "Microstructural effects on the permeability of periodic fibrous porous media" (secondary
  source); Gebart's paper itself was not available.
* Sangani, A. S. & Acrivos, A. (1982), Int. J. Multiphase Flow 8, 193-206, dilute expansion for a
  square array (also as printed in Yazdchi et al., Table 1, where K/d^2 = 1/(32 phi) (ln(1/phi) - 1.476
  + 2 phi - 1.774 phi^2 + 4.076 phi^3)).  The first two terms are Hasimoto (1959).  The series is truncated at O(phi^4); it is a
  dilute expansion and its accuracy at large phi was not verified here.
"""
import numpy as np

PHI_MAX_SQUARE = np.pi / 4.0
PHI_MAX_HEX = np.pi / (2.0 * np.sqrt(3.0))
C1_SQUARE = 16.0 / (9.0 * np.pi * np.sqrt(2.0))
C1_HEX = 16.0 / (9.0 * np.pi * np.sqrt(6.0))


def _nan_unless(ok, value):
    return np.where(ok, value, np.nan)[()]


def kozeny_carman(eps, a_s, K=5.0):
    """Kozeny-Carman permeability k = eps^3 / (K a_s^2 (1-eps)^2) [m^2].

    eps: porosity (-), domain 0 < eps < 1.  a_s: solid surface per solid volume [1/m] (spheres 6/d,
    long cylinders 4/d), domain > 0.  K: Kozeny constant (-), default 5.0 (Carman), domain > 0.
    Outside the domain: NaN.  Accuracy: empirical; typically used for eps roughly 0.3-0.6 in
    granular beds (expectation, not verified here)."""
    eps, a_s, K = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (eps, a_s, K)))
    ok = (eps > 0) & (eps < 1) & (a_s > 0) & (K > 0)
    with np.errstate(all="ignore"):
        return _nan_unless(ok, eps ** 3 / (K * a_s * a_s * (1.0 - eps) ** 2))


def fit_kozeny_constant(k, eps, a_s):
    """Kozeny constant K (-) that makes kozeny_carman match a measured permeability k [m^2], element by element.

    Exact inversion K = eps^3 / (k a_s^2 (1-eps)^2); for several measurements it returns one K per
    measurement (average them, or fit, as appropriate).  Domain: k > 0, 0 < eps < 1, a_s > 0 [1/m];
    outside it: NaN."""
    k, eps, a_s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (k, eps, a_s)))
    ok = (k > 0) & (eps > 0) & (eps < 1) & (a_s > 0)
    with np.errstate(all="ignore"):
        return _nan_unless(ok, eps ** 3 / (k * a_s * a_s * (1.0 - eps) ** 2))


def ergun_gradient(U, eps, d, mu, rho, A=150.0, B=1.75):
    """Ergun pressure gradient -dp/dx [Pa/m] for superficial velocity U [m/s].

    eps: bed porosity (-); d: particle diameter [m]; mu: viscosity [Pa s]; rho: density [kg/m^3];
    A, B: Ergun constants (-), defaults 150 and 1.75.  Domain: U >= 0, 0 < eps < 1, d > 0, mu >= 0, rho >= 0,
    A >= 0, B >= 0; outside it: NaN (negative U is rejected because the inertial term would lose its sign).
    Laminar limit (Re_p << A/B = 85.7): Blake-Kozeny, -dp/dx = mu U / k with k = eps^3 d^2 / (A (1-eps)^2);
    turbulent limit (Re_p >> A/B): Burke-Plummer."""
    U, eps, d, mu, rho, A, B = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (U, eps, d, mu, rho, A, B)))
    ok = (U >= 0) & (eps > 0) & (eps < 1) & (d > 0) & (mu >= 0) & (rho >= 0) & (A >= 0) & (B >= 0)
    with np.errstate(all="ignore"):
        g = (A * mu * U * (1 - eps) ** 2 / (eps ** 3 * d * d)
             + B * rho * U * U * (1 - eps) / (eps ** 3 * d))
    return _nan_unless(ok, g)


def _gebart(eps, d, c1, phi_max):
    eps, d = np.broadcast_arrays(np.asarray(eps, dtype=float), np.asarray(d, dtype=float))
    phi = 1.0 - eps
    ok = (eps > 0) & (eps < 1) & (d > 0) & (phi <= phi_max)
    with np.errstate(all="ignore"):
        base = np.sqrt(phi_max / phi) - 1.0
        k = c1 * np.where(base > 0, base, 0.0) ** 2.5 * (0.5 * d) ** 2
    return _nan_unless(ok, k)


def gebart_square(eps, d):
    """Gebart (1992) transverse permeability [m^2] of a square array of cylinders of diameter d [m].

    k = C1 (sqrt(phi_max/phi) - 1)^(5/2) R^2, phi = 1 - eps, R = d/2, C1 = 16/(9 pi sqrt2), phi_max = pi/4.
    Domain: 0 < eps < 1, d > 0, phi <= pi/4 (eps >= 0.2146); at phi = phi_max k = 0 exactly; for
    phi > phi_max (cylinders overlap): NaN.  Recommended: high solid fraction, close packing."""
    return _gebart(eps, d, C1_SQUARE, PHI_MAX_SQUARE)


def gebart_hexagonal(eps, d):
    """Gebart (1992) transverse permeability [m^2] of a hexagonal array of cylinders of diameter d [m].

    As gebart_square with C1 = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3) = 0.9069 (eps >= 0.0931)."""
    return _gebart(eps, d, C1_HEX, PHI_MAX_HEX)


def sangani_acrivos_square(eps, d):
    """Sangani & Acrivos (1982) dilute expansion [m^2] for transverse flow through a square array of
    cylinders of diameter d [m]: k = a^2/(8 phi) (-ln phi - 1.476 + 2 phi - 1.774 phi^2 + 4.076 phi^3),
    a = d/2, phi = 1 - eps.

    Domain: 0 < eps < 1, d > 0, phi <= pi/4 (beyond the square close packing the cylinders overlap);
    outside it: NaN.  Recommended: dilute, small phi (truncated at O(phi^4); accuracy at large phi not verified).
    Do not use near close packing: at phi = 0.78 it gives about 0.048 d^2 where Gebart gives about 7e-8 d^2."""
    eps, d = np.broadcast_arrays(np.asarray(eps, dtype=float), np.asarray(d, dtype=float))
    phi, a = 1.0 - eps, 0.5 * d
    ok = (eps > 0) & (eps < 1) & (d > 0) & (phi <= PHI_MAX_SQUARE)
    with np.errstate(all="ignore"):
        k = a * a / (8 * phi) * (-np.log(phi) - 1.476 + 2 * phi - 1.774 * phi ** 2 + 4.076 * phi ** 3)
    return _nan_unless(ok, k)
