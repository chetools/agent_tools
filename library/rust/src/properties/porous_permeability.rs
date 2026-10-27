//! Permeability and pressure-drop correlations for porous media.
//!
//! Rust twin of library/python/agent_tools/properties/porous_permeability.py (same formulas,
//! defaults and domains); ported from chetools/claude `porous-media-lbm/python/lbm.py`.
//!
//! Units (SI): `eps` dimensionless; `d` in m; `a_s` in 1/m; `k` in m^2; `u` in m/s; `mu` in Pa s;
//! `rho` in kg/m^3; pressure gradient in Pa/m; `kozeny`, `a`, `b` dimensionless.
//!
//! Out-of-domain policy: an input outside the mathematical/physical domain gives `NaN` (never a panic,
//! never a clipped value). Domains are stated per function. The recommended range of a correlation
//! is documented only.
//!
//! Sources: Kozeny (1927), Sitzungsber. Akad. Wiss. Wien IIa 136:271-306; Carman (1937), Trans.
//! Inst. Chem. Eng. 15:150-166 (constant K = 5); Ergun (1952), Chem. Eng. Prog. 48(2):89-94 (150 and
//! 1.75; the range of Ergun's data is not verified here); Gebart (1992), J. Compos. Mater. 26:1100-1133
//! (square C1 = 16/(9 pi sqrt2), phi_max = pi/4; hexagonal C1 = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3);
//! constants read from Table 1 of Yazdchi, Srivastava & Luding 2011, Int. J. Multiphase Flow, a secondary
//! source); Sangani & Acrivos (1982), Int. J. Multiphase Flow 8:193-206 (dilute series for a square
//! array; first two terms Hasimoto 1959; truncated at O(phi^4), accuracy at large phi not verified).
use std::f64::consts::PI;

/// Close-packing solid fraction of a square array of cylinders, pi/4.
pub const PHI_MAX_SQUARE: f64 = PI / 4.0;
/// Close-packing solid fraction of a hexagonal array of cylinders, pi/(2 sqrt 3).
pub const PHI_MAX_HEX: f64 = PI / (2.0 * 1.7320508075688772);

fn c1_square() -> f64 {
    16.0 / (9.0 * PI * 2.0_f64.sqrt())
}
fn c1_hex() -> f64 {
    16.0 / (9.0 * PI * 6.0_f64.sqrt())
}

/// Kozeny-Carman permeability `eps^3 / (K a_s^2 (1-eps)^2)` in m^2.
/// `eps`: porosity (-), domain 0 < eps < 1. `a_s`: solid surface per solid volume, 1/m (spheres 6/d,
/// long cylinders 4/d), domain > 0. `kozeny`: Kozeny constant (-), Carman's default is [`KOZENY_DEFAULT`] = 5,
/// domain > 0. Outside the domain: NaN.
pub fn kozeny_carman(eps: f64, a_s: f64, kozeny: f64) -> f64 {
    if !(eps > 0.0 && eps < 1.0 && a_s > 0.0 && kozeny > 0.0) {
        return f64::NAN;
    }
    eps.powi(3) / (kozeny * a_s * a_s * (1.0 - eps).powi(2))
}

/// Default Kozeny constant (Carman 1937).
pub const KOZENY_DEFAULT: f64 = 5.0;

/// Kozeny constant (-) that makes [`kozeny_carman`] match a measured permeability `k` (m^2):
/// `eps^3 / (k a_s^2 (1-eps)^2)`. One value per measurement. Domain: k > 0, 0 < eps < 1, a_s > 0 (1/m);
/// outside it: NaN.
pub fn fit_kozeny_constant(k: f64, eps: f64, a_s: f64) -> f64 {
    if !(k > 0.0 && eps > 0.0 && eps < 1.0 && a_s > 0.0) {
        return f64::NAN;
    }
    eps.powi(3) / (k * a_s * a_s * (1.0 - eps).powi(2))
}

/// Default Ergun viscous constant (Ergun 1952).
pub const ERGUN_A: f64 = 150.0;
/// Default Ergun inertial constant (Ergun 1952).
pub const ERGUN_B: f64 = 1.75;

/// Ergun pressure gradient -dp/dx in Pa/m for superficial velocity `u` (m/s), porosity `eps` (-),
/// particle diameter `d` (m), viscosity `mu` (Pa s), density `rho` (kg/m^3); constants `a`, `b`
/// default to [`ERGUN_A`], [`ERGUN_B`]. Domain: u >= 0, 0 < eps < 1, d > 0, mu, rho, a, b >= 0; outside: NaN.
/// Viscous and inertial terms are equal at Re_p = rho u d/(mu (1-eps)) = a/b = 85.7.
pub fn ergun_gradient(u: f64, eps: f64, d: f64, mu: f64, rho: f64, a: f64, b: f64) -> f64 {
    if !(u >= 0.0 && eps > 0.0 && eps < 1.0 && d > 0.0 && mu >= 0.0 && rho >= 0.0 && a >= 0.0 && b >= 0.0) {
        return f64::NAN;
    }
    a * mu * u * (1.0 - eps).powi(2) / (eps.powi(3) * d * d) + b * rho * u * u * (1.0 - eps) / (eps.powi(3) * d)
}

fn gebart(eps: f64, d: f64, c1: f64, phi_max: f64) -> f64 {
    let phi = 1.0 - eps;
    if !(eps > 0.0 && eps < 1.0 && d > 0.0 && phi <= phi_max) {
        return f64::NAN;
    }
    let base = (phi_max / phi).sqrt() - 1.0;
    let base = if base > 0.0 { base } else { 0.0 };
    c1 * base.powf(2.5) * (0.5 * d).powi(2)
}

/// Gebart (1992) transverse permeability (m^2) of a square cylinder array, cylinder diameter `d` (m):
/// `C1 (sqrt(phi_max/phi) - 1)^2.5 R^2`, phi = 1 - eps, C1 = 16/(9 pi sqrt2), phi_max = pi/4.
/// Domain: 0 < eps < 1, d > 0, phi <= pi/4 (k = 0 exactly at phi = phi_max); overlap (phi > phi_max): NaN.
/// Recommended: high solid fraction near close packing.
pub fn gebart_square(eps: f64, d: f64) -> f64 {
    gebart(eps, d, c1_square(), PHI_MAX_SQUARE)
}

/// Gebart (1992) transverse permeability (m^2) of a hexagonal cylinder array: as [`gebart_square`] with
/// C1 = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3) (eps >= 0.0931).
pub fn gebart_hexagonal(eps: f64, d: f64) -> f64 {
    gebart(eps, d, c1_hex(), PHI_MAX_HEX)
}

/// Sangani & Acrivos (1982) dilute expansion (m^2), transverse flow through a square array of cylinders
/// of diameter `d` (m): `a^2/(8 phi) (-ln phi - 1.476 + 2 phi - 1.774 phi^2 + 4.076 phi^3)`, a = d/2,
/// phi = 1 - eps. Domain: 0 < eps < 1, d > 0, phi <= pi/4; outside: NaN. Recommended: small phi.
pub fn sangani_acrivos_square(eps: f64, d: f64) -> f64 {
    let phi = 1.0 - eps;
    if !(eps > 0.0 && eps < 1.0 && d > 0.0 && phi <= PHI_MAX_SQUARE) {
        return f64::NAN;
    }
    let a = 0.5 * d;
    a * a / (8.0 * phi) * (-phi.ln() - 1.476 + 2.0 * phi - 1.774 * phi * phi + 4.076 * phi.powi(3))
}
