//! Shell-and-tube heat-exchanger capital cost correlations with dC/dA, and the capital recovery factor.
//!
//! Ported from chetools/claude heat-exchanger-network (hen.py `unit_cost`, `seider_pressure_factor`,
//! `turton_pressure_factor`, `capital_recovery_factor`; solver/src/lib.rs `unit_cost`). All costs are
//! in USD of the correlation's own cost-index year; escalate to another year by multiplying with
//! index_new / index_basis. Areas are heat-transfer areas in m^2.
//!
//! SEIDER set (Seider, Seader, Lewin, Widagdo et al., "Product and Process Design Principles", Wiley):
//! purchase cost of the base (carbon-steel) exchanger C_B = exp(c0 + c1 ln A + c2 ln^2 A), A in ft^2;
//! material factor F_M = a + (A/100 ft^2)^b (the book's shell-and-tube material-factor equation;
//! numbered 22.44 in the 3rd edition per the IDAES documentation; equation and page numbers were NOT
//! checked against the books);
//! pressure factor F_P = 0.9803 + 0.018 (P/100) + 0.0017 (P/100)^2, P in psig (IDAES code comment gives
//! the validity range 100-2000 psig; used here at any P, [`seider_pressure_below_range`] flags
//! P < 100 psig);
//! tube-length factor F_L (8, 12, 16, 20 ft: 1.25, 1.12, 1.05, 1.00).
//! The base-cost coefficients are those coded in BioSTEAM (units/design_tools), cost-index basis
//! CE = 567; the book's own purchase cost is C_P = F_P F_M F_L C_B. The bare-module cost used here,
//! `C_BM = C_B (F_BM + F_L F_P F_M - 1)`, F_BM = 3.17 (default), is BioSTEAM's installed-cost rule,
//! not a book equation. Valid area range as used by the source project: 150-12 000 ft^2
//! (13.9-1115 m^2): below it the cost of a 150 ft^2 unit is charged (slope 0), above it the
//! correlation is extrapolated and `above_range` is set.
//! Coefficient provenance (PR review findings; open items for the maintainer, no coefficient was changed):
//!   * floating head: BioSTEAM 12.0310 (CE 567) vs IDAES SSLW 11.9052 (CE 500) differ by ln(567/500) = 0.1258,
//!     so the costs agree to 4.9e-5 after the cost-index shift (checked in the tests). Not a contradiction.
//!   * fixed head: slope -0.9228 here (BioSTEAM; a Penn lecture snippet, Lewin FOCAPD 2004, agrees) against
//!     -0.8228 in IDAES. It is unconfirmed which is right; the IDAES value is probably a typo.
//!   * U-tube: the intercepts differ by 0.1658, more than the index shift 0.1258 (4.1 % cost difference);
//!     unresolved.
//!   * No book was consulted: equation numbers and the coefficients themselves are unchecked.
//! The material factors (a, b), F_P and F_L agree exactly with IDAES.
//!
//! TURTON set (Turton, Bailie, Whiting, Shaeiwitz, Akinc, "Analysis, Synthesis and Design of Chemical
//! Processes", Prentice Hall, Appendix A; 3rd/4th edition table and figure numbers not checked):
//! floating-head shell and tube, A in m^2, valid 10-1000 m^2, CEPCI = 397 (year 2001):
//! log10 Cp0 = K1 + K2 log10 A + K3 log10^2 A, K = (4.8306, -0.8509, 0.3187);
//! pressure factor (tube side, 5-140 barg) log10 F_P = C1 + C2 log10 P + C3 log10^2 P,
//! C = (0.03881, -0.11272, 0.08183), F_P = 1 below 5 barg;
//! bare module C_BM = Cp0 (B1 + B2 F_M F_P), B = (1.63, 1.66);
//! F_M for same-material shell and tube: CS 1.00, SS 2.73, Ni alloy (Monel) 3.73, Ti 11.4.
//! The independent implementation "eqcost" gives, for 100 m^2 CS at 20 barg, Cp0 25323 USD,
//! F_P 1.07321, C_BM 86393 USD; the tests check these. K, C and B are confirmed only by this third-party
//! worked example (not by the book; open item for the maintainer); F_M other than CS were not checked.
//!
//! Derivatives: dC_BM/dA in USD/m^2 at fixed material, pressure and length, exact (analytic); zero in
//! the clamped region below the lower area limit (the cost is flat there, as charged).

/// ft^2 per m^2 (as in the source project).
pub const M2_TO_FT2: f64 = 10.7639104;

/// Cost-index basis (Chemical Engineering Plant Cost Index) of the Seider base-cost coefficients.
pub const SEIDER_CE_BASIS: f64 = 567.0;
/// Lower area limit of the Seider correlation [ft^2].
pub const SEIDER_AMIN_FT2: f64 = 150.0;
/// Upper area limit of the Seider correlation [ft^2].
pub const SEIDER_AMAX_FT2: f64 = 12000.0;
/// Default bare-module factor (dimensionless).
pub const SEIDER_F_BM: f64 = 3.17;
/// Lower end of the stated validity range of the Seider pressure factor [psig].
pub const SEIDER_PMIN_PSIG: f64 = 100.0;

/// Exchanger type (Seider base-cost coefficient set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HxType {
    FloatingHead,
    FixedHead,
    UTube,
}

impl HxType {
    /// Base-cost coefficients (c0, c1, c2) of C_B [USD, CE 567] for A in ft^2.
    pub fn seider_cb(self) -> [f64; 3] {
        match self {
            HxType::FloatingHead => [12.0310, -0.8709, 0.09005],
            HxType::FixedHead => [11.4185, -0.9228, 0.09861],
            HxType::UTube => [11.5510, -0.9186, 0.09790],
        }
    }
}

/// Construction material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Material {
    /// Carbon steel.
    Cs,
    Brass,
    /// Chrome-moly steel.
    CrMo,
    /// Stainless steel.
    Ss,
    Monel,
    /// Titanium.
    Ti,
}

/// Result of a cost evaluation.
#[derive(Clone, Copy, Debug)]
pub struct HxCost {
    /// Bare-module cost [USD of the correlation's index year].
    pub cost_usd: f64,
    /// dC/dA [USD/m^2].
    pub dcost_da: f64,
    /// True if A was below the lower limit (cost of the limit area, slope 0).
    pub clamped_below: bool,
    /// True if A is above the upper limit (extrapolated).
    pub above_range: bool,
}

/// (a, b) of F_M = a + (A/100 ft^2)^b for a (shell, tube) pair; `None` for an unlisted pair.
pub fn seider_material_coeffs(shell: Material, tube: Material) -> Option<(f64, f64)> {
    use Material::*;
    match (shell, tube) {
        (Cs, Cs) => Some((0.0, 0.0)),
        (Cs, Brass) => Some((1.08, 0.05)),
        (Cs, Ss) => Some((1.75, 0.13)),
        (Cs, Monel) => Some((2.1, 0.13)),
        (Cs, Ti) => Some((5.2, 0.16)),
        (Cs, CrMo) => Some((1.55, 0.05)),
        (CrMo, CrMo) => Some((1.7, 0.07)),
        (Ss, Ss) => Some((2.7, 0.07)),
        (Monel, Monel) => Some((3.3, 0.08)),
        (Ti, Ti) => Some((9.6, 0.06)),
        _ => None,
    }
}

/// F_L for a tube length of exactly 8, 12, 16 or 20 ft (dimensionless); `None` otherwise.
pub fn seider_length_factor(tube_length_ft: u32) -> Option<f64> {
    match tube_length_ft {
        8 => Some(1.25),
        12 => Some(1.12),
        16 => Some(1.05),
        20 => Some(1.00),
        _ => None,
    }
}

/// F_P = 0.9803 + 0.018 (P/100) + 0.0017 (P/100)^2, P in psig (dimensionless result).
/// Stated validity 100-2000 psig; evaluated for any P (values below 100 psig are unchanged;
/// [`seider_pressure_below_range`] flags them).
pub fn seider_pressure_factor(p_psig: f64) -> f64 {
    let x = p_psig / 100.0;
    0.9803 + 0.018 * x + 0.0017 * x * x
}

/// True if `p_psig` [psig] is below the stated lower validity limit [`SEIDER_PMIN_PSIG`] = 100 psig of the
/// Seider pressure factor (F_P is then extrapolated, like `above_range` for the area).
pub fn seider_pressure_below_range(p_psig: f64) -> bool {
    p_psig < SEIDER_PMIN_PSIG
}

/// Bare-module cost `C_BM = C_B (F_BM + F_L F_P F_M - 1)` [USD, CE 567] and dC_BM/dA [USD/m^2].
///
/// `area_m2`: heat-transfer area [m^2], > 0. `fm_a`, `fm_b`: from [`seider_material_coeffs`].
/// `f_l`: [`seider_length_factor`]. `f_p`: [`seider_pressure_factor`]. `f_bm`: bare-module factor
/// (default [`SEIDER_F_BM`] = 3.17). Below 150 ft^2 the cost of 150 ft^2 is charged with slope 0
/// (`clamped_below`); above 12 000 ft^2 the fit is extrapolated (`above_range`). Invalid area
/// (<= 0 or not finite): NaN cost and slope.
pub fn seider_bare_module_cost(
    area_m2: f64,
    hx_type: HxType,
    fm_a: f64,
    fm_b: f64,
    f_l: f64,
    f_p: f64,
    f_bm: f64,
) -> HxCost {
    let [c0, c1, c2] = hx_type.seider_cb();
    if !(area_m2.is_finite() && area_m2 > 0.0) {
        return HxCost { cost_usd: f64::NAN, dcost_da: f64::NAN, clamped_below: false, above_range: false };
    }
    let mut af = area_m2 * M2_TO_FT2;
    let clamp = af < SEIDER_AMIN_FT2;
    if clamp {
        af = SEIDER_AMIN_FT2;
    }
    let ln_a = af.ln();
    let cb = (c0 + c1 * ln_a + c2 * ln_a * ln_a).exp();
    let fm = fm_a + (af / 100.0).powf(fm_b);
    let mult = f_bm + f_l * f_p * fm - 1.0;
    let cost = cb * mult;
    let above = af > SEIDER_AMAX_FT2;
    if clamp {
        return HxCost { cost_usd: cost, dcost_da: 0.0, clamped_below: true, above_range: false };
    }
    let dcb = cb * (c1 + 2.0 * c2 * ln_a) / af;
    let dfm = if fm_b != 0.0 { fm_b * (af / 100.0).powf(fm_b - 1.0) / 100.0 } else { 0.0 };
    let dcost = dcb * mult + cb * f_l * f_p * dfm;
    HxCost { cost_usd: cost, dcost_da: dcost * M2_TO_FT2, clamped_below: false, above_range: above }
}

/// Cost-index basis (CEPCI) of the Turton coefficients.
pub const TURTON_CEPCI_BASIS: f64 = 397.0;
/// (K1, K2, K3) of log10 Cp0, floating head, A in m^2.
pub const TURTON_K: [f64; 3] = [4.8306, -0.8509, 0.3187];
/// (B1, B2) of C_BM = Cp0 (B1 + B2 F_M F_P).
pub const TURTON_B: [f64; 2] = [1.63, 1.66];
/// (C1, C2, C3) of log10 F_P, P in barg.
pub const TURTON_C: [f64; 3] = [0.03881, -0.11272, 0.08183];
/// Lower area limit [m^2].
pub const TURTON_AMIN_M2: f64 = 10.0;
/// Upper area limit [m^2].
pub const TURTON_AMAX_M2: f64 = 1000.0;

/// F_M for a same-material shell and tube: Cs 1.00, Ss 2.73, Monel (Ni alloy) 3.73, Ti 11.4;
/// `None` for Brass and CrMo.
pub fn turton_material_factor(material: Material) -> Option<f64> {
    match material {
        Material::Cs => Some(1.00),
        Material::Ss => Some(2.73),
        Material::Monel => Some(3.73),
        Material::Ti => Some(11.4),
        _ => None,
    }
}

/// log10 F_P = C1 + C2 log10 P + C3 log10^2 P, P in barg (`c` defaults to [`TURTON_C`]);
/// F_P = 1 below 5 barg (dimensionless). Fitted for 5-140 barg; evaluated for any P >= 5.
pub fn turton_pressure_factor(p_barg: f64, c: [f64; 3]) -> f64 {
    if p_barg < 5.0 {
        return 1.0;
    }
    let y = p_barg.log10();
    10f64.powf(c[0] + c[1] * y + c[2] * y * y)
}

/// `C_BM = Cp0 (B1 + B2 F_M F_P)` [USD, CEPCI 397] and dC_BM/dA [USD/m^2].
///
/// `area_m2`: area [m^2] > 0. `f_m`: [`turton_material_factor`]. `f_p`: [`turton_pressure_factor`].
/// `k`, `b`: default [`TURTON_K`] (floating head), [`TURTON_B`]. Below `a_min` (default
/// [`TURTON_AMIN_M2`]) the cost of `a_min` is charged with slope 0 (`clamped_below`); above `a_max`
/// (default [`TURTON_AMAX_M2`]) the fit is extrapolated (`above_range`). Invalid area: NaN cost and slope.
pub fn turton_bare_module_cost(
    area_m2: f64,
    f_m: f64,
    f_p: f64,
    k: [f64; 3],
    b: [f64; 2],
    a_min: f64,
    a_max: f64,
) -> HxCost {
    if !(area_m2.is_finite() && area_m2 > 0.0) {
        return HxCost { cost_usd: f64::NAN, dcost_da: f64::NAN, clamped_below: false, above_range: false };
    }
    let clamp = area_m2 < a_min;
    let a = if clamp { a_min } else { area_m2 };
    let x = a.log10();
    let cp0 = 10f64.powf(k[0] + k[1] * x + k[2] * x * x);
    let mult = b[0] + b[1] * f_m * f_p;
    if clamp {
        return HxCost { cost_usd: cp0 * mult, dcost_da: 0.0, clamped_below: true, above_range: false };
    }
    HxCost {
        cost_usd: cp0 * mult,
        dcost_da: mult * cp0 * (k[1] + 2.0 * k[2] * x) / a,
        clamped_below: false,
        above_range: a > a_max,
    }
}

/// CRF = i (1+i)^n / ((1+i)^n - 1) [1/yr]: annual payment per unit of capital.
///
/// `rate`: interest rate i per year (dimensionless, > -1; 0.1 = 10 %), `years`: n > 0 (may be
/// non-integer). i = 0 gives 1/n. Invalid input gives NaN. Evaluated with `ln_1p`/`exp_m1`, so it
/// stays accurate for tiny rates (relative error about 1e-15 against 50-digit values, tested).
pub fn capital_recovery_factor(rate: f64, years: f64) -> f64 {
    if !(rate > -1.0 && years > 0.0 && rate.is_finite() && years.is_finite()) {
        return f64::NAN;
    }
    if rate == 0.0 {
        return 1.0 / years;
    }
    // same formula; (1+i)^n - 1 evaluated as exp_m1(n ln_1p(i)) so small rates do not cancel
    // (the textbook evaluation loses 4 digits at i = 1e-12, n = 20)
    let lg = years * rate.ln_1p();
    rate * lg.exp() / lg.exp_m1()
}
