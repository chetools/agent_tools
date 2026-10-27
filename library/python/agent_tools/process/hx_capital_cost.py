"""Shell-and-tube heat-exchanger capital cost correlations with dC/dA, and the capital recovery factor.

Ported from chetools/claude heat-exchanger-network (hen.py `unit_cost`, `seider_pressure_factor`,
`turton_pressure_factor`, `capital_recovery_factor`; solver/src/lib.rs `unit_cost`). All costs are in
USD of the correlation's own cost-index year (see below); escalate to another year by
multiplying with index_new / index_basis. Areas are heat-transfer areas in m^2.

SEIDER set (Seider, Seader, Lewin, Widagdo et al., "Product and Process Design Principles", Wiley):
  purchase cost of the base (carbon-steel) exchanger C_B = exp(c0 + c1 ln A + c2 ln^2 A), A in ft^2;
  material factor F_M = a + (A/100 ft^2)^b (the book's shell-and-tube material-factor equation;
  numbered 22.44 in the 3rd edition per the IDAES documentation, 16.44 in the 4th edition per the
  task statement; the equation numbers and page numbers were NOT checked against the books);
  pressure factor F_P = 0.9803 + 0.018 (P/100) + 0.0017 (P/100)^2, P in psig (the code comment in
  IDAES gives the validity range 100-2000 psig; used here at any P);
  tube-length factor F_L (8, 12, 16, 20 ft: 1.25, 1.12, 1.05, 1.00).
  The base-cost coefficients are those coded in BioSTEAM (units/design_tools), cost-index basis
  CE = 567; the book's own purchase cost is C_P = F_P F_M F_L C_B. The bare-module cost used here,
      C_BM = C_B (F_BM + F_L F_P F_M - 1),  F_BM = 3.17 (default),
  is BioSTEAM's installed-cost rule, not a book equation. Valid area range as used by the source
  project: 150-12 000 ft^2 (13.9-1115 m^2): below it the cost of a 150 ft^2 unit is charged (slope 0), above
  it the correlation is extrapolated and `above_range` is set.
  Cross-check: the IDAES SSLW module (3rd ed., CE = 500) lists a different floating-head intercept
  (11.9052 vs 12.0310 here), so the coefficient set is NOT confirmed against the book; the material
  factors (a, b), F_P and F_L agree exactly with IDAES.

TURTON set (Turton, Bailie, Whiting, Shaeiwitz, Akinc, "Analysis, Synthesis and Design of Chemical
Processes", Prentice Hall, Appendix A; 3rd/4th edition table and figure numbers not checked):
  floating-head shell and tube, A in m^2, valid 10-1000 m^2, CEPCI = 397 (year 2001):
  log10 Cp0 = K1 + K2 log10 A + K3 log10^2 A, K = (4.8306, -0.8509, 0.3187);
  pressure factor (shell and tube, tube side, 5-140 barg) log10 F_P = C1 + C2 log10 P + C3 log10^2 P,
  C = (0.03881, -0.11272, 0.08183), F_P = 1 below 5 barg;
  bare module C_BM = Cp0 (B1 + B2 F_M F_P), B = (1.63, 1.66);
  F_M for same-material shell and tube: CS 1.00, SS 2.73, Ni alloy (Monel) 3.73, Ti 11.4.
  The independent implementation "eqcost" gives, for 100 m^2 CS at 20 barg, Cp0 25323 USD,
  F_P 1.07321, C_BM 86393 USD; the tests check these. K, C, B and F_M other than CS were not checked
  against the book.

Derivatives: dC_BM/dA in USD/m^2 at fixed material, pressure and length, exact (analytic); zero in
the clamped region below the lower area limit (the cost is flat there, as charged).
"""
import math
from dataclasses import dataclass

M2_TO_FT2 = 10.7639104            # ft^2 per m^2 (as in the source project)

# ----------------------------------------------------------------------------- Seider
#: Cost-index basis (Chemical Engineering Plant Cost Index) of the Seider base-cost coefficients.
SEIDER_CE_BASIS = 567.0
#: Base-cost coefficients (c0, c1, c2) of C_B [USD, CE 567] for A in ft^2, by exchanger type.
SEIDER_CB = {
    "floating_head": (12.0310, -0.8709, 0.09005),
    "fixed_head": (11.4185, -0.9228, 0.09861),
    "u_tube": (11.5510, -0.9186, 0.09790),
}
#: Material factor coefficients F_M = a + (A/100 ft^2)^b by (shell, tube) material.
SEIDER_FM = {
    ("CS", "CS"): (0.0, 0.0),
    ("CS", "brass"): (1.08, 0.05),
    ("CS", "SS"): (1.75, 0.13),
    ("CS", "Monel"): (2.1, 0.13),
    ("CS", "Ti"): (5.2, 0.16),
    ("CS", "CrMo"): (1.55, 0.05),
    ("CrMo", "CrMo"): (1.7, 0.07),
    ("SS", "SS"): (2.7, 0.07),
    ("Monel", "Monel"): (3.3, 0.08),
    ("Ti", "Ti"): (9.6, 0.06),
}
SEIDER_LENGTH_FACTOR = {8: 1.25, 12: 1.12, 16: 1.05, 20: 1.00}   # tube length [ft] -> F_L
SEIDER_AMIN_FT2 = 150.0           # lower area limit [ft^2]
SEIDER_AMAX_FT2 = 12000.0         # upper area limit [ft^2]
SEIDER_F_BM = 3.17                # default bare-module factor (dimensionless)


@dataclass
class HxCost:
    """Result of a cost evaluation."""
    cost_usd: float          # bare-module cost [USD of the correlation's index year]
    dcost_da: float          # dC/dA [USD/m^2]
    clamped_below: bool      # True if A was below the lower limit (cost of the limit area, slope 0)
    above_range: bool        # True if A is above the upper limit (extrapolated)


def seider_material_coeffs(shell, tube):
    """(a, b) of F_M = a + (A/100 ft^2)^b for a (shell, tube) material pair
    (names: CS, brass, CrMo, SS, Monel, Ti). Raises ValueError for an unlisted pair."""
    try:
        return SEIDER_FM[(shell, tube)]
    except KeyError:
        raise ValueError(f"no Seider material factor for shell={shell!r}, tube={tube!r}") from None


def seider_length_factor(tube_length_ft):
    """F_L for a tube length of exactly 8, 12, 16 or 20 ft (dimensionless). ValueError otherwise."""
    try:
        return SEIDER_LENGTH_FACTOR[tube_length_ft]
    except KeyError:
        raise ValueError(f"tube length must be one of {sorted(SEIDER_LENGTH_FACTOR)} ft") from None


def seider_pressure_factor(p_psig):
    """F_P = 0.9803 + 0.018 (P/100) + 0.0017 (P/100)^2, P in psig (dimensionless result).
    Stated validity 100-2000 psig; evaluated for any P."""
    x = p_psig / 100.0
    return 0.9803 + 0.018 * x + 0.0017 * x * x


def seider_bare_module_cost(area_m2, hx_type, fm_a, fm_b, f_l, f_p, f_bm=SEIDER_F_BM):
    """Bare-module cost C_BM = C_B (F_BM + F_L F_P F_M - 1) [USD, CE 567] and dC_BM/dA [USD/m^2].

    area_m2: heat-transfer area [m^2], > 0. hx_type: 'floating_head', 'fixed_head' or 'u_tube'.
    fm_a, fm_b: from seider_material_coeffs. f_l: seider_length_factor. f_p: seider_pressure_factor.
    f_bm: bare-module factor, default 3.17. Below 150 ft^2 the cost of 150 ft^2 is charged with
    slope 0 (clamped_below); above 12 000 ft^2 the fit is extrapolated (above_range).
    Invalid area (<= 0 or not finite) or unknown hx_type: NaN cost and slope (unknown type: ValueError).
    """
    c0, c1, c2 = SEIDER_CB[hx_type]
    if not (math.isfinite(area_m2) and area_m2 > 0.0):
        return HxCost(math.nan, math.nan, False, False)
    af = area_m2 * M2_TO_FT2
    clamp = af < SEIDER_AMIN_FT2
    if clamp:
        af = SEIDER_AMIN_FT2
    ln_a = math.log(af)
    cb = math.exp(c0 + c1 * ln_a + c2 * ln_a * ln_a)
    fm = fm_a + (af / 100.0) ** fm_b
    mult = f_bm + f_l * f_p * fm - 1.0
    cost = cb * mult
    above = af > SEIDER_AMAX_FT2
    if clamp:
        return HxCost(cost, 0.0, True, False)
    dcb = cb * (c1 + 2.0 * c2 * ln_a) / af
    dfm = fm_b * (af / 100.0) ** (fm_b - 1.0) / 100.0 if fm_b != 0.0 else 0.0
    dcost = dcb * mult + cb * f_l * f_p * dfm
    return HxCost(cost, dcost * M2_TO_FT2, False, above)


# ----------------------------------------------------------------------------- Turton
#: Cost-index basis (CEPCI) of the Turton coefficients.
TURTON_CEPCI_BASIS = 397.0
TURTON_K = (4.8306, -0.8509, 0.3187)     # floating head, A in m^2
TURTON_B = (1.63, 1.66)
TURTON_C = (0.03881, -0.11272, 0.08183)  # pressure factor, P in barg
TURTON_FM = {"CS": 1.00, "SS": 2.73, "Monel": 3.73, "Ti": 11.4}   # same-material; Monel = Ni alloy
TURTON_AMIN_M2 = 10.0
TURTON_AMAX_M2 = 1000.0


def turton_material_factor(material):
    """F_M for a same-material shell and tube (CS, SS, Monel, Ti). ValueError otherwise."""
    try:
        return TURTON_FM[material]
    except KeyError:
        raise ValueError(f"no Turton material factor for {material!r}") from None


def turton_pressure_factor(p_barg, c=TURTON_C):
    """log10 F_P = C1 + C2 log10 P + C3 log10^2 P, P in barg; F_P = 1 below 5 barg
    (dimensionless). Fitted for 5-140 barg; evaluated for any P >= 5."""
    if p_barg < 5.0:
        return 1.0
    y = math.log10(p_barg)
    return 10.0 ** (c[0] + c[1] * y + c[2] * y * y)


def turton_bare_module_cost(area_m2, f_m, f_p, k=TURTON_K, b=TURTON_B, a_min=TURTON_AMIN_M2,
                            a_max=TURTON_AMAX_M2):
    """C_BM = Cp0 (B1 + B2 F_M F_P) [USD, CEPCI 397] and dC_BM/dA [USD/m^2].

    area_m2: area [m^2] > 0. f_m: turton_material_factor. f_p: turton_pressure_factor.
    k: (K1, K2, K3) of log10 Cp0 (default floating head), b: (B1, B2). Below a_min (10 m^2) the
    cost of a_min is charged with slope 0 (clamped_below); above a_max (1000 m^2) the fit is
    extrapolated (above_range). Invalid area: NaN cost and slope.
    """
    if not (math.isfinite(area_m2) and area_m2 > 0.0):
        return HxCost(math.nan, math.nan, False, False)
    clamp = area_m2 < a_min
    a = a_min if clamp else area_m2
    x = math.log10(a)
    cp0 = 10.0 ** (k[0] + k[1] * x + k[2] * x * x)
    mult = b[0] + b[1] * f_m * f_p
    if clamp:
        return HxCost(cp0 * mult, 0.0, True, False)
    return HxCost(cp0 * mult, mult * cp0 * (k[1] + 2.0 * k[2] * x) / a, False, a > a_max)


# ----------------------------------------------------------------------------- annualisation
def capital_recovery_factor(rate, years):
    """CRF = i (1+i)^n / ((1+i)^n - 1) [1/yr]: annual payment per unit of capital.

    rate: interest rate i per year (dimensionless, > -1; 0.1 = 10 %), years: n > 0 (may be
    non-integer). i = 0 gives 1/n. Invalid input gives NaN. Evaluated with log1p/expm1, so it stays
    accurate for tiny rates (relative error about 1e-15 against 50-digit values, tested).
    """
    if not (rate > -1.0 and years > 0.0) or not (math.isfinite(rate) and math.isfinite(years)):
        return math.nan
    if rate == 0.0:
        return 1.0 / years
    # same formula; (1+i)^n - 1 evaluated as expm1(n log1p(i)) so small rates do not cancel
    # (the textbook evaluation loses 4 digits at i = 1e-12, n = 20)
    lg = years * math.log1p(rate)
    return rate * math.exp(lg) / math.expm1(lg)
