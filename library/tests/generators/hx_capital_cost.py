"""Generator for reference_data/hx_capital_cost.json.

Contents
* published: numbers typed from sources (NOT computed by the library)
    - Seider material-factor table F_M = a + (A/100)^b and pressure factor 0.9803 + 0.018 x + 0.0017 x^2
      (x = P/100 psig) and tube-length factors (8, 12, 16, 20 ft: 1.25, 1.12, 1.05, 1.00): as listed in
      the IDAES SSLW costing module (idaes.models.costing.SSLW, which cites Seider, Seader, Lewin,
      Widagdo, 3rd ed., Eq. 22.44 for F_M), read 2026-10-27 at
      https://idaes-pse.readthedocs.io/en/main/_modules/idaes/models/costing/SSLW.html.
      The book itself (3rd or 4th ed., Eq. 16.44) was not available; page numbers are unchecked.
    - Turton test case of the independent implementation "eqcost": floating head, 100 m^2, carbon
      steel, 20 barg, CEPCI 397: Cp0 = 25323 USD, F_P = 1.07321, C_BM = 86393 USD (the tolerance in the
      tests, 3e-3 relative, is the source project's; the differences are due to rounded constants).
    - Seider base-cost coefficients (c0, c1, c2) as coded in BioSTEAM (CE 567); Turton K, B, C and
      F_M: from the source project. Not checked against the books; IDAES (3rd ed., CE 500) lists a
      different floating-head intercept (11.9052); see "Coefficient provenance" below.
  Coefficient provenance (PR review, open items for the maintainer; no coefficient was changed):
    - floating head: BioSTEAM 12.0310 (CE 567) vs IDAES SSLW 11.9052 (CE 500) differ by ln(567/500) = 0.1258, so
      the costs agree to 4.9e-5 after the cost-index shift (tested; "idaes_sslw_cb" above);
    - fixed head: slope -0.9228 (BioSTEAM; a Penn lecture snippet, Lewin FOCAPD 2004, agrees) vs -0.8228 (IDAES);
      unconfirmed which is right, the IDAES value is probably a typo;
    - U-tube: intercepts differ by 0.1658, not the index shift 0.1258 (4.1 % cost difference); unresolved;
    - Turton K, C, B confirmed only by a third-party worked example (eqcost, above), not by the book.
* seider_values / turton_values: C_BM and dC_BM/dA from an independent re-implementation below in
  the ft^2 domain, with the derivative by complex-step differentiation (no truncation error).
* crf: capital recovery factor from 50-digit mpmath.
Run: python library/tests/generators/hx_capital_cost.py
"""
import cmath
import itertools
import math

import mpmath as mp

from _common import write

mp.mp.dps = 50
M2_TO_FT2 = 10.7639104

SEIDER_CB = {"floating_head": (12.0310, -0.8709, 0.09005), "fixed_head": (11.4185, -0.9228, 0.09861),
             "u_tube": (11.5510, -0.9186, 0.09790)}
SEIDER_FM = [["CS", "CS", 0.0, 0.0], ["CS", "brass", 1.08, 0.05], ["CS", "SS", 1.75, 0.13],
             ["CS", "Monel", 2.1, 0.13], ["CS", "Ti", 5.2, 0.16], ["CS", "CrMo", 1.55, 0.05],
             ["CrMo", "CrMo", 1.7, 0.07], ["SS", "SS", 2.7, 0.07], ["Monel", "Monel", 3.3, 0.08],
             ["Ti", "Ti", 9.6, 0.06]]
SEIDER_FP = [0.9803, 0.0180, 0.0017]
SEIDER_FL = {8: 1.25, 12: 1.12, 16: 1.05, 20: 1.00}
TURTON_K, TURTON_B, TURTON_C = (4.8306, -0.8509, 0.3187), (1.63, 1.66), (0.03881, -0.11272, 0.08183)
TURTON_FM = {"CS": 1.00, "SS": 2.73, "Monel": 3.73, "Ti": 11.4}


def seider(a_m2, typ, a, b, fl, p_psig, f_bm=3.17):
    """C_BM(A) for complex a_m2 (>= 150 ft^2 branch only; clamp handled by the caller)."""
    af = a_m2 * M2_TO_FT2
    c0, c1, c2 = SEIDER_CB[typ]
    L = cmath.log(af)
    cb = cmath.exp(c0 + c1 * L + c2 * L * L)
    x = p_psig / 100.0
    fp = SEIDER_FP[0] + SEIDER_FP[1] * x + SEIDER_FP[2] * x * x
    fm = a + (af / 100.0) ** b
    return cb * (f_bm + fl * fp * fm - 1.0)


def turton(a_m2, fm, p_barg):
    x = cmath.log10(a_m2)
    cp0 = 10 ** (TURTON_K[0] + TURTON_K[1] * x + TURTON_K[2] * x * x)
    y = math.log10(p_barg) if p_barg >= 5 else 0.0
    fp = 1.0 if p_barg < 5 else 10 ** (TURTON_C[0] + TURTON_C[1] * y + TURTON_C[2] * y * y)
    return cp0 * (TURTON_B[0] + TURTON_B[1] * fm * fp)


def cstep(f, a, h=1e-30):
    v = f(complex(a, h))
    return v.real, v.imag / h


seider_values = []
for typ, (shell, tube, fa, fb), ft, p in itertools.product(
        SEIDER_CB, [SEIDER_FM[0], SEIDER_FM[4], SEIDER_FM[7]], (8, 20), (0.0, 300.0)):
    for area in (14.0, 20.0, 100.0, 500.0, 1115.0, 1500.0):   # 150 ft^2 = 13.9 m^2; 12000 ft^2 = 1114.8 m^2
        c, dc = cstep(lambda z: seider(z, typ, fa, fb, SEIDER_FL[ft], p), area)
        seider_values.append({"type": typ, "shell": shell, "tube": tube, "tube_ft": ft, "p_psig": p,
                              "area_m2": area, "cost": c, "dcost_da": dc})
# below the lower limit: cost of 150 ft^2 with zero slope
lo_area = 150.0 / M2_TO_FT2
for area in (1.0, 5.0, 13.0):
    for typ in SEIDER_CB:
        c, _ = cstep(lambda z: seider(z, typ, 5.2, 0.16, 1.12, 100.0), lo_area)
        seider_values.append({"type": typ, "shell": "CS", "tube": "Ti", "tube_ft": 12, "p_psig": 100.0,
                              "area_m2": area, "cost": c, "dcost_da": 0.0, "clamped": True})

turton_values = []
for (mat, fm), p in itertools.product(TURTON_FM.items(), (0.0, 5.0, 20.0, 100.0)):
    for area in (10.5, 25.0, 100.0, 400.0, 999.0, 1500.0):
        c, dc = cstep(lambda z: turton(z, fm, p), area)
        turton_values.append({"material": mat, "p_barg": p, "area_m2": area, "cost": c, "dcost_da": dc})
for area in (0.5, 5.0, 9.99):
    c, _ = cstep(lambda z: turton(z, 1.0, 20.0), 10.0)
    turton_values.append({"material": "CS", "p_barg": 20.0, "area_m2": area, "cost": c, "dcost_da": 0.0,
                          "clamped": True})

crf = []
for i, n in [(0.1, 10), (0.08, 15), (0.15, 20), (0.05, 25.5), (0.0, 10), (1e-9, 10), (1e-12, 20), (-0.02, 10),
             (2.0, 3)]:
    if i == 0:
        v = 1.0 / n
    else:
        g = (1 + mp.mpf(i)) ** n
        v = float(mp.mpf(i) * g / (g - 1))
    crf.append({"rate": i, "years": n, "crf": v})

published = {
    "seider_fm": SEIDER_FM, "seider_fp_coeffs": SEIDER_FP, "seider_fl": {str(k): v for k, v in SEIDER_FL.items()},
    "turton_eqcost": {"area_m2": 100.0, "p_barg": 20.0, "cp0": 25323.0, "fp": 1.07321, "cbm": 86393.0},
    # IDAES SSLW base-cost coefficients (3rd ed., CE 500) as read by the PR reviewer from idaes/models/costing/SSLW.py
    # (GitHub main) and its docs page; (c0, c1, c2) for A in ft^2. NOT re-read by this generator's author.
    "idaes_sslw_cb": {"ce_basis": 500, "floating_head": [11.9052, -0.8709, 0.09005],
                      "fixed_head": [11.2927, -0.8228, 0.09861], "u_tube": [11.3852, -0.9186, 0.09790]},
}
write("hx_capital_cost", "library/tests/generators/hx_capital_cost.py",
      {"derivative": "complex step h=1e-30", "crf": "mpmath 50 digits", "seider_ce_basis": 567,
       "turton_cepci_basis": 397, "seider_area_range_ft2": [150, 12000], "turton_area_range_m2": [10, 1000]},
      "Areas m^2, costs USD of the correlation's index year (Seider CE 567, Turton CEPCI 397), dcost_da USD/m^2, "
      "pressures psig (Seider) / barg (Turton), crf 1/yr.",
      {"published": published, "seider_values": seider_values, "turton_values": turton_values, "crf": crf},
      mpmath=mp.__version__)
