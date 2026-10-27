"""Tests for process.hx_capital_cost: published tables, complex-step reference values, finite differences, Rust."""
import math

import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.process.hx_capital_cost import (
    M2_TO_FT2, SEIDER_AMAX_FT2, SEIDER_AMIN_FT2, SEIDER_CB, SEIDER_CE_BASIS, SEIDER_PMIN_PSIG, TURTON_AMAX_M2, capital_recovery_factor,
    seider_bare_module_cost, seider_length_factor, seider_material_coeffs, seider_pressure_below_range, seider_pressure_factor,
    turton_bare_module_cost, turton_material_factor, turton_pressure_factor)

TOL_COST = 1e-12     # cost vs independent re-implementation, relative
TOL_DERIV = 1e-11    # dC/dA vs complex step, relative to max(|dC/dA|, 1e-3)
TOL_EQCOST = 3e-3    # Turton eqcost published case (source project's bound; eqcost rounds constants)
TOL_FD = 1e-6        # central finite difference, relative
TOL_CRF = 1e-13      # capital recovery factor vs 50-digit values, relative
TOL_INDEX_SHIFT = 6e-5   # floating head, BioSTEAM (CE 567) vs IDAES SSLW (CE 500) after the index shift (measured 4.9e-5)
TOL_RUST = 1e-13     # Python vs Rust example, relative

REF = load_reference("hx_capital_cost")["cases"]


def rel(a, b, floor=1.0):
    return abs(a - b) / max(abs(b), floor)


def seider(c, area):
    a, b = seider_material_coeffs(c["shell"], c["tube"])
    return seider_bare_module_cost(area, c["type"], a, b, seider_length_factor(c["tube_ft"]),
                                   seider_pressure_factor(c["p_psig"]))


def turton(c, area):
    return turton_bare_module_cost(area, turton_material_factor(c["material"]), turton_pressure_factor(c["p_barg"]))


def test_published_tables():
    p = REF["published"]
    for shell, tube, a, b in p["seider_fm"]:
        assert seider_material_coeffs(shell, tube) == (a, b)
    with pytest.raises(ValueError):
        seider_material_coeffs("Ti", "CS")
    c = p["seider_fp_coeffs"]
    worst = max(abs(seider_pressure_factor(P) - (c[0] + c[1] * P / 100 + c[2] * (P / 100) ** 2))
                for P in (0.0, 100.0, 600.0, 2000.0))
    for k, v in p["seider_fl"].items():
        assert seider_length_factor(int(k)) == v
    for bad in (lambda: seider_length_factor(13), lambda: turton_material_factor("brass")):
        with pytest.raises(ValueError):
            bad()
    print(f"VERIFY hx_capital_cost py_published_tables max_abs_err_FP={worst:e} fm_rows={len(p['seider_fm'])}")
    assert worst < 1e-15


def test_turton_eqcost_case():
    e = REF["published"]["turton_eqcost"]
    fp = turton_pressure_factor(e["p_barg"])
    res = turton_bare_module_cost(e["area_m2"], 1.0, fp)
    cp0 = 10 ** (4.8306 - 0.8509 * 2 + 0.3187 * 4)
    errs = (rel(fp, e["fp"]), rel(cp0, e["cp0"]), rel(res.cost_usd, e["cbm"]))
    print("VERIFY hx_capital_cost py_turton_eqcost rel_err_FP=%e rel_err_Cp0=%e rel_err_CBM=%e" % errs)
    assert max(errs) < TOL_EQCOST


def _values(name, fn, flag_above):
    ec = ed = 0.0
    for c in REF[name]:
        area = c["area_m2"]
        r = fn(c, area)
        e_c, e_d = rel(r.cost_usd, c["cost"]), rel(r.dcost_da, c["dcost_da"], 1e-3)
        assert e_c < TOL_COST and e_d < TOL_DERIV, (c, e_c, e_d)
        assert r.clamped_below == ("clamped" in c), c
        assert r.above_range == flag_above(area), c
        ec, ed = max(ec, e_c), max(ed, e_d)
    print(f"VERIFY hx_capital_cost py_{name} n={len(REF[name])} max_rel_err_cost={ec:e} max_rel_err_dcost={ed:e}")


def test_seider_values_and_ranges():
    _values("seider_values", seider, lambda a: a * M2_TO_FT2 > SEIDER_AMAX_FT2)


def test_turton_values_and_ranges():
    _values("turton_values", turton, lambda a: a > TURTON_AMAX_M2)


def test_derivative_vs_finite_differences_both_sides_of_clamp():
    worst = 0.0
    lo_s = SEIDER_AMIN_FT2 / M2_TO_FT2
    a, b = seider_material_coeffs("CS", "Ti")
    for typ in ("floating_head", "fixed_head", "u_tube"):
        def cost(area, typ=typ):
            return seider_bare_module_cost(area, typ, a, b, 1.12, 1.2)
        for area in (0.5 * lo_s, 0.97 * lo_s, 1.03 * lo_s, 20.0, 150.0, 1115.0, 2000.0):
            h = 1e-4 * area
            fd = (cost(area + h).cost_usd - cost(area - h).cost_usd) / (2 * h)
            e = abs(cost(area).dcost_da - fd) / max(abs(fd), 1.0)
            worst = max(worst, e)
            assert e < TOL_FD, (typ, area)
        assert rel(cost(lo_s * (1 - 1e-12)).cost_usd, cost(lo_s).cost_usd) < 1e-9     # continuous at the clamp
    fp = turton_pressure_factor(20.0)

    def tcost(area):
        return turton_bare_module_cost(area, 2.73, fp)
    for area in (5.0, 9.7, 10.3, 100.0, 1000.0, 3000.0):
        h = 1e-4 * area
        fd = (tcost(area + h).cost_usd - tcost(area - h).cost_usd) / (2 * h)
        e = abs(tcost(area).dcost_da - fd) / max(abs(fd), 1.0)
        worst = max(worst, e)
        assert e < TOL_FD, area
    assert rel(tcost(10.0 * (1 - 1e-12)).cost_usd, tcost(10.0).cost_usd) < 1e-9
    print(f"VERIFY hx_capital_cost py_fd_both_sides max_rel_diff={worst:e}")


def test_capital_recovery_factor_vs_mpmath():
    worst = 0.0
    for c in REF["crf"]:
        e = rel(capital_recovery_factor(c["rate"], c["years"]), c["crf"], 1e-300)
        worst = max(worst, e)
        assert e < TOL_CRF, (c, e)
    for i, n in [(-1.0, 10.0), (0.1, 0.0), (0.1, -3.0), (math.nan, 5.0)]:
        assert math.isnan(capital_recovery_factor(i, n))
    assert math.isnan(seider_bare_module_cost(0.0, "u_tube", 0.0, 0.0, 1.0, 1.0).cost_usd)
    assert math.isnan(turton_bare_module_cost(-1.0, 1.0, 1.0).dcost_da)
    print(f"VERIFY hx_capital_cost py_crf max_rel_err={worst:e}")


def test_python_vs_rust_example():
    py, rs = python_example("hx_capital_cost"), rust_example("hx_capital_cost")
    assert py["small_clamped"] and rs["small_clamped"]
    worst = max(rel(py[k], rs[k]) for k in ("seider_cost", "seider_dcost_da", "turton_cost", "turton_dcost_da", "crf"))
    print(f"VERIFY hx_capital_cost python_vs_rust max_rel_diff={worst:e}")
    assert worst < TOL_RUST


def test_unknown_exchanger_type_raises_value_error():
    with pytest.raises(ValueError):
        seider_bare_module_cost(100.0, "kettle", 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):          # also for an invalid area: the type is checked first
        seider_bare_module_cost(-1.0, "kettle", 0.0, 0.0, 1.0, 1.0)


def test_seider_pressure_below_range_flag():
    assert SEIDER_PMIN_PSIG == 100.0
    assert [seider_pressure_below_range(p) for p in (0.0, 99.999, 100.0, 300.0, 2000.0)] == [True, True, False, False, False]
    assert seider_pressure_factor(50.0) == 0.9803 + 0.018 * 0.5 + 0.0017 * 0.25     # values unchanged below range


def test_floating_head_intercepts_differ_by_cost_index_shift():
    """BioSTEAM 12.0310 (CE 567) vs IDAES SSLW 11.9052 (CE 500): the difference is ln(567/500), so the
    costs agree after the index shift. (Fixed-head slope and U-tube intercept differences are open items.)"""
    idaes = REF["published"]["idaes_sslw_cb"]
    c0, c1, c2 = SEIDER_CB["floating_head"]
    i0, i1, i2 = idaes["floating_head"]
    assert (c1, c2) == (i1, i2)
    shift = math.log(SEIDER_CE_BASIS / idaes["ce_basis"])
    worst = 0.0
    for af in (150.0, 1000.0, 12000.0):
        lg = math.log(af)
        bio = math.exp(c0 + c1 * lg + c2 * lg * lg)
        ide = math.exp(i0 + i1 * lg + i2 * lg * lg) * SEIDER_CE_BASIS / idaes["ce_basis"]
        worst = max(worst, abs(bio / ide - 1.0))
    print(f"VERIFY hx_capital_cost py_floating_head_index_shift intercept_diff_minus_ln_ratio={c0 - i0 - shift:e} "
          f"max_rel_cost_diff={worst:e}")
    assert abs(c0 - i0 - shift) < 1e-4 and worst < TOL_INDEX_SHIFT
