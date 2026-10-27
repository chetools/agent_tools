"""Tests for process.lmtd: Python vs 50-digit mpmath reference data, finite differences, and Rust."""
import math

import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.process.lmtd import LMTD_SERIES_THRESHOLD, lmtd

TOL_L = 1e-11               # L relative (measured worst case 5e-13, just above the switch)
TOL_D = 1e-11               # derivative absolute away from the switch (measured 5e-12)
TOL_D_JUST_ABOVE = 1e-8     # for 1e-4 <= rel diff < 1e-3: cancellation in the closed form (measured 4.9e-9)
TOL_FD = 5e-7               # central finite difference, absolute
TOL_RUST = 1e-14            # Python vs Rust example, relative


@pytest.mark.parametrize("group", ["ordinary_and_extreme", "switch_region"])
def test_against_mpmath(group):
    el = ed = 0.0
    for c in load_reference("lmtd")["cases"][group]:
        L, La, Lb = lmtd(c["a"], c["b"])
        e_l = abs(L - c["L"]) / c["L"]
        e_d = max(abs(La - c["dLda"]), abs(Lb - c["dLdb"]))
        tol_d = TOL_D_JUST_ABOVE if LMTD_SERIES_THRESHOLD <= c["rel_diff"] < 1e-3 else TOL_D
        assert e_l < TOL_L and e_d < tol_d, (c, e_l, e_d)
        el, ed = max(el, e_l), max(ed, e_d)
    print(f"VERIFY lmtd py_{group} max_rel_err_L={el:e} max_abs_err_dL={ed:e}")


def test_derivatives_vs_finite_differences_both_sides_of_switch():
    worst = 0.0
    for rel in (0.0, 1e-6, 5e-5, 9.9e-5, 1.01e-4, 2e-4, 1e-3, 0.5, 1.5):
        a, b = 20.0 * (1 + rel / 2), 20.0 * (1 - rel / 2)
        _, La, Lb = lmtd(a, b)
        h = 1e-4
        fa = (lmtd(a + h, b)[0] - lmtd(a - h, b)[0]) / (2 * h)
        fb = (lmtd(a, b + h)[0] - lmtd(a, b - h)[0]) / (2 * h)
        worst = max(worst, abs(La - fa), abs(Lb - fb))
        assert abs(La - fa) < TOL_FD and abs(Lb - fb) < TOL_FD, rel
    print(f"VERIFY lmtd py_fd_both_sides max_abs_diff={worst:e}")


def test_invalid_inputs_give_nan():
    for a, b in [(-1.0, 5.0), (5.0, 0.0), (math.nan, 1.0), (math.inf, 1.0)]:
        assert all(math.isnan(v) for v in lmtd(a, b)), (a, b)
    assert lmtd(30.0, 10.0)[0] == lmtd(10.0, 30.0)[0]
    print("VERIFY lmtd py_invalid_inputs nan_cases=4")


def test_python_vs_rust_example():
    py, rs = python_example("lmtd"), rust_example("lmtd")
    assert py["invalid_is_nan"] and rs["invalid_is_nan"]
    worst = max(abs(p - r) / max(abs(r), 1.0) for k in ("normal", "near_equal") for p, r in zip(py[k], rs[k]))
    print(f"VERIFY lmtd python_vs_rust max_rel_diff={worst:e}")
    assert worst < TOL_RUST
