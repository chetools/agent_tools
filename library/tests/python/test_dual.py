"""Tests for autodiff.dual: Python Dual vs analytic derivatives, central differences, and Rust."""
import numpy as np
import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.autodiff.dual import Dual, absval, cos, deriv, exp, ln, powi, sin, sqrt, val

TOL_ANALYTIC = 1e-12   # AD vs hand-derived closed forms, scaled by max(|ref|, 1)
TOL_FD = 1e-6          # AD vs central differences, h = 1e-5
TOL_RUST = 1e-12       # Python vs Rust example (same algorithm; libm rounding only)
H = 1e-5

FUNCS = {
    "exp_sin": lambda x: exp(sin(x)),
    "sqrt_ln": lambda x: sqrt(x) * ln(x),
    "powi_poly": lambda x: powi(x, 5) - 3.0 * powi(x, -2),
    "abs_cos": lambda x: x * absval(cos(x)),
    "rational": lambda x: (x * x + 1.0) / (x * x * x + 2.0),
    "arrhenius": lambda x: exp(x / (1.0 + x / 20.0)),
}


def err(a, b):
    return abs(a - b) / max(abs(b), 1.0)


def d1(f, x):
    return f(Dual(x, 1.0)).d


@pytest.mark.parametrize("name", list(FUNCS))
def test_derivatives(name):
    c = load_reference("dual")["cases"][name]
    f = FUNCS[name]
    e = dict(f=0.0, d1=0.0, d2=0.0, fd1=0.0, fd2=0.0)
    for x, fr, dr, d2r in zip(c["x"], c["f"], c["df"], c["d2f"]):
        a = f(Dual(x, 1.0))
        n = f(Dual(Dual(x, 1.0), Dual(1.0, 0.0)))
        e["f"] = max(e["f"], err(a.v, fr))
        e["d1"] = max(e["d1"], err(a.d, dr))
        e["d2"] = max(e["d2"], err(n.d.d, d2r))
        e["fd1"] = max(e["fd1"], err(a.d, (f(x + H) - f(x - H)) / (2 * H)))
        e["fd2"] = max(e["fd2"], err(n.d.d, (d1(f, x + H) - d1(f, x - H)) / (2 * H)))
    for k, label in [("f", "value_vs_analytic"), ("d1", "d1_vs_analytic"), ("d2", "d2_nested_vs_analytic"),
                     ("fd1", "d1_vs_central_fd"), ("fd2", "d2_vs_central_fd")]:
        print(f"VERIFY dual {name} py_max_err_{label}={e[k]:e}")
    assert max(e["f"], e["d1"], e["d2"]) < TOL_ANALYTIC
    assert max(e["fd1"], e["fd2"]) < TOL_FD


def test_operators_with_floats_and_edge_cases():
    x = Dual(2.0, 1.0)
    assert (3.0 + x).d == 1.0 and (3.0 - x).d == -1.0 and (3.0 * x).d == 3.0
    r = 1.0 / x                       # d/dx 1/x = -1/x^2
    assert r.v == 0.5 and r.d == -0.25
    assert (x ** 3).d == 12.0
    neg = absval(Dual(-2.0, 1.0))
    assert (neg.v, neg.d) == (2.0, -1.0)
    z = powi(Dual(0.0, 1.0), 0)
    assert (z.v, z.d) == (1.0, 0.0)
    assert deriv(3.0) == 0.0 and val(Dual(Dual(1.0, 2.0), Dual(3.0, 4.0))) == 1.0
    assert deriv(Dual(Dual(1.0, 2.0), Dual(3.0, 4.0))) == 3.0
    with pytest.raises(TypeError):
        x ** 0.5
    print("VERIFY dual py_edge_cases checks_passed=1")


def test_python_vs_rust_example():
    py, rs = python_example("dual"), rust_example("dual")
    assert set(py) == set(rs)
    worst = max(err(py[k], rs[k]) for k in py)
    print(f"VERIFY dual python_vs_rust max_rel_diff={worst:e}")
    assert worst < TOL_RUST
