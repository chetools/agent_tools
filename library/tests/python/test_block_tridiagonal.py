"""Tests for linalg.block_tridiagonal: Python vs numpy/solve_banded/Thomas reference data, Dual derivative, Rust."""
import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.autodiff.dual import Dual
from agent_tools.linalg.block_tridiagonal import block_tridiag_factor, block_tridiag_solve

TOL_DOMINANT = 1e-12      # dominant systems, solution vs numpy.linalg.solve, relative to max|x|
TOL_COND_FACTOR = 10.0    # weak / zero-diagonal-block systems: error <= factor * eps * cond
TOL_BACKWARD = 1e-14      # ||Ax-b|| / (||A|| ||x|| + ||b||), inf norms
TOL_THOMAS = 1e-13        # m = 1 vs classical Thomas
TOL_DUAL = 1e-11          # dx vs analytic, relative to max|dx|
TOL_RUST = 1e-13          # Python vs Rust example, relative (same operation order; expect ~0)
EPS = np.finfo(float).eps


def rel(x, ref):
    return np.max(np.abs(np.asarray(x) - ref)) / np.max(np.abs(ref))


def dense(c):
    """Assemble the dense matrix from the stored blocks (test-side only)."""
    m, nb = c["m"], c["nb"]
    L = np.array(c["lower"]).reshape(nb - 1, m, m)
    D = np.array(c["diag"]).reshape(nb, m, m)
    U = np.array(c["upper"]).reshape(nb - 1, m, m)
    A = np.zeros((m * nb, m * nb))
    for i in range(nb):
        A[i*m:(i+1)*m, i*m:(i+1)*m] = D[i]
        if i + 1 < nb:
            A[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = U[i]
            A[(i+1)*m:(i+2)*m, i*m:(i+1)*m] = L[i]
    return A


def run_case(c):
    A = dense(c)
    f = block_tridiag_factor(c["lower"], c["diag"], c["upper"], c["m"], c["nb"])
    assert f.success, f.message
    ex = eb = 0.0
    for k, (b, xr) in enumerate(zip(c["b"], c["x"])):
        x = np.array(block_tridiag_solve(f, b).x)
        ex = max(ex, rel(x, np.array(xr)))
        eb = max(eb, np.max(np.abs(A @ x - b)) / (np.linalg.norm(A, np.inf) * np.max(np.abs(x)) + np.max(np.abs(b))))
        if "x_thomas" in c:
            assert rel(x, np.array(c["x_thomas"][k])) < TOL_THOMAS
    return ex, eb, f.growth


def test_dominant_systems_and_thomas():
    for c in load_reference("block_tridiagonal")["cases"]["dominant"]:
        ex, eb, g = run_case(c)
        n = c["name"]
        print(f"VERIFY block_tridiagonal py_{n} max_rel_err_x_vs_numpy={ex:e}")
        print(f"VERIFY block_tridiagonal py_{n} max_backward_error={eb:e}")
        print(f"VERIFY block_tridiagonal py_{n} growth_max_norm_C={g:e}")
        assert ex < TOL_DOMINANT and eb < TOL_BACKWARD and g < 1.0


def test_hard_weak_dominance_and_in_block_pivoting():
    cases = load_reference("block_tridiagonal")["cases"]
    for c in cases["weak"] + cases["pivot"]:
        ex, eb, g = run_case(c)
        n = c["name"]
        print(f"VERIFY block_tridiagonal py_{n} max_rel_err_x_vs_numpy={ex:e}")
        print(f"VERIFY block_tridiagonal py_{n} max_backward_error={eb:e}")
        print(f"VERIFY block_tridiagonal py_{n} growth_max_norm_C={g:e}")
        assert ex < TOL_COND_FACTOR * EPS * c["cond"] and eb < TOL_BACKWARD


def test_singular_diagonal_block_is_reported():
    cases = load_reference("block_tridiagonal")["cases"]["fail"]
    for c in cases:
        f = block_tridiag_factor(c["lower"], c["diag"], c["upper"], c["m"], c["nb"])
        assert not f.success
        assert f"diagonal block {c['fail_block']}" in f.message and "singular" in f.message
        s = block_tridiag_solve(f, [1.0] * (c["m"] * c["nb"]))
        assert not s.success and s.x == []
    print(f"VERIFY block_tridiagonal py_fail cases_reported={len(cases)}")


def test_dual_solution_derivative():
    for c in load_reference("block_tridiagonal")["cases"]["param"]:
        p = Dual(c["p0"], 1.0)
        lin = lambda k: [a + b * p for a, b in zip(c[k + "0"], c[k + "1"])]
        f = block_tridiag_factor(lin("L"), lin("D"), lin("U"), c["m"], c["nb"])
        assert f.success
        s = block_tridiag_solve(f, lin("b"))
        ex = rel([z.v for z in s.x], np.array(c["x"]))
        edx = rel([z.d for z in s.x], np.array(c["dx"]))
        tag = f"m{c['m']}_nb{c['nb']}"
        print(f"VERIFY block_tridiagonal py_dual_{tag} rel_err_x={ex:e}")
        print(f"VERIFY block_tridiagonal py_dual_{tag} rel_err_dx_vs_analytic={edx:e}")
        assert ex < TOL_DOMINANT and edx < TOL_DUAL


def test_python_vs_rust_example():
    py, rs = python_example("block_tridiagonal"), rust_example("block_tridiagonal")
    assert py["success"] and rs["success"]
    assert py["bad_success"] is False and rs["bad_success"] is False
    assert py["bad_message"] == rs["bad_message"]
    ex = max(rel(py["x1"], np.array(rs["x1"])), rel(py["x2"], np.array(rs["x2"])))
    eg = abs(py["growth"] - rs["growth"]) / rs["growth"]
    print(f"VERIFY block_tridiagonal python_vs_rust rel_diff_x={ex:e}")
    print(f"VERIFY block_tridiagonal python_vs_rust rel_diff_growth={eg:e}")
    assert ex < TOL_RUST and eg < TOL_RUST
