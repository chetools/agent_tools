"""Tests for random.mulberry32: streams equal to Ettinger's JavaScript (reference from node)."""
import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.random.mulberry32 import Mulberry32

TOL_MEAN_UNIFORM, TOL_VAR_UNIFORM = 5e-3, 2e-3     # N = 1e5, fixed seed


def test_streams_are_bit_identical():
    n = 0
    for seed, c in load_reference("mulberry32")["cases"]["streams"].items():
        s = int(seed)
        a, b, d, e = (Mulberry32(s) for _ in range(4))
        for i, u in enumerate(c["u32"]):
            assert a.next_u32() == u and b.random() == c["random"][i]
            assert d.below(10) == c["below_10"][i] and e.below(1000) == c["below_1000"][i]
            n += 1
    # existing project value: heat-exchanger-network test_hen.py::test_mulberry32_reference_values
    g = Mulberry32(1)
    assert [g.next_u32() for _ in range(4)] == [2693262067, 11749833, 2265367787, 4213581821]
    assert Mulberry32(2**32 + 5).a == 5   # seed masked to 32 bits
    print(f"VERIFY mulberry32 py_streams mismatches_vs_stored_node_values=0 values_checked={n}")


def test_statistics():
    g = Mulberry32(2024)
    u = np.array([g.random() for _ in range(100_000)])
    assert u.min() >= 0.0 and u.max() < 1.0
    print(f"VERIFY mulberry32 py_uniform_1e5 mean_minus_half={u.mean() - 0.5:e} var_minus_1_12={u.var() - 1/12:e}")
    assert abs(u.mean() - 0.5) < TOL_MEAN_UNIFORM and abs(u.var() - 1 / 12) < TOL_VAR_UNIFORM


def test_python_vs_rust_example():
    py, rs = python_example("mulberry32"), rust_example("mulberry32")
    assert py == rs
    print("VERIFY mulberry32 python_vs_rust identical=1")
