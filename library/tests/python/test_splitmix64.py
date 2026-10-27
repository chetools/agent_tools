"""Tests for random.splitmix64: bit-exact streams from the published algorithm, statistics, Rust."""
import numpy as np
import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.random.splitmix64 import SplitMix64

TOL_NORMAL = 1e-14        # Box-Muller vs reference / Rust (libm rounding only), relative
TOL_MEAN_UNIFORM, TOL_VAR_UNIFORM = 5e-3, 2e-3     # N = 1e5, fixed seed; 5-8 standard errors
TOL_MEAN_NORMAL, TOL_VAR_NORMAL = 2e-2, 3e-2


def test_streams_are_bit_identical():
    r = load_reference("splitmix64")["cases"]
    n = 0
    for seed, c in r["streams"].items():
        g, h = SplitMix64(int(seed)), SplitMix64(int(seed))
        for u, f in zip(c["u64"], c["uniform"]):
            assert g.next_u64() == u and h.uniform() == f
            n += 1
    ss = r["seed_stream"]
    g = SplitMix64.from_seed_stream(ss["seed"], ss["stream"])
    assert g.s == ss["state"] and [g.next_u64() for _ in ss["u64"]] == ss["u64"]
    print(f"VERIFY splitmix64 py_streams u64_and_uniform_mismatches=0 values_checked={n}")


def test_normals_and_permutations():
    r = load_reference("splitmix64")["cases"]
    worst = 0.0
    for seed, v in r["normal"].items():
        g = SplitMix64(int(seed))
        for x in v:
            worst = max(worst, abs(g.normal() - x) / abs(x))
    print(f"VERIFY splitmix64 py_normal max_rel_diff_vs_reference={worst:e}")
    assert worst < TOL_NORMAL
    for key, want in r["permutation"].items():
        s, n = key.split(":")
        assert SplitMix64(int(s)).permutation(int(n)) == want
    assert SplitMix64(1).permutation(0) == []
    print(f"VERIFY splitmix64 py_permutation mismatches=0 cases={len(r['permutation'])}")


def test_statistics():
    g = SplitMix64(2024)
    u = np.array([g.uniform() for _ in range(100_000)])
    assert u.min() >= 0.0 and u.max() < 1.0
    print(f"VERIFY splitmix64 py_uniform_1e5 mean_minus_half={u.mean() - 0.5:e} var_minus_1_12={u.var() - 1/12:e}")
    assert abs(u.mean() - 0.5) < TOL_MEAN_UNIFORM and abs(u.var() - 1 / 12) < TOL_VAR_UNIFORM
    z = np.array([g.normal() for _ in range(100_000)])
    print(f"VERIFY splitmix64 py_normal_1e5 mean={z.mean():e} var_minus_1={z.var() - 1:e}")
    assert abs(z.mean()) < TOL_MEAN_NORMAL and abs(z.var() - 1) < TOL_VAR_NORMAL


def test_python_vs_rust_example():
    py, rs = python_example("splitmix64"), rust_example("splitmix64")
    assert py["u64"] == rs["u64"] and py["permutation"] == rs["permutation"]
    assert py["uniform"] == rs["uniform"]
    worst = max(abs(a - b) / abs(b) for a, b in zip(py["normal"], rs["normal"]))
    print(f"VERIFY splitmix64 python_vs_rust u64_uniform_perm_identical=1 normal_max_rel_diff={worst:e}")
    assert worst < TOL_NORMAL
