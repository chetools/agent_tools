"""workgroup_reduce Python reference vs independent methods (math.fsum exact sum, numpy max/min)."""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from gpu import workgroup_reduce as wr  # noqa: E402


@pytest.mark.parametrize("n", [1, 2, 255, 256, 1023, 1024, 1025, 65536, 262145, 300001])
def test_matches_fsum_and_numpy(n):
    x = wr.hash_values(n, seed=n, bias=0)
    ex = wr.exact_stats(x)
    # float64 two-pass tree: error bounded by ~(30) * 2^-53 * sum|x|; allow 1e-12 relative to sum|x|
    assert abs(wr.reduce(x, "sum") - ex["sum"]) <= 1e-12 * ex["sum_abs"]
    assert wr.reduce(x, "max") == np.max(x)
    assert wr.reduce(x, "min") == np.min(x)
    assert wr.reduce(x, "maxabs") == np.max(np.abs(x))


def test_hard_case_1e7_mixed_sign_wide_range():
    x = wr.hash_values(10_000_000, seed=6)
    ex = wr.exact_stats(x)
    # heavy cancellation expected: |sum| much smaller than sum|x|
    assert abs(ex["sum"]) < 1e-2 * ex["sum_abs"]
    assert abs(wr.reduce(x, "sum") - ex["sum"]) <= 1e-12 * ex["sum_abs"]
    assert wr.reduce(x, "max") == ex["max"] and wr.reduce(x, "min") == ex["min"]


def test_all_negative_max_is_negative():
    x = -np.arange(1, 5000, dtype=np.float64)
    assert wr.reduce(x, "max") == -1.0 and wr.reduce(x, "min") == -4999.0 and wr.reduce(x, "maxabs") == 4999.0


def test_hash_values_exact_in_float32():
    x = wr.hash_values(100000, seed=9)
    assert np.array_equal(x.astype(np.float32).astype(np.float64), x)


def test_numpy_order_differs_from_sequential_but_within_f32_bound():
    # documents the order dependence: float32 sequential vs pairwise differ, both within the f32 bound
    x = wr.hash_values(200000, seed=11).astype(np.float32)
    seq = np.float32(0)
    for v in x[:20000]:
        seq += v
    ex = math.fsum(x[:20000].astype(np.float64).tolist())
    sabs = math.fsum(np.abs(x[:20000]).astype(np.float64).tolist())
    assert abs(float(seq) - ex) <= 20000 * 2.0 ** -24 * sabs
    assert abs(float(np.sum(x[:20000])) - ex) <= 1e-5 * sabs


def test_invalid_args():
    with pytest.raises(ValueError):
        wr.reduce([1.0], "prod")
    with pytest.raises(ValueError):
        wr.reduce([], "sum")
