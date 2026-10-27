"""histogram_range Python reference vs independent methods (numpy.histogram, numpy.percentile)."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from gpu import histogram_range as hr  # noqa: E402


def _random_field(seed, n=200000):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mean=0.0, sigma=1.5, size=n) * rng.choice([-1.0, 1.0], size=n)
    return x.astype(np.float32)


@pytest.mark.parametrize("nbins", [16, 256, 2048])
def test_histogram_matches_numpy_histogram(nbins):
    x = _random_field(1)
    r = hr.histogram_range(x, nbins=nbins)
    v = np.abs(x.astype(np.float64))
    ref, _ = np.histogram(np.sqrt(v / v.max()), bins=nbins, range=(0.0, 1.0))
    # exact equality of integer counts; elements on a bin edge (ambiguous) are the only allowed discrepancy
    moved = int(np.abs(np.array(r["hist"]) - ref).sum() // 2)
    assert moved <= hr.ambiguous_count(x, nbins=nbins, margin=1e-12) + 0  # i.e. 0 unless an exact edge hit
    assert sum(r["hist"]) == r["count"] == x.size


@pytest.mark.parametrize("pct", [0.01, 0.5, 0.9, 0.995, 0.999, 1.0])
@pytest.mark.parametrize("nbins", [64, 2048])
def test_hi_within_one_bin_width_of_numpy_percentile(pct, nbins):
    x = _random_field(2)
    r = hr.histogram_range(x, nbins=nbins, pct=pct)
    p = np.quantile(np.abs(x.astype(np.float64)), pct, method="inverted_cdf")
    err = r["hi"] - p
    assert err >= -1e-12 * r["max"]
    assert err <= hr.bin_width(r["max"], nbins, r["bin"]) * (1 + 1e-12)
    assert r["bin"] >= 0 and r["max"] == np.max(np.abs(x))


def test_mask_and_nonfinite_excluded_match_numpy_on_subset():
    x, mask = hr.designed_field(50000, 128)
    r = hr.histogram_range(x, mask, nbins=128, pct=0.9)
    keep = np.isfinite(x) & (mask == 0)
    v = np.abs(x[keep].astype(np.float64))
    assert r["count"] == v.size and r["max"] == v.max() == 1.0
    ref, _ = np.histogram(np.sqrt(v / v.max()), bins=128, range=(0.0, 1.0))
    assert np.array_equal(np.array(r["hist"]), ref)
    p = np.quantile(v, 0.9, method="inverted_cdf")
    assert 0 <= r["hi"] - p <= hr.bin_width(1.0, 128, r["bin"]) * (1 + 1e-12)


def test_designed_field_bins_by_construction_have_no_ambiguity():
    nb = 128
    x, mask = hr.designed_field(60000, nb)
    assert hr.ambiguous_count(x, mask, nb) == 0
    h = hr._hash(60000, 7)
    c = (h >> np.uint32(26)).astype(int)
    b = (h & np.uint32(0xFFFF)).astype(int) % nb
    normal = c >= 4
    normal[12345] = False
    expected = np.bincount(b[normal], minlength=nb)
    expected[nb - 1] += 1                                   # the forced element 1.0
    assert np.array_equal(np.array(hr.histogram_range(x, mask, nb, 0.5)["hist"]), expected)


def test_symmetric_and_unsigned_range():
    x = _random_field(3, 5000)
    a = hr.histogram_range(x, pct=0.9, symmetric=True)
    b = hr.histogram_range(x, pct=0.9, symmetric=False)
    assert a["lo"] == -a["hi"] and b["lo"] == 0.0 and a["hi"] == b["hi"]


def test_degenerate_inputs():
    z = hr.histogram_range(np.zeros(10), nbins=8, symmetric=True)
    assert (z["lo"], z["hi"], z["max"], z["count"]) == (-1.0, 1.0, 0.0, 10)
    m = hr.histogram_range(np.ones(10), np.ones(10), nbins=8)
    assert (m["lo"], m["hi"], m["max"], m["count"]) == (0.0, 1.0, 0.0, 0)
    one = hr.histogram_range(np.array([-3.0]), nbins=8, pct=0.5)
    assert one["hi"] == 3.0 and one["max"] == 3.0                # single element: last bin -> hi = max


def test_pct_validation_and_json():
    with pytest.raises(ValueError):
        hr.histogram_range(np.ones(3), pct=0.0)
    with pytest.raises(ValueError):
        hr.histogram_range(np.ones(3), pct=1.5)
    json.dumps(hr.make_reference())                          # serialisable (plain floats/ints/lists)


def test_outlier_widens_bins_documented_pitfall():
    # bulk in [0, 1], one outlier 1000: the percentile of the bulk is resolved only to the bin width ~ 1000*(2b+1)/nb^2
    x = hr.heavy_tail_field(200000)
    r = hr.histogram_range(x, nbins=2048, pct=0.995)
    p = np.quantile(np.abs(x.astype(np.float64)), 0.995, method="inverted_cdf")
    assert 0 <= r["hi"] - p <= hr.bin_width(1000.0, 2048, r["bin"]) * (1 + 1e-12)
