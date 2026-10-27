"""float64 reference of library/wgsl/histogram_range.wgsl: robust colour-scale range of a field.

Steps (identical to the WGSL kernels hr_clear, hr_max, hr_hist, hr_range):
  1. counted entries: not masked (mask[i] == 0) and finite;
  2. mx = max |x| over counted entries;
  3. histogram of sqrt(|x| / mx) in ``nbins`` equal bins: bin = min(floor(sqrt(r) * nbins), nbins - 1);
  4. want = max(ceil(pct * count), 1); first bin with cumulative count >= want; hi = mx * ((bin + 1) / nbins)^2
     (upper edge of that bin, floored at 1e-30); range = [-hi, hi] if symmetric else [0, hi];
     count == 0 or mx == 0 gives hi = 1.

Units: x, lo, hi, max in the units of the field; pct is a fraction in (0, 1]; counts are dimensionless.
Defaults (as in the source project): nbins = 2048, pct = 0.995.

Quantisation error: the exact percentile p of the counted |x| (numpy.quantile, method="inverted_cdf") lies in
the selected bin, hence 0 <= hi - p <= mx * (2 * bin + 1) / nbins**2 <= 2 * mx / nbins  (see :func:`bin_width`).
Bin assignment in float32 (GPU) may differ from float64 for elements within ~1e-6 (relative to the bin
position) of a bin edge (:func:`ambiguous_count`).
"""
import math

import numpy as np

NBINS = 2048
PCT = 0.995
_FIN = np.uint32(0x7F800000)


def counted_mask(x, mask=None):
    """Boolean array of counted entries: finite and (mask is None or mask == 0)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if mask is not None:
        ok &= (np.asarray(mask) == 0)
    return ok


def bin_index(v, mx, nbins):
    """Bin of |value| v (>= 0) relative to the maximum mx."""
    r = v / mx if mx > 0 else np.zeros_like(v)
    return np.minimum(np.floor(np.sqrt(r) * nbins).astype(np.int64), nbins - 1)


def histogram_range(x, mask=None, nbins=NBINS, pct=PCT, symmetric=False):
    """Return dict(lo, hi, max, count, hist (list of nbins ints), bin (selected bin or -1))."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if not (0 < pct <= 1):
        raise ValueError("pct must be in (0, 1]")
    ok = counted_mask(x, None if mask is None else np.asarray(mask).ravel())
    v = np.abs(x[ok])
    cnt = int(v.size)
    mx = float(v.max()) if cnt else 0.0
    hist = np.zeros(nbins, dtype=np.int64)
    hi, sel = 1.0, -1
    if cnt:
        hist = np.bincount(bin_index(v, mx, nbins), minlength=nbins)
    if cnt > 0 and mx > 0:
        want = max(math.ceil(pct * cnt), 1)
        sel = int(np.searchsorted(np.cumsum(hist), want, side="left"))
        sel = min(sel, nbins - 1)
        q = (sel + 1) / nbins
        hi = max(mx * q * q, 1e-30)
    return {"lo": -hi if symmetric else 0.0, "hi": hi, "max": mx, "count": cnt,
            "hist": [int(h) for h in hist], "bin": sel}


def bin_width(mx, nbins, bin):
    """Width in value units of bin ``bin``: mx * (2 bin + 1) / nbins^2 (the error bound of hi)."""
    return mx * (2 * bin + 1) / nbins ** 2


def ambiguous_count(x, mask=None, nbins=NBINS, margin=1e-6):
    """Number of counted elements whose position pos = sqrt(|x|/mx) * nbins is within ``margin * pos`` of an
    integer (bin edge); these may fall in a neighbouring bin under float32 arithmetic (division, sqrt: a few ulp,
    about 2e-7 relative; the margin 1e-6 leaves a factor 5)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    ok = counted_mask(x, None if mask is None else np.asarray(mask).ravel())
    v = np.abs(x[ok])
    if v.size == 0 or v.max() == 0:
        return 0
    pos = np.sqrt(v / v.max()) * nbins
    d = np.abs(pos - np.rint(pos))
    return int(np.count_nonzero((d < margin * pos) & (np.rint(pos) < nbins) & (np.rint(pos) > 0)))


# ------------------------------------------------------------------ deterministic test inputs
def _hash(n, seed):
    i = np.arange(n, dtype=np.uint32)
    h = i * np.uint32(2654435761) + np.uint32(seed)
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x85EBCA6B)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0xC2B2AE35)
    h ^= h >> np.uint32(16)
    return h


def designed_field(n, nbins, seed=7):
    """Field whose bin assignment is unambiguous.  Returns (x float32 array, mask uint32 array).

    h = fmix32(i * 2654435761 + seed).  category c = h >> 26: c == 0 masked entry (mask = 1, value 1e30);
    c == 1 NaN; c == 2 +inf; c == 3 -inf; otherwise a normal entry with bin b = (h & 0xFFFF) % nbins, sign from
    bit 16 and |value| = (2b + 1)^2 / (4 nbins^2), i.e. exactly the middle of bin b for max = 1.
    Index 12345 is forced to value 1.0 (unmasked), so max = 1 and that element is in the last bin.
    """
    h = _hash(n, seed)
    c = (h >> np.uint32(26)).astype(np.int64)
    b = ((h & np.uint32(0xFFFF)).astype(np.int64)) % nbins
    sgn = 1.0 - 2.0 * ((h >> np.uint32(16)) & np.uint32(1)).astype(np.float64)
    val = sgn * ((2 * b + 1) * (2 * b + 1)) / (4.0 * nbins * nbins)
    x = val.copy()
    mask = np.zeros(n, dtype=np.uint32)
    x[c == 0] = 1e30
    mask[c == 0] = 1
    x[c == 1] = np.nan
    x[c == 2] = np.inf
    x[c == 3] = -np.inf
    x[12345] = 1.0
    mask[12345] = 0
    return x.astype(np.float32), mask


def heavy_tail_field(n, outlier=1000.0, seed=9):
    """Bulk |x| = u^4 with u uniform in (0, 1) (24-bit grid), random sign, plus one outlier at index 777.
    Only IEEE multiplications are used, so JS reproduces it bit for bit.  Returns float32 array."""
    h = _hash(n, seed)
    u = ((h & np.uint32(0xFFFFFF)).astype(np.float64) + 0.5) / 16777216.0
    a = u * u
    a = a * a
    sgn = 1.0 - 2.0 * ((h >> np.uint32(24)) & np.uint32(1)).astype(np.float64)
    x = sgn * a
    x[777] = outlier
    return x.astype(np.float32)


# (id, kind, n, nbins, maxGroupsX, pcts)
def make_reference():
    out = {"designed": {}, "heavy": {}, "degenerate": {}}
    # --- designed case: exact counts expected.  Dyadic pct with few bits: pct * count exact in f32.
    n, nb = 200000, 256
    x, m = designed_field(n, nb)
    d = {"n": n, "nbins": nb, "maxGroupsX": 100, "pcts": [0.5, 0.875, 0.96875, 1.0], "results": []}
    d["ambiguous"] = ambiguous_count(x, m, nb)
    for pct in d["pcts"]:
        r = histogram_range(x, m, nb, pct, symmetric=False)
        r["pct"] = pct
        r["p_exact"] = float(np.quantile(np.abs(x[counted_mask(x, m)]), pct, method="inverted_cdf"))
        d["results"].append(r)
    d["unmasked_count"] = histogram_range(x, None, nb, 0.5)["count"]
    d["unmasked_max"] = histogram_range(x, None, nb, 0.5)["max"]
    out["designed"] = d
    # --- heavy-tailed case with a 1000x outlier, default nbins = 2048 and pct = 0.995 (not dyadic)
    n, nb = 4_000_000, NBINS
    x = heavy_tail_field(n)
    h = {"n": n, "nbins": nb, "outlier": 1000.0, "ambiguous": ambiguous_count(x, None, nb), "results": []}
    for pct, sym in ((0.995, True), (0.5, False), (0.999, False)):
        r = histogram_range(x, None, nb, pct, symmetric=sym)
        r["pct"] = pct
        r["symmetric"] = sym
        r["p_exact"] = float(np.quantile(np.abs(x), pct, method="inverted_cdf"))
        h["results"].append(r)
    out["heavy"] = h
    # --- degenerate inputs
    z = histogram_range(np.zeros(1000, dtype=np.float32), None, 64, 0.5, symmetric=True)
    am = histogram_range(np.ones(1000, dtype=np.float32), np.ones(1000, dtype=np.uint32), 64, 0.5, symmetric=False)
    out["degenerate"] = {"zeros": {k: z[k] for k in ("lo", "hi", "max", "count")},
                         "all_masked": {k: am[k] for k in ("lo", "hi", "max", "count")}}
    return out
