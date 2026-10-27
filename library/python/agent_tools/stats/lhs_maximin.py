"""Maximin Latin hypercube design: best of `tries` random Latin hypercubes (Python twin of Rust).

Sources: McKay, Beckman & Conover, "A comparison of three methods for selecting values of input
variables in the analysis of output from a computer code", Technometrics 21 (1979) 239-245 (Latin
hypercube sampling); Johnson, Moore & Ylvisaker, "Minimax and maximin distance designs", J. Stat.
Plann. Inference 26 (1990) 131-148, and Morris & Mitchell, "Exploratory designs for computational
experiments", J. Stat. Plann. Inference 43 (1995) 381-402 (the maximin criterion; the search here
is the plain best-of-k random search, not their simulated-annealing search). Ported from
chetools/claude `sequential-design/python/sdl.py` (`lhs_maximin`, own code).
Rust twin: library/rust/src/stats/lhs_maximin.rs.

Algorithm. The unit range of every input is cut into n equal slices. One candidate design draws,
for each dimension in turn, a random permutation of the slices and then, for each point in turn, a
uniform position inside its slice: x = (slice + u) / n, u in [0, 1). So each input has exactly one
point per slice (the Latin property), at a random, not centred, position. Of `tries` independent
candidates the one whose closest pair of points (Euclidean distance in the unit cube) is farthest
apart is kept; on an exact tie the earlier candidate wins. Draw order per candidate: for dimension
0..d: permutation(n), then n uniforms. Squared distances are summed over dimensions in the same order
as in Rust, so designs are bit-identical to Rust for the same generator state.
All coordinates and distances are dimensionless (unit hypercube).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LhsResult:
    """Result of lhs_maximin."""
    success: bool                 # False when an input was invalid (see message)
    message: str                  # "ok" or the reason for failure
    points: np.ndarray            # (n, d) design in [0, 1), dimensionless; empty (0, d) on failure
    n: int                        # number of points
    d: int                        # number of dimensions
    min_distance: float           # smallest pairwise distance of the design (1.0 if n == 1; NaN on failure)
    best_try: int                 # zero-based index of the kept candidate (first to reach the best distance)
    n_candidates: int             # candidates generated and evaluated (= tries on success)


def lhs_random(rng, n, d):
    """One random Latin hypercube, array (n, d) in [0, 1): exactly one point in each slice
    [j/n, (j+1)/n) of each input, at a uniformly random position inside its slice.
    `rng` is a SplitMix64. Draws d * (2n - 1) uniforms (n >= 1). n or d = 0 gives an empty array."""
    X = np.empty((n, d))
    for i in range(d):
        perm = rng.permutation(n)
        for k in range(n):
            X[k, i] = (perm[k] + rng.uniform()) / n
    return X


def min_pairwise_distance(X):
    """Smallest pairwise Euclidean distance of the rows of X (n, d), in the unit of X.
    Returns 1.0 for n <= 1 (the source's convention for a single point). Squared differences are
    summed over the columns in order, as in Rust (this makes tie-breaking identical)."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n <= 1:
        return 1.0
    s = np.zeros((n, n))
    for c in range(d):
        t = X[:, c][:, None] - X[:, c][None, :]
        s = s + t * t
    return float(np.min(np.sqrt(s)[np.triu_indices(n, 1)]))


def lhs_maximin(rng, n, d, tries=50):
    """Maximin Latin hypercube of n points in d dimensions: the best (largest minimum pairwise
    distance) of `tries` random Latin hypercubes drawn from `rng` (a SplitMix64; the source uses
    SplitMix64.from_seed_stream(seed, stream)).

    n >= 1 points, d >= 1 dimensions, tries >= 1 candidates (default 50, as in the source).
    Output points are in [0, 1)^d, dimensionless; scale to physical ranges as lo + x * (hi - lo).
    Candidates are drawn one after another from the same stream, so with the same generator state a
    run with more tries starts with the candidates of a run with fewer: the achieved minimum
    distance never decreases when tries grows.
    Failure (success False, empty points): n, d or tries equal to 0."""
    if n < 1 or d < 1 or tries < 1:
        return LhsResult(False, f"n, d and tries must be at least 1 (got n={n}, d={d}, tries={tries})",
                         np.empty((0, max(d, 0))), n, d, float("nan"), 0, 0)
    best, best_dmin, best_try = None, -1.0, 0
    for t in range(tries):
        X = lhs_random(rng, n, d)
        dmin = min_pairwise_distance(X)
        if dmin > best_dmin:
            best, best_dmin, best_try = X, dmin, t
    return LhsResult(True, "ok", best, n, d, best_dmin, best_try, tries)
