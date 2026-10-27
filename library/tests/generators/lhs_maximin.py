"""Generator for reference_data/lhs_maximin.json.

Two kinds of reference, both independent of library/python/agent_tools:
 1. `designs`: designs produced by an independent re-implementation inside this script (plain Python
    integers for SplitMix64 per Vigna's splitmix64.c, the best-of-k LHS written from the algorithm
    description), with the minimum distance computed by scipy.spatial.distance.pdist. The library
    (Rust and Python) must reproduce the points bit for bit.
 2. `seed_study`: minimum pairwise distance over seeds 0..NSEEDS-1 for (n, d) in CASES:
    ours with tries = 1 and tries = 50 (independent implementation, stream (seed, 0)),
    scipy.stats.qmc.LatinHypercube(d, scramble=True, seed=seed) plain and optimization="random-cd".
    SciPy's plain LHS draws the same distribution as one candidate of ours (random position in slice),
    so "ours tries=1 vs scipy plain" is a fair distribution test; the others are comparisons only.
Run: python library/tests/generators/lhs_maximin.py
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from _common import write

M64 = (1 << 64) - 1
NSEEDS = 200
CASES = [(10, 2), (20, 5), (50, 10)]
DESIGNS = [(1, 3, 5, 1), (2, 2, 10, 3), (5, 2, 1, 0), (12, 3, 50, 42), (30, 4, 20, 11), (50, 10, 50, 7), (100, 6, 10, 5)]


class Sm:
    def __init__(self, seed, stream):
        self.s = (seed * 1000003 + stream) & M64

    def u64(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uni(self):
        return (self.u64() >> 11) / 9007199254740992.0

    def perm(self, n):
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(self.uni() * (i + 1))
            p[i], p[j] = p[j], p[i]
        return p


def one_lhs(g, n, d):
    X = np.empty((n, d))
    for i in range(d):
        p = g.perm(n)
        for k in range(n):
            X[k, i] = (p[k] + g.uni()) / n
    return X


def dmin_of(X):
    return 1.0 if len(X) < 2 else float(pdist(X).min())


def best_of(seed, n, d, tries):
    g = Sm(seed, 0)
    best, bd, bt = None, -1.0, 0
    for t in range(tries):
        X = one_lhs(g, n, d)
        dm = dmin_of(X)
        if dm > bd:
            best, bd, bt = X, dm, t
    return best, bd, bt


def latin_ok(X):
    n, d = X.shape
    return all(sorted(np.floor(X[:, i] * n).astype(int)) == list(range(n)) for i in range(d))


designs = []
for n, d, tries, seed in DESIGNS:
    X, dm, bt = best_of(seed, n, d, tries)
    assert latin_ok(X) and X.min() >= 0 and X.max() < 1
    designs.append({"n": n, "d": d, "tries": tries, "seed": seed, "points": X.ravel().tolist(),
                    "min_distance": dm, "best_try": bt})

study = {}
for n, d in CASES:
    rec = {"ours_1": [], "ours_50": [], "scipy_plain": [], "scipy_random_cd": []}
    for s in range(NSEEDS):
        rec["ours_1"].append(best_of(s, n, d, 1)[1])
        rec["ours_50"].append(best_of(s, n, d, 50)[1])
        rec["scipy_plain"].append(dmin_of(qmc.LatinHypercube(d=d, scramble=True, seed=s).random(n)))
        rec["scipy_random_cd"].append(dmin_of(qmc.LatinHypercube(d=d, scramble=True, optimization="random-cd",
                                                                 seed=s).random(n)))
    study[f"{n}x{d}"] = {"n": n, "d": d, **rec}
    print(n, d, {k: round(float(np.mean(v)), 4) for k, v in rec.items() if isinstance(v, list)})

write("lhs_maximin", "library/tests/generators/lhs_maximin.py",
      {"nseeds": NSEEDS, "cases_n_d": CASES, "designs_n_d_tries_seed": DESIGNS, "stream": 0,
       "scipy": "qmc.LatinHypercube(d, scramble=True, seed=s) and optimization='random-cd'",
       "min_distance": "scipy.spatial.distance.pdist(...).min()"},
      "Dimensionless unit cube. Points row-major flat n x d. Seed s uses SplitMix64 state s*1000003+0. "
      "seed_study lists are indexed by seed 0..nseeds-1.",
      {"designs": designs, "seed_study": study})
