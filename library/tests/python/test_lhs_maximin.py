"""Tests for stats.lhs_maximin: Latin property, designs vs an independent implementation, monotone
improvement with tries, comparison with scipy.stats.qmc.LatinHypercube, Rust vs Python."""
import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.random.splitmix64 import SplitMix64
from agent_tools.stats.lhs_maximin import lhs_maximin, lhs_random, min_pairwise_distance

TOL_DMIN = 1e-14      # min distance vs scipy pdist (summation order only), relative
Z_FAIR = 4.0          # ours (tries=1) vs scipy plain LHS: same distribution; max |z| of the mean difference
TOL_RUST = 0.0        # Rust vs Python designs: bit-identical (same draws, same summation order)


def is_latin(X):
    n, d = X.shape
    if X.min() < 0 or X.max() >= 1:
        return False
    return all(sorted(np.floor(X[:, i] * n).astype(int)) == list(range(n)) for i in range(d))


def gen(seed, stream=0):
    return SplitMix64.from_seed_stream(seed, stream)


def test_latin_property_many_cases():
    cases = bad = 0
    for n in (1, 2, 3, 5, 8, 13, 20, 33, 50, 100):
        for d in (1, 2, 3, 6, 10):
            for seed in range(6):
                r = lhs_maximin(gen(seed), n, d, 10)
                assert r.success and r.points.shape == (n, d) and r.n_candidates == 10
                bad += not is_latin(r.points)
                assert r.min_distance == min_pairwise_distance(r.points)
                bad += not is_latin(lhs_random(gen(seed, 1), n, d))
                cases += 2
    print(f"VERIFY lhs_maximin py_latin cases={cases} violations={bad}")
    assert bad == 0


def test_designs_match_independent_implementation():
    mism, worst = 0, 0.0
    for c in load_reference("lhs_maximin")["cases"]["designs"]:
        r = lhs_maximin(gen(c["seed"]), c["n"], c["d"], c["tries"])
        mism += int(np.sum(r.points.ravel() != np.array(c["points"])))
        assert r.best_try == c["best_try"]
        worst = max(worst, abs(r.min_distance - c["min_distance"]) / c["min_distance"])
        if c["n"] == 50 and c["d"] == 10:
            print(f"VERIFY lhs_maximin py_hard_n50_d10 min_distance={r.min_distance:.6f} best_try={r.best_try}")
    print(f"VERIFY lhs_maximin py_designs coordinate_mismatches={mism} max_rel_dmin_diff_vs_scipy_pdist={worst:e}")
    assert mism == 0 and worst < TOL_DMIN


def test_improves_monotonically_with_tries():
    ks = (1, 2, 5, 10, 20, 50, 100, 200)
    means = np.zeros(len(ks))
    seeds = 50
    for seed in range(seeds):
        prev = -1.0
        for j, k in enumerate(ks):
            dm = lhs_maximin(gen(seed), 30, 5, k).min_distance
            assert dm >= prev
            prev = dm
            means[j] += dm / seeds
    assert np.all(np.diff(means) > 0)
    print("VERIFY lhs_maximin py_monotone n30_d5 nondecreasing_every_seed=1 mean_min_distance "
          + " ".join(f"k{k}={m:.4f}" for k, m in zip(ks, means)))


def test_seed_study_vs_scipy():
    worst = 0.0
    for key, c in load_reference("lhs_maximin")["cases"]["seed_study"].items():
        n, d, ns = c["n"], c["d"], len(c["ours_1"])
        o1 = np.array([lhs_maximin(gen(s), n, d, 1).min_distance for s in range(ns)])
        o50 = np.array([lhs_maximin(gen(s), n, d, 50).min_distance for s in range(ns)])
        worst = max(worst, np.max(np.abs(o1 - c["ours_1"]) / o1), np.max(np.abs(o50 - c["ours_50"]) / o50))
        sp, cd = np.array(c["scipy_plain"]), np.array(c["scipy_random_cd"])
        z = (o1.mean() - sp.mean()) / np.sqrt(o1.var(ddof=1) / ns + sp.var(ddof=1) / ns)
        print(f"VERIFY lhs_maximin py_study_{key} mean_dmin ours_tries1={o1.mean():.4f} ours_tries50={o50.mean():.4f} "
              f"sd50={o50.std(ddof=1):.4f} scipy_plain={sp.mean():.4f} scipy_random_cd={cd.mean():.4f} "
              f"z_ours1_vs_scipy_plain={z:.2f} nseeds={ns}")
        assert abs(z) < Z_FAIR      # fair: same distribution as one scipy plain LHS
        assert o50.mean() > sp.mean()   # fair: best of 50 beats a single candidate
        # scipy optimization='random-cd' (a discrepancy criterion) is a comparison only, not asserted
    print(f"VERIFY lhs_maximin py_study max_rel_diff_vs_independent_impl={worst:e}")
    assert worst == 0.0


def test_invalid_input_and_single_point():
    for n, d, t in ((0, 3, 5), (4, 0, 5), (4, 3, 0)):
        r = lhs_maximin(gen(1), n, d, t)
        assert not r.success and r.points.size == 0 and np.isnan(r.min_distance) and r.message
    r = lhs_maximin(gen(1), 1, 4, 3)
    assert r.success and r.min_distance == 1.0 and r.points.shape == (1, 4)


def test_python_vs_rust_example():
    py, rs = python_example("lhs_maximin"), rust_example("lhs_maximin")
    worst = 0.0
    for k in ("small_points", "hard_points"):
        assert len(py[k]) == len(rs[k])
        worst = max(worst, max(abs(a - b) for a, b in zip(py[k], rs[k])))
    assert py["small_best_try"] == rs["small_best_try"] and py["hard_best_try"] == rs["hard_best_try"]
    dd = max(abs(py["small_min_distance"] - rs["small_min_distance"]), abs(py["hard_min_distance"] - rs["hard_min_distance"]))
    print(f"VERIFY lhs_maximin python_vs_rust max_coord_diff={worst:e} max_min_distance_diff={dd:e} "
          f"hard_min_distance={py['hard_min_distance']:.6f}")
    assert worst <= TOL_RUST and dd <= TOL_RUST
