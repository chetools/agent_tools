"""Tests for stats.imse_batch: greedy picks vs brute-force re-conditioning (generator, dense solves), the
rank-one downdate vs re-conditioning with the library GP, score = drop in average variance, and Python vs Rust."""
import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.stats.gp_matern52 import gp_build
from agent_tools.stats.imse_batch import imse_batch

# The fitted noise of the test GP sits on its lower bound (noise_sd/std(y) = 1e-3), so re-conditioning on a
# candidate next to a data point solves with cond(K) up to ~1e7-1e8, and the brute-force drops are differences of
# nearly equal variance averages: agreement to ~1e-8 relative is all that rounding allows.
TOL_RED = 1e-7        # reductions vs brute force, and own score vs own re-conditioning, relative to the largest score
TOL_FIRST = 1e-5      # every first-step score vs brute force (small scores lose digits to cancellation)
TOL_DOWNDATE = 1e-7   # downdated covariance vs re-conditioned, relative to max|C| (the source used 1e-7 absolute)
TOL_PY_RS = 1e-9      # Python vs Rust example, relative to the largest score


def setup():
    r = load_reference("imse_batch")["cases"]
    g = r["gp"]
    X = np.array(g["x"]).reshape(-1, 2)
    y = np.array(g["y"])
    return r, X, y, gp_build(X, y, g["ell"], g["signal_sd"], g["noise_sd"])


def test_vs_brute_force():
    r, X, y, gp = setup()
    for name in ("same_grid_8x8", "grid_vs_random_ref"):
        c = r[name]
        cand = np.array(c["candidates"]).reshape(-1, 2)
        ref = np.array(c["reference"]).reshape(-1, 2)
        res = imse_batch(gp.posterior_cov(cand, cand), gp.posterior_cov(ref, cand), gp.noise_sd ** 2, c["batch"])
        scale = max(c["first_scores"])
        er = np.max(np.abs(np.array(res.reduction) - c["reduction"])) / scale
        ef = np.max(np.abs(res.first_scores - c["first_scores"])) / scale
        print(f"VERIFY imse_batch py_{name} indices_equal={res.indices == c['indices']} rel_err_reduction={er:e} "
              f"rel_err_first_scores={ef:e} first_score_gap_rel={c['first_score_gap'] / scale:e}")
        assert res.indices == c["indices"] and er < TOL_RED and ef < TOL_FIRST


def test_downdate_equals_reconditioning():
    r, X, y, gp = setup()
    c = r["same_grid_8x8"]
    G = np.array(c["candidates"]).reshape(-1, 2)
    C = gp.posterior_cov(G, G)
    res = imse_batch(C, C, gp.noise_sd ** 2, 3)
    X3 = np.vstack([X, G[res.indices]])
    gp3 = gp_build(X3, np.r_[y, np.zeros(3)], gp.length_scales, gp.signal_sd, gp.noise_sd,
                   standardization=(gp.y_mean, gp.y_scale))
    C3 = gp3.posterior_cov(G, G)
    Cd = C.copy()
    for j in res.indices:
        u = Cd[:, j] / np.sqrt(Cd[j, j] + gp.noise_sd ** 2)
        Cd -= np.outer(u, u)
    err = np.max(np.abs(Cd - C3)) / np.max(np.abs(C))
    total = np.mean(np.diag(C)) - np.mean(np.diag(C3))
    es = abs(sum(res.reduction) - total) / total
    print(f"VERIFY imse_batch py_downdate_vs_reconditioning rel_err={err:e} sum_reduction_vs_total_drop_rel={es:e}")
    assert err < TOL_DOWNDATE and es < TOL_DOWNDATE


def test_score_is_drop_in_average_variance():
    r, X, y, gp = setup()
    G = np.array(r["same_grid_8x8"]["candidates"]).reshape(-1, 2)
    C = gp.posterior_cov(G, G)
    res = imse_batch(C, C, gp.noise_sd ** 2, 1)
    base = np.mean(np.diag(C))
    worst = 0.0
    for j in (0, 9, 30, 63):
        gpj = gp_build(np.vstack([X, G[j]]), np.r_[y, 0.0], gp.length_scales, gp.signal_sd, gp.noise_sd,
                       standardization=(gp.y_mean, gp.y_scale))
        drop = base - np.mean(np.diag(gpj.posterior_cov(G, G)))
        worst = max(worst, abs(res.first_scores[j] - drop) / max(res.first_scores))
    print(f"VERIFY imse_batch py_score_is_drop max_rel_err={worst:e}")
    assert worst < TOL_RED


def test_edge_cases():
    # a fully known noise-free candidate has score 0; replicates are allowed; inputs are not modified
    C = np.array([[0.0, 0.0], [0.0, 1.0]])
    C0 = C.copy()
    res = imse_batch(C, C, 0.0, 3)
    assert res.first_scores[0] == 0.0 and res.indices[0] == 1 and np.array_equal(C, C0)
    res = imse_batch(np.array([[1.0]]), np.array([[1.0]]), 0.5, 2)
    assert res.indices == [0, 0] and abs(res.reduction[0] - 1 / 1.5) < 1e-15
    print("VERIFY imse_batch py_edge_cases zero_variance,replicates,inputs_unchanged=ok")


def test_python_vs_rust_example():
    py, rs = python_example("imse_batch"), rust_example("imse_batch")
    for k in ("same", "other_ref"):
        assert py[k]["indices"] == rs[k]["indices"]
        s = max(rs[k]["first_scores"])
        d = max(np.max(np.abs(np.array(py[k]["reduction"]) - rs[k]["reduction"])),
                np.max(np.abs(np.array(py[k]["first_scores"]) - rs[k]["first_scores"]))) / s
        print(f"VERIFY imse_batch python_vs_rust_{k} indices_equal=True max_rel_diff={d:e}")
        assert d < TOL_PY_RS
