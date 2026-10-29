"""Tests for optimize.qp_box: KKT certificates, exact / enumerated / SciPy references, the muse fixture, failure modes, Rust."""
import itertools

import numpy as np
from conftest import load_reference, python_example, rust_example

from agent_tools.linalg.cholesky import cholesky, cholesky_solve
from agent_tools.optimize.qp_box import QpBoxOptions, qp_box

# KKT certificate, all relative to scale = 1 + max|g| + max|H| sum|x| (a bound on |Hx| and |g|):
TOL_STAT = 1e-13        # stationarity: max over free i of |(Hx+g)_i| / scale          (measured <= 3e-17)
TOL_SIGN = 1e-13        # multiplier sign: a bound-active variable's q-based multiplier >= -TOL_SIGN scale
TOL_FEAS = 0.0          # feasibility: lo <= x <= hi exactly (the blocking variable is set to its bound)
TOL_COMPL = 0.0         # complementarity: an active variable sits exactly on its bound, a free one has multiplier 0
TOL_MULT_REPORT = 1e-12  # reported multipliers vs the ones recomputed from q = Hx + g, relative to scale
# Against the references:
TOL_X_REF = 1e-9        # max |x - x_ref| (exact / enumerated); measured <= 2.5e-13 (cond 1e8, n = 30)
TOL_MULT_REF = 1e-8     # max |multiplier - reference|, relative to 1 + max reference multiplier; measured <= 5e-10 (cond 1e8)
TOL_F_REF = 1e-13       # |f - f_ref| relative to 1 + |x|^T |H| |x| + |g|^T |x| (rounding of f itself); measured <= 3e-17
TOL_X_SCIPY = 1e-6      # max |x - x_scipy|; SciPy L-BFGS-B ftol 1e-15 gtol 1e-12; measured <= 3.3e-7
TOL_X_SCIPY_ILL = 1e-4  # same for cond(H) = 1e8 (L-BFGS-B is accurate to about 5e-6 there; measured 5.2e-6)
TOL_X_MUSE = 1e-6       # muse test's own tolerance against its SciPy fixture (measured 5.7e-8, the muse prototype's 5.7e-8)
TOL_ZERO_MULT = 1e-8    # "zero multiplier" cases: |multiplier| <= this (g carries rounding of 1e-10 data)
TOL_EXAMPLE = 1e-12     # Python vs Rust example: x, f, multipliers
BOUND_TIE = 1e-9        # reference variable counts as "at a bound" when within this of it


def bounds(c):
    lo = np.array([-np.inf if v is None else v for v in c["lo"]])
    hi = np.array([np.inf if v is None else v for v in c["hi"]])
    return lo, hi


def problem(c):
    n = c["n"]
    lo, hi = bounds(c)
    return np.array(c["H"]).reshape(n, n), np.array(c["g"]), lo, hi


def scale_of(H, g, x):
    return 1 + np.max(np.abs(g)) + np.max(np.abs(H)) * np.sum(np.abs(x))


def certificate(H, g, lo, hi, r):
    """KKT residuals of result r, recomputed with NumPy (full H, independent of the routine's lower-triangle code)."""
    x = r.x
    q = H @ x + g
    scale = scale_of(H, g, x)
    free = r.active == 0
    stat = np.max(np.abs(q[free])) / scale if free.any() else 0.0
    fixed = lo == hi
    # multiplier recomputed from q: lower bound q_i, upper bound -q_i; a fixed variable may have either sign
    sgn = np.where(r.active == -1, 1.0, np.where(r.active == 1, -1.0, 0.0))
    m = sgn * q
    sign_viol = max(0.0, np.max(np.where(fixed | free, 0.0, -m))) / scale
    feas = max(0.0, np.max(lo - x), np.max(x - hi))
    bnd = np.where(r.active == -1, lo, np.where(r.active == 1, hi, x))
    compl = max(np.max(np.abs(x - bnd)), np.max(np.abs(r.multipliers[free])) if free.any() else 0.0)
    rep = np.max(np.abs(r.multipliers - np.where(fixed & (r.active != 0), np.abs(q), np.maximum(m, 0.0)))) / scale
    return {"stat": stat, "sign": sign_viol, "feas": feas, "compl": compl, "mult_report": rep, "scale": scale}


def check_certificate(H, g, lo, hi, r, label):
    c = certificate(H, g, lo, hi, r)
    assert r.success, (label, r.message)
    assert c["stat"] <= TOL_STAT and c["sign"] <= TOL_SIGN and c["feas"] <= TOL_FEAS and c["compl"] <= TOL_COMPL \
        and c["mult_report"] <= TOL_MULT_REPORT, (label, c)
    assert np.all(r.multipliers >= 0)
    return c


def f_scale(H, g, x):
    return 1 + np.abs(x) @ np.abs(H) @ np.abs(x) + np.abs(g) @ np.abs(x)


def active_agrees(r, ref_active, ref_mult, x_ref, lo, hi):
    """Active set equals the reference wherever it is determined: a free reference variable not at a bound must be
    free, a reference multiplier above TOL_ZERO_MULT fixes the side (zero-multiplier bounds may be reported either way)."""
    for i in range(len(x_ref)):
        at_bound = abs(x_ref[i] - lo[i]) <= BOUND_TIE or abs(x_ref[i] - hi[i]) <= BOUND_TIE
        if ref_active[i] == 0 and not at_bound:
            if r.active[i] != 0:
                return False
        elif ref_mult[i] > TOL_ZERO_MULT and lo[i] != hi[i]:
            if r.active[i] != ref_active[i]:
                return False
    return True


def test_reference_cases():
    for name, c in load_reference("qp_box")["cases"].items():
        H, g, lo, hi = problem(c)
        r = qp_box(H, g, lo, hi)
        cert = check_certificate(H, g, lo, hi, r, name)
        ref = c.get("exact") or c.get("enum")
        ex = np.max(np.abs(r.x - c["x_ref"]))
        assert ex <= TOL_X_REF, (name, ex)
        if ref:
            em = np.max(np.abs(r.multipliers - ref["multipliers"])) / (1 + np.max(ref["multipliers"]))
            ef = abs(r.f - ref["f"]) / f_scale(H, g, r.x)
            assert em <= TOL_MULT_REF and ef <= TOL_F_REF, (name, em, ef)
            assert active_agrees(r, ref["active"], ref["multipliers"], c["x_ref"], lo, hi), (name, r.active, ref["active"])
        else:
            em = ef = float("nan")
            assert (r.active == 0).all()
        es = np.max(np.abs(r.x - c["scipy"]["x"]))
        assert es <= (TOL_X_SCIPY_ILL if c["kind"] == "ill_conditioned" else TOL_X_SCIPY), (name, es)
        print(f"VERIFY qp_box py_{name} n={c['n']} x_ref={c['x_ref_source']} max_err_x_vs_ref={ex:.2e} "
              f"rel_err_mult={em:.2e} rel_err_f={ef:.2e} kkt_stat={cert['stat']:.1e} kkt_sign={cert['sign']:.1e} "
              f"kkt_feas={cert['feas']:.1e} diff_x_vs_scipy={es:.2e} nactive={int(np.count_nonzero(r.active))} "
              f"nit={r.nit} nfact={r.nfact} scipy_nit={c['scipy']['nit']}")


def test_unconstrained_equals_direct_cholesky_solve():
    for name in ("unconstrained_inf_n10", "unconstrained_wide_n10"):
        c = load_reference("qp_box")["cases"][name]
        H, g, lo, hi = problem(c)
        r = qp_box(H, g, lo, hi)
        ch = cholesky(H)
        xd = cholesky_solve(ch.l, -g)
        d = np.max(np.abs(r.x - xd))
        assert (r.active == 0).all() and np.all(r.multipliers == 0) and r.success
        assert d <= 1e-13 * (1 + np.max(np.abs(xd))), d      # the same factorization and solve: expect ~ 0
        # one step to the minimizer, one pass to certify (no new solve): probe + 1 free-set factorization
        assert (r.nit, r.nfact) == (2, 2), (r.nit, r.nfact)
        print(f"VERIFY qp_box py_{name}_vs_cholesky max_diff_x={d:.2e} nit={r.nit} nfact={r.nfact}")


def test_all_bounds_active():
    c = load_reference("qp_box")["cases"]["all_active_n5"]
    H, g, lo, hi = problem(c)
    r = qp_box(H, g, lo, hi)
    check_certificate(H, g, lo, hi, r, "all_active")
    assert (r.active != 0).all() and (r.multipliers > 0).all()
    # every variable at the bound its gradient pushes towards
    assert np.array_equal(r.x, np.where(r.active == -1, lo, hi))
    assert np.array_equal(r.active, -np.sign(g).astype(int))
    print(f"VERIFY qp_box py_all_active nactive=5 min_multiplier={r.multipliers.min():.2e} nit={r.nit} nfact={r.nfact}")


def test_degenerate_zero_multiplier():
    # (a) constructed: x_0 and x_1 at a bound with multiplier 0 to rounding of the data; x_3, x_4 genuinely active
    c = load_reference("qp_box")["cases"]["degenerate_zero_multiplier_n5"]
    H, g, lo, hi = problem(c)
    r = qp_box(H, g, lo, hi)
    check_certificate(H, g, lo, hi, r, "degenerate")
    kn = c["constructed"]
    assert np.max(np.abs(r.x - kn["x"])) <= 1e-9
    q = H @ r.x + g
    assert abs(q[0]) <= TOL_ZERO_MULT and abs(q[1]) <= TOL_ZERO_MULT          # multiplier zero at x_0, x_1
    assert abs(r.x[0] - lo[0]) <= 1e-9 and abs(r.x[1] - hi[1]) <= 1e-9        # and the points sit on their bounds
    assert r.active[3] == -1 and r.active[4] == 1 and abs(r.multipliers[3] - 1.5) < 1e-8 and abs(r.multipliers[4] - 0.7) < 1e-8
    # (b) start point 0 is on three bounds, gradient 0 there: converges at once, bounds stay active with multiplier 0
    c = load_reference("qp_box")["cases"]["start_at_bound_zero_gradient_n4"]
    H, g, lo, hi = problem(c)
    r2 = qp_box(H, g, lo, hi)
    check_certificate(H, g, lo, hi, r2, "start_at_bound")
    assert list(r2.active) == [-1, 0, -1, 1] and np.all(r2.multipliers == 0) and np.all(r2.x == 0)
    assert (r2.nit, r2.nfact) == (1, 2)
    print(f"VERIFY qp_box py_degenerate constructed_max_err_x={np.max(np.abs(r.x - kn['x'])):.2e} "
          f"zero_mult_residuals={abs(q[0]):.1e},{abs(q[1]):.1e} nit={r.nit} start_at_bound_nit={r2.nit} "
          f"start_at_bound_active={list(r2.active)}")


def test_fixed_variable_is_never_released():
    c = load_reference("qp_box")["cases"]["fixed_variable_n5"]
    H, g, lo, hi = problem(c)
    r = qp_box(H, g, lo, hi)
    check_certificate(H, g, lo, hi, r, "fixed")
    assert r.x[1] == 0.3 and r.x[3] == -0.2 and r.active[1] != 0 and r.active[3] != 0
    q = H @ r.x + g
    # side follows the sign of q: lower if q >= 0
    assert all((r.active[i] == -1) == (q[i] >= 0) for i in (1, 3))
    assert r.nit <= 3, r.nit       # the muse source releases a fixed variable with a wrong-signed multiplier and re-adds it later (6 iterations here)
    print(f"VERIFY qp_box py_fixed_variable nit={r.nit} nfact={r.nfact} active={list(r.active)} mult={np.round(r.multipliers, 6).tolist()}")


def test_ill_conditioned_free_set_does_not_stall():
    # Regression (found by the random sweep below): n = 2, cond(H) = 6e6, free minimizer at |x| ~ 2e2. The residual of the
    # Cholesky solve is ~ eps cond |x| ~ 1e-7 > tol_step = 1e-10, so a step re-solved after the full step never reaches
    # max|p| <= tol_step: the muse algorithm ran into its cap (4 n) and reported failure. Reference: exact 2 x 2 solve.
    from fractions import Fraction as Fr
    H = np.array([[49189.72355913, -122006.2183222], [-122006.2183222, 302621.52422222]])
    g = np.array([-127.21531714, 931.73530249])
    lo, hi = np.array([-np.inf, -np.inf]), np.array([np.inf, 0.37999599])
    a, b, d = Fr(H[0, 0]), Fr(H[1, 0]), Fr(H[1, 1])
    det = a * d - b * b
    x_exact = [float((-d * Fr(g[0]) + b * Fr(g[1])) / det), float((b * Fr(g[0]) - a * Fr(g[1])) / det)]
    r = qp_box(H, g, lo, hi)
    assert r.success, r.message
    dx = np.max(np.abs(r.x - x_exact))
    assert dx <= 1e-6 and (r.active == 0).all()       # error of any double precision solve: cond * eps * |x| ~ 2e-7
    assert (r.nit, r.nfact) == (2, 2), (r.nit, r.nfact)
    print(f"VERIFY qp_box py_ill_conditioned_free_set_does_not_stall cond=6e6 max_err_x_vs_exact_rational={dx:.2e} nit={r.nit} nfact={r.nfact}")


def enumerate_kkt(H, g, lo, hi):
    """Best feasible KKT point over all 3^n patterns (variable at lower / free / at upper)."""
    n = len(g)
    best = None
    for pat in itertools.product((-1, 0, 1), repeat=n):
        pat = np.array(pat)
        x = np.where(pat == -1, lo, np.where(pat == 1, hi, 0.0))
        if not np.all(np.isfinite(x)):
            continue
        F, A = np.where(pat == 0)[0], np.where(pat != 0)[0]
        if len(F):
            x[F] = np.linalg.solve(H[np.ix_(F, F)], -(g[F] + H[np.ix_(F, A)] @ x[A]))
        sc = 1 + np.max(np.abs(g)) + np.max(np.abs(H)) * np.sum(np.abs(x))
        if np.any(x < lo - 1e-10 * sc) or np.any(x > hi + 1e-10 * sc):
            continue
        q = H @ x + g
        if np.any(np.where(pat == -1, q, np.where(pat == 1, -q, 0.0)) < -1e-10 * sc):
            continue
        f = 0.5 * x @ H @ x + g @ x
        if best is None or f < best[0]:
            best = (f, x)
    return best


def test_random_small_problems_vs_enumeration():
    # 200 random problems, n = 1..6, bounds finite / one-sided / infinite, some lo == hi; cond 1 to 1e6
    rng = np.random.RandomState(7)
    worst, nit_max, nfixed = 0.0, 0, 0
    for k in range(200):
        n = 1 + k % 6
        Q, _ = np.linalg.qr(rng.randn(n, n))
        H = Q @ np.diag(np.logspace(0, rng.uniform(0, 6), n)) @ Q.T
        H = (H + H.T) / 2
        g = rng.randn(n) * 3 * np.sqrt(np.mean(np.diag(H)))
        lo = -rng.uniform(0.1, 1, n)
        hi = rng.uniform(0.1, 1, n)
        lo[rng.rand(n) < 0.2] = -np.inf
        hi[rng.rand(n) < 0.2] = np.inf
        fx = rng.rand(n) < 0.1
        lo[fx] = hi[fx] = rng.uniform(-0.5, 0.5, fx.sum())
        nfixed += int(fx.sum())
        r = qp_box(H, g, lo, hi)
        check_certificate(H, g, lo, hi, r, f"random{k}")
        f_best, x_best = enumerate_kkt(H, g, lo, hi)
        dx = np.max(np.abs(r.x - x_best)) / (1 + np.max(np.abs(x_best)))
        # x is determined to about cond * eps * |x|: cond up to 1e6 in this sweep
        assert dx <= 1e-8 and abs(r.f - f_best) <= 1e-12 * f_scale(H, g, r.x), (k, dx)
        worst, nit_max = max(worst, dx), max(nit_max, r.nit)
    print(f"VERIFY qp_box py_random_vs_enumeration problems=200 n=1..6 fixed_variables={nfixed} max_rel_err_x={worst:.2e} max_nit={nit_max}")


def test_random_problems_up_to_n50_certificate():
    rng = np.random.RandomState(11)
    nits, worst, count = [], 0.0, 0
    for n in (2, 5, 10, 20, 30, 40, 50):
        for cond in (1e0, 1e2, 1e4, 1e6, 1e8):
            Q, _ = np.linalg.qr(rng.randn(n, n))
            H = Q @ np.diag(np.logspace(0, np.log10(cond), n)) @ Q.T
            H = (H + H.T) / 2
            g = rng.randn(n) * 2 * np.sqrt(np.mean(np.diag(H)))
            lo, hi = -rng.uniform(0.05, 1, n), rng.uniform(0.05, 1, n)
            r = qp_box(H, g, lo, hi)
            c = check_certificate(H, g, lo, hi, r, f"n{n}_cond{cond}")
            assert r.nit <= max(50, 20 * n)
            worst = max(worst, c["stat"])
            nits.append((n, r.nit))
            count += 1
    print(f"VERIFY qp_box py_random_n50 problems={count} all_kkt_ok=1 max_kkt_stat={worst:.1e} "
          f"nit_at_n50={[i for k, i in nits if k == 50]}")


def test_muse_fixture():
    # chetools/muse twin-cstr/solver/tests/fixtures/qp.json: 26 random SPD box QPs n = 4..124 (SciPy L-BFGS-B reference);
    # the muse test's tolerance is 1e-6 max abs deviation
    cases = load_reference("qp_box_muse_fixture")["cases"]
    worst, nit_max = 0.0, 0
    for name, c in cases.items():
        H, g, lo, hi = problem(c)
        r = qp_box(H, g, lo, hi)
        check_certificate(H, g, lo, hi, r, name)
        d = np.max(np.abs(r.x - c["x_ref"]))
        assert d <= TOL_X_MUSE, (name, d)
        worst, nit_max = max(worst, d), max(nit_max, r.nit)
    print(f"VERIFY qp_box py_muse_fixture cases={len(cases)} max_abs_dev_vs_scipy={worst:.2e} max_nit={nit_max} "
          f"nit_n124={[qp_box(*problem(c)).nit for k, c in cases.items() if c['n'] == 124]}")
    assert len(cases) == 26


CAP_REPRODUCERS = [  # (n, H, g, hi) with lo = 0: nondegenerate, well-conditioned, found by hill climbing on the pass count
    (3, [[1.0472399095265337, 0.9421865156436513, -1.5521828673786728], [0.9421865156436513, 1.179456712389007, -1.4812687162226472], [-1.5521828673786728, -1.4812687162226472, 2.8891061401248]], [-0.2768828882302678, -0.4946297289368097, -0.4182782400244921], 1.602283819497114),
    (4, [[1.3802835278185264, -0.7516270309602238, -0.21645844042208112, -0.4591834757852036], [-0.7516270309602238, 1.1844470243729093, 0.8437979841461524, 0.6266504515422855], [-0.21645844042208112, 0.8437979841461524, 4.03710395165706, -0.9178952259079698], [-0.4591834757852036, 0.6266504515422855, -0.9178952259079698, 1.3951420125652543]], [-0.4292886726390505, -1.4503185691175424, 0.31810988987637895, -1.9702710108575923], 1.579564953358756),
    (5, [[0.8868805615949854, -0.14703752442530216, -0.2680469265695435, -0.1016354121942829, -0.4192624030048251], [-0.14703752442530216, 0.9936361398860136, 0.5205418672786014, -0.12310454567891946, -1.084242870466544], [-0.2680469265695435, 0.5205418672786014, 1.2024044824646984, 0.5703495039483478, -0.8522460281354195], [-0.1016354121942829, -0.12310454567891946, 0.5703495039483478, 1.6003587461216677, 0.23891391513417226], [-0.4192624030048251, -1.084242870466544, -0.8522460281354195, 0.23891391513417226, 1.9895783536657685]], [-0.10766510778311858, -0.3708280984436486, -0.8334775274509005, -0.9713629640398181, -0.10279924143925213], 1.618494052698689),
]


def test_default_cap_reproducers():
    # PR review: the old default cap 4 n failed on these ordinary problems (releases count as passes)
    out = []
    for n, H, g, hi in CAP_REPRODUCERS:
        H, g, lo, hi = np.array(H), np.array(g), np.zeros(n), np.full(n, hi)
        r = qp_box(H, g, lo, hi)
        assert r.success and r.nit > 4 * n and r.nit <= max(50, 20 * n), (n, r.nit)
        c = check_certificate(H, g, lo, hi, r, f"cap_n{n}")
        assert not qp_box(H, g, lo, hi, QpBoxOptions(max_iter=4 * n)).success
        out.append((n, r.nit, c["stat"]))
    x3 = qp_box(np.array(CAP_REPRODUCERS[0][1]), np.array(CAP_REPRODUCERS[0][2]), np.zeros(3), np.full(3, CAP_REPRODUCERS[0][3])).x
    assert np.max(np.abs(x3 - np.linalg.solve(np.array(CAP_REPRODUCERS[0][1]), -np.array(CAP_REPRODUCERS[0][2])))) < 1e-12
    print(f"VERIFY qp_box py_default_cap_reproducers (n,nit,kkt_stat)={[(n, k, float(f'{s:.1e}')) for n, k, s in out]} old_cap=4n_fails=ok")


def test_failures_and_options():
    o = QpBoxOptions()
    assert (o.max_iter, o.tol_step, o.tol_dual) == (None, 1e-10, 1e-9)
    H = np.array([[2.0, 0.5], [0.5, 1.0]])
    g, lo, hi = np.array([1.0, -2.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    ok = qp_box(H, g, lo, hi)
    assert ok.success and ok.message == "converged"
    # semidefinite and indefinite H: reported, x = clamp(0), nothing else done
    for Hbad in (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[1.0, 3.0], [3.0, 1.0]])):
        r = qp_box(Hbad, g, lo, hi)
        assert not r.success and "not positive definite" in r.message and r.nit == 0 and r.nfact == 1
        assert np.array_equal(r.x, [0.0, 0.0]) and np.isfinite(r.f)
    r = qp_box(np.array([[1.0, 1.0], [1.0, 1.0]]), g, [0.5, -1.0], [1.0, 1.0])   # clamp(0) respects the bounds
    assert np.array_equal(r.x, [0.5, 0.0])
    # invalid input: nfact 0, f nan
    bad = [(H, g, [1.0, 0.0], [0.0, 1.0]), (H, g, [np.nan, 0.0], hi), (H, g, lo, [1.0, np.nan]),
           (H, g, [np.inf, 0.0], [np.inf, 1.0]), (H, g, lo, [-np.inf, 1.0]), (np.array([[np.nan, 0.0], [0.0, 1.0]]), g, lo, hi),
           (H, [np.inf, 0.0], lo, hi), (H, g, [-1.0], hi), (np.ones((2, 3)), g, lo, hi)]
    for k, args in enumerate(bad):
        r = qp_box(*args)
        assert not r.success and "invalid input" in r.message and r.nfact == 0 and r.nit == 0 and np.isnan(r.f), (k, r.message)
    # invalid options
    for opts in (QpBoxOptions(max_iter=0), QpBoxOptions(tol_step=-1.0), QpBoxOptions(tol_step=float("nan")),
                 QpBoxOptions(tol_dual=float("inf")), QpBoxOptions(tol_dual=-1e-9)):
        r = qp_box(H, g, lo, hi, opts)
        assert not r.success and "invalid option" in r.message and r.nfact == 0
    # iteration cap: returns the feasible last iterate and says so
    c = load_reference("qp_box")["cases"]["random_n6"]
    Hc, gc, loc, hic = problem(c)
    r = qp_box(Hc, gc, loc, hic, QpBoxOptions(max_iter=2))
    assert not r.success and "max_iter" in r.message and r.nit == 2
    assert np.all(r.x >= loc) and np.all(r.x <= hic) and np.isfinite(r.f)
    full = qp_box(Hc, gc, loc, hic)
    assert r.f > full.f                     # not yet optimal; each iteration descends
    assert qp_box(Hc, gc, loc, hic, QpBoxOptions(max_iter=full.nit)).success
    assert not qp_box(Hc, gc, loc, hic, QpBoxOptions(max_iter=full.nit - 1)).success
    # only the lower triangle of H is read
    Hl = np.tril(Hc) + np.tril(Hc, -1).T
    Hjunk = np.tril(Hc) + 1e6 * np.triu(np.ones_like(Hc), 1)
    assert np.array_equal(qp_box(Hjunk, gc, loc, hic).x, qp_box(Hl, gc, loc, hic).x)
    # a looser tol_dual keeps a slightly wrong-signed bound: x_0 stays at its lower bound
    H1, g1 = np.array([[1.0]]), np.array([-1e-6])        # optimum x = 1e-6 inside [0, 1]; start clamp(0) = 0 is at the lower bound
    assert qp_box(H1, g1, [0.0], [1.0]).x[0] == 1e-6
    r = qp_box(H1, g1, [0.0], [1.0], QpBoxOptions(tol_dual=1e-5))
    assert r.success and r.x[0] == 0.0 and r.active[0] == -1 and r.multipliers[0] == 0.0
    print("VERIFY qp_box py_failure_modes semidefinite,indefinite,invalid_input(9),invalid_option(5),max_iter,lower_triangle_only,tol_dual=ok")


def test_python_vs_rust_example():
    py, rs = python_example("qp_box"), rust_example("qp_box")
    assert set(py) == set(rs) and len(py) == 9
    for k in py:
        a, b = py[k], rs[k]
        assert a["success"] and b["success"]
        assert (a["nit"], a["nfact"]) == (b["nit"], b["nfact"]) and a["active"] == b["active"], k
        dx = np.max(np.abs(np.array(a["x"]) - b["x"]))
        dm = np.max(np.abs(np.array(a["multipliers"]) - b["multipliers"]))
        df = abs(a["f"] - b["f"])
        print(f"VERIFY qp_box python_vs_rust_{k} max_diff_x={dx:.2e} max_diff_mult={dm:.2e} diff_f={df:.2e} "
              f"nit={a['nit']} nfact={a['nfact']} identical_active_set=1")
        assert dx <= TOL_EXAMPLE and dm <= TOL_EXAMPLE and df <= TOL_EXAMPLE * (1 + abs(a["f"])), k
