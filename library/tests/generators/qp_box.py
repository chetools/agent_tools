"""Generator for reference_data/qp_box.json (and the ported muse fixture reference_data/qp_box_muse_fixture.json).

Problem: minimize 0.5 x.H.x + g.x subject to lo <= x <= hi, H symmetric positive definite (dimensionless).
Infinite bounds are stored as JSON null (lo null = -inf, hi null = +inf). H is stored row-major flat.

Reference solutions, independent of the library:
  scipy      scipy.optimize.minimize(method="L-BFGS-B", jac=analytic, ftol=1e-15, gtol=1e-12, maxiter=1e5).
  enum       n <= 6: all 3^n active-set patterns (variable at lower bound / free / at upper bound); for each the
             free variables are solved by numpy.linalg.solve; KKT points (feasible, multipliers >= -1e-9 scale) are
             kept and the one of lowest objective is stored (strictly convex: the KKT point is unique, ties
             are the same x). Stores x, f, active (-1/0/+1), multipliers (>= 0).
  exact      the same enumeration in exact rational arithmetic (fractions.Fraction, the float data taken as
             exact rationals), with exact KKT tests (no tolerances); used for the ill-conditioned cases
             (cond(H) = 1e8), where SciPy's L-BFGS-B is accurate only to about 1e-3 in x. For n > 6 the active
             pattern guessed from SciPy is refined by an exact rational solve and then verified to be an exact KKT
             point (feasible, multipliers >= 0); the generator asserts this.
x_ref is the best available reference: exact, else enum, else scipy.
Cases are built by seeded numpy.random.RandomState streams (SEED = 20261029 + case index). "constructed" cases choose a
KKT point (x*, active set, multipliers) first and set g = q - H x*, so the answer is known a priori, including a
zero multiplier on an active bound (degeneracy).
Run: python library/tests/generators/qp_box.py
"""
import itertools
import json
import pathlib
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize

from _common import OUT, write

SEED = 20261029
INF = float("inf")


def spd(rng, n, cond, decimals):
    Q, _ = np.linalg.qr(rng.randn(n, n))
    H = Q @ np.diag(np.logspace(0, np.log10(cond), n)) @ Q.T
    H = np.round(H, decimals)
    H = np.triu(H) + np.triu(H, 1).T
    assert np.linalg.eigvalsh(H)[0] > 0
    return H


def scipy_ref(H, g, lo, hi):
    n = len(g)
    r = minimize(lambda x: (0.5 * x @ H @ x + g @ x, H @ x + g), np.clip(np.zeros(n), lo, hi), jac=True,
                 method="L-BFGS-B", bounds=[(None if np.isinf(a) else a, None if np.isinf(b) else b) for a, b in zip(lo, hi)],
                 options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 100000, "maxfun": 1000000})
    return {"x": r.x.tolist(), "fun": float(r.fun), "nit": int(r.nit), "nfev": int(r.nfev), "success": bool(r.success)}


def enum_ref(H, g, lo, hi):
    n = len(g)
    best, scale = None, max(1.0, np.max(np.abs(g)), np.max(np.abs(H)) * max(np.max(np.abs(lo[np.isfinite(lo)])) if np.isfinite(lo).any() else 1, 1))
    kkt = []
    for pat in itertools.product((-1, 0, 1), repeat=n):
        pat = np.array(pat)
        x = np.zeros(n)
        ok = True
        for i in range(n):
            if pat[i] == -1:
                x[i] = lo[i]
            elif pat[i] == 1:
                x[i] = hi[i]
            ok = ok and np.isfinite(x[i])
        if not ok:
            continue
        F = np.where(pat == 0)[0]
        A = np.where(pat != 0)[0]
        if len(F):
            x[F] = np.linalg.solve(H[np.ix_(F, F)], -(g[F] + H[np.ix_(F, A)] @ x[A]))
        if np.any(x < lo - 1e-9 * scale) or np.any(x > hi + 1e-9 * scale):
            continue
        q = H @ x + g
        mult = np.where(pat == -1, q, np.where(pat == 1, -q, 0.0))
        if np.any(mult < -1e-9 * scale):
            continue
        kkt.append((float(0.5 * x @ H @ x + g @ x), x, pat, np.maximum(mult, 0.0)))
    assert kkt, "no KKT point found"
    kkt.sort(key=lambda t: t[0])
    for t in kkt:
        assert np.max(np.abs(t[1] - kkt[0][1])) < 1e-6 * scale, "KKT points differ: not strictly convex?"
    f, x, pat, mult = kkt[0]
    return {"x": x.tolist(), "f": f, "active": pat.tolist(), "multipliers": mult.tolist(), "n_kkt_patterns": len(kkt)}


def fsolve(M, b):
    """Exact Gauss-Jordan solve of M x = b in Fractions (M SPD, no pivoting needed beyond nonzero pivots)."""
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c] / a[c][c]
                a[r] = [u - f * v for u, v in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def exact_for_pattern(H, g, lo, hi, pat):
    """Exact rational x for a pattern, and exact KKT test. Returns (x, mult, ok)."""
    n = len(g)
    H = [[Fraction(v) for v in row] for row in H]
    g = [Fraction(v) for v in g]
    bound = lambda i: Fraction(lo[i]) if pat[i] == -1 else Fraction(hi[i])
    if any(pat[i] != 0 and not np.isfinite(lo[i] if pat[i] == -1 else hi[i]) for i in range(n)):
        return None
    x = [bound(i) if pat[i] != 0 else Fraction(0) for i in range(n)]
    F = [i for i in range(n) if pat[i] == 0]
    A = [i for i in range(n) if pat[i] != 0]
    if F:
        rhs = [-(g[i] + sum(H[i][j] * x[j] for j in A)) for i in F]
        xf = fsolve([[H[i][j] for j in F] for i in F], rhs)
        for k, i in enumerate(F):
            x[i] = xf[k]
    for i in range(n):
        if (np.isfinite(lo[i]) and x[i] < Fraction(lo[i])) or (np.isfinite(hi[i]) and x[i] > Fraction(hi[i])):
            return None
    q = [sum(H[i][j] * x[j] for j in range(n)) + g[i] for i in range(n)]
    mult = [q[i] if pat[i] == -1 else (-q[i] if pat[i] == 1 else Fraction(0)) for i in range(n)]
    if any(m < 0 for m in mult):
        return None
    return x, mult


def exact_enum(H, g, lo, hi):
    n = len(g)
    for pat in itertools.product((-1, 0, 1), repeat=n):
        r = exact_for_pattern(H, g, lo, hi, pat)
        if r is not None:
            x, mult = r
            Hf = [[Fraction(v) for v in row] for row in H]
            f = sum(Fraction(g[i]) * x[i] for i in range(n)) + sum(x[i] * Hf[i][j] * x[j] for i in range(n) for j in range(n)) / 2
            return {"x": [float(v) for v in x], "f": float(f), "active": list(pat), "multipliers": [float(m) for m in mult]}
    raise AssertionError("no exact KKT pattern")


def exact_refine(H, g, lo, hi, xs):
    n = len(g)
    tol = 1e-9
    pat = tuple(-1 if xs[i] <= lo[i] + tol * max(1, abs(lo[i])) else (1 if xs[i] >= hi[i] - tol * max(1, abs(hi[i])) else 0) for i in range(n))
    r = exact_for_pattern(H, g, lo, hi, pat)
    assert r is not None, "SciPy's active pattern is not an exact KKT point"
    x, mult = r
    Hf = [[Fraction(v) for v in row] for row in H]
    f = sum(Fraction(g[i]) * x[i] for i in range(n)) + sum(x[i] * Hf[i][j] * x[j] for i in range(n) for j in range(n)) / 2
    return {"x": [float(v) for v in x], "f": float(f), "active": list(pat), "multipliers": [float(m) for m in mult]}


def tojson(v):
    return [None if not np.isfinite(t) else float(t) for t in v]


def case(name, kind, H, g, lo, hi, exact=None, enum=None, note=""):
    n = len(g)
    c = {"kind": kind, "n": n, "H": H.ravel().tolist(), "g": list(map(float, g)), "lo": tojson(lo), "hi": tojson(hi),
         "scipy": scipy_ref(H, g, lo, hi), "note": note}
    if enum is None:
        enum = n <= 6 and exact is None
    if enum:
        c["enum"] = enum_ref(H, g, lo, hi)
    if exact == "enum":
        c["exact"] = exact_enum(H, g, lo, hi)
    elif exact == "refine":
        c["exact"] = exact_refine(H, g, lo, hi, np.array(c["scipy"]["x"]))
    ref = c.get("exact") or c.get("enum")
    c["x_ref"] = ref["x"] if ref else c["scipy"]["x"]
    c["x_ref_source"] = "exact" if "exact" in c else ("enum" if "enum" in c else "scipy")
    c["active_ref"] = ref["active"] if ref else None
    return c


def constructed(rng, H, active, mult_list):
    """Pick a KKT point: bounds around x*, active pattern, multipliers; g = q - H x*."""
    n = len(active)
    xs = rng.uniform(-1, 1, n)
    lo = xs - rng.uniform(0.3, 1.0, n)
    hi = xs + rng.uniform(0.3, 1.0, n)
    q = np.zeros(n)
    for i in range(n):
        if active[i] == -1:
            xs[i] = lo[i]
            q[i] = mult_list[i]
        elif active[i] == 1:
            xs[i] = hi[i]
            q[i] = -mult_list[i]
    g = q - H @ xs
    return g, lo, hi, xs


cases = {}
idx = 0


def rs():
    global idx
    idx += 1
    return np.random.RandomState(SEED + idx)


# random with finite bounds, mixed active set; enumeration available
for n in (4, 6):
    rng = rs()
    H = spd(rng, n, 1e2, 8)
    g = np.round(3 * rng.randn(n), 8)
    lo = np.round(-rng.uniform(0.2, 1, n), 8)
    hi = np.round(rng.uniform(0.2, 1, n), 8)
    cases[f"random_n{n}"] = case(f"random_n{n}", "random", H, g, lo, hi, note="mixed active set")

# infinite bounds: one-sided and free
rng = rs()
n = 6
H = spd(rng, n, 1e2, 8)
g = np.round(3 * rng.randn(n), 8)
lo = np.array([-INF, 0.0, -INF, -0.3, -INF, 0.0])
hi = np.array([INF, INF, 0.5, INF, INF, 0.4])
cases["infinite_bounds_n6"] = case("infinite_bounds_n6", "infinite_bounds", H, g, lo, hi, note="mixed one-sided and infinite bounds")

# unconstrained: all bounds infinite, and wide finite bounds; reference = numpy.linalg.solve
for name, big in (("unconstrained_inf_n10", INF), ("unconstrained_wide_n10", 1e3)):
    rng = rs()
    n = 10
    H = spd(rng, n, 1e3, 8)
    g = np.round(rng.randn(n), 8)
    c = case(name, "unconstrained", H, g, np.full(n, -big), np.full(n, big), note="no bound active; x = -H^-1 g (numpy.linalg.solve)")
    xs = np.linalg.solve(H, -g)
    c["x_ref"] = xs.tolist()
    c["x_ref_source"] = "numpy.linalg.solve"
    c["active_ref"] = [0] * n
    cases[name] = c

# all bounds active: large linear term pushes every variable to a bound
rng = rs()
n = 5
H = spd(rng, n, 1e2, 8)
g = np.round(rng.choice([-1, 1], n) * rng.uniform(80, 120, n), 8)
lo = np.round(-rng.uniform(0.2, 1, n), 8)
hi = np.round(rng.uniform(0.2, 1, n), 8)
cases["all_active_n5"] = case("all_active_n5", "all_active", H, g, lo, hi, note="every variable ends at a bound")

# degenerate: active bounds with zero multiplier (constructed), mixed with a free variable and nonzero multipliers
rng = rs()
n = 5
H = spd(rng, n, 1e2, 8)
active = [-1, 1, 0, -1, 1]
mult = [0.0, 0.0, 0.0, 1.5, 0.7]
g, lo, hi, xs = constructed(rng, H, active, mult)
g, lo, hi = np.round(g, 10), np.round(lo, 10), np.round(hi, 10)
cases["degenerate_zero_multiplier_n5"] = case("degenerate_zero_multiplier_n5", "degenerate", H, g, lo, hi,
                                              note="constructed: x_0 at lower and x_1 at upper bound with multiplier exactly 0 (to rounding of g), x_3, x_4 active with positive multipliers, x_2 free")
cases["degenerate_zero_multiplier_n5"]["constructed"] = {"active": active, "multipliers": mult, "x": xs.tolist()}

# degenerate: starting point 0 is exactly at a bound and stationary there
n = 4
rng = rs()
H = spd(rng, n, 1e2, 8)
g = np.array([0.0, 0.0, 0.0, 0.0]) - H @ np.zeros(n)
lo = np.array([0.0, -1.0, 0.0, -1.0])
hi = np.array([1.0, 1.0, 1.0, 0.0])
cases["start_at_bound_zero_gradient_n4"] = case("start_at_bound_zero_gradient_n4", "degenerate", H, g, lo, hi,
                                                note="g = 0: the minimizer is x = 0, which sits at a bound for x_0, x_2 (lower) and x_3 (upper), all with multiplier 0")

# fixed variable lo == hi
rng = rs()
n = 5
H = spd(rng, n, 1e2, 8)
g = np.round(2 * rng.randn(n), 8)
lo = np.round(-rng.uniform(0.2, 1, n), 8)
hi = np.round(rng.uniform(0.2, 1, n), 8)
lo[1] = hi[1] = 0.3
lo[3] = hi[3] = -0.2
cases["fixed_variable_n5"] = case("fixed_variable_n5", "fixed", H, g, lo, hi, note="x_1 and x_3 fixed (lo == hi); their multipliers have either sign")

# hard: cond 1e8, exact rational enumeration (n = 6)
rng = rs()
n = 6
H = spd(rng, n, 1e8, 4)
g = np.round(rng.randn(n) * 3e3, 6)
lo = np.round(-rng.uniform(0.1, 1, n), 6)
hi = np.round(rng.uniform(0.1, 1, n), 6)
cases["cond1e8_n6"] = case("cond1e8_n6", "ill_conditioned", H, g, lo, hi, exact="enum", note="cond(H) = 1e8 (eigenvalues 1 to 1e8); reference by exact rational enumeration of all 729 patterns")

# hard: cond 1e8, n = 30 and n = 50, pattern from SciPy refined and verified in exact rational arithmetic
for n in (30, 50):
    for attempt in range(20):
        rng = rs()
        H = spd(rng, n, 1e8, 4)
        g = np.round(rng.randn(n) * 3e3, 6)
        lo = np.round(-rng.uniform(0.05, 0.5, n), 6)
        hi = np.round(rng.uniform(0.05, 0.5, n), 6)
        try:
            cases[f"cond1e8_n{n}"] = case(f"cond1e8_n{n}", "ill_conditioned", H, g, lo, hi, exact="refine",
                                          note="cond(H) = 1e8; reference: SciPy's active pattern refined and verified as an exact KKT point in rational arithmetic")
            break
        except AssertionError:
            continue
    else:
        raise SystemExit("no seed passed exact verification")

# random sweep n = 3 .. 50 with condition numbers 1 to 1e4
for n, cond in ((3, 1e0), (5, 1e1), (8, 1e2), (12, 1e3), (16, 1e4), (20, 1e2), (25, 1e3), (30, 1e4), (40, 1e2), (50, 1e3)):
    rng = rs()
    H = spd(rng, n, cond, 8)
    gscale = 2 * np.sqrt(np.mean(np.diag(H)))
    g = np.round(gscale * rng.randn(n), 8)
    lo = np.round(-rng.uniform(0.1, 1, n), 8)
    hi = np.round(rng.uniform(0.1, 1, n), 8)
    cases[f"sweep_n{n}_cond{cond:.0e}"] = case(f"sweep_n{n}", "sweep", H, g, lo, hi, exact="refine", note="random; reference refined from SciPy's pattern, verified exactly")

for k, c in cases.items():
    c["n_active_ref"] = None if c["active_ref"] is None else int(np.count_nonzero(c["active_ref"]))
    print(k, c["n"], c["x_ref_source"], c["n_active_ref"], "scipy-ref max diff", np.max(np.abs(np.array(c["x_ref"]) - c["scipy"]["x"])))

write("qp_box", "library/tests/generators/qp_box.py",
      {"seed": SEED, "rng": "numpy.random.RandomState(SEED + case index)",
       "scipy_reference": "scipy.optimize.minimize L-BFGS-B ftol=1e-15 gtol=1e-12 maxiter=1e5",
       "enumeration": "all 3^n active-set patterns, numpy.linalg.solve (n <= 6); exact: fractions.Fraction",
       "infinite_bounds": "JSON null (lo null = -inf, hi null = +inf)"},
      "Dimensionless QPs: minimize 0.5 x.H.x + g.x subject to lo <= x <= hi, H row-major flat SPD. Cases carry x_ref (exact, "
      "enumerated or SciPy: see x_ref_source), active_ref (-1 lower, 0 free, +1 upper) and the SciPy result for comparison.",
      cases)

# ---- the ported muse fixture -------------------------------------------------------------------------------------
src = pathlib.Path("/home/user/chetools/muse/twin-cstr/solver/tests/fixtures/qp.json")
fx = json.load(open(src))
muse_cases = {}
for i, c in enumerate(fx["cases"]):
    n = c["n"]
    muse_cases[f"muse_{i:02d}_n{n}"] = {"n": n, "H": [v for row in c["B"] for v in row], "g": c["g"], "lo": c["lb"], "hi": c["ub"],
                                        "x_ref": c["x_ref"], "proto_dev": c["proto_dev"]}
write("qp_box_muse_fixture", "copied from chetools/muse twin-cstr/solver/tests/fixtures/qp.json by library/tests/generators/qp_box.py",
      {"origin": "chetools/muse, twin-cstr/solver/tests/fixtures/qp.json (26 random positive-definite box QPs, n = 4, 8, 16, 32, 124)",
       "reference": "scipy.optimize.minimize L-BFGS-B ftol=1e-14 gtol=1e-12 (per the muse test header qp_solver.rs; generated there by /tmp/gen_qp_fixtures.py, not available)",
       "proto_dev": "deviation of the muse numpy active-set prototype from the reference, kept from the source"},
      "Dimensionless. Same layout as qp_box.json except H is the muse key B and bounds lb/ub are renamed lo/hi; all bounds finite.",
      muse_cases)
