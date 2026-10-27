"""Generator for reference_data/block_tridiagonal.json.

Cases (blocks rounded to 6 decimals; the rounded blocks ARE the test matrix):
  dominant: random block-diagonally-dominant systems (m, nb) in {(1,5),(2,50),(3,200),(5,40)}, 3 right-hand
            sides each. Reference x = numpy.linalg.solve on the assembled dense matrix, cross-checked against
            scipy.linalg.solve_banded (bandwidth 2m-1) in this script. For m = 1 the classical Thomas
            algorithm (scalar, written here) is stored as x_thomas.
  weak:     weakly dominant 1D Laplacian blocks (m=1, nb=200: tridiag(-1,2,-1)) and a block Laplacian (m=2).
  pivot:    m=2 diagonal blocks with zero diagonal entries ([[0,d],[e,0]] + weak coupling): needs pivoting
            inside the blocks; scalar Thomas without pivoting would divide by zero.
  fail:     nonsingular matrices whose first diagonal block is singular (no cross-block pivoting): block
            Thomas must report failure although numpy solves them (x stored).
  param:    A(p) = A0 + p A1, b(p) = b0 + p b1; dx = A^-1 (b1 - A1 x) via numpy, checked by central differences.
The generator also asserts the block diagonal dominance condition max_i(||M_i^-1 L|| + ||M_i^-1 U||) < 1 for
the 'dominant' and 'pivot' cases using an independent numpy block elimination (inf-norm).
Run: python library/tests/generators/block_tridiagonal.py
"""
import numpy as np
import scipy.linalg as sla

from _common import write

SEED = 20261028
rng = np.random.RandomState(SEED)
R = lambda a: np.round(a, 6)


def assemble(L, D, U):
    nb, m = D.shape[0], D.shape[1]
    A = np.zeros((m * nb, m * nb))
    for i in range(nb):
        A[i*m:(i+1)*m, i*m:(i+1)*m] = D[i]
        if i + 1 < nb:
            A[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = U[i]
            A[(i+1)*m:(i+2)*m, i*m:(i+1)*m] = L[i]
    return A


def banded_solve(A, b, m):
    k = 2 * m - 1
    n = A.shape[0]
    ab = np.zeros((2 * k + 1, n))
    for d in range(-k, k + 1):                       # diagonal offset d: ab[k - d, j] = A[j - d, j]
        for j in range(n):
            i = j - d
            if 0 <= i < n:
                ab[k - d, j] = A[i, j]
    return sla.solve_banded((k, k), ab, b)


def dominance(L, D, U):
    """max_i ||M_i^-1 L_{i-1}||_inf + ||M_i^-1 U_i||_inf via independent numpy block elimination."""
    nb = D.shape[0]
    worst, C = 0.0, None
    for i in range(nb):
        M = D[i] - (L[i-1] @ C if i else 0.0)
        Mi = np.linalg.inv(M)
        t = np.linalg.norm(Mi @ L[i-1], np.inf) if i else 0.0
        if i + 1 < nb:
            C = Mi @ U[i]
            t += np.linalg.norm(C, np.inf)
        worst = max(worst, t)
    return worst


def thomas(a, b, c, d):
    """Classical scalar Thomas algorithm: sub a[1:], diag b, super c[:-1], rhs d."""
    n = len(b)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / den
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


def case(name, L, D, U, nrhs=3, check_banded=True, with_thomas=True, extra=None):
    nb, m = D.shape[0], D.shape[1]
    A = assemble(L, D, U)
    B = R(rng.randn(nrhs, m * nb))
    X = [np.linalg.solve(A, b) for b in B]
    if check_banded:
        for b, x in zip(B, X):
            xb = banded_solve(A, b, m)
            assert np.max(np.abs(x - xb)) < 1e-9 * np.max(np.abs(x)) * np.linalg.cond(A), name
    row = {"name": name, "m": m, "nb": nb, "lower": L.ravel().tolist(), "diag": D.ravel().tolist(),
           "upper": U.ravel().tolist(), "b": B.tolist(), "x": [x.tolist() for x in X],
           "cond": float(np.linalg.cond(A))}
    if m == 1 and with_thomas:   # classical Thomas is undefined (0/0) for the 'fail' cases
        a = np.r_[0.0, L.ravel()]
        c = np.r_[U.ravel(), 0.0]
        row["x_thomas"] = [thomas(a, D.ravel(), c, b).tolist() for b in B]
        for xt, x in zip(row["x_thomas"], X):
            assert np.max(np.abs(np.array(xt) - x)) < 1e-10 * np.max(np.abs(x)) * row["cond"], name
    if extra:
        row.update(extra)
    return row


dominant = []
for m, nb in [(1, 5), (2, 50), (3, 200), (5, 40)]:
    L = R(rng.randn(nb - 1, m, m))
    U = R(rng.randn(nb - 1, m, m))
    D = R(rng.randn(nb, m, m) + (4.0 * m + 2.0) * np.eye(m))
    dom = dominance(L, D, U)
    assert dom < 1.0, (m, nb, dom)
    dominant.append(case(f"dominant_m{m}_nb{nb}", L, D, U, extra={"dominance": dom}))

weak = []
nb = 200
weak.append(case("laplacian_m1_nb200", -np.ones((nb - 1, 1, 1)), 2.0 * np.ones((nb, 1, 1)), -np.ones((nb - 1, 1, 1))))
nb = 100
Ib = np.eye(2)
Dl = np.array([[2.0, -0.5], [-0.5, 2.0]])           # block Laplacian: -I, 2I + coupling, -I (weakly dominant)
weak.append(case("laplacian_m2_nb100", -np.tile(Ib, (nb - 1, 1, 1)), np.tile(Dl, (nb, 1, 1)),
                 -np.tile(Ib, (nb - 1, 1, 1))))

pivot = []
nb, m = 30, 2
d = rng.uniform(1.5, 2.5, nb)
e = rng.uniform(1.5, 2.5, nb)
D = np.zeros((nb, 2, 2))
D[:, 0, 1], D[:, 1, 0] = d, e
D = R(D)
L = R(0.1 * rng.randn(nb - 1, 2, 2))
U = R(0.1 * rng.randn(nb - 1, 2, 2))
dom = dominance(L, D, U)
assert dom < 1.0, dom
pivot.append(case("zero_diag_blocks_m2_nb30", L, D, U, extra={"dominance": dom}))

fail = []
fail.append(case("scalar_swap_m1_nb2", np.array([[[1.0]]]), np.array([[[0.0]], [[0.0]]]), np.array([[[1.0]]]), with_thomas=False))
D = np.array([[[1.0, 2.0], [2.0, 4.0]], [[3.0, 0.0], [0.0, 3.0]]])           # D_0 rank 1
L = np.array([[[1.0, 0.0], [0.0, 1.0]]])
U = np.array([[[1.0, 0.0], [0.0, 1.0]]])
fail.append(case("singular_first_block_m2_nb2", L, D, U))
for c in fail:
    c["fail_block"] = 0

param = []
for m, nb in [(2, 6), (3, 10)]:
    L0, L1 = R(rng.randn(nb - 1, m, m)), R(rng.randn(nb - 1, m, m))
    U0, U1 = R(rng.randn(nb - 1, m, m)), R(rng.randn(nb - 1, m, m))
    D0 = R(rng.randn(nb, m, m) + (4.0 * m + 2.0) * np.eye(m))
    D1 = R(rng.randn(nb, m, m))
    b0, b1 = R(rng.randn(m * nb)), R(rng.randn(m * nb))
    p0 = 0.3
    A = lambda p: assemble(L0 + p * L1, D0 + p * D1, U0 + p * U1)
    Ap, A1 = A(p0), assemble(L1, D1, U1)
    x = np.linalg.solve(Ap, b0 + p0 * b1)
    dx = np.linalg.solve(Ap, b1 - A1 @ x)
    h = 1e-6
    fd = (np.linalg.solve(A(p0 + h), b0 + (p0 + h) * b1) - np.linalg.solve(A(p0 - h), b0 + (p0 - h) * b1)) / (2 * h)
    assert np.max(np.abs(fd - dx)) < 1e-7 * np.max(np.abs(dx)), "analytic derivative check"
    param.append({"m": m, "nb": nb, "p0": p0, "L0": L0.ravel().tolist(), "L1": L1.ravel().tolist(),
                  "D0": D0.ravel().tolist(), "D1": D1.ravel().tolist(), "U0": U0.ravel().tolist(),
                  "U1": U1.ravel().tolist(), "b0": b0.tolist(), "b1": b1.tolist(),
                  "x": x.tolist(), "dx": dx.tolist()})

write("block_tridiagonal", "library/tests/generators/block_tridiagonal.py",
      {"seed": SEED, "rng": "numpy.random.RandomState", "rounding_decimals": 6,
       "reference_x": "numpy.linalg.solve on the assembled dense matrix",
       "cross_check": "scipy.linalg.solve_banded((2m-1, 2m-1)); classical Thomas for m = 1"},
      "Dimensionless. Blocks are row-major m*m, stored back to back: lower[k] = block (k+1,k), upper[k] = "
      "block (k,k+1), diag has nb blocks. 'b' and 'x' are lists of right-hand sides / solutions. 'dominance' = "
      "max_i(||M_i^-1 L||+||M_i^-1 U||) in the inf-norm (< 1 required, asserted). 'param': A(p) = A0 + p A1, "
      "b(p) = b0 + p b1; dx = A^-1 (b1 - A1 x), checked by central differences (h=1e-6).",
      {"dominant": dominant, "weak": weak, "pivot": pivot, "fail": fail, "param": param})
