"""Independent checks of library/python/gpu/multigrid_poisson_cc.py (float64 reference).

1. The 7-point operator equals an independently assembled sparse matrix (Kronecker sums of 1D operators).
2. The multigrid solution equals a direct sparse solve (scipy.sparse.linalg.spsolve of the bordered system that
   fixes the constant null space), walls-z and periodic-z, incl. NZ = 1, NZ = 2 and a non-default option set.
3. Manufactured analytic solution (cosine modes, zero-flux compatible): second-order convergence in h.
4. Convergence factor per V-cycle; mean-removal (solvability); z-uniform preservation; size validation.
"""
import pathlib
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from gpu import multigrid_poisson_cc as mg  # noqa: E402


def op1d(m, periodic):
    """1D second-difference operator (without 1/h^2): zero-flux ends, or periodic (np.roll, so m = 2 doubles the coupling)."""
    if m == 1:
        return sp.csr_matrix((1, 1))
    I = np.eye(m)
    if periodic:
        T = np.roll(I, 1, axis=0) + np.roll(I, -1, axis=0) - 2 * I
    else:
        T = np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1) - 2 * I
        T[0, 0] = T[-1, -1] = -1
    return sp.csr_matrix(T)


def matrix(n, nz, h, bc_z):
    """Dense-free assembly with the index order (k*n + j)*n + i."""
    Tx, Tz = op1d(n, False), op1d(nz, bc_z == "periodic")
    Ix, Iz = sp.identity(n), sp.identity(nz)
    A = sp.kron(Iz, sp.kron(Ix, Tx)) + sp.kron(Iz, sp.kron(Tx, Ix)) + sp.kron(Tz, sp.kron(Ix, Ix))
    return (A / h ** 2).tocsc()


def direct(b, h, bc_z):
    """Zero-mean solution of A x = b - mean(b) via the bordered system [[A, 1], [1^T, 0]]."""
    nz, n = b.shape[0], b.shape[1]
    A = matrix(n, nz, h, bc_z)
    m = A.shape[0]
    one = sp.csr_matrix(np.ones((m, 1)))
    K = sp.bmat([[A, one], [one.T, None]], format="csc")
    rhs = np.concatenate([(b - b.mean()).ravel(), [0.0]])
    return spla.spsolve(K, rhs)[:m].reshape(b.shape)


CASES = [(16, 8, "walls"), (16, 8, "periodic"), (32, 16, "walls"), (32, 16, "periodic"),
         (16, 1, "walls"), (8, 2, "periodic"), (8, 2, "walls"), (4, 4, "periodic"), (64, 4, "walls")]


@pytest.mark.parametrize("n,nz,bc", CASES)
def test_operator_matches_sparse_matrix(n, nz, bc):
    h = 0.37
    x = np.random.default_rng(0).standard_normal((nz, n, n))
    ref = (matrix(n, nz, h, bc) @ x.ravel()).reshape(x.shape)
    err = np.abs(mg.laplacian(x, h, bc) - ref).max() / np.abs(ref).max()
    assert err < 1e-13, err


@pytest.mark.parametrize("n,nz,bc", CASES)
def test_solution_matches_direct_solve(n, nz, bc):
    h = 0.1
    b = np.random.default_rng(1).standard_normal((nz, n, n)) + 0.7      # nonzero mean on purpose
    res = mg.solve(b, h, bc, cycles=60, rtol=1e-13)
    xd = direct(b, h, bc)
    err = np.abs(res["x"] - xd).max() / np.abs(xd).max()
    print(f"{n}x{n}x{nz} {bc}: |x_mg - x_direct|/|x| = {err:.2e}, cycles {res['cycles']}")
    assert res["success"] and res["converged"]
    assert err < 1e-9, err
    assert abs(res["x"].mean()) < 1e-13


def test_nondefault_options_and_start_guess():
    n, nz, h = 16, 4, 0.1
    rng = np.random.default_rng(2)
    b = rng.standard_normal((nz, n, n))
    res = mg.solve(b, h, "walls", cycles=80, nu1=1, nu2=3, ncoarse=10, x0=rng.standard_normal((nz, n, n)), rtol=1e-13)
    xd = direct(b, h, "walls")
    assert res["converged"] and np.abs(res["x"] - xd).max() / np.abs(xd).max() < 1e-9


@pytest.mark.parametrize("bc", ["periodic", "walls"])
def test_manufactured_cosine_solution_is_second_order(bc):
    """u = cos(a pi x/L) cos(b pi y/L) cos(c pi z/Lz) (walls) or cos(2 pi c z/Lz) (periodic); cell-centre values."""
    errs = []
    for n in (16, 32, 64):
        nz, h = n // 2, 1.0 / n                                    # box 1 x 1 x 1/2
        Lx, Lz = n * h, nz * h
        k, j, i = np.meshgrid(np.arange(nz), np.arange(n), np.arange(n), indexing="ij")
        x, y, z = (i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h
        kz = 2 * np.pi / Lz if bc == "periodic" else np.pi / Lz
        kx, ky = 2 * np.pi / Lx, 3 * np.pi / Lx
        u = np.cos(kx * x) * np.cos(ky * y) * np.cos(kz * z)       # even about every wall, zero mean
        b = -(kx ** 2 + ky ** 2 + kz ** 2) * u                     # analytic laplacian
        res = mg.solve(b, h, bc, cycles=40, rtol=1e-13)
        errs.append(np.abs(res["x"] - u).max() / np.abs(u).max())
    orders = [np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])]
    print(bc, "errors", errs, "orders", orders)
    assert errs[2] < 5e-3
    assert all(1.9 < o < 2.1 for o in orders), orders


@pytest.mark.parametrize("bc", ["periodic", "walls"])
def test_convergence_factor_per_vcycle(bc):
    """Measured on the stored-test sizes: asymptotic factor about 0.13-0.15 (3D) and 0.07 (2D); bound 0.3."""
    for n, nz in [(32, 16), (64, 32), (32, 1)]:
        b = np.random.default_rng(3).standard_normal((nz, n, n))
        res = mg.solve(b, 1.0 / n, bc, cycles=10)
        r = np.array(res["residual_rms"])
        f = r[1:] / r[:-1]
        gm = (r[-1] / r[3]) ** (1 / (len(r) - 4))
        print(f"{bc} {n}x{n}x{nz}: factors {np.round(f, 3)}, geometric mean cycles 3-10: {gm:.3f}")
        assert np.all(f < 0.5) and gm < 0.3


def test_default_two_cycles_reports_history():
    b = np.random.default_rng(4).standard_normal((8, 16, 16))
    res = mg.solve(b, 0.1)
    assert res["cycles"] == 2 and len(res["residual_rms"]) == 3 and not res["converged"] and res["success"]


def test_solvability_mean_removal():
    n, nz, h = 16, 8, 0.1
    b = np.random.default_rng(5).standard_normal((nz, n, n))
    on = mg.solve(b + 5.0, h, "periodic", cycles=30)           # mean of b removed: converges
    assert on["residual_rms"][-1] < 1e-8 * on["residual_rms"][0]
    np.testing.assert_allclose(on["x"], mg.solve(b, h, "periodic", cycles=30)["x"], atol=1e-10)
    off = mg.solve(b + 5.0, h, "periodic", cycles=30, remove_mean_b=False)   # inconsistent: residual stalls at the mean
    assert off["residual_rms"][-1] > 4.9


def test_z_uniform_input_stays_z_uniform_periodic():
    """The lesson of the source: i+j colouring in every layer keeps a z-independent problem z-independent."""
    n, nz = 32, 16
    b = np.repeat(np.random.default_rng(6).standard_normal((1, n, n)), nz, axis=0)
    x = mg.solve(b, 0.05, "periodic", cycles=4)["x"]
    assert np.abs(x - x[0]).max() == 0.0
    xw = mg.solve(b, 0.05, "walls", cycles=60)["x"]           # walls: transient z-differences, decay with cycles
    assert np.abs(xw - xw[0]).max() / np.abs(xw).max() < 1e-12


@pytest.mark.parametrize("n,nz", [(6, 4), (3, 4), (2, 2), (16, 3), (16, 0), (12, 8)])
def test_size_validation(n, nz):
    with pytest.raises(ValueError):
        mg.validate_grid(n, nz)


def test_option_validation():
    b = np.zeros((4, 8, 8))
    with pytest.raises(ValueError):
        mg.solve(b, 0.1, bc_z="dirichlet")
    with pytest.raises(ValueError):
        mg.solve(b, -1.0)
    with pytest.raises(ValueError):
        mg.solve(np.zeros((4, 8, 6)), 0.1)
