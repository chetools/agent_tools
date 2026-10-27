"""Independent checks of library/python/gpu/particle_advect_rk2.py (float64 reference).

(a) uniform and linear velocity fields: trilinear interpolation is exact, so RK2 equals the closed-form RK2
    amplification (I + dt M + dt^2 M^2/2)^n applied to the start point, and differs from the exact solution
    expm(M T) by the stated truncation error (T dt^2/6) M^3 expm(M T) x0, second order in dt;
(b) solid-body rotation sampled on the grid vs the exact circle (RK2 radius/phase error in closed form);
(c) interpolation vs scipy.interpolate.RegularGridInterpolator, including the wall ghost (explicit padded array)
    and the periodic seam (wrapped padded array).
"""
import pathlib
import sys

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import expm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from gpu import particle_advect_rk2 as pa  # noqa: E402


def centres(n, h):
    return (np.arange(n) + 0.5) * h


def field_from(fn, nx, ny, nz, h):
    """V[c,k,j,i] = fn(x,y,z)[c] at the cell centres."""
    Z, Y, X = np.meshgrid(centres(nz, h), centres(ny, h), centres(nx, h), indexing="ij")
    return np.stack(fn(X, Y, Z))


# ---------------------------------------------------------------- (a) uniform and linear fields
def test_uniform_field_exact_any_step_count():
    nx = ny = nz = 16; h = 1.0 / 16
    b = np.array([0.1, -0.05, 0.2])           # L/time
    V = field_from(lambda X, Y, Z: (X * 0 + b[0], X * 0 + b[1], X * 0 + b[2]), nx, ny, nz, h)
    q0 = np.array([[0.4, 0.6, 0.9], [0.5, 0.5, 0.95]])
    T = 1.5
    for nsub in (1, 3, 17):
        r = pa.advect(V, q0, T, h, periodic=True, nsub=nsub)
        exact = q0 + T * b
        exact[:, 2] %= nz * h                 # periodic z wraps
        err = np.abs(r["pos"] - exact)
        err[:, 2] = np.minimum(err[:, 2], nz * h - err[:, 2])
        assert err.max() < 1e-13, (nsub, err.max())
        assert np.all(r["flags"] & pa.FLAG_ZWRAP)   # both particles crossed the z seam
        assert not np.any(r["flags"] & 7)


M = np.array([[0.05, -0.30, 0.02],
              [0.25, -0.04, 0.0],
              [0.0, 0.03, 0.06]])             # 1/time; a gentle rotation + strain + z coupling
B = np.array([0.01, 0.02, 0.0])               # L/time


def linear_setup():
    nx = ny = nz = 24; h = 1.0 / 24
    c0 = np.array([0.5, 0.5, 0.5])
    # u = M (x - c0) + B
    V = field_from(lambda X, Y, Z: tuple(M[c, 0] * (X - c0[0]) + M[c, 1] * (Y - c0[1]) + M[c, 2] * (Z - c0[2]) + B[c]
                                         for c in range(3)), nx, ny, nz, h)
    return V, h, c0


def exact_linear(x0, c0, T):
    """x' = M (x - c0) + B  =>  s = x - c0, s' = M s + B; s(T) = e^{MT} s0 + M^{-1}(e^{MT} - I) B."""
    s0 = x0 - c0
    E = expm(M * T)
    return c0 + E @ s0 + np.linalg.solve(M, (E - np.eye(3)) @ B)


@pytest.mark.parametrize("walls", [False, True])
def test_linear_field_rk2_matches_closed_form_and_second_order(walls):
    V, h, c0 = linear_setup()
    x0 = np.array([0.55, 0.45, 0.5])
    T = 2.0
    errs, preds = [], []
    for n in (8, 16, 32, 64, 128):
        r = pa.advect(V, x0[None], T, h, periodic=not walls, nsub=n)
        assert not np.any(r["flags"])
        d = T / n
        # closed form of the RK2 recursion for a linear field (interpolation is exact in the interior)
        A = np.eye(3) + d * M + 0.5 * d * d * M @ M
        s = x0 - c0
        # affine recursion: s <- A s + d (I + d M/2) B
        s_n = s.copy(); c_b = d * (np.eye(3) + 0.5 * d * M) @ B
        for _ in range(n):
            s_n = A @ s_n + c_b
        assert np.abs(r["pos"][0] - (c0 + s_n)).max() < 1e-12
        e = np.abs(r["pos"][0] - exact_linear(x0, c0, T))
        errs.append(np.linalg.norm(e))
        # leading truncation term: (T d^2 / 6) M^3 (e^{MT} (x0 - c0 + M^-1 B) - M^-1 B)   [x' = M s + B]
        s_star = np.linalg.solve(M, B)
        lead = (T * d * d / 6.0) * M @ M @ M @ (expm(M * T) @ (x0 - c0 + s_star))
        preds.append(np.linalg.norm(lead))
    errs, preds = np.array(errs), np.array(preds)
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(np.abs(orders - 2.0) < 0.1), orders           # second order
    assert abs(errs[-1] / preds[-1] - 1.0) < 0.05, (errs[-1], preds[-1])   # matches the stated leading error


def test_uniform_and_linear_particles_stay_in_interior_no_ghost():
    """The linear tests above must not touch the ghost cells (where the field is no longer linear)."""
    V, h, c0 = linear_setup()
    r = pa.advect(V, np.array([[0.55, 0.45, 0.5]]), 2.0, h, nsub=64)
    assert (r["pos"][0, :2] > 0.5 * h).all() and (r["pos"][0, :2] < 1 - 0.5 * h).all()


# ---------------------------------------------------------------- (b) solid-body rotation
def test_solid_body_rotation_circle():
    nx = ny = nz = 32; h = 1.0 / 32
    Om = 2.0                                   # rad/time
    V = field_from(lambda X, Y, Z: (-Om * (Y - 0.5), Om * (X - 0.5), X * 0), nx, ny, nz, h)
    r0 = 0.25
    x0 = np.array([[0.5 + r0, 0.5, 0.3]])
    T = 0.5 * np.pi / Om                       # a quarter turn
    errs = []
    for n in (8, 16, 32, 64, 128):
        res = pa.advect(V, x0, T, h, nsub=n)
        q = res["pos"][0]
        # exact: (0.5, 0.5 + r0)
        errs.append(np.hypot(q[0] - 0.5, q[1] - (0.5 + r0)))
        # RK2 for rotation multiplies s = (x - xc) + i (y - yc) by m = 1 + i th - th^2/2, th = Om T / n
        th = Om * T / n
        s = (x0[0, 0] - 0.5) + 1j * (x0[0, 1] - 0.5)
        s = s * (1 + 1j * th - 0.5 * th * th) ** n
        assert abs(q[0] - (0.5 + s.real)) < 1e-12 and abs(q[1] - (0.5 + s.imag)) < 1e-12
        assert abs(q[2] - 0.3) < 1e-14         # w = 0
        # radius error |m|^n - 1 ~ n th^4 / 8, phase error ~ n th^3 / 6: stated truncation error
        pred = r0 * np.hypot(n * th ** 4 / 8, n * th ** 3 / 6)
        if n >= 32:
            assert abs(errs[-1] / pred - 1) < 0.05, (n, errs[-1], pred)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 2.0) < 0.05), orders
    assert errs[-1] < 1e-4 * r0 * 10


def test_solid_body_rotation_full_turn_radius_drift_bounded():
    nx = ny = nz = 32; h = 1.0 / 32
    Om = 2.0
    V = field_from(lambda X, Y, Z: (-Om * (Y - 0.5), Om * (X - 0.5), X * 0), nx, ny, nz, h)
    x0 = np.array([[0.5 + 0.2, 0.5, 0.5], [0.5, 0.5 - 0.3, 0.5]])
    res = pa.advect(V, x0, 2 * np.pi / Om, h, nsub=400)
    assert np.abs(res["pos"] - x0).max() < 2e-4


# ---------------------------------------------------------------- (c) interpolation vs scipy
def random_field(nx, ny, nz, seed=3):
    return np.random.default_rng(seed).normal(size=(3, nz, ny, nx))


def test_interior_points_vs_regular_grid_interpolator():
    nx, ny, nz, h = 9, 7, 11, 0.1
    V = random_field(nx, ny, nz)
    rgi = RegularGridInterpolator((centres(nz, h), centres(ny, h), centres(nx, h)), np.moveaxis(V, 0, -1), method="linear")
    rng = np.random.default_rng(5)
    lo, hi = 0.5 * h, np.array([nx - 0.5, ny - 0.5, nz - 0.5]) * h
    pts = lo + rng.random((4000, 3)) * (hi - lo)                       # x, y, z
    got = pa.velocity_at(V, pts, h, periodic=False)
    want = rgi(pts[:, ::-1])
    assert np.abs(got - want).max() < 1e-13


def padded_reference(V, h, periodic):
    """Same ghost rule written as an explicit padded array, interpolated by scipy: odd mirror in x and y (and z for
    walls), wrapped in z for periodic."""
    _, nz, ny, nx = V.shape
    P = np.pad(V, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")   # placeholder, ghosts set below
    P[:, :, 0, 1:-1] = -V[:, :, 0, :]; P[:, :, -1, 1:-1] = -V[:, :, -1, :]
    P[:, :, 1:-1, 0] = -V[:, :, :, 0]; P[:, :, 1:-1, -1] = -V[:, :, :, -1]
    P[:, :, 0, 0] = V[:, :, 0, 0]; P[:, :, 0, -1] = V[:, :, 0, -1]
    P[:, :, -1, 0] = V[:, :, -1, 0]; P[:, :, -1, -1] = V[:, :, -1, -1]   # corners: two sign flips
    zc = np.concatenate([[-0.5], np.arange(nz) + 0.5, [nz + 0.5]]) * h
    top = np.concatenate([P[:, -1:], P, P[:, :1]], axis=1) if periodic else np.concatenate([-P[:, :1], P, -P[:, -1:]], axis=1)
    yc = np.concatenate([[-0.5], np.arange(ny) + 0.5, [ny + 0.5]]) * h
    xc = np.concatenate([[-0.5], np.arange(nx) + 0.5, [nx + 0.5]]) * h
    return RegularGridInterpolator((zc, yc, xc), np.moveaxis(top, 0, -1), method="linear")


@pytest.mark.parametrize("periodic", [True, False])
def test_wall_ghost_and_seam_vs_padded_scipy(periodic):
    nx, ny, nz, h = 8, 6, 7, 0.125
    V = random_field(nx, ny, nz, seed=7 + periodic)
    rgi = padded_reference(V, h, periodic)
    rng = np.random.default_rng(9)
    L = np.array([nx, ny, nz]) * h
    pts = rng.random((6000, 3)) * L                                    # whole box, including the outer half cells
    # a share of points hugging the walls / the seam
    pts[:1000, 0] = rng.random(1000) * 0.5 * h
    pts[1000:2000, 1] = L[1] - rng.random(1000) * 0.5 * h
    pts[2000:3000, 2] = rng.random(1000) * 0.5 * h
    pts[3000:4000, 2] = L[2] - rng.random(1000) * 0.5 * h
    got = pa.velocity_at(V, pts, h, periodic=periodic)
    want = rgi(pts[:, ::-1])
    assert np.abs(got - want).max() < 1e-13


@pytest.mark.parametrize("periodic", [True, False])
def test_velocity_exactly_zero_on_x_and_y_walls(periodic):
    nx, ny, nz, h = 8, 6, 7, 0.125
    V = random_field(nx, ny, nz, seed=1)
    z = np.linspace(0, nz * h, 50, endpoint=False)
    for fixed in ("x0", "x1", "y0", "y1"):
        pts = np.zeros((50, 3)); pts[:, 2] = z
        pts[:, 0] = np.linspace(0, nx * h, 50); pts[:, 1] = np.linspace(0, ny * h, 50)
        if fixed == "x0": pts[:, 0] = 0
        if fixed == "x1": pts[:, 0] = nx * h
        if fixed == "y0": pts[:, 1] = 0
        if fixed == "y1": pts[:, 1] = ny * h
        assert np.abs(pa.velocity_at(V, pts, h, periodic)).max() < 1e-14
    if not periodic:
        pts = np.random.default_rng(2).random((50, 3)) * [nx * h, ny * h, 0]
        assert np.abs(pa.velocity_at(V, pts, h, False)).max() < 1e-14
        pts[:, 2] = nz * h
        assert np.abs(pa.velocity_at(V, pts, h, False)).max() < 1e-14


# ---------------------------------------------------------------- clamp-with-flag policy, substep count
def test_clamp_flags_and_periodic_wrap():
    nx = ny = nz = 8; h = 0.125
    V = random_field(nx, ny, nz, seed=4)
    q0 = np.random.default_rng(6).random((500, 3)) * nx * h
    # very large substeps (about 10 h each): many particles leave the box and must be clamped to a wall with a flag
    r = pa.advect(V, q0, 2.0, h, periodic=False, nsub=2)
    L = nx * h
    assert (r["pos"] >= 0).all() and (r["pos"] <= L).all()
    for axis, bit in ((0, pa.FLAG_X), (1, pa.FLAG_Y), (2, pa.FLAG_ZWALL)):
        f = (r["flags"] & bit) != 0
        assert f.sum() > 0
    # a particle flagged in x is on an x wall at the end only if it never re-entered; at least some are
    fx = (r["flags"] & pa.FLAG_X) != 0
    assert np.any(np.isin(r["pos"][fx, 0], [0.0, L]))
    assert not np.any(r["flags"] & pa.FLAG_ZWRAP)
    r = pa.advect(V, q0, 2.0, h, periodic=True, nsub=2)
    assert np.any(r["flags"] & pa.FLAG_ZWRAP) and not np.any(r["flags"] & pa.FLAG_ZWALL)
    assert (r["pos"][:, 2] >= 0).all() and (r["pos"][:, 2] < L).all()


def test_substep_count_formula():
    assert pa.substep_count(1.0, 2.0, 0.1) == 40           # 2/(0.5*0.1)
    assert pa.substep_count(1.0, 2.0, 0.1, 1.0) == 20
    assert pa.substep_count(1.0, 0.0, 0.1) == 1
