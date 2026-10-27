"""Independent checks of the marching_cubes Python reference (library/python/gpu/marching_cubes.py).

Independent methods: scikit-image's marching_cubes and mesh_surface_area, numpy.gradient, scipy cKDTree, and exact
areas/volumes of a sphere, torus and cylinder.  Nothing here compares the reference with itself.
"""
import pathlib
import sys
from collections import Counter

import numpy as np
import pytest
from scipy.spatial import cKDTree
from skimage import measure

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from gpu import marching_cubes as mc  # noqa: E402


def grid(n, lo, hi):
    h = (hi - lo) / (n - 1)
    x = lo + h * np.arange(n)
    z, y, xx = np.meshgrid(x, x, x, indexing="ij")          # field[k, j, i]
    return xx, y, z, h


def sphere(n, r=0.8):
    x, y, z, h = grid(n, -1.2, 1.2)
    return np.sqrt(x**2 + y**2 + z**2), h, r


def torus(n, R=0.6, a=0.25):
    x, y, z, h = grid(n, -1.0, 1.0)
    return (np.hypot(x, y) - R) ** 2 + z**2, h, a * a        # surface: f = a^2


# ------------------------------------------------------------------ case table
def test_table_consistent_with_corner_signs():
    """Every case uses exactly the edges whose end corners differ in sign, triangles are whole and non-degenerate."""
    table = mc.triangle_table()
    assert len(table) == 256
    nonempty = 0
    for case, edges in enumerate(table):
        assert len(edges) % 3 == 0
        crossing = {e for e in range(12) if ((case >> mc.EDGE_A[e]) & 1) != ((case >> mc.EDGE_B[e]) & 1)}
        assert set(edges) == crossing, case
        for t in range(0, len(edges), 3):
            assert len(set(edges[t:t + 3])) == 3, case
        # every crossing edge belongs to two triangle corners of the fan, i.e. appears at least once
        nonempty += bool(edges)
        assert set(table[255 - case]) == set(edges), case      # complement flips inside/outside, same edges
    assert nonempty == 254 and table[0] == [] and table[255] == []


# ------------------------------------------------------------------ against scikit-image
@pytest.mark.parametrize("n", [16, 32, 64])
def test_sphere_matches_skimage(n):
    """Smooth field: same triangle count, area, and vertex set as skimage (same linear edge interpolation)."""
    f, h, r = sphere(n)
    res = mc.marching_cubes(f, h, (-1.2,) * 3, r)
    v, faces, _, _ = measure.marching_cubes(f.transpose(2, 1, 0), r, spacing=(h, h, h))
    assert res["count"] == len(faces)
    assert mc.mesh_area(res["tris"]) == pytest.approx(measure.mesh_surface_area(v, faces), rel=1e-6)
    u = mc.unique_vertices(res)[:, :3]
    vs = v - 1.2
    assert cKDTree(u).query(vs)[0].max() < 1e-4 * h
    assert cKDTree(vs).query(u)[0].max() < 1e-4 * h


def test_saddle_field_vs_skimage():
    """Closed random field with ~900 ambiguous cells: area within 5 % of both skimage variants.

    The ambiguity resolution differs between tables (measured +2.6 % for skimage's method='lorensen' and -1.2 %
    for its default 'lewiner'), so triangle counts and areas legitimately differ here; only a loose area bound
    is asserted.  The smooth-field tests above are the tight comparison.
    """
    f = np.pad(np.random.default_rng(1).random((13, 12, 14)), 1, constant_values=1.0)
    assert mc.ambiguous_face_cells(f, 0.5) > 500
    A = mc.mesh_area(mc.marching_cubes(f, 1.0, (0, 0, 0), 0.5)["tris"])
    for method in ("lorensen", "lewiner"):
        v, faces, _, _ = measure.marching_cubes(f.transpose(2, 1, 0), 0.5, method=method)
        assert abs(measure.mesh_surface_area(v, faces) / A - 1) < 5e-2


def test_saddle_surface_is_watertight():
    """Closed random field (border outside): every welded edge is shared by exactly two triangles, no cracks."""
    for seed in range(3):
        f = np.pad(np.random.default_rng(seed).random((13, 12, 14)), 1, constant_values=1.0)
        res = mc.marching_cubes(f, 1.0, (0, 0, 0), 0.5)
        t = np.round(res["tris"], 6)
        c = Counter()
        for tr in t:
            k = [tuple(p) for p in tr]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                c[frozenset((k[a], k[b]))] += 1
        assert set(c.values()) == {2}


def test_gradient_matches_numpy_gradient():
    rng = np.random.default_rng(3)
    f = rng.random((7, 9, 11))
    h = 0.37
    gx, gy, gz = mc.node_gradient(f, h)
    ref = np.gradient(f, h, edge_order=1)            # axes (k, j, i) -> (gz, gy, gx)
    for a, b in ((gz, ref[0]), (gy, ref[1]), (gx, ref[2])):
        assert np.abs(a - b).max() < 1e-12
    # periodic z: analytic derivative of sin(2 pi k / nz) is second-order accurate
    nz = 32
    z = np.arange(nz) * (2 * np.pi / nz)
    g = np.broadcast_to(np.sin(z)[:, None, None], (nz, 3, 3)).copy()
    gzp = mc.node_gradient(g, 2 * np.pi / nz, periodic_z=True)[2]
    assert np.abs(gzp - np.cos(z)[:, None, None]).max() < 0.01


# ------------------------------------------------------------------ analytic areas and volumes
def _order(errs):
    return np.log(errs[0] / errs[-1]) / np.log(2 ** (len(errs) - 1))


def test_sphere_area_volume_converge():
    A0, V0 = 4 * np.pi * 0.8**2, 4 / 3 * np.pi * 0.8**3
    eA, eV = [], []
    for n in (17, 33, 65):                       # h = 0.15, 0.075, 0.0375
        f, h, r = sphere(n)
        res = mc.marching_cubes(f, h, (-1.2,) * 3, r)
        eA.append(abs(mc.mesh_area(res["tris"]) / A0 - 1))
        eV.append(abs(mc.mesh_volume(res["tris"]) / V0 - 1))
    assert eA[2] < 1.5e-3 and eV[2] < 2.5e-3                 # observed 7e-4 and 1.3e-3 at n = 64
    assert _order(eA) > 1.8 and _order(eV) > 1.8             # second order in h
    assert all(a > b for a, b in zip(eA, eA[1:])) and all(a > b for a, b in zip(eV, eV[1:]))


def test_torus_area_volume_converge():
    R, a = 0.6, 0.25
    A0, V0 = 4 * np.pi**2 * R * a, 2 * np.pi**2 * R * a**2
    eA, eV = [], []
    for n in (33, 65, 129):                      # h = 0.0625, 0.03125, 0.0156
        f, h, s = torus(n)
        res = mc.marching_cubes(f, h, (-1.0,) * 3, s)
        eA.append(abs(mc.mesh_area(res["tris"]) / A0 - 1))
        eV.append(abs(mc.mesh_volume(res["tris"]) / V0 - 1))
    assert eA[2] < 1e-3 and eV[2] < 2.5e-3                   # observed 4e-4 and 1.2e-3 at n = 129
    assert _order(eA) > 1.7 and _order(eV) > 1.7


def test_periodic_cylinder_area_converges():
    """Cylinder along z, periodic: area per period 2 pi r L exactly; checks the wrap cells (open z ends would lose a layer)."""
    r, errs = 0.6, []
    for nz in (6, 12, 24):
        h = 1.5 / nz
        n = int(2.2 / h) + 1
        x = -1.1 + h * np.arange(n)
        z = np.arange(nz) * h
        kk, jj, ii = np.meshgrid(z, x, x, indexing="ij")
        res = mc.marching_cubes(np.hypot(ii, jj), h, (-1.1, -1.1, 0.0), r, periodic_z=True)
        errs.append(abs(mc.mesh_area(res["tris"]) / (2 * np.pi * r * 1.5) - 1))
        assert res["tris"][..., 2].max() <= 1.5 + 1e-12 and res["tris"][..., 2].min() >= 0.0
    assert errs[-1] < 2e-3 and errs[0] > errs[-1]


# ------------------------------------------------------------------ orientation, cap, edge cases
def test_normals_outward_and_winding():
    f, h, r = sphere(24)
    res = mc.marching_cubes(f, h, (-1.2,) * 3, r)
    p, n = res["tris"], res["normals"]
    assert (np.einsum("tvk,tvk->tv", p, n) > 0).all()                       # normals point away from the centre
    assert mc.mesh_volume(p, orient=False) < 0                              # region f < iso, documented winding
    face_n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (np.einsum("tk,tk->t", face_n, n.mean(axis=1)) < 0).all()        # winding opposes the normals


def test_cap_counts_everything_stores_cap():
    f, h, r = sphere(16)
    full = mc.marching_cubes(f, h, (-1.2,) * 3, r)
    cap = mc.marching_cubes(f, h, (-1.2,) * 3, r, max_tris=100)
    assert cap["count"] == full["count"] and cap["overflow"] and len(cap["tris"]) == 100
    assert not full["overflow"]
    assert np.array_equal(cap["tris"], full["tris"][:100])


def test_degenerate_inputs():
    assert mc.marching_cubes(np.full((4, 4, 4), 2.0), iso=2.0)["count"] == 0      # f == iso is outside
    assert mc.marching_cubes(np.zeros((1, 4, 4)))["count"] == 0
    f = np.full((2, 2, 2), 1.0)
    f[0, 0, 0] = -1.0                                                           # one inside corner: one triangle
    res = mc.marching_cubes(f, h=2.0, origin=(1.0, -1.0, 3.0), iso=0.0)
    assert res["count"] == 1
    expected = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]) * 0.5 + [1.0, -1.0, 3.0]   # midpoints of the 3 edges
    assert sorted(map(tuple, res["tris"][0])) == sorted(map(tuple, expected))


def test_color_interpolation_is_linear_field():
    """A linear colour field is reproduced exactly at the vertices (interpolation along edges is linear)."""
    f, h, r = sphere(20)
    x, y, z, _ = grid(20, -1.2, 1.2)
    c = 2 * x - y + 0.5 * z
    res = mc.marching_cubes(f, h, (-1.2,) * 3, r, color=c)
    p = res["tris"]
    # the field values are only linear along each cell edge, so the vertex colour equals the colour field
    # evaluated at the vertex position to O(h^2): bound by 2 h * |grad c|
    exact = 2 * p[..., 0] - p[..., 1] + 0.5 * p[..., 2]
    assert np.abs(res["cvals"] - exact).max() < 1e-12
