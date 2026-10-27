"""Marching cubes isosurface extraction: float64 numpy reference for library/wgsl/marching_cubes.wgsl.

Mirrors the WGSL kernel step by step (same case table, corner/edge numbering, interpolation, normals, defaults).
Readable first, fast second: the cube classification is vectorised, the triangle generation loops over the
surface cells only, exactly as one GPU thread per cell does.

Grid and units
    ``field[k, j, i]`` (shape (nz, ny, nx), x fastest in memory, same as the GPU buffer
    ``FA[(k*ny + j)*nx + i]``) holds the scalar field at the node (ox + i*h, oy + j*h, oz + k*h).
    ``h`` is the node spacing (any consistent length unit; output positions use the same unit, default 1.0),
    ``origin`` = (ox, oy, oz) in that unit (default (0, 0, 0)), ``iso`` is in field units (default 0.0).
    Spacing is equal in x, y and z; nx, ny, nz may differ.
    Nodes with ``field < iso`` are "inside" (bit set in the case index); ``field == iso`` counts as outside.

Output of :func:`marching_cubes` (a dict)
    ``count``      number of triangles the surface needs (int), also when capped
    ``overflow``   True when ``count > max_tris`` (the GPU sets its overflow flag)
    ``tris``       float64 (T, 3, 3) vertex positions, length unit of ``h``; T = min(count, max_tris)
    ``normals``    float64 (T, 3, 3) unit normals from the field gradient, pointing toward increasing field
    ``cvals``      float64 (T, 3) second field interpolated to the vertices (zeros when ``color`` is None)
    Triangles are in cell order (k, j, i); the GPU order is nondeterministic, so compare order-independent
    quantities (count, area, volume, vertex sets).  When capped, the GPU keeps an arbitrary subset.

Winding: with inside = (field < iso) the right-hand-rule normal of each triangle points toward DECREASING field
(opposite to ``normals``); :func:`mesh_volume` therefore returns a negative value for a region ``field < iso``
unless ``orient=True``.

Case table: Lorensen & Cline (SIGGRAPH 1987), corner/edge numbering of P. Bourke; read from the WGSL file
(single source of truth).  Ambiguous faces use the table's fixed choice (no asymptotic decider).
"""
import pathlib
import re

import numpy as np

_WGSL = pathlib.Path(__file__).resolve().parents[2] / "wgsl" / "marching_cubes.wgsl"

# corner offsets (i, j, k) and edge -> corner pairs; identical to CO / EA / EB in the WGSL file
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
EDGE_A = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
EDGE_B = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]

_TABLE = None


def triangle_table():
    """256 lists of edge ids (multiples of 3), decoded from the 4-bit packed TRI array of the WGSL file."""
    global _TABLE
    if _TABLE is None:
        text = _WGSL.read_text()
        body = text.split("array<u32, 512>(", 1)[1].split(");", 1)[0]
        words = [int(w) for w in re.findall(r"(\d+)u", body)]
        if len(words) != 512:
            raise ValueError(f"expected 512 packed words, found {len(words)}")
        table = []
        for case in range(256):
            edges = []
            for m in range(16):
                t = case * 16 + m
                e = (words[t >> 3] >> ((t & 7) * 4)) & 15
                if e == 15:
                    break
                edges.append(e)
            table.append(edges)
        _TABLE = table
    return _TABLE


def node_gradient(f, h, periodic_z=False):
    """Gradient of f (shape (nz, ny, nx)) per length unit, returned as (gx, gy, gz).

    Central differences; one-sided over one cell at open edges (same as numpy.gradient, edge_order=1);
    periodic wrap in z when ``periodic_z``.
    """
    def d(a, axis, periodic):
        if periodic:
            return (np.roll(a, -1, axis) - np.roll(a, 1, axis)) / (2.0 * h)
        out = np.empty_like(a)
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        sl = lambda s: tuple(s)
        out[sl([*lo[:axis], slice(1, -1), *lo[axis + 1:]])] = (
            a[sl([*lo[:axis], slice(2, None), *lo[axis + 1:]])] - a[sl([*lo[:axis], slice(0, -2), *lo[axis + 1:]])]) / (2.0 * h)
        out[sl([*lo[:axis], 0, *lo[axis + 1:]])] = (a[sl([*lo[:axis], 1, *lo[axis + 1:]])] - a[sl([*lo[:axis], 0, *lo[axis + 1:]])]) / h
        out[sl([*lo[:axis], -1, *lo[axis + 1:]])] = (a[sl([*lo[:axis], -1, *lo[axis + 1:]])] - a[sl([*lo[:axis], -2, *lo[axis + 1:]])]) / h
        return out
    return d(f, 2, False), d(f, 1, False), d(f, 0, periodic_z)


def marching_cubes(field, h=1.0, origin=(0.0, 0.0, 0.0), iso=0.0, color=None, periodic_z=False, max_tris=None):
    """Extract the isosurface ``field == iso``.  See the module docstring for units, layout and output."""
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError("field must have shape (nz, ny, nx)")
    nz, ny, nx = f.shape
    g = None if color is None else np.asarray(color, dtype=np.float64)
    if g is not None and g.shape != f.shape:
        raise ValueError("color must have the same shape as field")
    table = triangle_table()
    ncz = nz if periodic_z else nz - 1
    ncx, ncy = nx - 1, ny - 1
    empty = {"count": 0, "overflow": False, "tris": np.zeros((0, 3, 3)), "normals": np.zeros((0, 3, 3)),
             "cvals": np.zeros((0, 3))}
    if min(ncx, ncy, ncz) < 1:
        return empty
    # step 1: cube index of every cell (cells (kc, cj, ci); with periodic_z the last layer wraps to node 0)
    kidx = np.arange(ncz)
    k1idx = (kidx + 1) % nz if periodic_z else kidx + 1
    cube = np.zeros((ncz, ncy, ncx), dtype=np.int64)
    for c, (ox, oy, oz) in enumerate(CORNERS):
        kk = k1idx if oz else kidx
        v = f[kk][:, oy:oy + ncy, ox:ox + ncx]
        cube |= (v < iso).astype(np.int64) << c
    active = np.argwhere((cube != 0) & (cube != 255))     # row-major (kc, cj, ci) = GPU cell order
    gx, gy, gz = node_gradient(f, h, periodic_z)
    grad = np.stack([gx, gy, gz], axis=-1)                 # (nz, ny, nx, 3)
    o = np.asarray(origin, dtype=np.float64)
    tris, nrms, cvs, count = [], [], [], 0
    for kc, cj, ci in active:
        k0, k1 = kc, (kc + 1) % nz if periodic_z else kc + 1
        edges = table[cube[kc, cj, ci]]
        for q in range(len(edges) // 3):
            count += 1
            if max_tris is not None and count > max_tris:
                continue                                    # counted, not stored (GPU: not written)
            tri_p, tri_n, tri_c = [], [], []
            for e in edges[3 * q:3 * q + 3]:
                a, b = EDGE_A[e], EDGE_B[e]
                ia = (ci + CORNERS[a][0], cj + CORNERS[a][1], k0 if CORNERS[a][2] == 0 else k1)
                ib = (ci + CORNERS[b][0], cj + CORNERS[b][1], k0 if CORNERS[b][2] == 0 else k1)
                va, vb = f[ia[2], ia[1], ia[0]], f[ib[2], ib[1], ib[0]]
                den = vb - va
                s = min(max((iso - va) / den, 0.0), 1.0) if abs(den) > 1e-30 else 0.5
                # node z position uses the layer index of the cell (wrap cell lies above node nz-1)
                pa = o + h * np.array([ia[0], ia[1], kc + CORNERS[a][2]], dtype=np.float64)
                pb = o + h * np.array([ib[0], ib[1], kc + CORNERS[b][2]], dtype=np.float64)
                tri_p.append(pa + s * (pb - pa))
                gv = grad[ia[2], ia[1], ia[0]] * (1 - s) + grad[ib[2], ib[1], ib[0]] * s
                tri_n.append(gv / max(np.linalg.norm(gv), 1e-30))
                tri_c.append(0.0 if g is None else g[ia[2], ia[1], ia[0]] * (1 - s) + g[ib[2], ib[1], ib[0]] * s)
            tris.append(tri_p); nrms.append(tri_n); cvs.append(tri_c)
    if not tris:
        return {**empty, "count": count, "overflow": bool(max_tris is not None and count > max_tris)}
    return {"count": count, "overflow": bool(max_tris is not None and count > max_tris),
            "tris": np.array(tris), "normals": np.array(nrms), "cvals": np.array(cvs)}


def mesh_area(tris):
    """Total triangle area (length unit of h squared)."""
    t = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def mesh_volume(tris, orient=True):
    """Volume enclosed by a CLOSED mesh by the divergence theorem, sum of p0 . (p1 x p2) / 6 (h cubed).

    Signed by the winding (negative for the region field < iso, see module docstring); ``orient=True``
    returns the absolute value.  Meaningless for an open surface (grid-boundary contact, periodic wrap).
    """
    t = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    v = float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    return abs(v) if orient else v


def unique_vertices(res, decimals=7):
    """Unique vertices (position rounded to ``decimals``) as an (M, 7) array: x y z nx ny nz cval, sorted."""
    p = res["tris"].reshape(-1, 3)
    n = res["normals"].reshape(-1, 3)
    c = res["cvals"].reshape(-1, 1)
    allv = np.hstack([p, n, c])
    _, first = np.unique(np.round(p, decimals), axis=0, return_index=True)
    return allv[first]


def ambiguous_face_cells(field, iso=0.0, periodic_z=False):
    """Number of cells having at least one ambiguous face (diagonal corners inside, the other diagonal outside)."""
    f = np.asarray(field, dtype=np.float64) < iso
    nz, ny, nx = f.shape
    ncz = nz if periodic_z else nz - 1
    kidx = np.arange(ncz)
    k1 = (kidx + 1) % nz if periodic_z else kidx + 1
    b = [f[(k1 if oz else kidx)][:, oy:oy + ny - 1, ox:ox + nx - 1] for ox, oy, oz in CORNERS]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    amb = np.zeros(b[0].shape, dtype=bool)
    for p, q, r, s in faces:      # corners in ring order p-q-r-s
        amb |= (b[p] == b[r]) & (b[q] == b[s]) & (b[p] != b[q])
    return int(amb.sum())


# ---------------------------------------------------------------------------------------- test data
def _f32(a):
    """float32-rounded values as shortest-repr Python floats (exact round trip through JSON and Float32Array)."""
    return [float(str(np.float32(x))) for x in np.asarray(a).ravel()]


def _grid(nx, ny, nz, h, origin):
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return (origin[0] + i * h, origin[1] + j * h, origin[2] + k * h)


def _case(name, field, h, origin, iso, color=None, periodic_z=False, max_tris=None, closed=False, note=""):
    f32 = np.asarray(field, dtype=np.float32)
    c32 = None if color is None else np.asarray(color, dtype=np.float32)
    res = marching_cubes(f32, h, origin, iso, c32, periodic_z, max_tris)
    full = res if max_tris is None else marching_cubes(f32, h, origin, iso, c32, periodic_z, None)
    nz, ny, nx = f32.shape
    out = {"name": name, "note": note, "nx": nx, "ny": ny, "nz": nz, "h": h, "origin": list(origin),
           "iso": iso, "periodic_z": bool(periodic_z), "max_tris": max_tris, "field": _f32(f32),
           "color": None if c32 is None else _f32(c32), "count": int(full["count"]),
           "area": mesh_area(full["tris"]), "closed": bool(closed),
           "volume": mesh_volume(full["tris"]) if closed else None,
           "vertices": np.round(unique_vertices(full), 7).ravel().tolist()}
    return out


def make_reference():
    """Test inputs and float64 expected outputs for tests/gpu/marching_cubes.test.js (JSON-serialisable)."""
    cases = []
    # 1. sphere on a non-cubic grid, with a colour field (x + 2y): closed surface, volume by divergence theorem
    h, org = 0.1, (-1.2, -1.0, -1.4)
    x, y, z = _grid(24, 20, 28, h, org)
    cases.append(_case("sphere_noncubic_colour", np.sqrt(x**2 + y**2 + z**2), h, org, 0.8, color=x + 2 * y,
                       closed=True, note="24x20x28 nodes, r=0.8, colour x+2y"))
    # 2. overflow: same sphere, cap far below the triangle count
    sph = cases[0]
    # (the field, colour and vertex list are not repeated: "same_as" names the case they are taken from)
    cases.append({**{k: sph[k] for k in ("nx", "ny", "nz", "h", "origin", "iso", "periodic_z")},
                  "name": "overflow_cap", "note": "same sphere, max_tris = 100", "max_tris": 100, "same_as": sph["name"],
                  "count": sph["count"], "area": sph["area"], "closed": False, "volume": None})
    # 3./4. Schwarz P surface cos x + cos y + cos z = iso: many ambiguous (saddle) faces; one period in z
    nzp = 24
    hp = 2 * np.pi / nzp
    x, y, z = _grid(28, 20, nzp, hp, (0.0, 0.0, 0.0))
    schwarz = np.cos(x) + np.cos(y) + np.cos(z)
    cases.append(_case("schwarz_boundary", schwarz, hp, (0.0, 0.0, 0.0), 0.15, note="open at grid boundary, saddles"))
    cases.append(_case("schwarz_periodic_z", schwarz, hp, (0.0, 0.0, 0.0), 0.15, periodic_z=True,
                       note="same field, wrap cells in z"))
    cases[-1]["field"] = None
    cases[-1]["same_as"] = "schwarz_boundary"                # field taken from that case; own vertices and counts
    # 5. random field: ambiguous (saddle) faces in many cells; positions/count/area only (normals are noisy)
    rnd = np.random.default_rng(20261027).random((13, 12, 14))
    amb = ambiguous_face_cells(rnd.astype(np.float32), 0.5)
    cases.append(_case("random_saddles", rnd, 1.0, (0.0, 0.0, 0.0), 0.5, note="13x12x14 uniform random, seed 20261027"))
    # 6. circular cylinder along z, periodic in z (period nz*h): area per period = 2 pi r L analytic
    h, org = 0.125, (-1.0, -1.0, 0.5)
    x, y, z = _grid(17, 17, 12, h, org)
    cases.append(_case("cylinder_periodic_z", np.sqrt(x**2 + y**2), h, org, 0.6, periodic_z=True,
                       note="r=0.6, period 1.5 in z"))
    # 7. one cell, one inside corner (case 1): exact positions; and an empty field
    cases.append(_case("single_cell", np.array([[[0, 1], [1, 1]], [[1, 1], [1, 1]]]).astype(float) * 2 - 0.5 + 0.0, 2.0,
                       (1.0, -1.0, 3.0), 1.0, note="2x2x2 nodes, corner 0 inside"))
    cases.append(_case("constant_field", np.full((5, 6, 7), 3.0), 1.0, (0, 0, 0), 3.0,
                       note="f == iso everywhere is outside: no triangles"))
    return {"cases": cases, "ambiguous_cells_random_saddles": amb,
            "precision": "float64 reference on the float32-rounded field"}
