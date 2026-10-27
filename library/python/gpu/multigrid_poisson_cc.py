"""Cell-centred geometric multigrid (V-cycles) for the 3D Poisson problem, float64 reference.

Solves  laplacian(x) = b  on a box of N x N x NZ cells of edge h with the 7-point cell-centred operator

    (sum of existing neighbours - count * x) / h**2 = b

* x, y: zero-flux (Neumann) walls: a missing neighbour is skipped and not counted.
* z:    ``bc_z="periodic"`` (cell above the last layer is layer 0) or ``bc_z="walls"`` (zero-flux).
  With NZ = 1 there is no z coupling.

All-Neumann or periodic problems are singular (constants are the null space).  Solvability: b must have zero
mean, and x is fixed up to a constant.  ``remove_mean_b`` subtracts mean(b) from a copy of b; ``remove_mean_x``
subtracts mean(x) from the returned solution.  Both default True.

Arrays have shape (NZ, N, N), axes (k, j, i) = (z, y, x); the C-order flattening is the WGSL index
(k * N + j) * N + i.  Units: h in metres (or any consistent length unit); if x has unit [X], b has
[X]/[length]**2 and the residual b - L x has the same unit as b.

Size constraints (same as the WGSL kernel, checked by ``validate_grid``): N a power of two >= 4 (coarsest level
N = 4), NZ a power of two >= 1 (z is halved with x and y while it has >= 2 layers, then stays at 1).

Defaults (nu1 = nu2 = 2 pre/post red-black sweeps, ncoarse = 40 sweeps on the coarsest level, cycles = 2)
equal the source (chetools/claude cylinder-cavity-3d, python/fv3d.py: vcycles = 2, nu1 = 2, nu2 = 2, ncoarse = 40).
The smoother colours by the parity of i+j in EVERY layer and updates a whole z column from old values (Jacobi in
z); an (i+j+k) colouring would make z-independent problems z-dependent through round-off-level asymmetry.

Extracted from chetools/claude cylinder-cavity-3d/python/fv3d.py (neighbour_sum, neighbour_count, laplacian,
smooth, restrict, prolong, vcycle, solve_pressure).  Source licence/terms: same project owner.
"""
import numpy as np

DEFAULTS = dict(bc_z="periodic", cycles=2, nu1=2, nu2=2, ncoarse=40, rtol=0.0,
                remove_mean_b=True, remove_mean_x=True)
N_COARSEST = 4


def validate_grid(n, nz):
    """Raise ValueError unless N is a power of two >= 4 and NZ a power of two >= 1."""
    def pow2(v):
        return isinstance(v, (int, np.integer)) and v >= 1 and (v & (v - 1)) == 0
    if not pow2(n) or n < N_COARSEST:
        raise ValueError(f"N must be a power of two >= {N_COARSEST}, got {n!r}")
    if not pow2(nz):
        raise ValueError(f"NZ must be a power of two >= 1, got {nz!r}")


def _check_bc(bc_z):
    if bc_z not in ("periodic", "walls"):
        raise ValueError(f"bc_z must be 'periodic' or 'walls', got {bc_z!r}")


def neighbour_sum(x, bc_z="periodic"):
    """Sum of the existing neighbours of every cell (zero-flux walls: missing ones skipped; periodic z wraps)."""
    s = np.zeros_like(x)
    s[:, 1:, :] += x[:, :-1, :]
    s[:, :-1, :] += x[:, 1:, :]
    s[:, :, 1:] += x[:, :, :-1]
    s[:, :, :-1] += x[:, :, 1:]
    if x.shape[0] >= 2:                            # with one layer there is no z coupling
        if bc_z == "periodic":
            s += np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0)
        else:
            s[1:] += x[:-1]
            s[:-1] += x[1:]
    return s


def neighbour_count(shape, bc_z="periodic"):
    """Number of existing neighbours of every cell, shape (NZ, N, N)."""
    nz, n = shape[0], shape[1]
    c = np.full((nz, n, n), 4.0)
    c[:, 0, :] -= 1
    c[:, -1, :] -= 1
    c[:, :, 0] -= 1
    c[:, :, -1] -= 1
    if nz >= 2:
        c += 2.0
        if bc_z == "walls":
            c[0] -= 1
            c[-1] -= 1
    return c


def laplacian(x, h, bc_z="periodic"):
    """L x with zero normal gradient at the walls; h in length units, result in [x]/length**2."""
    return (neighbour_sum(x, bc_z) - neighbour_count(x.shape, bc_z) * x) / (h * h)


def smooth(x, b, h, sweeps, bc_z="periodic"):
    """Red-black Gauss-Seidel in place: x_c = (sum of neighbours - h^2 b) / count, colour = parity of i+j."""
    nz, n = x.shape[0], x.shape[1]
    cnt = neighbour_count(x.shape, bc_z)
    _, jj, ii = np.meshgrid(np.arange(nz), np.arange(n), np.arange(n), indexing="ij")
    red = (ii + jj) % 2 == 0
    for _ in range(sweeps):
        for colour in (red, ~red):
            x[colour] = ((neighbour_sum(x, bc_z) - h * h * b) / cnt)[colour]
    return x


def restrict(r):
    """Fine residual -> coarse right-hand side: average of the 4 children in x, y and of 2 layers in z (if NZ >= 2)."""
    rc = 0.25 * (r[:, 0::2, 0::2] + r[:, 1::2, 0::2] + r[:, 0::2, 1::2] + r[:, 1::2, 1::2])
    if r.shape[0] >= 2:
        rc = 0.5 * (rc[0::2] + rc[1::2])
    return rc


def prolong(xc, nz, bc_z="periodic"):
    """Coarse correction -> fine grid with nz layers: trilinear (weights 3/4, 1/4 per direction), zero gradient at walls."""
    n = xc.shape[1]
    I = np.arange(2 * n) // 2
    off = np.where(np.arange(2 * n) % 2 == 0, -1, 1)
    I2 = np.clip(I + off, 0, n - 1)
    rows = 0.75 * xc[:, I, :] + 0.25 * xc[:, I2, :]
    xf = 0.75 * rows[:, :, I] + 0.25 * rows[:, :, I2]
    nzc = xc.shape[0]
    if nz == nzc:                                  # z was not coarsened on this level
        return xf
    K = np.arange(2 * nzc) // 2
    offk = np.where(np.arange(2 * nzc) % 2 == 0, -1, 1)
    K2 = (K + offk) % nzc if bc_z == "periodic" else np.clip(K + offk, 0, nzc - 1)
    return 0.75 * xf[K] + 0.25 * xf[K2]


def vcycle(x, b, h, nu1=2, nu2=2, ncoarse=40, bc_z="periodic"):
    """One V-cycle on x (in place).  Coarsest level (N = 4): ncoarse sweeps, then mean(x) removed."""
    nz, n = x.shape[0], x.shape[1]
    if n <= N_COARSEST:
        smooth(x, b, h, ncoarse, bc_z)
        x -= x.mean()
        return x
    smooth(x, b, h, nu1, bc_z)
    r = b - laplacian(x, h, bc_z)
    nzc = nz // 2 if nz >= 2 else 1                # semi-coarsening: z only while it has >= 2 layers
    xc = vcycle(np.zeros((nzc, n // 2, n // 2)), restrict(r), 2 * h, nu1, nu2, ncoarse, bc_z)
    x += prolong(xc, nz, bc_z)
    smooth(x, b, h, nu2, bc_z)
    return x


def residual_rms(x, b, h, bc_z="periodic"):
    """Root-mean-square of b - L x over all cells (unit of b)."""
    r = b - laplacian(x, h, bc_z)
    return float(np.sqrt(np.mean(r * r)))


def solve(b, h, bc_z=DEFAULTS["bc_z"], cycles=DEFAULTS["cycles"], nu1=DEFAULTS["nu1"], nu2=DEFAULTS["nu2"],
          ncoarse=DEFAULTS["ncoarse"], x0=None, rtol=DEFAULTS["rtol"],
          remove_mean_b=DEFAULTS["remove_mean_b"], remove_mean_x=DEFAULTS["remove_mean_x"]):
    """Run up to `cycles` V-cycles.  Returns a dict:

    x             solution, shape (NZ, N, N), unit [X]
    residual_rms  list, length cycles_used + 1: RMS of b - L x for the starting guess and after every V-cycle
                  (b with its mean removed if remove_mean_b), unit of b
    cycles        V-cycles actually done
    converged     True if rtol > 0 and the final RMS <= rtol * initial RMS (rtol = 0: always run `cycles`, False)
    success       True if every residual is finite
    message       text

    b: (NZ, N, N) array in [X]/length**2.  h: cell edge, length unit.  x0: starting guess (default 0).
    rtol: relative residual tolerance for early stop (default 0 = off).  Defaults match the source (2 cycles).
    """
    b = np.array(b, dtype=float)                   # copy
    if b.ndim != 3 or b.shape[1] != b.shape[2]:
        raise ValueError("b must have shape (NZ, N, N)")
    validate_grid(b.shape[1], b.shape[0])
    _check_bc(bc_z)
    if cycles < 0 or nu1 < 0 or nu2 < 0 or ncoarse < 1:
        raise ValueError("cycles, nu1, nu2 must be >= 0 and ncoarse >= 1")
    if not h > 0:
        raise ValueError("h must be > 0")
    if remove_mean_b:
        b -= b.mean()
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    if x.shape != b.shape:
        raise ValueError("x0 must have the same shape as b")
    hist = [residual_rms(x, b, h, bc_z)]
    done = 0
    converged = False
    while done < cycles:
        if rtol > 0 and hist[-1] <= rtol * hist[0]:
            break
        vcycle(x, b, h, nu1, nu2, ncoarse, bc_z)
        done += 1
        hist.append(residual_rms(x, b, h, bc_z))
    if rtol > 0:
        converged = hist[-1] <= rtol * hist[0]
    if remove_mean_x:
        x -= x.mean()
    ok = bool(np.all(np.isfinite(hist)) and np.all(np.isfinite(x)))
    msg = ("ok" if ok else "non-finite values") + f"; {done} V-cycle(s), RMS residual {hist[0]:.3e} -> {hist[-1]:.3e}"
    return dict(x=x, residual_rms=hist, cycles=done, converged=converged, success=ok, message=msg)


# ----------------------------------------------------------------------------- test data (also used by the GPU test)
def hash_field(n, nz, seed):
    """Deterministic pseudo-random field in [-1, 1), shape (nz, n, n); integer hash, identical in the JS test."""
    k, j, i = np.meshgrid(np.arange(nz, dtype=np.uint64), np.arange(n, dtype=np.uint64),
                          np.arange(n, dtype=np.uint64), indexing="ij")
    m = np.uint64(0xFFFFFFFF)
    v = ((i * np.uint64(73856093)) ^ (j * np.uint64(19349663)) ^ (k * np.uint64(83492791))
         ^ ((np.uint64(seed) * np.uint64(2654435761)) & m)) & m
    v ^= v >> np.uint64(16)
    v = (v * np.uint64(0x45d9f3b)) & m
    v ^= v >> np.uint64(16)
    v = (v * np.uint64(0x45d9f3b)) & m
    v ^= v >> np.uint64(16)
    return v.astype(np.float64) / 2147483648.0 - 1.0


def test_rhs(n, nz, seed, h):
    """b = hash noise + one smooth cosine mode, in [X]/length**2 (mean NOT removed).  JS mirror: testRhs()."""
    k, j, i = np.meshgrid(np.arange(nz), np.arange(n), np.arange(n), indexing="ij")
    smooth_mode = 3.0 * np.cos(np.pi * (i + 0.5) / n) * np.cos(2.0 * np.pi * (j + 0.5) / n)
    return hash_field(n, nz, seed) + smooth_mode + 0.25


CASES = [
    # name, n, nz, bc_z, cycles, h, extra options
    dict(name="walls_16x16x8", n=16, nz=8, bc_z="walls", cycles=8, h=0.1, seed=1),
    dict(name="periodic_16x16x8", n=16, nz=8, bc_z="periodic", cycles=8, h=0.1, seed=2),
    dict(name="walls_32x32x1_2D", n=32, nz=1, bc_z="walls", cycles=8, h=0.05, seed=3),
    dict(name="periodic_8x8x2", n=8, nz=2, bc_z="periodic", cycles=6, h=0.25, seed=4),
    dict(name="options_16x16x4_walls_x0", n=16, nz=4, bc_z="walls", cycles=5, h=0.1, seed=5,
         nu1=1, nu2=3, ncoarse=10, x0_seed=6),
    dict(name="periodic_64x64x32_large", n=64, nz=32, bc_z="periodic", cycles=6, h=1.0 / 64, seed=7, stride=2),
    dict(name="walls_64x64x32_large", n=64, nz=32, bc_z="walls", cycles=6, h=1.0 / 64, seed=8, stride=2),
]


def make_reference():
    """JSON-serialisable reference data for tests/gpu/multigrid_poisson_cc.test.js (float64)."""
    out = {"description": "multigrid_poisson_cc: b = test_rhs(n, nz, seed, h) (mean not removed), x0 = 0.3*hash_field "
                          "(x0_seed) or 0; stored: solution (stride-subsampled for big grids), RMS residual history.",
           "cases": []}
    for c in CASES:
        n, nz, h = c["n"], c["nz"], c["h"]
        b = test_rhs(n, nz, c["seed"], h)
        x0 = 0.3 * hash_field(n, nz, c["x0_seed"]) if "x0_seed" in c else None
        opt = {k: c[k] for k in ("nu1", "nu2", "ncoarse") if k in c}
        res = solve(b, h, c["bc_z"], c["cycles"], x0=x0, **opt)
        s = c.get("stride", 1)
        xs = res["x"][::s, ::s, ::s]
        entry = {k: c[k] for k in ("name", "n", "nz", "bc_z", "cycles", "h", "seed")}
        entry.update({k: c[k] for k in ("nu1", "nu2", "ncoarse", "x0_seed") if k in c})
        entry.update(stride=s, x=[float(v) for v in xs.ravel()], x_shape=list(xs.shape),
                     x_rms=float(np.sqrt(np.mean(res["x"] ** 2))), x_max=float(np.abs(res["x"]).max()),
                     residual_rms=[float(v) for v in res["residual_rms"]], cycles_done=int(res["cycles"]))
        out["cases"].append(entry)
    return out
