"""Particle advection through a gridded 3D velocity field: float64 reference of library/wgsl/particle_advect_rk2.wgsl.

Advects massless particles with RK2 (explicit midpoint) in equal substeps.  The velocity is trilinearly
interpolated between cell centres.  Mirrors the WGSL kernel step by step; this is the readable teaching version
and the test oracle for the GPU kernel.

Units: any consistent length unit L (metres in SI) for ``h`` and positions; velocity in L/time; ``dt`` in the
matching time unit.  The box is [0, nx*h] x [0, ny*h] x [0, nz*h].

Interpolation convention (identical to chetools/claude cylinder-cavity-3d trails.wgsl and tracers.js):
    cell (i, j, k) has its centre at x_i = (i + 1/2) h, y_j = (j + 1/2) h, z_k = (k + 1/2) h; V[c, k, j, i] is the
    velocity component c (0, 1, 2 = x, y, z) there.  The fractional index is f = x/h - 1/2.
    A stencil index one cell outside the grid takes a ghost value:
      x, y (always walls) and z when ``periodic=False``: ghost = -(edge value) for all components (odd mirror),
          so the interpolated velocity is exactly 0 on every wall plane (no-slip);
      z when ``periodic=True``: the index wraps.
    Sampling points are clamped to the box (walls) or wrapped (periodic z) before interpolation.

Out-of-domain policy: after every substep x, y (and z for walls) are CLAMPED to the box and a bit is set in the
flags word; periodic z is wrapped into [0, nz*h).  Flags bits (OR over all substeps of the call):
    1: x clamped   2: y clamped   4: z clamped (walls)   8: z wrapped (periodic).
Nothing is re-seeded or dropped.

Truncation error: RK2 midpoint is second order.  For a frozen field the local error is
(dt^3/6) * (u'' u + ...) = O(dt^3) and the global error O(dt^2); for the linear field u = M x + b it is
exp(M T) - [I + dt M + (dt M)^2/2]^n = (T dt^2 / 6) M^3 exp(M T) + O(dt^3) (T = n dt).
"""
import math

import numpy as np

DEFAULT_MAX_FRAC = 0.5
FLAG_X, FLAG_Y, FLAG_ZWALL, FLAG_ZWRAP = 1, 2, 4, 8


def substep_count(dt, umax, h, max_frac=DEFAULT_MAX_FRAC):
    """Equal substeps so that no particle moves more than ``max_frac*h`` per substep at speed ``umax``.

    dt: time span [time]; umax: top speed [L/time]; h: cell size [L].  Returns max(1, ceil(dt*umax/(max_frac*h) - 1e-9)).
    """
    if not (dt >= 0 and h > 0 and max_frac > 0 and umax >= 0):
        raise ValueError("substep_count: invalid argument")
    return max(1, math.ceil(dt * umax / (max_frac * h) - 1e-9))


def _vg(V, c, i, j, k, periodic):
    """Velocity component c at integer cell indices (arrays), with the odd-mirror / periodic ghost rule."""
    _, nz, ny, nx = V.shape
    s = np.ones(i.shape)
    lo, hi = i < 0, i >= nx
    i = np.where(lo, 0, np.where(hi, nx - 1, i)); s = np.where(lo | hi, -s, s)
    lo, hi = j < 0, j >= ny
    j = np.where(lo, 0, np.where(hi, ny - 1, j)); s = np.where(lo | hi, -s, s)
    lo, hi = k < 0, k >= nz
    if periodic:
        k = np.where(lo, nz - 1, np.where(hi, 0, k))
    else:
        k = np.where(lo, 0, np.where(hi, nz - 1, k)); s = np.where(lo | hi, -s, s)
    return s * V[c, k, j, i]


def velocity_at(V, pos, h, periodic=True):
    """Trilinear velocity at points ``pos`` (np, 3) [L]; V (3, nz, ny, nx) [L/time]; returns (np, 3) [L/time]."""
    V = np.asarray(V, dtype=np.float64)
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    _, nz, ny, nx = V.shape
    lx, ly, lz = nx * h, ny * h, nz * h
    x = np.clip(pos[:, 0], 0.0, lx)
    y = np.clip(pos[:, 1], 0.0, ly)
    z = pos[:, 2] - lz * np.floor(pos[:, 2] / lz) if periodic else np.clip(pos[:, 2], 0.0, lz)
    fx, fy, fz = x / h - 0.5, y / h - 0.5, z / h - 0.5
    i0, j0, k0 = (np.floor(f).astype(np.int64) for f in (fx, fy, fz))
    tx, ty, tz = fx - i0, fy - j0, fz - k0
    out = np.empty((pos.shape[0], 3))
    for c in range(3):
        def g(di, dj, dk):
            return _vg(V, c, i0 + di, j0 + dj, k0 + dk, periodic)
        a0 = _lerp(_lerp(g(0, 0, 0), g(1, 0, 0), tx), _lerp(g(0, 1, 0), g(1, 1, 0), tx), ty)
        a1 = _lerp(_lerp(g(0, 0, 1), g(1, 0, 1), tx), _lerp(g(0, 1, 1), g(1, 1, 1), tx), ty)
        out[:, c] = _lerp(a0, a1, tz)
    return out


def _lerp(a, b, t):
    return a + (b - a) * t


def rk2_step(V, pos, dt, h, periodic=True):
    """One RK2 midpoint step of length dt [time] with the field frozen; no wall handling (see advect)."""
    u1 = velocity_at(V, pos, h, periodic)
    u2 = velocity_at(V, pos + 0.5 * dt * u1, h, periodic)
    return pos + dt * u2


def advect(V, pos, dt_total, h, periodic=True, nsub=None, umax=None, max_frac=DEFAULT_MAX_FRAC, want_speed=False):
    """Advect particles over the time span ``dt_total`` [time].

    V: velocity (3, nz, ny, nx) [L/time]; pos: (np, 3) [L]; h: cell size [L].
    Substeps: give ``nsub`` (>= 0; 0 moves nothing), or ``umax`` [L/time] (then nsub = substep_count(dt_total, umax,
    h, max_frac)).  Defaults: periodic z, max_frac = 0.5.
    Returns dict(pos (np,3) [L], speed (np,) [L/time] or None, flags (np,) int, nsub).
    """
    V = np.asarray(V, dtype=np.float64)
    q = np.array(pos, dtype=np.float64, ndmin=2)
    _, nz, ny, nx = V.shape
    lx, ly, lz = nx * h, ny * h, nz * h
    if nsub is None:
        if umax is None:
            raise ValueError("give nsub, or umax")
        nsub = 0 if dt_total == 0 else substep_count(dt_total, umax, h, max_frac)
    dt = dt_total / nsub if nsub else 0.0
    flags = np.zeros(q.shape[0], dtype=np.int64)
    for _ in range(nsub):
        q = rk2_step(V, q, dt, h, periodic)
        for axis, hi, bit in ((0, lx, FLAG_X), (1, ly, FLAG_Y)):
            out = (q[:, axis] < 0) | (q[:, axis] > hi)
            flags[out] |= bit
            q[:, axis] = np.clip(q[:, axis], 0.0, hi)
        if periodic:
            out = (q[:, 2] < 0) | (q[:, 2] >= lz)
            flags[out] |= FLAG_ZWRAP
            z = q[:, 2] - lz * np.floor(q[:, 2] / lz)
            q[:, 2] = np.where(out & (z >= lz), 0.0, z)    # rounding: lz - tiny + lz = lz is the same point as 0
        else:
            out = (q[:, 2] < 0) | (q[:, 2] > lz)
            flags[out] |= FLAG_ZWALL
            q[:, 2] = np.clip(q[:, 2], 0.0, lz)
    speed = np.linalg.norm(velocity_at(V, q, h, periodic), axis=1) if want_speed else None
    return {"pos": q, "speed": speed, "flags": flags, "nsub": nsub}


# ------------------------------------------------------------------------------------------ test data
def _lcg(seed, n):
    """n uniform numbers in [0,1) from a 32-bit LCG (exactly reproducible in JavaScript, see the GPU test)."""
    s = seed & 0xFFFFFFFF
    out = np.empty(n)
    for m in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[m] = (s >> 8) / 16777216.0
    return out


def seed_positions(count, nx, ny, nz, h, seed):
    """Deterministic start positions [L] (float64 values rounded to float32, as the GPU receives them).

    Particle p uses u1, u2, u3 from the LCG; p % 5 selects: 0 anywhere; 1 within 0.3 h of the x = 0 wall;
    2 within 0.3 h of the x = nx h wall; 3 within 0.3 h of the z = 0 / nz h seam (side by u2); 4 within 0.3 h of
    the y = 0 wall.  The 0.3 h strips are inside the outermost cell centres' half cell, where the ghost rule acts.
    """
    u = _lcg(seed, 3 * count).reshape(count, 3)
    lx, ly, lz = nx * h, ny * h, nz * h
    pos = np.stack([u[:, 0] * lx, u[:, 1] * ly, u[:, 2] * lz], axis=1)
    m = np.arange(count) % 5
    pos[m == 1, 0] = u[m == 1, 0] * 0.3 * h
    pos[m == 2, 0] = lx - u[m == 2, 0] * 0.3 * h
    pos[m == 3, 2] = np.where(u[m == 3, 1] < 0.5, u[m == 3, 2] * 0.3 * h, lz - u[m == 3, 2] * 0.3 * h)
    pos[m == 4, 1] = u[m == 4, 1] * 0.3 * h
    return pos.astype(np.float32).astype(np.float64)


def make_field(nx, ny, nz, h, amp, seed):
    """Smooth no-penetration-agnostic test field (3, nz, ny, nx) [L/time], rounded to float32 values.

    Sum of low-wavenumber sines with random phases (seeded); amplitude about ``amp``.  Not divergence free and
    not zero at walls: the kernel's ghost rule makes it vanish on the walls anyway.
    """
    r = _lcg(seed, 3 * 4 * 4).reshape(3, 4, 4)
    k = (np.arange(nz) + 0.5) * h; j = (np.arange(ny) + 0.5) * h; i = (np.arange(nx) + 0.5) * h
    Z, Y, X = np.meshgrid(k, j, i, indexing="ij")
    L = (nx * h, ny * h, nz * h)
    V = np.zeros((3, nz, ny, nx))
    for c in range(3):
        for m in range(4):
            wx, wy, wz = (1 + int(4 * r[c, m, 0])), (1 + int(3 * r[c, m, 1])), (1 + int(3 * r[c, m, 2]))
            V[c] += np.sin(2 * np.pi * (wx * X / L[0] + wy * Y / L[1] + wz * Z / L[2]) + 6.28 * r[c, m, 3])
    return (amp * V / 4.0 * 2.0).astype(np.float32).astype(np.float64)


def _r(a, digits=9):
    """Round to ``digits`` significant digits (9 reproduces float32 values exactly) to keep the JSON small."""
    return [float(f"{x:.{digits}g}") for x in np.ravel(a)]


def _case(name, nx, ny, nz, h, periodic, count, nsub, dt, amp, seed, stride, want_speed=True):
    h = float(np.float32(h))
    dt = float(np.float32(dt))
    V = make_field(nx, ny, nz, h, amp, seed)
    q0 = seed_positions(count, nx, ny, nz, h, seed + 1)
    res = advect(V, q0, dt * nsub, h, periodic=periodic, nsub=nsub, want_speed=want_speed)
    idx = np.arange(0, count, stride)
    return {
        "name": name, "nx": nx, "ny": ny, "nz": nz, "h": h, "periodic": periodic, "count": count,
        "nsub": nsub, "dt_sub": dt, "dt_total": dt * nsub, "want_speed": want_speed,
        "field_seed": seed, "pos_seed": seed + 1, "field_amp": amp,
        "velocity": _r(V),
        "stride": stride,
        "pos": _r(res["pos"][idx], 12),
        "speed": (_r(res["speed"][idx], 12) if want_speed else []),
        "flags": res["flags"][idx].tolist(),
        "flag_counts": [int(np.sum((res["flags"] & b) != 0)) for b in (1, 2, 4, 8)],
        "max_speed": float(np.abs(V).max()),
    }


def make_reference():
    """Test inputs and f64 expected outputs for tests/gpu/particle_advect_rk2.test.js (JSON-serialisable).

    The velocity field is stored; particle start positions are regenerated by the test from the LCG (seed_positions)
    and expected outputs are stored for every ``stride``-th particle (all of them for the small cases).
    """
    return {"cases": [
        # anisotropic grid, periodic z, 8 substeps; every particle stored
        _case("periodic z, 14x10x12, 2000 particles", 14, 10, 12, 0.05, True, 2000, 8, 0.02, 1.0, 11, 1),
        # walled z, same grid
        _case("walled z, 14x10x12, 2000 particles", 14, 10, 12, 0.05, False, 2000, 8, 0.02, 1.0, 12, 1),
        # hard: 1e5 particles, 24^3, 40 substeps (about 0.5 h per substep at the field's top speed), periodic z
        _case("hard: 1e5 particles, 24^3, 40 substeps, periodic z", 24, 24, 24, 1.0 / 24, True, 100000, 40, 0.0125, 1.0, 21, 25),
        # hard: very large substeps (2 substeps of 4 h per unit speed): many particles leave through walls (flags)
        _case("hard: huge substeps, walled z, clamp flags", 16, 16, 16, 1.0 / 16, False, 20000, 2, 0.3, 1.0, 31, 4),
        # sampling only (nsub = 0): |u| at the start positions, a pure interpolation test
        _case("sampling only (nsub=0), 14x10x12", 14, 10, 12, 0.05, True, 2000, 0, 0.0, 1.0, 41, 1),
    ]}
