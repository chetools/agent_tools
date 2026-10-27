// particle_advect_rk2.wgsl: advect massless particles through a gridded 3D velocity field (RK2 midpoint).
//
// WHAT IT COMPUTES
//   For every particle p, `nsub` explicit RK2 (midpoint) substeps of equal length dt, so the particle moves over
//   the time span nsub*dt:   u1 = u(q);  u2 = u(q + dt/2 u1);  q <- q + dt u2.
//   The velocity is trilinearly interpolated between cell centres (see "Interpolation convention").
//   Optionally the speed |u| at the final position is written to the 4th component of the particle.
//   nsub = 0 moves nothing and only samples |u| at the given positions (a pure interpolation test).
//
// UNITS
//   Lengths: any consistent length unit (metres in SI); h is the cell size in that unit.
//   Velocity: length / time, in the same length unit.  dt and the time span: the matching time unit.
//   Positions are physical coordinates in the length unit, in the box [0, nx h] x [0, ny h] x [0, nz h].
//
// INTERPOLATION CONVENTION (identical to cylinder-cavity-3d trails.wgsl vel()/vg())
//   Cell (i,j,k) has its centre at ((i+1/2) h, (j+1/2) h, (k+1/2) h); the stored velocity (3 components) is the
//   value at that centre.  The fractional index is f = x/h - 1/2; i0 = floor(f), t = f - i0, and the value is the
//   trilinear blend of the 8 surrounding centres.  A stencil index one cell outside the grid uses a ghost value:
//     x and y (always walls):  ghost = -(edge cell value), all three components (odd mirror), so the
//                              interpolated velocity is exactly 0 on the planes x=0, x=nx h, y=0, y=ny h;
//     z, periodic = 1:         index wraps (-1 -> nz-1, nz -> 0);
//     z, periodic = 0:         odd mirror like x and y (velocity exactly 0 on z=0 and z=nz h).
//   Sampling points are first clamped to the box (x, y, and z when walled) or wrapped (z when periodic).
//
// OUT-OF-DOMAIN POLICY (clamp with a flag)
//   After every substep: x and y (and z when walled) are CLAMPED to the box walls; periodic z is wrapped into
//   [0, nz h).  A particle clamped to a wall stays there (the velocity is 0 on the wall).  The per-particle
//   flags buffer records, OR-accumulated over all substeps of the call (it is overwritten at each call):
//     bit 0 (1): x clamped   bit 1 (2): y clamped   bit 2 (4): z clamped (walled z)   bit 3 (8): z wrapped.
//   Nothing is re-seeded or dropped; the caller decides what to do with flagged particles.
//
// BUFFER LAYOUT AND INDEX ORDER
//   binding 0  uniform  P (48 bytes, see struct below)
//   binding 1  storage, read        V: array<f32>, velocity [c][k][j][i], index ((c*nz + k)*ny + j)*nx + i,
//                                   c = 0,1,2 for (u, v, w) in x, y, z; length 3*nx*ny*nz
//   binding 2  storage, read_write  PS: array<vec4<f32>>, one entry per particle: (x, y, z, |u|).  x,y,z in and
//                                   out; |u| written only when want_speed = 1 (else left unchanged)
//   binding 3  storage, read_write  FL: array<u32>, one flag word per particle (see above)
//
// LIMITS
//   workgroup_size 64, one thread per particle; dispatch ceil(np/64) workgroups in x (host: np <= 64*65535 on
//   default limits).  nx, ny, nz >= 1 (any size, not only powers of two).  Index arithmetic is u32.
//
// PRECISION
//   f32 throughout.  Positions are good to about 1e-7 * (box size) per step; expect roughly 1e-6 h to 1e-5 h
//   after a few tens of substeps (measured in library/tests/gpu/particle_advect_rk2.test.js).  A tiny negative z
//   that rounds to exactly nz h on wrapping is stored as 0 (the same point).
//
// Extracted from chetools/claude cylinder-cavity-3d/web/trails.wgsl (cs_advect, vel, vg); the trail storage and
// drawing, the re-seeding RNG, particle lifetimes, the substep schedule table and the cylinder hiding are removed.

struct P {
  nx: i32, ny: i32, nz: i32, np: u32,
  h: f32, dt: f32, nsub: u32, periodic: u32,      // dt: length of one substep (time unit); periodic: z periodic?
  want_speed: u32, pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> G: P;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> PS: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> FL: array<u32>;

// velocity component c at cell (i,j,k); an index may lie one cell outside the grid (ghost rule above)
fn vg(c: i32, i: i32, j: i32, k: i32) -> f32 {
  let nx = G.nx; let ny = G.ny; let nz = G.nz;
  var ii = i; var jj = j; var kk = k; var s = 1.0;
  if (ii < 0) { ii = 0; s = -s; } else if (ii >= nx) { ii = nx - 1; s = -s; }
  if (jj < 0) { jj = 0; s = -s; } else if (jj >= ny) { jj = ny - 1; s = -s; }
  if (kk < 0) { if (G.periodic != 0u) { kk = nz - 1; } else { kk = 0; s = -s; } }
  else if (kk >= nz) { if (G.periodic != 0u) { kk = 0; } else { kk = nz - 1; s = -s; } }
  return s * V[u32(((c * nz + kk) * ny + jj) * nx + ii)];
}

// trilinear interpolation of the velocity between cell centres, at the physical point p0
fn vel(p0: vec3<f32>) -> vec3<f32> {
  let lx = f32(G.nx) * G.h; let ly = f32(G.ny) * G.h; let lz = f32(G.nz) * G.h;
  let x = clamp(p0.x, 0.0, lx); let y = clamp(p0.y, 0.0, ly);
  var z = p0.z;
  if (G.periodic != 0u) { z = z - lz * floor(z / lz); } else { z = clamp(z, 0.0, lz); }
  let fx = x / G.h - 0.5; let fy = y / G.h - 0.5; let fz = z / G.h - 0.5;
  let i0 = i32(floor(fx)); let j0 = i32(floor(fy)); let k0 = i32(floor(fz));
  let tx = fx - f32(i0); let ty = fy - f32(j0); let tz = fz - f32(k0);
  var out = vec3<f32>(0.0);
  for (var c = 0; c < 3; c++) {
    let a0 = mix(mix(vg(c, i0, j0, k0), vg(c, i0 + 1, j0, k0), tx), mix(vg(c, i0, j0 + 1, k0), vg(c, i0 + 1, j0 + 1, k0), tx), ty);
    let a1 = mix(mix(vg(c, i0, j0, k0 + 1), vg(c, i0 + 1, j0, k0 + 1), tx), mix(vg(c, i0, j0 + 1, k0 + 1), vg(c, i0 + 1, j0 + 1, k0 + 1), tx), ty);
    out[c] = mix(a0, a1, tz);
  }
  return out;
}

@compute @workgroup_size(64)
fn cs_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= G.np) { return; }
  let lx = f32(G.nx) * G.h; let ly = f32(G.ny) * G.h; let lz = f32(G.nz) * G.h;
  var q = PS[p].xyz;
  var flags = 0u;
  for (var s = 0u; s < G.nsub; s++) {
    let u1 = vel(q);
    let u2 = vel(q + 0.5 * G.dt * u1);
    q += G.dt * u2;
    if (q.x < 0.0 || q.x > lx) { q.x = clamp(q.x, 0.0, lx); flags |= 1u; }
    if (q.y < 0.0 || q.y > ly) { q.y = clamp(q.y, 0.0, ly); flags |= 2u; }
    if (G.periodic != 0u) {
      if (q.z < 0.0 || q.z >= lz) {
        q.z = q.z - lz * floor(q.z / lz);
        if (q.z >= lz) { q.z = 0.0; }                     // rounding: lz - tiny + lz rounds to lz; same point as 0
        flags |= 8u;
      }
    } else if (q.z < 0.0 || q.z > lz) { q.z = clamp(q.z, 0.0, lz); flags |= 4u; }
  }
  var w = PS[p].w;
  if (G.want_speed != 0u) { w = length(vel(q)); }
  PS[p] = vec4<f32>(q, w);
  FL[p] = flags;
}
