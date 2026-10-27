// GPU test for particle_advect_rk2: f32 WGSL kernel vs the stored float64 reference positions, speeds and flags.
// Start positions are regenerated here with the same 32-bit LCG as python/gpu/particle_advect_rk2.py
// (seed_positions); the reference stores expected outputs for every `stride`-th particle.
import { createParticleAdvector } from "../../js/particle_advect_rk2.js";

function lcg(seed, n) {
  let s = seed >>> 0; const out = new Float64Array(n);
  for (let m = 0; m < n; m++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; out[m] = (s >>> 8) / 16777216; }
  return out;
}
function seedPositions(count, nx, ny, nz, h, seed) {
  const u = lcg(seed, 3 * count), lx = nx * h, ly = ny * h, lz = nz * h, pos = new Float32Array(3 * count);
  for (let p = 0; p < count; p++) {
    const u0 = u[3 * p], u1 = u[3 * p + 1], u2 = u[3 * p + 2], m = p % 5;
    let x = u0 * lx, y = u1 * ly, z = u2 * lz;
    if (m === 1) x = u0 * 0.3 * h;
    else if (m === 2) x = lx - u0 * 0.3 * h;
    else if (m === 3) z = u1 < 0.5 ? u2 * 0.3 * h : lz - u2 * 0.3 * h;
    else if (m === 4) y = u1 * 0.3 * h;
    pos[3 * p] = x; pos[3 * p + 1] = y; pos[3 * p + 2] = z;
  }
  return pos;
}

// Tolerances, f32 GPU vs f64 reference: max position error in units of h, max |u| error relative to the field's top
// speed.  Per case, 4-9x over the measured values (3.6e-5, 1.1e-4, 1.1e-3, 2.3e-4 h; speeds 2.5e-5..3.4e-4).
// The test field is compressible, so f32 round-off (~1e-7 of the box size per interpolation) is amplified along
// the paths that converge on sinks: the longest, densest case has the largest error.  A numpy float32 emulation of
// the algorithm shows the same errors (3.6e-5, 1.1e-4, 1.1e-3 h), i.e. they are f32 round-off, not kernel bugs.
const TOL_POS = [2e-4, 5e-4, 5e-3, 2e-3, 1e-5], TOL_SPEED = [2e-4, 1e-3, 2e-3, 1.5e-3, 5e-6];

export async function run(device, ref) {
  const rows = [];
  for (const [ci, c] of ref.cases.entries()) {
    const adv = await createParticleAdvector(device, { nx: c.nx, ny: c.ny, nz: c.nz, h: c.h, periodic: c.periodic, maxParticles: c.count });
    adv.setVelocity(Float32Array.from(c.velocity));
    adv.setPositions(seedPositions(c.count, c.nx, c.ny, c.nz, c.h, c.pos_seed));
    await adv.advect({ dt: c.dt_total, nsub: c.nsub, wantSpeed: c.want_speed });
    const out = await adv.read();
    const Lz = c.nz * c.h;
    let ePos = 0, eSp = 0, nFlagBad = 0, nBad = 0;
    const idx = c.flags.map((_, m) => m * c.stride);
    idx.forEach((p, m) => {
      for (let a = 0; a < 3; a++) {
        let d = Math.abs(out.xyz[3 * p + a] - c.pos[3 * m + a]);
        if (a === 2 && c.periodic) d = Math.min(d, Math.abs(d - Lz));        // z = 0 and z = Lz are one point
        ePos = Math.max(ePos, d / c.h);
      }
      if (c.want_speed) eSp = Math.max(eSp, Math.abs(out.speed[p] - c.speed[m]) / c.max_speed);
      if (out.flags[p] !== c.flags[m]) nFlagBad++;
    });
    // all particles (not only the stored subset): inside the box, flag totals vs the reference's totals
    const lim = [c.nx * c.h, c.ny * c.h, Lz];
    for (let p = 0; p < c.count; p++) for (let a = 0; a < 3; a++) {
      const v = out.xyz[3 * p + a], hi = Math.fround(lim[a]) * (1 + 1e-6);
      if (!(v >= 0 && v <= hi)) nBad++;
    }
    const cnt = [1, 2, 4, 8].map(b => out.flags.reduce((s, f) => s + ((f & b) ? 1 : 0), 0));
    // totals over all particles: tolerance 0.1 % of N (rounding at a wall threshold may flip a few)
    const eCnt = Math.max(...cnt.map((n, i) => Math.abs(n - c.flag_counts[i]))) / c.count;
    adv.destroy();
    rows.push({ name: `${c.name}: max position error / h (${idx.length} stored)`, err: ePos, tol: TOL_POS[ci], pass: ePos <= TOL_POS[ci] });
    if (c.want_speed) rows.push({ name: `${c.name}: max |u| error / top speed`, err: eSp, tol: TOL_SPEED[ci], pass: eSp <= TOL_SPEED[ci] });
    rows.push({ name: `${c.name}: flag words differing from f64 (stored)`, err: nFlagBad, tol: 0, pass: nFlagBad === 0 });
    rows.push({ name: `${c.name}: flag totals, max |diff| / N (all particles)`, err: eCnt, tol: 1e-3, pass: eCnt <= 1e-3 });
    rows.push({ name: `${c.name}: particles outside the box (all)`, err: nBad, tol: 0, pass: nBad === 0 });
  }
  return rows;
}
