// GPU test for multigrid_poisson_cc: GPU (f32) vs the stored float64 reference (solution, residual history).
// Reference data: tests/reference_data/gpu/multigrid_poisson_cc.json (library/python/gpu/multigrid_poisson_cc.py).
// Tolerances (f32 vs f64, iterative, accumulating over <= 8 V-cycles; measured errors are 3-10x smaller):
//   solution      max|x_gpu - x_ref| / max|x_ref|                       <= 2e-5
//   residual hist max over cycles with ref RMS >= 1e-3 * initial of |r_gpu - r_ref| / r_ref   <= 1e-3
//     (later entries sit near the f32 floor of b - L x, where only the order of magnitude is meaningful)
//   z-uniform     max z-plane difference / max|x|, z-independent b: periodic z, 4 cycles   <= 1e-6 (stays z-independent
//                 exactly, the point of the i+j colouring); walls z, 40 cycles   <= 5e-6 (the end layers have fewer
//                 neighbours, so the Jacobi-in-z smoother makes transient z-differences, 8e-5 after 4 cycles in the
//                 f64 reference too, that decay to the f32 floor)
//   exact         cycles done (integer) equals the reference
import { createMultigrid } from "../../js/multigrid_poisson_cc.js";

const TOL_X = 2e-5, TOL_RES = 1e-3, RES_FLOOR = 1e-3;

// mirror of hash_field / test_rhs in library/python/gpu/multigrid_poisson_cc.py (integer hash, exact)
function hashField(n, nz, seed) {
  const out = new Float64Array(n * n * nz);
  const s = Math.imul(seed, 2654435761) >>> 0;     // (seed * 2654435761) mod 2^32
  for (let k = 0; k < nz; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    let v = (Math.imul(i, 73856093) ^ Math.imul(j, 19349663) ^ Math.imul(k, 83492791) ^ s) >>> 0;
    v = (v ^ (v >>> 16)) >>> 0; v = Math.imul(v, 0x45d9f3b) >>> 0;
    v = (v ^ (v >>> 16)) >>> 0; v = Math.imul(v, 0x45d9f3b) >>> 0;
    v = (v ^ (v >>> 16)) >>> 0;
    out[(k * n + j) * n + i] = v / 2147483648 - 1;
  }
  return out;
}
function testRhs(n, nz, seed) {
  const b = hashField(n, nz, seed);
  for (let k = 0; k < nz; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
    b[(k * n + j) * n + i] += 3 * Math.cos(Math.PI * (i + 0.5) / n) * Math.cos(2 * Math.PI * (j + 0.5) / n) + 0.25;
  return b;
}

export async function run(device, ref) {
  const rows = [];
  for (const c of ref.cases) {
    const mg = await createMultigrid(device, { n: c.n, nz: c.nz, h: c.h, bcZ: c.bc_z, cycles: c.cycles,
      ...(c.nu1 !== undefined ? { nu1: c.nu1, nu2: c.nu2, ncoarse: c.ncoarse } : {}) });
    const x0 = c.x0_seed !== undefined ? hashField(c.n, c.nz, c.x0_seed).map(v => 0.3 * v) : null;
    const t0 = performance.now();
    const res = await mg.solve(testRhs(c.n, c.nz, c.seed), { x0 });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    mg.destroy();
    // solution on the stored (strided) points
    const [sz, sy, sx] = c.x_shape, s = c.stride; let e = 0, q = 0;
    for (let k = 0; k < sz; k++) for (let j = 0; j < sy; j++) for (let i = 0; i < sx; i++) {
      const g = res.x[((k * s) * c.n + j * s) * c.n + i * s];
      e = Math.max(e, Math.abs(g - c.x[(k * sy + j) * sx + i]));
    }
    rows.push({ name: `${c.name}: solution max error / max|x| (${secs} s)`, err: e / c.x_max, tol: TOL_X, pass: e / c.x_max <= TOL_X });
    // RMS of the whole field
    let ss = 0; for (const v of res.x) ss += v * v;
    const rmsErr = Math.abs(Math.sqrt(ss / res.x.length) - c.x_rms) / c.x_rms;
    rows.push({ name: `${c.name}: solution RMS rel. error`, err: rmsErr, tol: TOL_X, pass: rmsErr <= TOL_X });
    // residual history
    let eh = 0; const r0 = c.residual_rms[0];
    c.residual_rms.forEach((rr, m) => { if (rr >= RES_FLOOR * r0) eh = Math.max(eh, Math.abs(res.residualRMS[m] - rr) / rr); });
    rows.push({ name: `${c.name}: residual history rel. error (${res.residualRMS.length} entries)`, err: eh, tol: TOL_RES,
      pass: eh <= TOL_RES && res.residualRMS.length === c.residual_rms.length });
    rows.push({ name: `${c.name}: cycles done (exact)`, err: Math.abs(res.cycles - c.cycles_done), tol: 0, pass: res.cycles === c.cycles_done });
  }
  // z-uniform input must give a z-uniform solution (red-black colouring independent of k)
  for (const [bcZ, cycles, tol] of [["periodic", 4, 1e-6], ["walls", 40, 5e-6]]) {
    const n = 32, nz = 16, plane = testRhs(n, 1, 11), b = new Float64Array(n * n * nz);
    for (let k = 0; k < nz; k++) b.set(plane, k * n * n);
    const mg = await createMultigrid(device, { n, nz, h: 0.05, bcZ, cycles });
    const res = await mg.solve(b); mg.destroy();
    let d = 0, m = 0;
    for (let k = 1; k < nz; k++) for (let p = 0; p < n * n; p++) d = Math.max(d, Math.abs(res.x[k * n * n + p] - res.x[p]));
    for (const v of res.x) m = Math.max(m, Math.abs(v));
    rows.push({ name: `z-uniform b, ${bcZ} z, ${cycles} cycles: max plane difference / max|x|`, err: d / m, tol, pass: d / m <= tol });
  }
  // validation errors must throw
  for (const [n, nz] of [[6, 4], [3, 4], [16, 3], [16, 0]]) {
    let thrown = false;
    try { await createMultigrid(device, { n, nz }); } catch (e) { thrown = /power of two/.test(e.message); }
    rows.push({ name: `invalid size N=${n}, NZ=${nz} throws`, err: thrown ? 0 : 1, tol: 0, pass: thrown });
  }
  return rows;
}
