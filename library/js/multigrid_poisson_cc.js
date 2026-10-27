// multigrid_poisson_cc.js -- host wrapper for library/wgsl/multigrid_poisson_cc.wgsl (area `gpu`).
// Cell-centred geometric multigrid V-cycles for  laplacian(x) = b  on an N x N x NZ box of cells of edge h:
// x, y zero-flux walls; z periodic or zero-flux walls.  f32 on the GPU; Python float64 reference:
// library/python/gpu/multigrid_poisson_cc.py (same defaults and algorithm).
//
// Array layout (Float32Array, length N*N*NZ): index (k * N + j) * N + i, i = x fastest, j = y, k = z.
// Units: h in metres (or any consistent length unit); x in [X]; b and the residual in [X]/length^2.
// Size constraints (checked, throws): N power of two >= 4; NZ power of two >= 1.
// Extracted from chetools/claude cylinder-cavity-3d (web/fv_gpu.js levels + V-cycle dispatch order).
// No DOM access.  Usage:
//   const mg = await createMultigrid(device, { n: 32, nz: 16, h: 0.1, bcZ: "walls" });
//   const res = await mg.solve(bFloat32Array);   // { x, residualRMS, cycles, converged, success, message }
//   mg.destroy();
import { loadWGSL, compile, makePipelines, bindGroup, storage, uniform, readBuffer, runPass, checkBufferSize }
  from "./gpu_util.js";

/** Option defaults, identical to the Python reference (DEFAULTS in multigrid_poisson_cc.py). */
export const DEFAULTS = Object.freeze({ bcZ: "periodic", cycles: 2, nu1: 2, nu2: 2, ncoarse: 40, rtol: 0,
  removeMeanB: true, removeMeanX: true, h: 1 });
export const N_COARSEST = 4;

const isPow2 = v => Number.isInteger(v) && v >= 1 && (v & (v - 1)) === 0;
/** Throws unless N is a power of two >= 4 and NZ a power of two >= 1. */
export function validateGrid(n, nz) {
  if (!isPow2(n) || n < N_COARSEST) throw new Error(`multigrid: N must be a power of two >= ${N_COARSEST}, got ${n}`);
  if (!isPow2(nz)) throw new Error(`multigrid: NZ must be a power of two >= 1, got ${nz}`);
}

const BINDS = [[0, "uniform"], [1, "storage"], [2, "storage"], [3, "storage"], [4, "storage"], [5, "storage"]];
const KERNELS = ["mg_smooth", "mg_residual", "mg_restrict", "mg_prolong", "mg_coarse", "mg_mean_b", "mg_mean_x"];

/**
 * Create a solver for one grid size.
 * opts: n (cells in x and y), nz (cells in z), h (cell edge, length unit, default 1),
 *   bcZ "periodic" | "walls" (default periodic), nu1, nu2 (pre/post red-black sweeps per level, default 2),
 *   ncoarse (sweeps on the coarsest level n = 4, default 40), cycles (default 2), rtol (relative RMS residual
 *   tolerance for early stop, default 0 = off), removeMeanB, removeMeanX (default true).
 * Returns { n, nz, h, buffers: {x, b, r}, solve, setRhs, setSolution, readSolution, residualRMS, encodeVcycle,
 *   encodeSolve, destroy }.  buffers.x/b are the level-0 storage buffers (f32), usable by other kernels.
 */
export async function createMultigrid(device, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const { n, nz, h } = o;
  validateGrid(n, nz);
  if (o.bcZ !== "periodic" && o.bcZ !== "walls") throw new Error(`multigrid: bcZ must be "periodic" or "walls", got ${o.bcZ}`);
  if (!(h > 0)) throw new Error("multigrid: h must be > 0");
  for (const k of ["cycles", "nu1", "nu2"]) if (!Number.isInteger(o[k]) || o[k] < 0) throw new Error(`multigrid: ${k} must be an integer >= 0`);
  if (!Number.isInteger(o.ncoarse) || o.ncoarse < 1) throw new Error("multigrid: ncoarse must be an integer >= 1");
  if (device.limits.maxComputeInvocationsPerWorkgroup < 256 || device.limits.maxComputeWorkgroupSizeX < 256)
    throw new Error("multigrid: device needs workgroups of 256 invocations");
  if (device.limits.maxStorageBuffersPerShaderStage < 5) throw new Error("multigrid: device needs 5 storage buffers per stage");
  checkBufferSize(device, 4 * n * n * nz, `multigrid level-0 buffer (${n}x${n}x${nz} f32)`);

  const module = await compile(device, await loadWGSL("multigrid_poisson_cc"), "multigrid_poisson_cc");
  const { pipes, layouts } = await makePipelines(device, module, Object.fromEntries(KERNELS.map(k => [k, BINDS])));
  const owned = [];
  const buf = bytes => { const b = storage(device, bytes); owned.push(b); return b; };
  const periodic = o.bcZ === "periodic" ? 1 : 0;

  // levels: (n, nz), (n/2, nz/2), ... while n >= 4; z is halved while it has >= 2 layers
  const levels = [];
  for (let m = n, z = nz, l = 0; m >= N_COARSEST; m /= 2, l++, z = z >= 2 ? z / 2 : 1) {
    const cells = z * m * m;
    levels.push({ n: m, nz: z, nzc: z >= 2 ? z / 2 : 1, x: buf(4 * cells), b: buf(4 * cells), r: buf(4 * cells) });
  }
  const dummy = [buf(16), buf(16)];
  levels.forEach((lv, l) => {
    const next = levels[l + 1];
    lv.groups = [0, 1].map(colour => {
      const a = new ArrayBuffer(32), I = new Int32Array(a), F = new Float32Array(a);
      I.set([lv.n, lv.nz, lv.nzc, colour]);
      F[4] = (h * 2 ** l) * (h * 2 ** l);
      I[5] = next ? 0 : o.ncoarse; I[6] = periodic;
      const ub = uniform(device, a); owned.push(ub);
      return bindGroup(device, layouts.mg_smooth, { 0: ub, 1: lv.x, 2: lv.b, 3: lv.r,
        4: next ? next.x : dummy[0], 5: next ? next.b : dummy[1] });
    });
  });
  const L0 = levels[0], cells0 = n * n * nz;

  function dispatch(pass, name, l, colour = 0) {
    const lv = levels[l], nc = lv.n / 2;
    pass.setPipeline(pipes[name]);
    pass.setBindGroup(0, lv.groups[colour]);
    if (name === "mg_coarse" || name === "mg_mean_b" || name === "mg_mean_x") pass.dispatchWorkgroups(1);
    else if (name === "mg_smooth") pass.dispatchWorkgroups(Math.ceil(lv.n / 16), Math.ceil(lv.n / 16));
    else if (name === "mg_restrict") pass.dispatchWorkgroups(Math.ceil(nc / 8), Math.ceil(nc / 8), Math.ceil(lv.nzc / 4));
    else pass.dispatchWorkgroups(Math.ceil(lv.n / 8), Math.ceil(lv.n / 8), Math.ceil(lv.nz / 4));
  }
  function vcycle(pass, l, nu1, nu2) {
    if (l === levels.length - 1) { dispatch(pass, "mg_coarse", l); return; }
    for (let k = 0; k < nu1; k++) { dispatch(pass, "mg_smooth", l, 0); dispatch(pass, "mg_smooth", l, 1); }
    dispatch(pass, "mg_residual", l);
    dispatch(pass, "mg_restrict", l);
    vcycle(pass, l + 1, nu1, nu2);
    dispatch(pass, "mg_prolong", l);
    for (let k = 0; k < nu2; k++) { dispatch(pass, "mg_smooth", l, 0); dispatch(pass, "mg_smooth", l, 1); }
  }

  const toF32 = (a, what) => {
    if (a.length !== cells0) throw new Error(`multigrid: ${what} has ${a.length} values, expected ${cells0} (= ${n}*${n}*${nz})`);
    return a instanceof Float32Array ? a : Float32Array.from(a);
  };
  const api = {
    n, nz, h, options: o, buffers: { x: L0.x, b: L0.b, r: L0.r },
    /** Upload b (Float32Array/Float64Array/Array, N*N*NZ values in [X]/length^2). */
    setRhs(b) { device.queue.writeBuffer(L0.b, 0, toF32(b, "b")); },
    /** Upload an initial guess, or null for zeros. */
    setSolution(x) {
      if (x) device.queue.writeBuffer(L0.x, 0, toF32(x, "x0"));
      else device.queue.writeBuffer(L0.x, 0, new Float32Array(cells0));
    },
    /** Record one V-cycle on level 0 into an open compute pass (no readback). */
    encodeVcycle(pass, nu1 = o.nu1, nu2 = o.nu2) { vcycle(pass, 0, nu1, nu2); },
    /** Record [mean removal of b] + `cycles` V-cycles + [mean removal of x] into an open compute pass (no residuals). */
    encodeSolve(pass, cycles = o.cycles) {
      if (o.removeMeanB) dispatch(pass, "mg_mean_b", 0);
      for (let c = 0; c < cycles; c++) vcycle(pass, 0, o.nu1, o.nu2);
      if (o.removeMeanX) dispatch(pass, "mg_mean_x", 0);
    },
    async readSolution() { return readBuffer(device, L0.x, Float32Array, 4 * cells0); },
    /** RMS of b - L x for the current x, b (recomputed on the GPU, summed in f64 on the host), unit of b. */
    async residualRMS() {
      await runPass(device, pass => dispatch(pass, "mg_residual", 0));
      const r = await readBuffer(device, L0.r, Float32Array, 4 * cells0);
      let s = 0; for (let i = 0; i < cells0; i++) s += r[i] * r[i];
      return Math.sqrt(s / cells0);
    },
    /**
     * Solve for b with up to `cycles` V-cycles (per-call overrides: cycles, rtol, x0).  Returns
     * { x: Float32Array, residualRMS: number[] (initial + after each cycle; b mean-removed if removeMeanB),
     *   cycles, converged, success, message }.  Reads the residual back after every cycle.
     */
    async solve(b, { cycles = o.cycles, rtol = o.rtol, x0 = null } = {}) {
      api.setRhs(b); api.setSolution(x0);
      if (o.removeMeanB) await runPass(device, pass => dispatch(pass, "mg_mean_b", 0));
      const hist = [await api.residualRMS()];
      let done = 0;
      while (done < cycles) {
        if (rtol > 0 && hist[hist.length - 1] <= rtol * hist[0]) break;
        await runPass(device, pass => vcycle(pass, 0, o.nu1, o.nu2));
        done++;
        hist.push(await api.residualRMS());
      }
      const converged = rtol > 0 && hist[hist.length - 1] <= rtol * hist[0];
      if (o.removeMeanX) await runPass(device, pass => dispatch(pass, "mg_mean_x", 0));
      const x = await api.readSolution();
      let finite = hist.every(Number.isFinite);
      for (let i = 0; finite && i < cells0; i++) finite = Number.isFinite(x[i]);
      const message = (finite ? "ok" : "non-finite values") + `; ${done} V-cycle(s), RMS residual ${hist[0].toExponential(3)} -> ${hist[hist.length - 1].toExponential(3)}`;
      return { x, residualRMS: hist, cycles: done, converged, success: finite, message };
    },
    destroy() { owned.forEach(b => b.destroy()); owned.length = 0; },
  };
  return api;
}
