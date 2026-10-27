// Host wrapper for library/wgsl/histogram_range.wgsl: robust colour-scale range (max |x|, histogram, percentile).
// ES module, no DOM access.  See the WGSL header for buffer layouts, binning, error bound and limits.
//
//   import { createRangeFinder } from "./histogram_range.js";
//   const rf = await createRangeFinder(device, { nbins: 2048 });
//   const r = await rf.range(fieldBuf, n, { pct: 0.995, symmetric: true, maskBuf: bodyMask });
//   // r = { lo, hi, max, count }  in the units of the field values
//   const hist = await rf.readHistogram();     // Uint32Array(nbins) of the last call
//
// createRangeFinder(device, opts):
//   opts.nbins       histogram bins, integer >= 1 (default 2048, as in the source project)
//   opts.maxGroupsX  max workgroups per row of the 2D dispatch (default device.limits.maxComputeWorkgroupsPerDimension)
// range(fieldBuf, n, o):   fieldBuf = GPUBuffer (STORAGE) of n f32; o.pct (default 0.995, in (0, 1]);
//   o.symmetric (default false: range [0, hi]; true: [-hi, hi]); o.maskBuf = GPUBuffer (STORAGE) of n u32, entries
//   with a non-zero value are ignored (default none).  Returns Promise<{ lo, hi, max, count }>.
// rangeToBuffer(fieldBuf, n, o, outBuf): same without read-back; outBuf (STORAGE, >= 16 bytes) receives
//   f32 [lo, hi, max, count], ready to be bound by a colouring shader.
// Quantisation error of hi: 0 <= hi - exact percentile <= max * (2 bin + 1) / nbins^2 <= 2 max / nbins.
import { loadWGSL, compile, makePipelines, bindGroup, storage, uniform, readBuffer, runPass, checkBufferSize } from "./gpu_util.js";

export async function createRangeFinder(device, opts = {}) {
  const nbins = opts.nbins ?? 2048;
  if (!Number.isInteger(nbins) || nbins < 1) throw new Error(`nbins must be an integer >= 1, got ${nbins}`);
  const maxGroupsX = opts.maxGroupsX ?? device.limits.maxComputeWorkgroupsPerDimension;
  if (!(maxGroupsX >= 1) || maxGroupsX > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error(`maxGroupsX must be in [1, ${device.limits.maxComputeWorkgroupsPerDimension}]`);
  checkBufferSize(device, 4 * (nbins + 2), "histogram");
  const module = await compile(device, await loadWGSL("histogram_range"), "histogram_range");
  const { pipes, layouts } = await makePipelines(device, module, {
    hr_clear: [[0, "uniform"], [3, "storage"]],
    hr_max: [[0, "uniform"], [1, "read-only-storage"], [2, "read-only-storage"], [3, "storage"]],
    hr_hist: [[0, "uniform"], [1, "read-only-storage"], [2, "read-only-storage"], [3, "storage"]],
    hr_range: [[0, "uniform"], [3, "storage"], [4, "storage"]],
  });
  const st = storage(device, 4 * (nbins + 2));
  const dummyMask = storage(device, 16);
  let ownOut = null;

  async function encode(fieldBuf, n, o, outBuf) {
    if (!Number.isInteger(n) || n < 1 || n > 2 ** 32 - 1) throw new Error(`n must be an integer in [1, 2^32), got ${n}`);
    const pct = o.pct ?? 0.995;
    if (!(pct > 0 && pct <= 1)) throw new Error(`pct must be in (0, 1], got ${pct}`);
    checkBufferSize(device, 4 * n, "field");
    if (fieldBuf.size < 4 * n) throw new Error(`field buffer holds ${fieldBuf.size / 4} floats, n = ${n}`);
    if (o.maskBuf && o.maskBuf.size < 4 * n) throw new Error(`mask buffer holds ${o.maskBuf.size / 4} words, n = ${n}`);
    const groups = Math.ceil(n / 64), gx = Math.min(groups, maxGroupsX), gy = Math.ceil(groups / gx);
    if (gy > device.limits.maxComputeWorkgroupsPerDimension) throw new Error(`n = ${n} needs too many workgroups`);
    const ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = n; u[1] = gx; u[2] = o.maskBuf ? 1 : 0; u[3] = o.symmetric ? 1 : 0; u[4] = nbins; f[5] = pct;
    const ub = uniform(device, ab), mask = o.maskBuf || dummyMask;
    const bgS = bindGroup(device, layouts.hr_clear, { 0: ub, 3: st });
    const bgF = bindGroup(device, layouts.hr_max, { 0: ub, 1: fieldBuf, 2: mask, 3: st });
    const bgH = bindGroup(device, layouts.hr_hist, { 0: ub, 1: fieldBuf, 2: mask, 3: st });
    const bgR = bindGroup(device, layouts.hr_range, { 0: ub, 3: st, 4: outBuf });
    await runPass(device, pass => {
      pass.setPipeline(pipes.hr_clear); pass.setBindGroup(0, bgS); pass.dispatchWorkgroups(1);
      pass.setPipeline(pipes.hr_max); pass.setBindGroup(0, bgF); pass.dispatchWorkgroups(gx, gy);
      pass.setPipeline(pipes.hr_hist); pass.setBindGroup(0, bgH); pass.dispatchWorkgroups(gx, gy);
      pass.setPipeline(pipes.hr_range); pass.setBindGroup(0, bgR); pass.dispatchWorkgroups(1);
    });
    await device.queue.onSubmittedWorkDone();
    ub.destroy();
  }

  return {
    nbins,
    async rangeToBuffer(fieldBuf, n, o, outBuf) { await encode(fieldBuf, n, o || {}, outBuf); },
    async range(fieldBuf, n, o = {}) {
      if (!ownOut) ownOut = storage(device, 16);
      await encode(fieldBuf, n, o, ownOut);
      const [lo, hi, max, count] = await readBuffer(device, ownOut, Float32Array, 16);
      return { lo, hi, max, count };
    },
    /** Histogram (Uint32Array of nbins counts) left by the last call. */
    async readHistogram() { return (await readBuffer(device, st, Uint32Array, 4 * (nbins + 2))).slice(2, 2 + nbins); },
    destroy() { st.destroy(); dummyMask.destroy(); if (ownOut) ownOut.destroy(); ownOut = null; },
  };
}
