// Host wrapper for library/wgsl/workgroup_reduce.wgsl: sum / max / min / max-abs of an f32 storage buffer.
// ES module, no DOM access.  See the WGSL header for the buffer layout, limits and precision.
//
//   import { createReducer } from "./workgroup_reduce.js";
//   const red = await createReducer(device);                   // options below
//   const buf = storage(device, new Float32Array([...]));      // any GPUBuffer with STORAGE usage, f32 elements
//   const total = await red.reduce(buf, n, "sum");             // number (the f32 result as a JS double)
//   red.destroy();
//
// createReducer(device, opts):
//   opts.maxGroupsX  maximum workgroups per row of the 2D dispatch (default device.limits.maxComputeWorkgroupsPerDimension,
//                    65535 on all current devices).  Lower it only to exercise the 2D path with small arrays.
// Methods:
//   reduce(xBuf, n, op = "sum")       -> Promise<number>   op: "sum" | "max" | "min" | "maxabs"; n >= 1 elements, f32.
//   reduceToBuffer(xBuf, n, op, outBuf)  encodes + submits both passes, result in outBuf[0] (f32, STORAGE usage);
//                                        no read-back, so the result can feed another kernel.
//   destroy()
// Units: results carry the units of the input values.
import { loadWGSL, compile, makePipelines, bindGroup, storage, uniform, readBuffer, runPass, checkBufferSize } from "./gpu_util.js";

export const OPS = { sum: 0, max: 1, min: 2, maxabs: 3 };
const BLOCK = 1024;      // elements per workgroup of reduce_partial (256 threads x 4)

export async function createReducer(device, opts = {}) {
  const maxGroupsX = opts.maxGroupsX ?? device.limits.maxComputeWorkgroupsPerDimension;
  if (!(maxGroupsX >= 1) || maxGroupsX > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error(`maxGroupsX must be in [1, ${device.limits.maxComputeWorkgroupsPerDimension}]`);
  const module = await compile(device, await loadWGSL("workgroup_reduce"), "workgroup_reduce");
  const { pipes, layouts } = await makePipelines(device, module, {
    reduce_partial: [[0, "uniform"], [1, "read-only-storage"], [2, "storage"]],
    reduce_final: [[0, "uniform"], [2, "storage"], [3, "storage"]],
  });
  let part = null, partCap = 0, ownOut = null;

  function plan(n) {
    if (!Number.isInteger(n) || n < 1) throw new Error(`n must be an integer >= 1, got ${n}`);
    if (n > 2 ** 32 - 1) throw new Error("n exceeds the u32 index range");
    checkBufferSize(device, 4 * n, "input");
    const nparts = Math.ceil(n / BLOCK), wgx = Math.min(nparts, maxGroupsX), wgy = Math.ceil(nparts / wgx);
    if (wgy > device.limits.maxComputeWorkgroupsPerDimension) throw new Error(`n = ${n} needs too many workgroups`);
    return { nparts, wgx, wgy };
  }

  async function encode(xBuf, n, op, outBuf) {
    const code = OPS[op];
    if (code === undefined) throw new Error(`unknown op "${op}" (use sum, max, min, maxabs)`);
    if (xBuf.size < 4 * n) throw new Error(`input buffer holds ${xBuf.size / 4} floats, n = ${n}`);
    const { nparts, wgx, wgy } = plan(n);
    if (nparts > partCap) { if (part) part.destroy(); part = storage(device, 4 * nparts); partCap = nparts; }
    const u = uniform(device, new Uint32Array([n, code, wgx, nparts]).buffer);
    const bg1 = bindGroup(device, layouts.reduce_partial, { 0: u, 1: xBuf, 2: part });
    const bg2 = bindGroup(device, layouts.reduce_final, { 0: u, 2: part, 3: outBuf });
    await runPass(device, pass => {
      pass.setPipeline(pipes.reduce_partial); pass.setBindGroup(0, bg1); pass.dispatchWorkgroups(wgx, wgy);
      pass.setPipeline(pipes.reduce_final); pass.setBindGroup(0, bg2); pass.dispatchWorkgroups(1);
    });
    await device.queue.onSubmittedWorkDone();
    u.destroy();
  }

  return {
    async reduceToBuffer(xBuf, n, op, outBuf) { await encode(xBuf, n, op, outBuf); },
    async reduce(xBuf, n, op = "sum") {
      if (!ownOut) ownOut = storage(device, 16);
      await encode(xBuf, n, op, ownOut);
      return (await readBuffer(device, ownOut, Float32Array, 16))[0];
    },
    destroy() { if (part) part.destroy(); if (ownOut) ownOut.destroy(); part = ownOut = null; partCap = 0; },
  };
}
