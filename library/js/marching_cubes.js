// marching_cubes.js: WebGPU host wrapper for library/wgsl/marching_cubes.wgsl (isosurface extraction, compute only).
// ES module, no DOM access.  Python float64 reference: library/python/gpu/marching_cubes.py (same defaults).
//
//   import { createMarchingCubes } from "./marching_cubes.js";
//   const mc = await createMarchingCubes(device, { nx, ny, nz, h: 0.1, origin: [0, 0, 0], iso: 0.5 });
//   mc.setField(float32Array);                  // length nx*ny*nz, index (k*ny + j)*nx + i, x fastest
//   await mc.run();                             // fills mc.vertexBuffer, mc.counterBuffer, mc.argsBuffer
//   const mesh = await mc.read();               // { triangles, stored, overflow, vertices: Float32Array }
//
// Units: h and origin in any consistent length unit (output positions use it); iso in field units.
// Options (defaults identical to the Python reference where one exists):
//   nx, ny, nz   grid nodes per axis (>= 2, may differ; spacing h is the same in all three directions)
//   h = 1        node spacing;  origin = [0, 0, 0] position of node (0,0,0);  iso = 0 isovalue (f < iso is inside)
//   periodicZ = false   treat the field as periodic in z with period nz*h (adds the wrap cells)
//   hasColor = false    also interpolate a second field (setColor) to the vertices
//   maxTris = 500000    triangle capacity.  Python: unbounded (max_tris = None).  The default is clamped to the
//                       device's storage-buffer limit; an explicit value that does not fit throws.
//   vertexBuffer = null supply your own vertex buffer (usage STORAGE, >= maxTris*96 bytes; it may be larger, the
//                       kernel never writes beyond maxTris triangles).  Otherwise one is created.
// Vertex record (32 bytes = 8 floats): x y z cval nx ny nz pad; three vertices per triangle.
// Overflow: when the surface needs more than maxTris triangles, the counter still reports the full count,
// overflow is set, and only maxTris triangles (an arbitrary subset) are stored.
import { loadWGSL, compile, makePipelines, bindGroup, storage, uniform, readBuffer, runPass, checkBufferSize }
  from "./gpu_util.js";

const VERT_BYTES = 32, TRI_BYTES = 3 * VERT_BYTES;

export async function createMarchingCubes(device, opts = {}) {
  const o = { h: 1, origin: [0, 0, 0], iso: 0, periodicZ: false, hasColor: false, maxTris: undefined,
    vertexBuffer: null, ...opts };
  const { nx, ny, nz } = o;
  for (const [k, v] of [["nx", nx], ["ny", ny], ["nz", nz]])
    if (!Number.isInteger(v) || v < 2) throw new Error(`marching_cubes: ${k} must be an integer >= 2 (got ${v})`);
  if (!(o.h > 0)) throw new Error("marching_cubes: h must be > 0");
  if (!Array.isArray(o.origin) || o.origin.length !== 3) throw new Error("marching_cubes: origin must be [ox, oy, oz]");
  const nodes = nx * ny * nz;
  const cells = (nx - 1) * (ny - 1) * (o.periodicZ ? nz : nz - 1);
  if (cells >= 2 ** 31) throw new Error("marching_cubes: too many cells for 32-bit cell indices");
  checkBufferSize(device, nodes * 4, "scalar field");
  const lim = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const capLimit = Math.floor(lim / TRI_BYTES);
  let maxTris = o.maxTris;
  if (maxTris === undefined) maxTris = Math.min(500000, capLimit);
  if (!Number.isInteger(maxTris) || maxTris < 1) throw new Error("marching_cubes: maxTris must be an integer >= 1");
  if (maxTris > capLimit) throw new Error(`marching_cubes: maxTris ${maxTris} needs ${(maxTris * TRI_BYTES / 2 ** 20).toFixed(1)}` +
    ` MiB but this device allows ${(lim / 2 ** 20).toFixed(1)} MiB per storage buffer`);
  const groups = Math.ceil(cells / 64), gx = Math.min(groups, 65535), gy = Math.ceil(groups / 65535);
  if (gy > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error(`marching_cubes: ${cells} cells need ${gy} workgroup rows, device allows ${device.limits.maxComputeWorkgroupsPerDimension}`);

  const code = await loadWGSL("marching_cubes");
  const module = await compile(device, code, "marching_cubes.wgsl");
  const S = "storage", R = "read-only-storage";
  const layoutSpec = [[0, "uniform"], [1, R], [2, R], [3, S], [4, S], [5, S]];
  const { pipes, layouts } = await makePipelines(device, module, { cs_mc: layoutSpec, cs_args: layoutSpec });

  const field = storage(device, nodes * 4);
  const color = o.hasColor ? storage(device, nodes * 4) : field;      // unused (has_c = 0) when absent
  let vbuf = o.vertexBuffer;
  if (vbuf) {
    if (vbuf.size < maxTris * TRI_BYTES) throw new Error(`marching_cubes: vertexBuffer is ${vbuf.size} bytes, needs ${maxTris * TRI_BYTES}`);
  } else vbuf = storage(device, maxTris * TRI_BYTES);
  const counter = storage(device, 8);                                   // [triangle count, overflow flag]
  const args = storage(device, 16, GPUBufferUsage.INDIRECT);            // drawIndirect args
  const ubuf = uniform(device, new ArrayBuffer(48));
  const bg = bindGroup(device, layouts.cs_mc, { 0: ubuf, 1: field, 2: color, 3: vbuf, 4: counter, 5: args });

  const api = { nx, ny, nz, cells, maxTris, fieldBuffer: field, colorBuffer: o.hasColor ? color : null, vertexBuffer: vbuf,
    counterBuffer: counter, argsBuffer: args, params: o };

  function writeParams() {
    const ab = new ArrayBuffer(48), i = new Int32Array(ab), u = new Uint32Array(ab), f = new Float32Array(ab);
    i[0] = nx; i[1] = ny; i[2] = nz; u[3] = maxTris;
    f[4] = o.h; f[5] = o.iso; u[6] = o.periodicZ ? 1 : 0; u[7] = o.hasColor ? 1 : 0;
    f[8] = o.origin[0]; f[9] = o.origin[1]; f[10] = o.origin[2];
    device.queue.writeBuffer(ubuf, 0, ab);
  }
  function upload(buf, data, what) {
    if (!(data instanceof Float32Array) || data.length !== nodes)
      throw new Error(`marching_cubes: ${what} must be a Float32Array of length nx*ny*nz = ${nodes}`);
    device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  /** Upload the scalar field (Float32Array, nx*ny*nz, index (k*ny + j)*nx + i). */
  api.setField = (data) => upload(field, data, "field");
  /** Upload the colour field (needs hasColor: true). */
  api.setColor = (data) => { if (!o.hasColor) throw new Error("marching_cubes: created without hasColor"); upload(color, data, "color"); };
  /** Change the isovalue (field units) and/or origin ([x, y, z] length units) between runs. */
  api.setParams = ({ iso, origin, h } = {}) => {
    if (iso !== undefined) o.iso = iso;
    if (origin !== undefined) o.origin = origin;
    if (h !== undefined) { if (!(h > 0)) throw new Error("marching_cubes: h must be > 0"); o.h = h; }
  };
  /** Extract the surface: zero the counter, run cs_mc then cs_args.  Resolves when submitted and validated. */
  api.run = async () => {
    writeParams();
    device.queue.writeBuffer(counter, 0, new Uint32Array([0, 0]));
    await runPass(device, (pass) => {
      pass.setBindGroup(0, bg);
      pass.setPipeline(pipes.cs_mc); pass.dispatchWorkgroups(gx, gy, 1);
      pass.setPipeline(pipes.cs_args); pass.dispatchWorkgroups(1, 1, 1);
    });
  };
  /** Read back { triangles (needed), stored, overflow, vertices } (vertices: Float32Array, 8 floats per vertex). */
  api.read = async () => {
    const c = await readBuffer(device, counter, Uint32Array, 8);
    const stored = Math.min(c[0], maxTris);
    const vertices = stored ? await readBuffer(device, vbuf, Float32Array, stored * TRI_BYTES) : new Float32Array(0);
    return { triangles: c[0], stored, overflow: c[1] !== 0, vertices };
  };
  api.destroy = () => {
    field.destroy(); if (o.hasColor) color.destroy(); if (!opts.vertexBuffer) vbuf.destroy();
    counter.destroy(); args.destroy(); ubuf.destroy();
  };
  return api;
}

/** Total area (length unit squared) of a vertex array from read(): float64 accumulation. */
export function meshArea(v) {
  let a = 0;
  for (let t = 0; t + 23 < v.length; t += 24) {
    const [p0, p1, p2] = [t, t + 8, t + 16];
    const ux = v[p1] - v[p0], uy = v[p1 + 1] - v[p0 + 1], uz = v[p1 + 2] - v[p0 + 2];
    const wx = v[p2] - v[p0], wy = v[p2 + 1] - v[p0 + 1], wz = v[p2 + 2] - v[p0 + 2];
    a += 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
  }
  return a;
}

/** Volume enclosed by a closed mesh (divergence theorem), absolute value (length unit cubed). */
export function meshVolume(v) {
  let s = 0;
  for (let t = 0; t + 23 < v.length; t += 24) {
    const [p0, p1, p2] = [t, t + 8, t + 16];
    s += v[p0] * (v[p1 + 1] * v[p2 + 2] - v[p1 + 2] * v[p2 + 1]) + v[p0 + 1] * (v[p1 + 2] * v[p2] - v[p1] * v[p2 + 2])
      + v[p0 + 2] * (v[p1] * v[p2 + 1] - v[p1 + 1] * v[p2]);
  }
  return Math.abs(s / 6);
}
