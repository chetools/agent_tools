// Shared WebGPU host helpers for the routines in library/wgsl/ (area `gpu`).
// ES module, browser only.  Every routine's host wrapper (library/js/<id>.js) builds on these, so that
// compilation errors, validation errors and device limits fail loudly instead of silently producing zeros.

/** Request an adapter and device.  Returns { adapter, device }; keep the adapter referenced. */
export async function getDevice(requiredLimits = {}) {
  if (!globalThis.navigator || !navigator.gpu) throw new Error("WebGPU is not available in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const limits = {};
  for (const [k, v] of Object.entries(requiredLimits)) {
    if (!(k in adapter.limits)) throw new Error(`unknown limit ${k}`);
    if (adapter.limits[k] < v) throw new Error(`adapter limit ${k} = ${adapter.limits[k]} < required ${v}`);
    limits[k] = v;
  }
  const device = await adapter.requestDevice({ requiredLimits: limits });
  return { adapter, device };
}

/** Fetch WGSL source relative to this module (library/js/ -> library/wgsl/<name>.wgsl). */
export async function loadWGSL(name) {
  const url = new URL(`../wgsl/${name}.wgsl`, import.meta.url);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`cannot load ${url}: HTTP ${r.status}`);
  return r.text();
}

/** Compile a shader module and throw on any compilation error. */
export async function compile(device, code, label = "") {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === "error");
  if (errs.length) throw new Error(`WGSL ${label}: ` + errs.map(m => `line ${m.lineNum}: ${m.message}`).join("; "));
  return module;
}

/**
 * Compute pipelines with explicit bind-group layouts.
 * kernels: { entryPoint: [ [binding, "uniform" | "storage" | "read-only-storage"], ... ] }
 * Returns { pipes, layouts } keyed by entry point.
 */
export async function makePipelines(device, module, kernels) {
  const C = GPUShaderStage.COMPUTE, pipes = {}, layouts = {};
  device.pushErrorScope("validation");
  for (const [name, binds] of Object.entries(kernels)) {
    const layout = device.createBindGroupLayout({
      entries: binds.map(([binding, type]) => ({ binding, visibility: C, buffer: { type } })) });
    layouts[name] = layout;
    pipes[name] = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: name } });
  }
  const err = await device.popErrorScope();
  if (err) throw new Error("WebGPU pipeline: " + err.message);
  return { pipes, layouts };
}

/** Bind group for `layout` from { binding: GPUBuffer }. */
export function bindGroup(device, layout, buffers) {
  return device.createBindGroup({ layout, entries: Object.entries(buffers).map(
    ([b, buffer]) => ({ binding: Number(b), resource: { buffer } })) });
}

/** Storage buffer (also copy src/dst), optionally initialised from a typed array. Size padded to 16 bytes. */
export function storage(device, bytesOrData, extraUsage = 0) {
  const data = typeof bytesOrData === "number" ? null : bytesOrData;
  const bytes = data ? data.byteLength : bytesOrData;
  const size = Math.max(16, Math.ceil(bytes / 16) * 16);
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
    GPUBufferUsage.COPY_DST | extraUsage });
  if (data) device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

/** Uniform buffer from an ArrayBuffer (size padded to 16 bytes). */
export function uniform(device, arrayBuffer) {
  const size = Math.max(16, Math.ceil(arrayBuffer.byteLength / 16) * 16);
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, arrayBuffer);
  return buf;
}

/** Copy a GPU buffer back.  type: Float32Array | Uint32Array | Int32Array.  Throws on a lost device. */
export async function readBuffer(device, buf, type = Float32Array, bytes = buf.size) {
  const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new type(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy();
  return out;
}

/** Run `fn(pass)` inside one compute pass and submit it; surfaces validation errors. */
export async function runPass(device, fn) {
  device.pushErrorScope("validation");
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  fn(pass);
  pass.end();
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error("WebGPU dispatch: " + err.message);
}

/** Check that a storage buffer of `bytes` fits this device. */
export function checkBufferSize(device, bytes, what = "buffer") {
  const lim = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  if (bytes > lim) throw new Error(`${what} needs ${(bytes / 2 ** 20).toFixed(1)} MiB but this device allows ` +
    `${(lim / 2 ** 20).toFixed(1)} MiB`);
}
