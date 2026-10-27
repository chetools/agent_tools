// particle_advect_rk2.js: host wrapper for library/wgsl/particle_advect_rk2.wgsl (RK2 particle advection through
// a cell-centred 3D velocity field on a uniform grid; trilinear interpolation; odd-mirror walls).
//
//   const adv = await createParticleAdvector(device, { nx, ny, nz, h, periodic: true, maxParticles });
//   adv.setVelocity(Float32Array(3*nz*ny*nx));      // [c][k][j][i], length/time
//   adv.setPositions(xyz);                          // Float32Array(3*np): x,y,z in length units, np <= maxParticles
//   await adv.advect({ dt: 0.1, nsub: 8 });         // or { dt, umax, maxFrac }: nsub = substepCount(...)
//   const { xyz, speed, flags } = await adv.read(); // speed only if advect({..., wantSpeed: true})
//
// Units: length unit L (h in L), velocity L/time, dt in the time unit.  Domain [0,nx h]x[0,ny h]x[0,nz h].
// Defaults (identical to library/python/gpu/particle_advect_rk2.py): periodic = true, maxFrac = 0.5,
// wantSpeed = false.  See the WGSL header for the interpolation convention and the clamp-with-flag policy.
import { loadWGSL, compile, makePipelines, bindGroup, storage, uniform, readBuffer, runPass, checkBufferSize }
  from "./gpu_util.js";

/**
 * Number of equal substeps for a time span dt when no particle may move more than maxFrac*h per substep at speed
 * umax (length/time): max(1, ceil(dt*umax/(maxFrac*h) - 1e-9)).  Same formula as the Python reference.
 */
export function substepCount(dt, umax, h, maxFrac = 0.5) {
  if (!(dt >= 0) || !(h > 0) || !(maxFrac > 0) || !(umax >= 0)) throw new Error("substepCount: invalid argument");
  return Math.max(1, Math.ceil(dt * umax / (maxFrac * h) - 1e-9));
}

export async function createParticleAdvector(device, opts) {
  const { nx, ny, nz, h, maxParticles } = opts;
  const periodic = opts.periodic ?? true;
  for (const [k, v] of Object.entries({ nx, ny, nz })) if (!Number.isInteger(v) || v < 1) throw new Error(`${k} must be a positive integer`);
  if (!(h > 0)) throw new Error("h must be > 0");
  if (!Number.isInteger(maxParticles) || maxParticles < 1) throw new Error("maxParticles must be a positive integer");
  const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
  if (Math.ceil(maxParticles / 64) > maxGroups)
    throw new Error(`maxParticles = ${maxParticles} needs ${Math.ceil(maxParticles / 64)} workgroups; device allows ${maxGroups} per dimension (max ${64 * maxGroups} particles)`);
  if (3 * nx * ny * nz >= 2 ** 32) throw new Error("velocity grid too large for u32 indexing");
  checkBufferSize(device, 3 * nx * ny * nz * 4, "velocity buffer");
  checkBufferSize(device, maxParticles * 16, "particle buffer");
  if (device.limits.maxStorageBuffersPerShaderStage < 3) throw new Error("device allows fewer than 3 storage buffers per stage");

  const module = await compile(device, await loadWGSL("particle_advect_rk2"), "particle_advect_rk2");
  const { pipes, layouts } = await makePipelines(device, module,
    { cs_advect: [[0, "uniform"], [1, "read-only-storage"], [2, "storage"], [3, "storage"]] });
  const vbuf = storage(device, 3 * nx * ny * nz * 4);
  const pbuf = storage(device, maxParticles * 16);
  const fbuf = storage(device, maxParticles * 4);
  let np = 0;

  return {
    nx, ny, nz, h, periodic, maxParticles,
    /** v: Float32Array(3*nz*ny*nx), layout [c][k][j][i], length/time. */
    setVelocity(v) {
      if (v.length !== 3 * nx * ny * nz) throw new Error(`velocity length ${v.length} != 3*nz*ny*nx = ${3 * nx * ny * nz}`);
      device.queue.writeBuffer(vbuf, 0, v.buffer, v.byteOffset, v.byteLength);
    },
    /** xyz: Float32Array(3*np), np <= maxParticles; physical coordinates in the length unit. */
    setPositions(xyz) {
      if (xyz.length % 3) throw new Error("positions length must be a multiple of 3");
      np = xyz.length / 3;
      if (np > maxParticles) throw new Error(`${np} particles > maxParticles ${maxParticles}`);
      const a = new Float32Array(4 * np);
      for (let p = 0; p < np; p++) { a[4 * p] = xyz[3 * p]; a[4 * p + 1] = xyz[3 * p + 1]; a[4 * p + 2] = xyz[3 * p + 2]; }
      device.queue.writeBuffer(pbuf, 0, a);
    },
    get count() { return np; },
    /** Advect over the time span dt (time unit) in `nsub` equal substeps, or nsub = substepCount(dt, umax, h, maxFrac). */
    async advect({ dt, nsub, umax, maxFrac = 0.5, wantSpeed = false }) {
      if (!(dt >= 0)) throw new Error("dt must be >= 0");
      if (nsub === undefined) {
        if (umax === undefined) throw new Error("give nsub, or umax (and optionally maxFrac)");
        nsub = dt === 0 ? 0 : substepCount(dt, umax, h, maxFrac);
      }
      if (!Number.isInteger(nsub) || nsub < 0) throw new Error("nsub must be a non-negative integer");
      const ab = new ArrayBuffer(48), dv = new DataView(ab);
      dv.setInt32(0, nx, true); dv.setInt32(4, ny, true); dv.setInt32(8, nz, true); dv.setUint32(12, np, true);
      dv.setFloat32(16, h, true); dv.setFloat32(20, nsub ? dt / nsub : 0, true); dv.setUint32(24, nsub, true);
      dv.setUint32(28, periodic ? 1 : 0, true); dv.setUint32(32, wantSpeed ? 1 : 0, true);
      const ub = uniform(device, ab);
      const bg = bindGroup(device, layouts.cs_advect, { 0: ub, 1: vbuf, 2: pbuf, 3: fbuf });
      await runPass(device, pass => {
        pass.setPipeline(pipes.cs_advect); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(np / 64));
      });
      ub.destroy();
      return nsub;
    },
    /** Returns { xyz: Float32Array(3*np), speed: Float32Array(np), flags: Uint32Array(np) } (see the WGSL flags). */
    async read() {
      const a = await readBuffer(device, pbuf, Float32Array, np * 16);
      const flags = await readBuffer(device, fbuf, Uint32Array, np * 4);
      const xyz = new Float32Array(3 * np), speed = new Float32Array(np);
      for (let p = 0; p < np; p++) { xyz[3 * p] = a[4 * p]; xyz[3 * p + 1] = a[4 * p + 1]; xyz[3 * p + 2] = a[4 * p + 2]; speed[p] = a[4 * p + 3]; }
      return { xyz, speed, flags };
    },
    destroy() { vbuf.destroy(); pbuf.destroy(); fbuf.destroy(); },
  };
}
