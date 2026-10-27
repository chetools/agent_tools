// GPU test for marching_cubes: float32 WGSL against the float64 Python reference (tests/reference_data/gpu/marching_cubes.json).
// Triangle order from the atomic counter is nondeterministic, so only order-independent quantities are compared:
// exact triangle count, total area, enclosed volume (closed surfaces), and the vertex sets (two-sided Hausdorff
// distance in units of h, plus normals and colour values at matched vertices).
import { createMarchingCubes, meshArea, meshVolume } from "../../js/marching_cubes.js";

const TOL = { area: 1e-6, volume: 1e-6, hausdorff: 1e-5, normal: 1e-5, cval: 1e-6 };   // measured on SwiftShader: area 1e-7, volume 1.1e-7, Hausdorff 2.3e-6 h, normal 1.0e-7, colour 1.5e-7

// nearest-vertex search through a hash grid of cell size `cs`; refs are rows of `stride` floats starting with x y z
function makeIndex(arr, stride, cs) {
  const m = new Map();
  const key = (i, j, k) => `${i},${j},${k}`;
  for (let v = 0; v * stride < arr.length; v++) {
    const o = v * stride, k = key(Math.floor(arr[o] / cs), Math.floor(arr[o + 1] / cs), Math.floor(arr[o + 2] / cs));
    (m.get(k) || m.set(k, []).get(k)).push(v);
  }
  return (x, y, z) => {
    const ci = Math.floor(x / cs), cj = Math.floor(y / cs), ck = Math.floor(z / cs);
    let best = Infinity, bi = -1;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++)
      for (const v of m.get(key(ci + a, cj + b, ck + c)) || []) {
        const o = v * stride, d = Math.hypot(arr[o] - x, arr[o + 1] - y, arr[o + 2] - z);
        if (d < best) { best = d; bi = v; }
      }
    return [best, bi];
  };
}

function compare(gpu, ref, cs) {
  const verts = ref.vertices;                                    // 7 floats: x y z nx ny nz cval
  const nearRef = makeIndex(verts, 7, cs), nearGpu = makeIndex(gpu, 8, cs);
  let hG = 0, hR = 0, nErr = 0, cErr = 0;
  for (let o = 0; o + 7 < gpu.length; o += 8) {
    const [d, r] = nearRef(gpu[o], gpu[o + 1], gpu[o + 2]);
    hG = Math.max(hG, d);
    if (r >= 0) {
      const q = r * 7;
      nErr = Math.max(nErr, Math.hypot(gpu[o + 4] - verts[q + 3], gpu[o + 5] - verts[q + 4], gpu[o + 6] - verts[q + 5]));
      cErr = Math.max(cErr, Math.abs(gpu[o + 3] - verts[q + 6]));
    }
  }
  for (let q = 0; q + 6 < verts.length; q += 7) hR = Math.max(hR, nearGpu(verts[q], verts[q + 1], verts[q + 2])[0]);
  return { hausdorff: Math.max(hG, hR) / ref.h, hG: hG / ref.h, normal: nErr, cval: cErr };
}

function resolve(ref) {                                        // fill in fields shared between cases ("same_as")
  const by = Object.fromEntries(ref.cases.map(c => [c.name, c]));
  for (const c of ref.cases) if (c.same_as) { c.field = by[c.same_as].field; c.color = by[c.same_as].color;
    if (!c.vertices) c.vertices = by[c.same_as].vertices; }
}

async function extract(device, c, extra = {}) {
  const mc = await createMarchingCubes(device, { nx: c.nx, ny: c.ny, nz: c.nz, h: c.h, origin: c.origin, iso: c.iso,
    periodicZ: c.periodic_z, hasColor: !!c.color, maxTris: c.max_tris ?? 100000, ...extra });
  mc.setField(new Float32Array(c.field));
  if (c.color) mc.setColor(new Float32Array(c.color));
  await mc.run();
  const mesh = await mc.read();
  return { mc, mesh };
}

const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);

export async function run(device, ref) {
  resolve(ref);
  const rows = [], row = (name, err, tol) => rows.push({ name, err, tol, pass: err <= tol });
  const exact = (name, got, want) => rows.push({ name, err: got === want ? 0 : Math.abs(got - want) || 1, tol: 0, pass: got === want });
  for (const c of ref.cases) {
    if (c.name === "overflow_cap") continue;
    const { mc, mesh } = await extract(device, c);
    exact(`${c.name}: triangle count (${c.count})`, mesh.triangles, c.count);
    exact(`${c.name}: overflow flag off`, mesh.overflow ? 1 : 0, 0);
    row(`${c.name}: area rel. error`, c.count ? rel(meshArea(mesh.vertices), c.area) : meshArea(mesh.vertices), TOL.area);
    if (c.closed) row(`${c.name}: volume rel. error (divergence theorem)`, rel(meshVolume(mesh.vertices), c.volume), TOL.volume);
    if (c.count) {
      const k = compare(mesh.vertices, c, c.h);
      row(`${c.name}: vertex-set Hausdorff / h`, k.hausdorff, TOL.hausdorff);
      if (!c.name.startsWith("random")) row(`${c.name}: normal max error at matched vertices`, k.normal, TOL.normal);
      if (c.color) row(`${c.name}: colour max error at matched vertices`, k.cval, TOL.cval);
    }
    mc.destroy();
  }

  // capacity overflow: own vertex buffer with a sentinel-filled guard region behind the cap
  const o = ref.cases.find(c => c.name === "overflow_cap"), cap = o.max_tris, guard = 64, SENT = 0xdeadbeef;
  const vbuf = device.createBuffer({ size: (cap + guard) * 96, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vbuf, 0, new Uint32Array((cap + guard) * 24).fill(SENT));
  const { mc, mesh } = await extract(device, o, { vertexBuffer: vbuf, maxTris: cap });
  exact(`overflow: reported triangle count (${o.count})`, mesh.triangles, o.count);
  exact("overflow: flag set", mesh.overflow ? 1 : 0, 1);
  exact(`overflow: stored triangles = cap (${cap})`, mesh.stored, cap);
  const all = await (async () => { const { readBuffer } = await import("../../js/gpu_util.js"); return readBuffer(device, vbuf, Uint32Array, vbuf.size); })();
  let touched = 0; for (let i = cap * 24; i < all.length; i++) if (all[i] !== SENT) touched++;
  exact("overflow: words written beyond the cap", touched, 0);
  let holes = 0; for (let i = 0; i < cap * 24; i++) if (all[i] === SENT) holes++;
  exact("overflow: unwritten (sentinel) words inside the cap", holes, 0);
  const args = await (await import("../../js/gpu_util.js")).readBuffer(device, mc.argsBuffer, Uint32Array, 16);
  exact("overflow: indirect vertexCount = 3*cap, instanceCount 1", args[0] === 3 * cap && args[1] === 1 ? 0 : 1, 0);
  const k = compare(mesh.vertices, o, o.h);
  row("overflow: stored vertices lie on the reference surface (Hausdorff GPU->ref / h)", k.hG, TOL.hausdorff);
  mc.destroy(); vbuf.destroy();
  return rows;
}
