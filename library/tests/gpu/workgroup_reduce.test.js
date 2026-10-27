// GPU test for workgroup_reduce: sum/max/min/maxabs against exact references (math.fsum, numpy) stored in
// tests/reference_data/gpu/workgroup_reduce.json.  Inputs are regenerated here by the same integer hash as
// python/gpu/workgroup_reduce.py (hash_values), all values exactly representable in f32.
// Stated tolerance: sum   |GPU - exact| <= 1e-6 * sum(|x|)   (measured <= 2e-8; rigorous worst-case f32 bound 3.5e-6 at n = 1e7)
//                   max, min, maxabs: exact equality.
import { storage } from "../../js/gpu_util.js";
import { createReducer } from "../../js/workgroup_reduce.js";

const SUM_RTOL = 1e-6;

function hashValues(n, seed, emin, emax, bias) {
  const x = new Float32Array(n), ne = emax - emin + 1;
  for (let i = 0; i < n; i++) {
    let h = (Math.imul(i, 2654435761) + seed) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x85EBCA6B) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 0xC2B2AE35) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    const m = (h & 0xFFFFFF) - 8388608 + bias, e = ((h >>> 24) % ne) + emin;
    x[i] = m * 2 ** (e - 10);
  }
  return x;
}

export async function run(device, ref) {
  const rows = [];
  for (const c of ref.cases) {
    const x = hashValues(c.n, c.seed, c.emin, c.emax, c.bias);
    const buf = storage(device, x);
    const red = await createReducer(device, c.maxGroupsX ? { maxGroupsX: c.maxGroupsX } : {});
    for (const op of ["sum", "max", "min", "maxabs"]) {
      const g = await red.reduce(buf, c.n, op);
      if (op === "sum") {
        const err = Math.abs(g - c.sum) / c.sum_abs;
        rows.push({ name: `${c.name}: sum (err relative to sum|x|; |sum|/sum|x| = ${(Math.abs(c.sum) / c.sum_abs).toExponential(1)})`,
          err, tol: SUM_RTOL, pass: err <= SUM_RTOL });
      } else {
        const err = Math.abs(g - c[op]);
        rows.push({ name: `${c.name}: ${op} (exact)`, err, tol: 0, pass: g === c[op] });
      }
    }
    red.destroy(); buf.destroy();
  }
  return rows;
}
