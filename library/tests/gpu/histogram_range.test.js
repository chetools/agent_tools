// GPU test for histogram_range against the float64 reference (python/gpu/histogram_range.py), whose results are
// stored in tests/reference_data/gpu/histogram_range.json; the inputs are regenerated here with the same integer
// hash (all arithmetic is IEEE multiplication/division, so the f32 inputs are bit-identical to numpy's).
//
// Tolerances (basis: histogram bin width, see the WGSL header):
//   designed case (every element in the middle of a bin, no boundary-ambiguous element): histogram counts, count and
//     max EXACT (integers / exactly representable); hi equals the float64 hi to 1e-6 relative (f32 q*q*max);
//     |hi - exact percentile| <= bin width max (2 bin + 1) / nbins^2.
//   heavy-tailed case (random values, 4e6 elements, default nbins = 2048): the number of elements that moved bin
//     is <= the number of boundary-ambiguous elements (listed in the row name); |hi - exact percentile| <= width of
//     the bin one above the reference bin (one-bin slack for f32 percentile rank / boundary elements).
import { storage } from "../../js/gpu_util.js";
import { createRangeFinder } from "../../js/histogram_range.js";

function hash(i, seed) {
  let h = (Math.imul(i, 2654435761) + seed) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0; h = Math.imul(h, 0x85EBCA6B) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0; h = Math.imul(h, 0xC2B2AE35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}
function designedField(n, nb, seed = 7) {
  const x = new Float32Array(n), mask = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const h = hash(i, seed), c = h >>> 26, b = (h & 0xFFFF) % nb, sgn = ((h >>> 16) & 1) ? -1 : 1;
    let v = (sgn * ((2 * b + 1) * (2 * b + 1))) / (4 * nb * nb);
    if (c === 0) { v = 1e30; mask[i] = 1; } else if (c === 1) v = NaN; else if (c === 2) v = Infinity; else if (c === 3) v = -Infinity;
    x[i] = v;
  }
  x[12345] = 1.0; mask[12345] = 0;
  return { x, mask };
}
function heavyField(n, outlier = 1000, seed = 9) {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = hash(i, seed), u = ((h & 0xFFFFFF) + 0.5) / 16777216;
    let a = u * u; a = a * a;
    x[i] = ((h >>> 24) & 1 ? -1 : 1) * a;
  }
  x[777] = outlier;
  return x;
}
const binWidth = (mx, nb, bin) => mx * (2 * bin + 1) / (nb * nb);
function histDiff(a, b) {            // [number of mismatching bins, elements moved = sum|diff| / 2]
  let bins = 0, s = 0;
  for (let i = 0; i < b.length; i++) { const d = Math.abs(a[i] - b[i]); if (d) bins++; s += d; }
  return [bins, s / 2];
}

export async function run(device, ref) {
  const rows = [];
  // ---------------- designed case
  {
    const d = ref.designed, { x, mask } = designedField(d.n, d.nbins);
    const xb = storage(device, x), mb = storage(device, mask);
    const rf = await createRangeFinder(device, { nbins: d.nbins, maxGroupsX: d.maxGroupsX });
    const tag = `designed n=${d.n} nbins=${d.nbins} (${d.ambiguous} ambiguous)`;
    for (const r of d.results) {
      const sym = r.pct === 0.875;                       // exercise both range kinds
      const g = await rf.range(xb, d.n, { pct: r.pct, symmetric: sym, maskBuf: mb });
      const h = await rf.readHistogram();
      const [bins] = histDiff(h, r.hist);
      rows.push({ name: `${tag}, pct ${r.pct}: histogram bins differing`, err: bins, tol: 0, pass: bins === 0 });
      rows.push({ name: `${tag}, pct ${r.pct}: count and max exact`, err: Math.abs(g.count - r.count) + Math.abs(g.max - r.max),
        tol: 0, pass: g.count === r.count && g.max === r.max });
      const eh = Math.abs(g.hi - r.hi) / r.hi;
      rows.push({ name: `${tag}, pct ${r.pct}: hi vs float64 hi (relative)`, err: eh, tol: 1e-6, pass: eh <= 1e-6 });
      const w = binWidth(r.max, d.nbins, r.bin), ep = g.hi - r.p_exact;
      rows.push({ name: `${tag}, pct ${r.pct}: hi - exact percentile (0..bin width ${w.toExponential(2)})`,
        err: Math.abs(ep), tol: w * (1 + 1e-6), pass: ep >= -1e-6 * r.max && ep <= w * (1 + 1e-6) });
      const lo = sym ? -g.hi : 0;
      rows.push({ name: `${tag}, pct ${r.pct}: lo is ${sym ? "-hi (symmetric)" : "0 (unsigned)"}`, err: Math.abs(g.lo - lo), tol: 0, pass: g.lo === lo });
    }
    const gu = await rf.range(xb, d.n, { pct: 0.5 });               // no mask: huge masked values now count
    rows.push({ name: `${tag}: without mask, count and max (1e30) exact`,
      err: Math.abs(gu.count - d.unmasked_count) + Math.abs(gu.max - d.unmasked_max), tol: 0,
      pass: gu.count === d.unmasked_count && gu.max === d.unmasked_max });
    rf.destroy(); xb.destroy(); mb.destroy();
  }
  // ---------------- heavy-tailed case with a 1000x outlier
  {
    const d = ref.heavy, x = heavyField(d.n), xb = storage(device, x);
    const rf = await createRangeFinder(device, { nbins: d.nbins });
    const tag = `heavy-tail n=${d.n} nbins=${d.nbins} (${d.ambiguous} ambiguous)`;
    for (const r of d.results) {
      const g = await rf.range(xb, d.n, { pct: r.pct, symmetric: r.symmetric });
      const h = await rf.readHistogram();
      const [, moved] = histDiff(h, r.hist);
      rows.push({ name: `${tag}, pct ${r.pct}: elements moved between bins (<= ambiguous)`, err: moved, tol: d.ambiguous, pass: moved <= d.ambiguous });
      rows.push({ name: `${tag}, pct ${r.pct}: count and max exact`, err: Math.abs(g.count - r.count) + Math.abs(g.max - r.max),
        tol: 0, pass: g.count === r.count && g.max === r.max });
      const w = binWidth(r.max, d.nbins, r.bin + 1), ep = g.hi - r.p_exact;
      rows.push({ name: `${tag}, pct ${r.pct}: hi - exact percentile (<= width of bin ${r.bin + 1}: ${w.toExponential(2)})`,
        err: Math.abs(ep), tol: w, pass: ep >= -1e-6 * r.max && ep <= w });
      const lo = r.symmetric ? -g.hi : 0;
      rows.push({ name: `${tag}, pct ${r.pct}: lo`, err: Math.abs(g.lo - lo), tol: 0, pass: g.lo === lo });
    }
    rf.destroy(); xb.destroy();
  }
  // ---------------- degenerate inputs
  {
    const rf = await createRangeFinder(device, { nbins: 64 });
    const z = storage(device, new Float32Array(1000)), ones = storage(device, new Float32Array(1000).fill(1));
    const m1 = storage(device, new Uint32Array(1000).fill(1));
    for (const [nm, g, e] of [["all zeros, symmetric", await rf.range(z, 1000, { pct: 0.5, symmetric: true }), ref.degenerate.zeros],
                              ["all masked", await rf.range(ones, 1000, { pct: 0.5, maskBuf: m1 }), ref.degenerate.all_masked]]) {
      const err = Math.abs(g.lo - e.lo) + Math.abs(g.hi - e.hi) + Math.abs(g.max - e.max) + Math.abs(g.count - e.count);
      rows.push({ name: `degenerate ${nm}: lo, hi, max, count exact`, err, tol: 0, pass: err === 0 });
    }
    rf.destroy();
  }
  return rows;
}
