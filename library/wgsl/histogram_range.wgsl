// histogram_range.wgsl -- robust colour-scale range of an f32 field: max |x|, a histogram of |x|, and the
// percentile of |x| read from that histogram.  Optionally entries flagged by a u32 mask are ignored.
//
// Four kernels, run in this order in one compute pass (the host wrapper does this):
//   hr_clear   workgroup_size(64), dispatch ONE workgroup : zero ST[0 .. nbins+2)
//   hr_max     workgroup_size(64), dispatch ceil(n/64)    : ST[0] = bits of max |x| over counted entries (atomicMax)
//   hr_hist    workgroup_size(64), dispatch ceil(n/64)    : histogram of sqrt(|x| / max) in nbins equal bins
//   hr_range   workgroup_size(1),  dispatch ONE workgroup : percentile -> RNG[0..4)
//
// Buffer layout (group 0)
//   binding 0  uniform           HP { n, wgx, use_mask, symmetric, nbins, pct, _, _ }   (see struct)
//   binding 1  storage, read     FLD  : array<f32>, n elements (any index order; values in any unit)
//   binding 2  storage, read     MASK : array<u32>, n elements; entry i is IGNORED when MASK[i] != 0.  Only read
//                                       when use_mask = 1; otherwise bind any buffer of >= 4 bytes (it is not read).
//   binding 3  storage, rw       ST   : array<atomic<u32>>, >= nbins + 2 elements: [0] max-bits, [1] count,
//                                       [2 + b] histogram bin b.  The histogram stays valid after hr_range.
//   binding 4  storage, rw       RNG  : array<f32>, >= 4 elements: [lo, hi, max, count] (units of FLD; count as f32)
// Bindings used per entry point: hr_clear {0,3}, hr_max {0,1,2,3}, hr_hist {0,1,2,3}, hr_range {0,3,4}.
//
// Counted entries: not masked and finite (NaN and +-inf are skipped, detected by the exponent bits so that
// fast-math cannot remove the test).  Counted entries are reduced as |x|.
//
// Binning.  r = |x| / max in [0, 1];  bin = min(floor(sqrt(r) * nbins), nbins - 1).  Bin b therefore holds
// |x| in [max (b/nbins)^2, max ((b+1)/nbins)^2]: the square-root scale resolves the small values of a
// heavy-tailed field.  Percentile: want = max(ceil(pct * count), 1); the first bin whose cumulative count reaches
// want gives  hi = max ((bin+1)/nbins)^2  (its UPPER edge, floored at 1e-30).  Output range: lo = -hi if symmetric = 1
// else 0.  If count = 0 or max = 0 the range is hi = 1 (lo = -1 or 0).
//
// Quantisation error.  The exact percentile value p (numpy.quantile, method="inverted_cdf") lies in the same bin,
// so  0 <= hi - p <= max (2 bin + 1) / nbins^2  <= 2 max / nbins.  That bin width is the tolerance basis.  With
// nbins = 2048 the error is <= 0.1 % of max.  A single large outlier raises max and widens every bin of the bulk.
//
// f32 precision.  max is exact.  Bin assignment uses an f32 division and sqrt (a few ulp, about 2e-7 relative), so
// an element whose position sqrt(r) * nbins lies within about 1e-6 * position of an integer may land in a
// neighbouring bin compared with float64: counts equal the float64 reference exactly unless such
// boundary-ambiguous elements exist.  pct * count is formed in f32; it is
// exact for count < 2^24 and a pct with a few significant bits (0.5, 0.875, ...), otherwise want may differ by 1.
// Limit: count is an atomic u32 (fine up to n < 2^32).
//
// Extracted from chetools/claude cylinder-cavity-3d/web/render.wgsl (cs_clear, cs_max, cs_hist, cs_range, counted,
// used_range); the cylinder "in_body" exclusion is replaced by the optional MASK buffer.

struct HP {
  n: u32,          // number of elements
  wgx: u32,        // workgroups per row of the 2D dispatch of hr_max / hr_hist
  use_mask: u32,   // 1: ignore entries with MASK[i] != 0
  symmetric: u32,  // 1: range [-hi, hi]; 0: range [0, hi]
  nbins: u32,      // number of histogram bins (>= 1)
  pct: f32,        // percentile in (0, 1], fraction of counted entries
  _p0: u32,
  _p1: u32,
};
@group(0) @binding(0) var<uniform> HU: HP;
@group(0) @binding(1) var<storage, read> FLD: array<f32>;
@group(0) @binding(2) var<storage, read> MASK: array<u32>;
@group(0) @binding(3) var<storage, read_write> ST: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> RNG: array<f32>;

fn cell_index(wid: vec3<u32>, li: u32) -> u32 { return (wid.y * HU.wgx + wid.x) * 64u + li; }

fn counted(idx: u32) -> bool {
  if (HU.use_mask != 0u && MASK[idx] != 0u) { return false; }
  return (bitcast<u32>(FLD[idx]) & 0x7f800000u) != 0x7f800000u;       // finite
}

@compute @workgroup_size(64)
fn hr_clear(@builtin(local_invocation_index) li: u32) {
  let nb = HU.nbins + 2u;
  for (var b = li; b < nb; b += 64u) { atomicStore(&ST[b], 0u); }
}

@compute @workgroup_size(64)
fn hr_max(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = cell_index(wid, li);
  if (idx >= HU.n) { return; }
  if (!counted(idx)) { return; }
  atomicMax(&ST[0], bitcast<u32>(abs(FLD[idx])));       // non-negative finite floats order like their bits
}

@compute @workgroup_size(64)
fn hr_hist(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = cell_index(wid, li);
  if (idx >= HU.n) { return; }
  if (!counted(idx)) { return; }
  let mx = bitcast<f32>(atomicLoad(&ST[0]));
  let r = select(0.0, abs(FLD[idx]) / mx, mx > 0.0);
  let b = min(u32(sqrt(r) * f32(HU.nbins)), HU.nbins - 1u);
  atomicAdd(&ST[2u + b], 1u);
  atomicAdd(&ST[1], 1u);
}

@compute @workgroup_size(1)
fn hr_range() {
  let mx = bitcast<f32>(atomicLoad(&ST[0]));
  let cnt = atomicLoad(&ST[1]);
  var hi = 1.0;
  if (cnt > 0u && mx > 0.0) {
    let want = max(u32(ceil(HU.pct * f32(cnt))), 1u);
    var acc = 0u; var bin = HU.nbins - 1u;
    for (var b = 0u; b < HU.nbins; b++) {
      acc += atomicLoad(&ST[2u + b]);
      if (acc >= want) { bin = b; break; }
    }
    let q = f32(bin + 1u) / f32(HU.nbins);
    hi = max(mx * q * q, 1.0e-30);
  }
  RNG[0] = select(0.0, -hi, HU.symmetric != 0u);
  RNG[1] = hi;
  RNG[2] = mx;
  RNG[3] = f32(cnt);
}
