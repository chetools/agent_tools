// workgroup_reduce.wgsl -- sum / max / min / max-abs reduction of an f32 array of arbitrary length to one scalar.
//
// Two passes (one compute pass, two dispatches):
//   reduce_partial  workgroup_size(256): every workgroup reduces a block of 1024 consecutive elements (each
//                   thread handles 4 elements, 256 apart, so loads are coalesced) with a shared-memory tree and
//                   writes ONE partial to PART[workgroup_linear_id].
//   reduce_final    workgroup_size(256), dispatched as ONE workgroup: threads stride over all partials, then a tree
//                   reduction writes OUT[0].
//
// Buffer layout (group 0)
//   binding 0  uniform           RP { n, op, wgx, nparts }   (see struct)
//   binding 1  storage, read     X    : array<f32>, n elements, any index order (the reduction is order-free)
//   binding 2  storage, rw       PART : array<f32>, >= nparts elements (scratch)
//   binding 3  storage, rw       OUT  : array<f32>, >= 1 element; OUT[0] = result
// Bindings used by each entry point: reduce_partial {0,1,2}, reduce_final {0,2,3}.
//
// Lengths: any n >= 1 (a final partial block is padded with the operation's identity).  The partial pass is
// dispatched as a 2D grid so that more than 65535 workgroups (n > 67 107 840) work: the host picks
// nparts = ceil(n/1024), wgx = min(nparts, maxComputeWorkgroupsPerDimension), wgy = ceil(nparts/wgx), and the
// kernel linearises workgroup_id as wid.y * wgx + wid.x.  Surplus workgroups of the last row do nothing.
// reduce_final handles any nparts (strided loop).
//
// op: 0 = sum, 1 = max, 2 = min, 3 = max |x|.
// Limits: WGSL has no infinity literal, so the max/min identity is +-3.4028234e38 (the largest finite f32); an
// array holding only values equal to that, or +-inf, is not supported.  NaN inputs give undefined results for
// max/min and propagate for sum.  The ordering of the max/min results is exact; only the sum is rounded.
//
// f32 precision: the sum is accumulated in f32 in a fixed tree order that differs from numpy's pairwise order and
// from a sequential loop.  Compare sums with a tolerance RELATIVE TO sum(|x|), not to |sum(x)| (cancellation makes
// the latter arbitrarily small).  Every element passes through at most 4 + 8 + ceil(nparts/256) + 8 f32 additions,
// so |error| <= that count * 2^-24 * sum|x| (about 3.5e-6 * sum|x| for n = 1e7); typical errors are far smaller.
//
// Extracted from chetools/claude cylinder-cavity-3d/web/lbm.wgsl (wg_reduce4, reduce) and web/fv.wgsl
// (stats, stats_final), generalised to one channel, four operations, arbitrary n and 2D dispatch.

struct RP {
  n: u32,        // number of elements in X (>= 1)
  op: u32,       // 0 sum, 1 max, 2 min, 3 max |x|
  wgx: u32,      // workgroups per row of the 2D dispatch of reduce_partial
  nparts: u32,   // number of partials = ceil(n / 1024)
};
@group(0) @binding(0) var<uniform> RU: RP;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;

const WG: u32 = 256u;
const PER_THREAD: u32 = 4u;
const F32_MAX: f32 = 3.4028234e38;

var<workgroup> sh: array<f32, 256>;

fn identity(op: u32) -> f32 {
  if (op == 1u) { return -F32_MAX; }
  if (op == 2u) { return F32_MAX; }
  return 0.0;                                   // sum and max |x|
}

// load transform: |x| for op 3, x otherwise
fn prep(v: f32, op: u32) -> f32 { return select(v, abs(v), op == 3u); }

fn combine(a: f32, b: f32, op: u32) -> f32 {
  if (op == 0u) { return a + b; }
  if (op == 2u) { return min(a, b); }
  return max(a, b);                             // max and max |x| (values already non-negative for op 3)
}

// tree reduction of sh[0..256) -> sh[0]
fn wg_tree(lid: u32, op: u32) {
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    workgroupBarrier();
    if (lid < s) { sh[lid] = combine(sh[lid], sh[lid + s], op); }
  }
  workgroupBarrier();
}

@compute @workgroup_size(256)
fn reduce_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let op = RU.op;
  let w = wid.y * RU.wgx + wid.x;
  let base = w * (WG * PER_THREAD);
  var acc = identity(op);
  for (var k = 0u; k < PER_THREAD; k++) {
    let i = base + k * WG + lid;
    if (w < RU.nparts && i < RU.n) { acc = combine(acc, prep(X[i], op), op); }
  }
  sh[lid] = acc;
  wg_tree(lid, op);
  if (lid == 0u && w < RU.nparts) { PART[w] = sh[0]; }
}

@compute @workgroup_size(256)
fn reduce_final(@builtin(local_invocation_index) lid: u32) {
  let op = RU.op;
  var acc = identity(op);
  for (var p = lid; p < RU.nparts; p += WG) { acc = combine(acc, PART[p], op); }
  sh[lid] = acc;
  wg_tree(lid, op);
  if (lid == 0u) { OUT[0] = sh[0]; }
}
