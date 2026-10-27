// multigrid_poisson_cc.wgsl -- cell-centred geometric multigrid V-cycle for the 3D Poisson problem
//
//     laplacian(x) = b       on a box of N x N x NZ cells of edge h,   (x, y: zero-flux walls;  z: periodic OR zero-flux walls)
//
// Discretisation (7-point, cell-centred, second order):  (sum of existing neighbours - count * x) / h^2 = b.
// "Existing neighbours": a missing neighbour across a zero-flux wall is skipped (and not counted); in periodic z
// the cell above the last layer is layer 0.  With nz = 1 there is no z coupling at all.
// With zero-flux walls in x, y AND z, or periodic z, the operator is singular (constants are its null space):
// b must have zero mean, and x is determined up to a constant.  The host wrapper removes the mean of b and of x
// (kernels mg_mean_b, mg_mean_x) so that the problem is well posed.
//
// Buffer layout (all f32, one value per cell, index = (k * n + j) * n + i, i fastest, i = x, j = y, k = z):
//   binding 0  uniform  Level   (struct below, 32 bytes, i32 and f32 fields)
//   binding 1  storage  xf   solution / iterate on this level, size n*n*nz           (read_write)
//   binding 2  storage  bf   right-hand side on this level, size n*n*nz              (read_write)
//   binding 3  storage  rf   residual  b - L x on this level, size n*n*nz            (read_write)
//   binding 4  storage  xc   next-coarser level's x, size (n/2)^2 * nzc               (read_write)
//   binding 5  storage  bc   next-coarser level's b, size (n/2)^2 * nzc               (read_write)
// Units: if x carries the unit [X] and h is a length [m] (or any consistent length unit), then b has [X]/[m]^2.
// The uniform field h2 is the squared cell edge of THIS level (h * 2^level)^2 in [m^2].
//
// Level struct fields: n (cells per side in x and y), nz (layers), nzc (layers of the coarser level),
//   colour (0 | 1, red-black colour processed by mg_smooth), h2, sweeps (red-black sweeps of mg_coarse),
//   periodic (1: periodic z, 0: zero-flux z walls), reserved (0).
//
// Entry points and workgroup sizes (dispatch sizes in brackets):
//   mg_smooth    @workgroup_size(16,16)   one red-black colour; threads over (i, j) columns   [ceil(n/16), ceil(n/16)]
//   mg_residual  @workgroup_size(8,8,4)   rf = bf - L xf over fine cells                       [ceil(n/8), ceil(n/8), ceil(nz/4)]
//   mg_restrict  @workgroup_size(8,8,4)   bc = restricted rf, xc = 0, over coarse cells        [ceil(nc/8), ceil(nc/8), ceil(nzc/4)]
//   mg_prolong   @workgroup_size(8,8,4)   xf += trilinear prolongation of xc, over fine cells  [ceil(n/8), ceil(n/8), ceil(nz/4)]
//   mg_coarse    @workgroup_size(256)     coarsest level n = 4: `sweeps` red-black sweeps, then mean(xf) removed   [1]
//   mg_mean_b / mg_mean_x  @workgroup_size(256)  subtract the mean of bf / xf (all n*n*nz cells)               [1]
//
// Limits: N must be a power of two >= 4 (coarsest level n = 4); NZ must be a power of two >= 1 (z is
// halved together with x and y while it has >= 2 layers, then stays at 1: semi-coarsening).  The coarsest level
// has 16 columns, one per thread of the single mg_coarse workgroup.  Validation is done in the host wrapper.
//
// Red-black colouring is by the parity of i+j IN EVERY LAYER (not i+j+k) and a whole z-column is updated by
// one thread from the OLD values (Jacobi in z).  This keeps a z-independent problem z-independent to round-off;
// an (i+j+k) colouring breaks that (it treats neighbouring layers differently).
//
// Precision: f32.  The smoother, residual and transfer operators are local; errors accumulate over the cycles.
// Measured on SwiftShader (N up to 64, NZ up to 32): solution within 2e-6 of max|x| of the f64 reference after
// 6-8 cycles, residual history within 1.1e-4 relative; a residual floor of about
// 1e-6 times |b| (f32 cancellation in b - L x); see tests/gpu/multigrid_poisson_cc.test.js.
//
// Extracted from chetools/claude cylinder-cavity-3d/web/fv.wgsl (multigrid section: nsum, smooth_column, mg_smooth,
// mg_residual, mg_restrict, plane, mg_prolong, mg_coarse), generalised: own uniform struct and bind group, z boundary
// option, mean-removal kernels.

struct Level { n: i32, nz: i32, nzc: i32, colour: i32, h2: f32, sweeps: i32, periodic: i32, reserved: i32 };
@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read_write> xf: array<f32>;
@group(0) @binding(2) var<storage, read_write> bf: array<f32>;
@group(0) @binding(3) var<storage, read_write> rf: array<f32>;
@group(0) @binding(4) var<storage, read_write> xc: array<f32>;
@group(0) @binding(5) var<storage, read_write> bc: array<f32>;

// (neighbour sum, neighbour count) of cell (i, j, k): same order of additions as the Python neighbour_sum
fn nsum(i: i32, j: i32, k: i32) -> vec2<f32> {
  let n = L.n; let nz = L.nz; let id = (k * n + j) * n + i; let nn = n * n;
  var s = 0.0; var c = 0.0;
  if (j > 0) { s += xf[id - n]; c += 1.0; }
  if (j < n - 1) { s += xf[id + n]; c += 1.0; }
  if (i > 0) { s += xf[id - 1]; c += 1.0; }
  if (i < n - 1) { s += xf[id + 1]; c += 1.0; }
  if (nz >= 2) {
    if (L.periodic == 1) {
      s += xf[((k + nz - 1) % nz) * nn + j * n + i] + xf[((k + 1) % nz) * nn + j * n + i]; c += 2.0;
    } else {
      if (k > 0) { s += xf[id - nn]; c += 1.0; }
      if (k < nz - 1) { s += xf[id + nn]; c += 1.0; }
    }
  }
  return vec2<f32>(s, c);
}

// One red-black colour for the whole (i, j) column, with z-independent colouring (colour = parity of
// i+j in every layer).  A cell and its z-neighbours share a colour, so they must all be updated from
// the OLD values (Jacobi in z, as numpy does): one thread owns the column and remembers the old
// value of the layer below (and of layer 0 for the periodic wrap).
fn smooth_column(i: i32, j: i32) {
  let n = L.n; let nz = L.nz; let nn = n * n; let col = j * n + i;
  let first_old = xf[col];
  var prev_old = 0.0;
  for (var k = 0; k < nz; k++) {
    let id = k * nn + col;
    let xo = xf[id];
    var s = 0.0; var c = 0.0;
    if (j > 0) { s += xf[id - n]; c += 1.0; }
    if (j < n - 1) { s += xf[id + n]; c += 1.0; }
    if (i > 0) { s += xf[id - 1]; c += 1.0; }
    if (i < n - 1) { s += xf[id + 1]; c += 1.0; }
    if (nz >= 2) {
      if (L.periodic == 1) {
        var zm = prev_old; if (k == 0) { zm = xf[(nz - 1) * nn + col]; }
        var zp = first_old; if (k < nz - 1) { zp = xf[id + nn]; }
        s += zm + zp; c += 2.0;
      } else {
        if (k > 0) { s += prev_old; c += 1.0; }
        if (k < nz - 1) { s += xf[id + nn]; c += 1.0; }
      }
    }
    xf[id] = (s - L.h2 * bf[id]) / c;
    prev_old = xo;
  }
}

@compute @workgroup_size(16, 16)
fn mg_smooth(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y);
  if (i >= L.n || j >= L.n || (i + j) % 2 != L.colour) { return; }
  smooth_column(i, j);
}

@compute @workgroup_size(8, 8, 4)
fn mg_residual(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z); let n = L.n;
  if (i >= n || j >= n || k >= L.nz) { return; }
  let s = nsum(i, j, k); let id = (k * n + j) * n + i;
  rf[id] = bf[id] - (s.x - s.y * xf[id]) / L.h2;
}

@compute @workgroup_size(8, 8, 4)
fn mg_restrict(@builtin(global_invocation_id) g: vec3<u32>) {  // over coarse cells
  let I = i32(g.x); let J = i32(g.y); let K = i32(g.z); let n = L.n; let nc = n / 2;
  if (I >= nc || J >= nc || K >= L.nzc) { return; }
  var v = 0.0;
  if (L.nz >= 2) {
    let o0 = (2 * K * n + 2 * J) * n + 2 * I; let o1 = o0 + n * n;
    let a = 0.25 * (rf[o0] + rf[o0 + n] + rf[o0 + 1] + rf[o0 + n + 1]);
    let b = 0.25 * (rf[o1] + rf[o1 + n] + rf[o1 + 1] + rf[o1 + n + 1]);
    v = 0.5 * (a + b);
  } else {
    let o = (K * n + 2 * J) * n + 2 * I;
    v = 0.25 * (rf[o] + rf[o + n] + rf[o + 1] + rf[o + n + 1]);
  }
  let m = (K * nc + J) * nc + I;
  bc[m] = v; xc[m] = 0.0;
}

fn plane(K: i32, i: i32, j: i32) -> f32 {      // 2D trilinear value of coarse layer K at fine (j, i)
  let nc = L.n / 2;
  let I = i / 2; let J = j / 2;
  let I2 = clamp(I + select(1, -1, i % 2 == 0), 0, nc - 1);
  let J2 = clamp(J + select(1, -1, j % 2 == 0), 0, nc - 1);
  let o = K * nc * nc;
  let a = 0.75 * xc[o + J * nc + I] + 0.25 * xc[o + J2 * nc + I];
  let c = 0.75 * xc[o + J * nc + I2] + 0.25 * xc[o + J2 * nc + I2];
  return 0.75 * a + 0.25 * c;
}

@compute @workgroup_size(8, 8, 4)
fn mg_prolong(@builtin(global_invocation_id) g: vec3<u32>) {   // over fine cells
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z); let n = L.n;
  if (i >= n || j >= n || k >= L.nz) { return; }
  var v = 0.0;
  if (L.nz == L.nzc) { v = plane(k, i, j); }
  else {
    let K = k / 2; let off = select(1, -1, k % 2 == 0);
    var K2 = clamp(K + off, 0, L.nzc - 1);
    if (L.periodic == 1) { K2 = (K + off + L.nzc) % L.nzc; }
    v = 0.75 * plane(K, i, j) + 0.25 * plane(K2, i, j);
  }
  xf[(k * n + j) * n + i] += v;
}

// coarsest grid (n = 4): one workgroup, one thread per (i, j) column, all sweeps, then the mean is removed
var<workgroup> msum: array<f32, 256>;
@compute @workgroup_size(256)
fn mg_coarse(@builtin(local_invocation_index) li: u32) {
  let n = L.n; let nz = L.nz; let col = i32(li);
  let ncol = n * n;
  let i = col % n; let j = col / n;
  for (var sweep = 0; sweep < L.sweeps; sweep++) {
    for (var colour = 0; colour < 2; colour++) {
      if (col < ncol && (i + j) % 2 == colour) { smooth_column(i, j); }
      storageBarrier();
      workgroupBarrier();
    }
  }
  var cs = 0.0;
  if (col < ncol) { for (var k = 0; k < nz; k++) { cs += xf[k * ncol + col]; } }
  msum[li] = cs;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { msum[li] += msum[li + s]; }
    workgroupBarrier();
  }
  let mean = msum[0] / f32(ncol * nz);
  if (col < ncol) { for (var k = 0; k < nz; k++) { xf[k * ncol + col] -= mean; } }
}

// Remove the mean (over all n*n*nz cells) of bf (solvability of the singular problem) ...
@compute @workgroup_size(256)
fn mg_mean_b(@builtin(local_invocation_index) li: u32) {
  let total = u32(L.n * L.n * L.nz);
  var s = 0.0;
  for (var m = li; m < total; m += 256u) { s += bf[m]; }
  msum[li] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (li < h) { msum[li] += msum[li + h]; }
    workgroupBarrier();
  }
  let mean = msum[0] / f32(total);
  for (var m = li; m < total; m += 256u) { bf[m] -= mean; }
}

// ... and of xf (fixes the free constant of the solution).
@compute @workgroup_size(256)
fn mg_mean_x(@builtin(local_invocation_index) li: u32) {
  let total = u32(L.n * L.n * L.nz);
  var s = 0.0;
  for (var m = li; m < total; m += 256u) { s += xf[m]; }
  msum[li] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (li < h) { msum[li] += msum[li + h]; }
    workgroupBarrier();
  }
  let mean = msum[0] / f32(total);
  for (var m = li; m < total; m += 256u) { xf[m] -= mean; }
}
