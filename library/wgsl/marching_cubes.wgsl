// marching_cubes.wgsl: isosurface extraction of a scalar field on a regular 3D grid (compute only).
//
// WHAT IT COMPUTES
//   For the level set { f = iso } of a node-centred field f sampled on an nx*ny*nz grid it emits a triangle
//   list (3 vertices per triangle).  Each vertex carries its position, a unit normal from the field gradient
//   (pointing toward INCREASING f, i.e. out of the region f < iso) and, optionally, a second "colour" field g
//   linearly interpolated to the vertex.  One thread handles one cell; triangles are appended with an
//   atomic counter, so the triangle ORDER is nondeterministic (the SET of triangles is deterministic up to
//   f32 rounding).  Winding: the classic table (inside = f < iso) gives triangles whose right-hand-rule
//   normal points toward DECREASING f, so the signed volume sum(1/6 p0.(p1 x p2)) is negative for a region
//   f < iso; use abs() or the stored normals to orient.
//
// GRID, UNITS AND INDEX ORDER
//   Nodes (i,j,k), 0<=i<nx, 0<=j<ny, 0<=k<nz, sit at  x = ox + i*h, y = oy + j*h, z = oz + k*h.
//   h is the node spacing and (ox,oy,oz) the origin, in ANY consistent length unit; output positions use the
//   same unit.  Spacing is the same in x, y and z (cubic voxels); nx, ny, nz may differ (non-cubic grids).
//   Field buffers FA (and FB) are f32, x fastest:  FA[(k*ny + j)*nx + i].  FA is in arbitrary field units and
//   `iso` is in the same units; the normal is dimensionless; the colour field keeps FB's units.
//   Cells: (nx-1)*(ny-1)*(nz-1).  periodic_z = 1 adds the wrap cells between node nz-1 and node 0, so that
//   the field is treated as periodic with period nz*h in z; their vertices lie in [oz+(nz-1)h, oz+nz*h]
//   (no clipping needed).  Gradient differences wrap in z likewise; in x and y (and in z when periodic_z = 0)
//   they are one-sided at the grid boundary.  Surfaces touching the grid boundary are open there (no caps).
//
// BUFFERS (group 0, own bind group)
//   0  uniform  MC     parameters (struct below, 48 bytes)
//   1  storage read        FA   f32[nx*ny*nz]  scalar field
//   2  storage read        FB   f32[nx*ny*nz]  colour field (bind FA again, or any buffer, when has_c = 0)
//   3  storage read_write  VW   Vert[3*max_tris]  32 bytes per vertex: pos.xyz, cval, nrm.xyz, pad
//   4  storage read_write  CNT  atomic<u32>[2]   CNT[0] = number of triangles the surface needs (counts
//                              past the cap), CNT[1] = overflow flag (1 when CNT[0] > max_tris).  Zero both
//                              words before each run.  Only triangles with index < max_tris are written, so
//                              there is no out-of-bounds write; the stored mesh is then incomplete.
//   5  storage read_write  ARGS u32[4]  indirect draw args { vertexCount = 3*min(CNT[0],max_tris), 1, 0, 0 }
//
// ENTRY POINTS
//   cs_mc    @workgroup_size(64): dispatch (min(G,65535), ceil(G/65535), 1) with G = ceil(cells/64).
//   cs_args  @workgroup_size(1) : dispatch (1,1,1), after cs_mc (separate pass).
//
// LIMITS
//   nx, ny, nz >= 2 for a non-empty result; nx*ny*nz and 3*max_tris*32 bytes must fit the device storage-buffer
//   limits (the host wrapper checks).  Cell indices are i32 (cells < 2^31).
//
// PRECISION (f32)
//   Vertex positions are interpolated in f32: absolute error about 1e-7 * (|o| + n*h) plus the f32 rounding
//   of the field.  The cube classification (f < iso) is exact on the f32 inputs, so the triangle count agrees
//   exactly with a float64 reference run on the same f32-rounded field.  A node with f == iso exactly counts as
//   outside (f >= iso).
//
// CASE AND EDGE TABLES
//   Classic 256-case table of Lorensen & Cline, "Marching cubes: a high resolution 3D surface construction
//   algorithm", Computer Graphics 21(4) (SIGGRAPH '87), 163-169, with the corner and edge numbering and
//   table layout published by Paul Bourke ("Polygonising a scalar field", paulbourke.net/geometry/polygonise/).
//   Corner c: offsets (0,0,0) (1,0,0) (1,1,0) (0,1,0) (0,0,1) (1,0,1) (1,1,1) (0,1,1); bit c of the case index
//   is set when f(corner c) < iso.  Edge e joins corners EA[e] -> EB[e]: bottom ring 0-1,1-2,2-3,3-0, top ring
//   4-5,5-6,6-7,7-4, verticals 0-4,1-5,2-6,3-7.  TRI packs the triangle list of each case as 4-bit edge ids,
//   16 nibbles per case (15 = end), 8 nibbles per u32.
//   LICENCE: the source file this table was extracted from (cylinder-cavity-3d/web/iso.wgsl) states only the
//   attribution above, not a licence.  The table is an unlicensed, widely republished transcription of the
//   Lorensen-Cline / Bourke table; no public-domain declaration was found in that file, and none is claimed here.
//   Ambiguous cases are resolved by the table's fixed choice (no asymptotic decider), so on saddle faces the
//   surface can differ in topology from other implementations and, rarely, show a small crack between cells.
//
// Extracted from chetools/claude cylinder-cavity-3d/web/iso.wgsl (cs_mc, cs_args, grad, fa, fb, wrapk).
// Changes: own uniform and bind group; origin and non-cubic nx,ny,nz; absolute iso only; periodic z is an option
// (default off) with one wrap cell per column and no clipping; counter + overflow flag; per-triangle cap test
// (the original dropped a whole straddling cell); gradients divided by their true span; rendering removed.

struct MC {
  nx: i32, ny: i32, nz: i32, max_tris: u32,
  h: f32, iso: f32, periodic_z: u32, has_c: u32,
  ox: f32, oy: f32, oz: f32, pad: f32,
};
struct Vert { pos: vec3<f32>, cval: f32, nrm: vec3<f32>, pad: f32 };

@group(0) @binding(0) var<uniform> P: MC;
@group(0) @binding(1) var<storage, read> FA: array<f32>;
@group(0) @binding(2) var<storage, read> FB: array<f32>;
@group(0) @binding(3) var<storage, read_write> VW: array<Vert>;
@group(0) @binding(4) var<storage, read_write> CNT: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> ARGS: array<u32>;

// 4-bit packed triangle table: entry t = case*16 + m, value 15 = end
var<private> TRI: array<u32, 512> = array<u32, 512>(
  4294967295u,4294967295u,4294964096u,4294967295u,4294965520u,4294967295u,4279800705u,4294967295u,
  4294965793u,4294967295u,4288811904u,4294967295u,4287760937u,4294967295u,2592744322u,4294967288u,
  4294963891u,4294967295u,4278944432u,4294967295u,4289929361u,4294967295u,2310607537u,4294967291u,
  4282036643u,4294967295u,3098018208u,4294967290u,2879074451u,4294967289u,4290292361u,4294967295u,
  4294965364u,4294967295u,4282609716u,4294967295u,4285827344u,4294967295u,924272916u,4294967281u,
  4285827617u,4294967295u,557856579u,4294967290u,1210096169u,4294967287u,1920543138u,4294920563u,
  4281022280u,4294967295u,37926731u,4294967284u,846496009u,4294967291u,3115621236u,4294906514u,
  2276143523u,4294967284u,28580529u,4294921076u,3115358324u,4294914234u,3113962356u,4294967290u,
  4294964313u,4294967295u,4281861209u,4294967295u,4278522960u,4294967295u,324240472u,4294967285u,
  4282751521u,4294967295u,2493650947u,4294967285u,69491237u,4294967282u,1397896610u,4294935604u,
  4289930329u,4294967295u,2495087280u,4294967285u,844170320u,4294967291u,2189763858u,4294924363u,
  1496427322u,4294967284u,2820146580u,4294945665u,3048230981u,4294914234u,2326288453u,4294967291u,
  4288108665u,4294967295u,1966444601u,4294967283u,1366362224u,4294967287u,4285870929u,4294967295u,
  443914361u,4294967282u,889557530u,4294915920u,1481802248u,4294911399u,1395991970u,4294967287u,
  3013113239u,4294967282u,690460505u,4294948640u,1904282418u,4294931736u,393326891u,4294967285u,
  443910233u,4294947747u,3079688309u,4037779472u,1513140395u,4034228352u,4284183979u,4294967295u,
  4294964586u,4294967295u,4285158272u,4294967295u,4285157641u,4294967295u,2777224065u,4294967286u,
  4279641441u,4294967295u,56759649u,4294967288u,543200617u,4294967286u,623401109u,4294935094u,
  4283870002u,4294967295u,1778563083u,4294967285u,2779982096u,4294967286u,3106477733u,4294949010u,
  355822390u,4294967283u,1348144000u,4294929233u,1617102515u,4294939909u,2612619606u,4294967288u,
  4287055525u,4294967295u,1446461492u,4294967290u,1214926993u,4294967287u,1903760746u,4294920563u,
  1947558422u,4294967288u,56775969u,4294931508u,1615894344u,4294926853u,592017719u,4183976281u,
  1783132851u,4294967285u,606553765u,4294948640u,847726864u,4294928987u,1227591977u,4138027899u,
  1398486856u,4294929233u,23812885u,4215294843u,810944848u,4148704950u,1958308182u,4294941561u,
  4288964938u,4294967295u,2158577316u,4294967283u,1174839562u,4294967280u,1751220536u,4294943076u,
  1648499009u,4294967284u,1116870659u,4294919721u,4284630048u,4294967295u,608338488u,4294967286u,
  726051146u,4294967283u,2495095424u,4294928970u,1616970419u,4294943076u,2225168710u,4055413025u,
  425956457u,4294915763u,1795260856u,4047913361u,1611019955u,4294967284u,4287019078u,4294967295u,
  2561177255u,4294967290u,2423915376u,4294944618u,1903826794u,4294903832u,1897375594u,4294967283u,
  2173048353u,4294932105u,1981360482u,4181145865u,106983559u,4294967282u,4280771127u,4294967295u,
  2324081458u,4294932105u,2427922178u,4155090791u,2709983361u,4214400871u,1785835819u,4294907745u,
  427198104u,4130420406u,4285968784u,4294967295u,3009441927u,4294903472u,4294964919u,4294967295u,
  4294966119u,4294967295u,4284987395u,4294967295u,4284987664u,4294967295u,2064877848u,4294967286u,
  4286276122u,4294967295u,3061856801u,4294967287u,3063554194u,4294967287u,2319067062u,4294936995u,
  4285686567u,4294967295u,637958151u,4294967280u,275981938u,4294967289u,2439516769u,4294928264u,
  829531770u,4294967287u,2175211130u,4294934551u,2695300912u,4294933097u,2827647591u,4294967289u,
  4285052038u,4294967295u,1080048483u,4294967286u,157584232u,4294967281u,959878729u,4294927281u,
  2727044230u,4294967281u,1622161953u,4294927371u,548817076u,4294941225u,1227072410u,4083442611u,
  1680114472u,4294967282u,4280697408u,4294967295u,1111695505u,4294935366u,1109660817u,4294967286u,
  1209434904u,4294908518u,106995738u,4294967284u,2793620324u,4086993667u,4283065498u,4294967295u,
  4290147732u,4294967295u,2069447552u,4294967286u,1728336133u,4294967291u,1396934267u,4294922548u,
  1730258009u,4294967291u,2158106550u,4294924611u,614751079u,4294910026u,591738947u,4135265701u,
  1160147751u,4294967289u,1617429593u,4294932578u,1365717603u,4294902864u,309880870u,4166080584u,
  1902224473u,4294931222u,23534177u,4099475335u,811223556u,4201871978u,1166703207u,4294944842u,
  2342217110u,4294967289u,1345719139u,4294924550u,280299696u,4294948437u,894657462u,4294967281u,
  3115686433u,4294927800u,2427847600u,4196505174u,140883339u,4112524709u,2723374006u,4294915491u,
  1703041413u,4294912050u,1611044441u,4294967282u,1695029329u,4163250232u,4284556881u,4294967295u,
  2208699953u,4137195094u,1499504666u,4294903376u,4289091632u,4294967295u,4294964826u,4294967295u,
  4290083419u,4294967295u,945273435u,4294967280u,2444908469u,4294967280u,2306581882u,4294906753u,
  1461170715u,4294967281u,1903301504u,4294947445u,158504313u,4294933282u,2511499863u,4069098034u,
  1934830162u,4294967285u,2015723560u,4294922917u,893014281u,4294912567u,2014483081u,4065809058u,
  4283905329u,4294967295u,1897334656u,4294967285u,894669577u,4294967287u,4286142345u,4294967295u,
  3129627781u,4294967288u,2768983045u,4294902715u,2829355280u,4294923435u,995402938u,4094891156u,
  2992120146u,4294935880u,1413155648u,4212464427u,2996110624u,4119561541u,4282066249u,4294967295u,
  1126513234u,4294920245u,608326309u,4294967280u,2208641699u,4178609477u,2437042853u,4294911122u,
  1396016456u,4294967281u,4283438400u,4294967295u,154502472u,4294923013u,4294964553u,4294967295u,
  2847492020u,4294967291u,3111732096u,4294949527u,1095441313u,4294947952u,2709795859u,4105876596u,
  692819892u,4294910363u,427529337u,4085259051u,1109701755u,4294967280u,941929595u,4294918708u,
  848767634u,4294939767u,709334953u,4144105351u,1193949811u,4194568730u,4282876577u,4294967295u,
  393298324u,4294967283u,2154906004u,4294907777u,4281627396u,4294967295u,4294965124u,4294967295u,
  4287342761u,4294967295u,2612607235u,4294967290u,2827618832u,4294967291u,4288920083u,4294967295u,
  3113949985u,4294967288u,565786883u,4294941481u,4289760032u,4294967295u,4294966051u,4294967295u,
  2326276146u,4294967289u,4280877737u,4294967295u,279455794u,4294937112u,4294963873u,4294967295u,
  4286683185u,4294967295u,4294963600u,4294967295u,4294965296u,4294967295u,4294967295u,4294967295u,
);
var<private> EA: array<u32, 12> = array<u32, 12>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u);
var<private> EB: array<u32, 12> = array<u32, 12>(1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 4u, 5u, 6u, 7u);
var<private> CO: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
  vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(1, 1, 0), vec3<i32>(0, 1, 0),
  vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(1, 1, 1), vec3<i32>(0, 1, 1));

fn fa(i: i32, j: i32, k: i32) -> f32 { return FA[u32((k * P.ny + j) * P.nx + i)]; }
fn fb(i: i32, j: i32, k: i32) -> f32 { return FB[u32((k * P.ny + j) * P.nx + i)]; }
fn wrapk(k: i32) -> i32 {
  if (P.periodic_z != 0u) { return (k + P.nz) % P.nz; }
  return clamp(k, 0, P.nz - 1);
}
// gradient of FA at node (i,j,k) in field units per length unit (central differences, one-sided at open edges)
fn grad(i: i32, j: i32, k: i32) -> vec3<f32> {
  let xp = min(i + 1, P.nx - 1); let xm = max(i - 1, 0);
  let yp = min(j + 1, P.ny - 1); let ym = max(j - 1, 0);
  let kp = wrapk(k + 1); let km = wrapk(k - 1);
  let gx = (fa(xp, j, k) - fa(xm, j, k)) / (f32(xp - xm) * P.h);
  let gy = (fa(i, yp, k) - fa(i, ym, k)) / (f32(yp - ym) * P.h);
  var sz = kp - km;
  if (P.periodic_z != 0u) { sz = 2; }
  let gz = (fa(i, j, kp) - fa(i, j, km)) / (f32(max(sz, 1)) * P.h);
  return vec3<f32>(gx, gy, gz);
}

@compute @workgroup_size(64)
fn cs_mc(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = i32((wid.y * 65535u + wid.x) * 64u + li);
  let ncx = P.nx - 1; let ncy = P.ny - 1;
  var ncz = P.nz - 1;
  if (P.periodic_z != 0u) { ncz = P.nz; }
  if (ncx < 1 || ncy < 1 || ncz < 1) { return; }
  if (idx >= ncx * ncy * ncz) { return; }
  let ci = idx % ncx; let cj = (idx / ncx) % ncy; let kc = idx / (ncx * ncy);
  let k0 = kc; var k1 = kc + 1;
  if (P.periodic_z != 0u && kc == P.nz - 1) { k1 = 0; }
  let iso = P.iso;
  var val: array<f32, 8>;
  var cube = 0u;
  for (var c = 0u; c < 8u; c++) {
    let o = CO[c];
    let k = select(k0, k1, o.z == 1);
    val[c] = fa(ci + o.x, cj + o.y, k);
    if (val[c] < iso) { cube |= 1u << c; }
  }
  if (cube == 0u || cube == 255u) { return; }
  var nt = 0u;
  for (var m = 0u; m < 15u; m += 3u) {
    let t = cube * 16u + m;
    if (((TRI[t >> 3u] >> ((t & 7u) * 4u)) & 15u) == 15u) { break; }
    nt++;
  }
  let base = atomicAdd(&CNT[0], nt);
  if (base + nt > P.max_tris) { atomicStore(&CNT[1], 1u); }
  let z0 = P.oz + f32(kc) * P.h;
  for (var q = 0u; q < nt; q++) {
    if (base + q >= P.max_tris) { break; }
    for (var v = 0u; v < 3u; v++) {
      let t = cube * 16u + q * 3u + v;
      let e = (TRI[t >> 3u] >> ((t & 7u) * 4u)) & 15u;
      let a = EA[e]; let b = EB[e];
      let oa = CO[a]; let ob = CO[b];
      let va = val[a]; let vb = val[b];
      let den = vb - va;
      let s = select(0.5, clamp((iso - va) / den, 0.0, 1.0), abs(den) > 1.0e-30);
      let ka = select(k0, k1, oa.z == 1); let kb = select(k0, k1, ob.z == 1);
      let pa = vec3<f32>(P.ox + f32(ci + oa.x) * P.h, P.oy + f32(cj + oa.y) * P.h, z0 + f32(oa.z) * P.h);
      let pb = vec3<f32>(P.ox + f32(ci + ob.x) * P.h, P.oy + f32(cj + ob.y) * P.h, z0 + f32(ob.z) * P.h);
      var o: Vert;
      o.pos = mix(pa, pb, s);
      let g = mix(grad(ci + oa.x, cj + oa.y, ka), grad(ci + ob.x, cj + ob.y, kb), s);
      o.nrm = g / max(length(g), 1.0e-30);
      o.cval = 0.0;
      if (P.has_c != 0u) { o.cval = mix(fb(ci + oa.x, cj + oa.y, ka), fb(ci + ob.x, cj + ob.y, kb), s); }
      o.pad = 0.0;
      VW[(base + q) * 3u + v] = o;
    }
  }
}

@compute @workgroup_size(1)
fn cs_args() {
  let t = min(atomicLoad(&CNT[0]), P.max_tris);
  ARGS[0] = t * 3u; ARGS[1] = 1u; ARGS[2] = 0u; ARGS[3] = 0u;
}
