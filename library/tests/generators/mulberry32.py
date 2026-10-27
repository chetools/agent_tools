"""Generator for reference_data/mulberry32.json.

The stream is computed here from the published algorithm with Python integer arithmetic, independently
of library/python (this file does not import it): Tommy Ettinger's public-domain mulberry32 (JavaScript,
Math.imul version, 2017). Also computed in node, for Ettinger's JavaScript and for the variant in
chetools/claude heat-exchanger-network/web/gpu.js, and asserted equal.

Run: python library/tests/generators/mulberry32.py
"""
import json
import math
import subprocess

from _common import node_version, write

M64 = (1 << 64) - 1
M32 = 0xFFFFFFFF
N = 20


def sm64_stream(state, n):
    out = []
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & M64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out.append(z ^ (z >> 31))
    return out


def mb32_stream(a, n):
    out = []
    for _ in range(n):
        a = (a + 0x6D2B79F5) & M32
        t = a
        t = (((t ^ (t >> 15)) * (t | 1)) & M32)
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32))
        out.append((t ^ (t >> 14)) & M32)
    return out


JS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const M = (1n << 64n) - 1n;
function sm64(state, n) {
  let s = BigInt(state); const out = [];
  for (let i = 0; i < n; i++) {
    s = (s + 0x9E3779B97F4A7C15n) & M; let z = s;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & M;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & M;
    out.push((z ^ (z >> 31n)).toString());
  }
  return out;
}
function mulberry32(a) {   // Ettinger, as published
  return function () {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0);
  };
}
function mulberryGpu(a) {  // variant in heat-exchanger-network/web/gpu.js
  a = a | 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0);
  };
}
const res = {sm64: {}, mb32: {}, mb32_gpu: {}};
for (const s of input.sm64) res.sm64[s] = sm64(s, input.n);
for (const s of input.mb32) {
  const r = mulberry32(s), g = mulberryGpu(s);
  res.mb32[s] = Array.from({length: input.n}, r);
  res.mb32_gpu[s] = Array.from({length: input.n}, g);
}
console.log(JSON.stringify(res));
"""

SM_SEEDS = [0, 1, 42, 1234567, M64]
MB_SEEDS = [0, 1, 42, 7, M32]

node = node_version()
js = None
if node:
    inp = json.dumps({"sm64": [str(s) for s in SM_SEEDS], "mb32": MB_SEEDS, "n": N})
    out = subprocess.run(["node", "-e", JS], input=inp, capture_output=True, text=True, check=True).stdout
    js = json.loads(out)

# ---- Mulberry32
mb_cases = {"streams": {}}
for s in MB_SEEDS:
    u = mb32_stream(s, N)
    if js:
        assert u == js["mb32"][str(s)], "node (Ettinger) disagrees"
        assert u == js["mb32_gpu"][str(s)], "node (gpu.js variant) disagrees"
    mb_cases["streams"][str(s)] = {"u32": u, "random": [x / 4294967296.0 for x in u],
                                   "below_10": [x % 10 for x in u], "below_1000": [x % 1000 for x in u]}
# existing project value (heat-exchanger-network test_hen.py / checks.rs), seed 1
assert mb_cases["streams"]["1"]["u32"][:4] == [2693262067, 11749833, 2265367787, 4213581821]
write("mulberry32", "library/tests/generators/mulberry32.py",
      {"n_outputs": N, "seeds": MB_SEEDS, "algorithm": "Ettinger mulberry32 (Math.imul version, public domain)",
       "node_cross_check": "Ettinger JS and the gpu.js variant, both in node" if js else "node not available"},
      "Dimensionless. random = u32/2^32; below_n = u32 % n. Seed 1 first four outputs also equal "
      "the values in heat-exchanger-network test_hen.py::test_mulberry32_reference_values.",
      mb_cases, node=node)
