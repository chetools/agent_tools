"""Generator for reference_data/splitmix64.json.

The stream is computed here from the published algorithm with Python integer arithmetic, independently
of library/python (this file does not import it): Steele, Lea & Flood (OOPSLA 2014); constants of
Vigna's splitmix64.c. Also computed in node (BigInt) and asserted equal; the first outputs are checked
against published test vectors recalled from memory (not independently sourced).

Run: python library/tests/generators/splitmix64.py
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

# ---- SplitMix64
sm_cases = {"streams": {}}
for s in SM_SEEDS:
    u = sm64_stream(s, N)
    if js:
        assert [str(x) for x in u] == js["sm64"][str(s)], "node BigInt disagrees"
    sm_cases["streams"][str(s)] = {
        "u64": u,
        "uniform": [(x >> 11) * 2.0 ** -53 for x in u],
    }
# published test vectors, as recalled from memory (checked here, not independently sourced)
assert sm64_stream(0, 1)[0] == 0xE220A8397B1DCDAF
assert sm64_stream(1234567, 5) == [6457827717110365317, 3203168211198807973,
                                   9817491932198370423, 4593380528125082431, 16408922859458223821]


def uniforms(state, k):
    return [(x >> 11) * 2.0 ** -53 for x in sm64_stream(state, k)]


# normals, permutations: functions of the uniform stream, written out independently
def normals(state, k):
    u = uniforms(state, 2 * k)
    return [math.sqrt(-2.0 * math.log(1.0 - u[2 * i])) * math.cos(2.0 * math.pi * u[2 * i + 1])
            for i in range(k)]


def perm(state, n):
    u = iter(uniforms(state, max(n - 1, 0)))
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(next(u) * (i + 1))
        p[i], p[j] = p[j], p[i]
    return p


sm_cases["normal"] = {str(s): normals(s, 10) for s in (0, 42, 1234567)}
sm_cases["permutation"] = {f"{s}:{n}": perm(s, n) for s in (7, 99) for n in (1, 2, 10, 52)}
sm_cases["seed_stream"] = {"seed": 3, "stream": 2, "state": (3 * 1000003 + 2) & M64,
                           "u64": sm64_stream((3 * 1000003 + 2) & M64, 5)}
write("splitmix64", "library/tests/generators/splitmix64.py",
      {"n_outputs": N, "seeds": SM_SEEDS,
       "algorithm": "Steele, Lea & Flood 2014; Vigna splitmix64.c constants",
       "node_cross_check": "BigInt implementation in node" if js else "node not available"},
      "Dimensionless. u64 outputs are decimal-exact integers. 'uniform' = (u64>>11)*2^-53. "
      "normal = Box-Muller sqrt(-2 ln(1-u1)) cos(2 pi u2) using two uniforms per variate; "
      "permutation = Fisher-Yates from the top (i = n-1..1, j = int(u*(i+1))), key 'seed:n'. "
      "Seed 0 first output and seed 1234567 first five outputs also match the test vectors "
      "recalled from memory (not independently sourced).",
      sm_cases, node=node)

