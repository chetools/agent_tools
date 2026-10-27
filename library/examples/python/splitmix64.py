"""Example: SplitMix64 stream for state 42: integers, uniforms, normals, a shuffle.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.random.splitmix64 import SplitMix64  # noqa: E402

r = SplitMix64(42)
u64s = [r.next_u64() for _ in range(5)]
uniforms = [r.uniform() for _ in range(5)]
normals = [r.normal() for _ in range(5)]
perm = r.permutation(10)
print(json.dumps({"u64": u64s, "uniform": uniforms, "normal": normals, "permutation": perm}))
