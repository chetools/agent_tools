"""Example: Mulberry32 stream for seed 1: integers, uniforms, bounded integers.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.random.mulberry32 import Mulberry32  # noqa: E402

r = Mulberry32(1)
print(json.dumps({"u32": [r.next_u32() for _ in range(4)], "random": [r.random() for _ in range(4)],
                  "below_10": [r.below(10) for _ in range(6)]}))
