"""Example: maximin Latin hypercube, 12 points in 3 dimensions, 50 candidates, seed 42 stream 0
(SplitMix64.from_seed_stream), and the hard case n = 50, d = 10 with seed 7.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.random.splitmix64 import SplitMix64  # noqa: E402
from agent_tools.stats.lhs_maximin import lhs_maximin  # noqa: E402

a = lhs_maximin(SplitMix64.from_seed_stream(42, 0), 12, 3, 50)
b = lhs_maximin(SplitMix64.from_seed_stream(7, 0), 50, 10, 50)
print(json.dumps({"small_points": a.points.ravel().tolist(), "small_min_distance": a.min_distance,
                  "small_best_try": a.best_try, "hard_points": b.points.ravel().tolist(),
                  "hard_min_distance": b.min_distance, "hard_best_try": b.best_try}))
