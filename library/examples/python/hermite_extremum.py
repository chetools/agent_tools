"""Example: max/min of the Hermite interpolant of sin(t) on coarse knots, evaluator values, a tie
and an invalid input. Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.interpolate.cubic_hermite import cubic_hermite  # noqa: E402
from agent_tools.interpolate.hermite_extremum import hermite_extremum  # noqa: E402

t = np.array([0.0, 0.9, 1.7, 2.6, 3.9, 4.6, 5.9, 7.2])
y, dy = np.sin(t), np.cos(t)
mx, mn = hermite_extremum(t, y, dy), hermite_extremum(t, y, dy, "min")
ev = cubic_hermite(t, y, dy, np.array([-0.5, 1.0, 3.0, 8.0]))
tie = hermite_extremum([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
bad = hermite_extremum([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
print(json.dumps({"max_t": mx.t, "max_value": mx.value, "max_interval": mx.interval,
                  "min_t": mn.t, "min_value": mn.value, "min_interval": mn.interval,
                  "eval": ev.tolist(), "tie_t": tie.t, "tie_interval": tie.interval,
                  "bad_success": bad.success}))
