"""Example: pinch targets of Kemp's four-stream problem (dT_min = 10 K) and an invalid stream.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.process.pinch_targets import pinch_targets  # noqa: E402

hot = [(170.0, 60.0, 3.0), (150.0, 30.0, 1.5)]
cold = [(20.0, 135.0, 2.0), (80.0, 140.0, 4.0)]
r = pinch_targets(hot, cold, 10.0)
try:
    pinch_targets([(10.0, 50.0, 1.0)], [], 10.0)   # hot stream that heats up
    bad_rejected = False
except ValueError:
    bad_rejected = True
print(json.dumps({"qh_min": r.qh_min, "qc_min": r.qc_min, "pinch_hot": r.pinch_hot,
                  "t_hot_scale": r.t_hot_scale, "heat_flow": r.heat_flow,
                  "hot_composite_h": list(r.hot_composite[1]), "cold_composite_h": list(r.cold_composite[1]),
                  "bad_rejected": bad_rejected}))
