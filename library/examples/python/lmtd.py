"""Example: LMTD and partial derivatives for a normal, a near-equal and an invalid case.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.process.lmtd import lmtd  # noqa: E402

normal = lmtd(30.0, 10.0)
near = lmtd(20.0, 20.00001)
bad = lmtd(-1.0, 5.0)
print(json.dumps({"normal": list(normal), "near_equal": list(near), "invalid_is_nan": bad[0] != bad[0]}))
