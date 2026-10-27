"""Example: Cholesky factor and solve of a 3x3 SPD system, and a non-SPD matrix.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.linalg.cholesky import cholesky, cholesky_solve  # noqa: E402

a = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
r = cholesky(a)
x = cholesky_solve(r.l, np.array([1.0, 2.0, 3.0]))
bad = cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
print(json.dumps({"l": r.l.ravel().tolist(), "x": x.tolist(), "success": r.success,
                  "bad_success": bad.success}))
