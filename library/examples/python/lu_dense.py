"""Example: solve a 4x4 system by LU with partial pivoting, and report a singular matrix.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.linalg.lu_dense import lu_factor, lu_solve  # noqa: E402

a = [0.0, 2.0, 1.0, -1.0,
     3.0, -1.0, 2.0, 0.5,
     1.0, 1.0, -4.0, 2.0,
     2.0, 0.0, 1.0, 5.0]
b = [1.0, 2.0, 3.0, 4.0]
f = lu_factor(a, 4)
s = lu_solve(f, b)
sing = lu_factor([1.0, 2.0, 2.0, 4.0], 2)
print(json.dumps({"x": s.x, "piv": f.piv, "success": s.success,
                  "singular_success": sing.success, "singular_message": sing.message}))
