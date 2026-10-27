"""Example: solve a block-tridiagonal system (m = 2, nb = 6) by block Thomas, solve a second
right-hand side with the same factorization, and report a singular first diagonal block.
Prints one JSON object (same keys as the Rust example)."""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.linalg.block_tridiagonal import block_tridiag_factor, block_tridiag_solve  # noqa: E402

m, nb = 2, 6
lower, upper, diag = [], [], []
for i in range(nb):
    t = float(i)
    diag += [4.0 + 0.1 * t, 0.5, -0.3, 5.0 - 0.2 * t]
    if i + 1 < nb:
        lower += [0.5, 0.1 * t, -0.2, 0.7]
        upper += [-0.6, 0.2, 0.1 * t - 0.3, 0.4]
b1 = [math.sin(i + 1.0) for i in range(m * nb)]
b2 = [1.0 / (i + 1.0) for i in range(m * nb)]
f = block_tridiag_factor(lower, diag, upper, m, nb)
s1, s2 = block_tridiag_solve(f, b1), block_tridiag_solve(f, b2)
# [[0,1],[1,0]] with m = 1: nonsingular, but the first diagonal block is zero
bad = block_tridiag_factor([1.0], [0.0, 0.0], [1.0], 1, 2)
print(json.dumps({"x1": s1.x, "x2": s2.x, "growth": f.growth, "success": f.success,
                  "bad_success": bad.success, "bad_message": bad.message}))
