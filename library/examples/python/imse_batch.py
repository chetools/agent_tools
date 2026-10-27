"""Example: greedy IMSE batches for a GP with fixed hyperparameters (length scales 0.3, 0.5,
signal_sd 1, noise_sd 0.05) conditioned on 6 samples of f(x) = sin(6 x1) + x2^2 (inputs from
SplitMix64 seed 3). Case "same": candidates = reference = 6 x 6 grid on [0,1]^2, batch 4.
Case "other_ref": the same candidates, 25 reference points from SplitMix64 seed 4, batch 3.
Prints one JSON object (same keys as the Rust example)."""
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.random.splitmix64 import SplitMix64  # noqa: E402
from agent_tools.stats.gp_matern52 import gp_build  # noqa: E402
from agent_tools.stats.imse_batch import imse_batch  # noqa: E402

rng = SplitMix64(3)
X = np.array([[rng.uniform(), rng.uniform()] for _ in range(6)])
y = np.array([math.sin(6.0 * a) + b * b for a, b in X])
gp = gp_build(X, y, [0.3, 0.5], 1.0, 0.05)
cand = np.array([[i / 5, j / 5] for j in range(6) for i in range(6)])
rr = SplitMix64(4)
ref = np.array([[rr.uniform(), rr.uniform()] for _ in range(25)])
c_cc = gp.posterior_cov(cand, cand)
out = {}
for name, r_pts, b in (("same", cand, 4), ("other_ref", ref, 3)):
    res = imse_batch(c_cc, gp.posterior_cov(r_pts, cand), gp.noise_sd ** 2, b)
    out[name] = {"indices": res.indices, "reduction": res.reduction, "first_scores": res.first_scores.tolist()}
print(json.dumps(out))
