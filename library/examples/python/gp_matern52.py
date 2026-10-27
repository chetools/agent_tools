"""Example: fit a Matern-5/2 ARD GP to 15 noisy samples of f(x) = sin(6 x1) + x2^2 on [0,1]^2
(inputs from SplitMix64 seed 11, noise 0.05 N(0,1)), with the default starts plus 2 random starts
(seed 5); predict at 4 points and give the posterior covariance between 3 of them.
Prints one JSON object (same keys as the Rust example)."""
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.random.splitmix64 import SplitMix64  # noqa: E402
from agent_tools.stats.gp_matern52 import GpOptions, gp_fit  # noqa: E402

rng = SplitMix64(11)
X, y = [], []
for _ in range(15):                                  # same draw order as Rust: x1, x2, noise
    a, b = rng.uniform(), rng.uniform()
    X.append([a, b])
    y.append(math.sin(6.0 * a) + b * b + 0.05 * rng.normal())
X, y = np.array(X), np.array(y)
r = gp_fit(X, y, GpOptions(n_random_starts=2, seed=5))
G = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.3], [0.33, 0.77]])
mean, var = r.model.predict(G)
cov = r.model.posterior_cov(G[:3], G[:3])
print(json.dumps({"length_scales": r.model.length_scales.tolist(), "signal_sd": r.model.signal_sd,
                  "noise_sd": r.model.noise_sd, "nll_std": r.nll_std,
                  "log_marginal_likelihood": r.model.log_marginal_likelihood, "start_nll": r.start_nll,
                  "best_start": r.best_start, "success": r.success, "nfev": r.nfev, "nit": r.nit,
                  "mean": mean.tolist(), "var": var.tolist(), "cov": cov.ravel().tolist()}))
