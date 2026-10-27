"""Generator for reference_data/imse_batch.json (run after gp_matern52.py, whose smooth2d case it reads).

Reference: brute-force greedy IMSE, independent of the library's rank-one downdate. At every step, for every
candidate j, the GP (Matern-5/2 kernel by the Bessel-function formula, scipy.special.kv, as in
gp_matern52.py) is re-conditioned with np.linalg.solve on the design plus the earlier picks plus j (noise_sd
on every observation, plus the 1e-8 standardised jitter), and the drop of the mean latent posterior variance
over the reference points is measured (y^2 units). The candidate with the largest drop is picked.
Cases: same_grid_8x8 (candidates = reference = 8 x 8 grid on [0,1]^2, batch 3, the source project's use);
grid_vs_random_ref (candidates 10 x 10 cell-centred grid, 40 random reference points, batch 4).
Run: python library/tests/generators/imse_batch.py
"""
import json
import math

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

from _common import OUT, write

JITTER = 1e-8
SEED = 20261031


def kmat(A, B, ell, sf2):
    a = math.sqrt(5.0) * cdist(A / ell, B / ell)
    out = np.full(a.shape, sf2)
    m = a > 0
    out[m] = sf2 * 2 ** (1 - 2.5) / gamma(2.5) * a[m] ** 2.5 * kv(2.5, a[m])
    return out


def brute_force_greedy(X, y, ell, signal_sd, noise_sd, cand, ref, batch):
    ys = float(np.std(y))
    sf2, sn2 = (signal_sd / ys) ** 2, (noise_sd / ys) ** 2

    def mean_ref_var(Xa):
        K = kmat(Xa, Xa, ell, sf2) + (sn2 + JITTER) * np.eye(len(Xa))
        Ks = kmat(ref, Xa, ell, sf2)
        v = sf2 - np.sum(Ks * np.linalg.solve(K, Ks.T).T, axis=1)
        return ys * ys * float(np.mean(v))

    chosen, drops, first = [], [], None
    for step in range(batch):
        Xc = np.vstack([X] + ([cand[chosen]] if chosen else []))
        base = mean_ref_var(Xc)
        d = np.array([base - mean_ref_var(np.vstack([Xc, cand[j:j + 1]])) for j in range(len(cand))])
        if step == 0:
            first = d.copy()
        j = int(np.argmax(d))
        chosen.append(j)
        drops.append(float(d[j]))
    srt = np.sort(first)[::-1]
    return chosen, drops, first.tolist(), float(srt[0] - srt[1])


gp_ref = json.load(open(OUT / "gp_matern52.json"))["cases"]["smooth2d"]
X = np.array(gp_ref["x"]).reshape(-1, 2)
y = np.array(gp_ref["y"])
fit = gp_ref["fitted"]
ell, sd, nsd = np.array(fit["ell"]), fit["signal_sd"], fit["noise_sd"]
rng = np.random.RandomState(SEED)
g8 = np.array([[i / 7, j / 7] for j in range(8) for i in range(8)])
g10 = np.array([[(i + 0.5) / 10, (j + 0.5) / 10] for j in range(10) for i in range(10)])
refp = rng.rand(40, 2)
cases = {"gp": {"d": 2, "x": X.ravel().tolist(), "y": y.tolist(), "ell": ell.tolist(), "signal_sd": sd,
                "noise_sd": nsd}}
for name, cand, ref, batch in (("same_grid_8x8", g8, g8, 3), ("grid_vs_random_ref", g10, refp, 4)):
    ch, dr, first, gap = brute_force_greedy(X, y, ell, sd, nsd, cand, ref, batch)
    cases[name] = {"candidates": cand.ravel().tolist(), "reference": ref.ravel().tolist(), "batch": batch,
                   "indices": ch, "reduction": dr, "first_scores": first, "first_score_gap": gap}
    print(name, ch, dr, "gap", gap)
write("imse_batch", "library/tests/generators/imse_batch.py",
      {"seed": SEED, "rng": "numpy.random.RandomState",
       "reference": "brute-force greedy re-conditioning with np.linalg.solve, Bessel-function Matern kernel"},
      "GP data x, y and hyperparameters are those of gp_matern52.json case smooth2d (best L-BFGS-B fit), y "
      "standardised by its mean and population std. Reductions and scores are drops of the mean latent posterior "
      "variance over the reference points, y^2 units. first_score_gap is the gap between the best and "
      "second-best first score (a tie would make index comparisons fragile).",
      cases)
