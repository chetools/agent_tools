"""Greedy batch selection by the integrated mean squared error (IMSE) of a Gaussian process
(Python reference of the Rust `stats::imse_batch`).

Source: the sequential-design heuristic of Seo, Wallat, Graepel & Obermayer, "Gaussian process
regression: active data selection and test point rejection" (IJCNN 2000) and the IMSE criterion
of Sacks, Welch, Mitchell & Wynn, "Design and analysis of computer experiments", Statistical
Science 4 (1989); the rank-one update of the posterior covariance after a hypothetical observation
is the Gaussian conditioning identity (Rasmussen & Williams, "Gaussian Processes for Machine
Learning", 2006, Appendix A.2); the covariance does not depend on the measured value. Ported from
chetools/claude `sequential-design/python/sdl.py` (`greedy_batch`, rule "imse"; same authors),
generalised so that the candidate points and the reference points where the error is averaged may
differ. Rust twin: library/rust/src/stats/imse_batch.rs.

Method: let C be the posterior covariance of the function values. Observing the candidate j with
noise variance s2 reduces the covariance to C - C[:,j] C[j,:] / (C[j,j] + s2). The average posterior
variance over the nr reference points drops by score_j = sum_r C[r,j]^2 / (C[j,j] + s2) / nr. One
step picks the candidate with the largest score (the first one on a tie), then applies the rank-one
downdate for it (the measured value is not needed). Repeating `batch` times gives a batch chosen
jointly (greedily), without refitting the GP. A candidate can be picked again (a replicate) when its
remaining variance is still the best use of an experiment.

Only the covariances among the candidates and between reference points and candidates are needed;
the reference-reference block is not. Cost: O(batch (nc^2 + nr nc)).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ImseResult:
    """Result of `imse_batch`.

    indices: chosen candidate indices, in selection order (`batch` entries, repeats possible).
    reduction: for each pick, the drop of the average posterior variance over the reference points
        caused by that pick given the earlier ones (units of the covariance matrices). Their sum is
        the total drop for the batch.
    first_scores: score (drop of the average variance if only that candidate were added) of every
        candidate at the first step, length nc (same units as `reduction`).
    """
    indices: list
    reduction: list
    first_scores: np.ndarray


def imse_batch(cov_cand, cov_ref_cand, noise_variance, batch):
    """Choose `batch` of the nc candidates by greedy IMSE reduction.

    cov_cand: (nc, nc) posterior covariance between the candidates, symmetric (units u, e.g. y^2);
        only its diagonal and columns are read.
    cov_ref_cand: (nr, nc) posterior covariance between the nr reference points (rows) and the
        candidates (columns), same units. When the reference points are the candidates, pass the
        same matrix as cov_cand.
    noise_variance: variance of one new observation (same units, >= 0).
    batch: number of picks (any value, also larger than nc).

    Both matrices are copied, not modified. Raises ValueError on inconsistent shapes, nc = 0 or
    nr = 0, a negative or non-finite noise_variance. A candidate with C[j,j] + noise_variance <= 0
    (fully known and noise-free) gets score 0, and if such a candidate is picked no downdate is
    applied (all scores are 0 from then on).
    """
    cc = np.array(cov_cand, dtype=float)
    rc = np.array(cov_ref_cand, dtype=float)
    if cc.ndim != 2 or cc.shape[0] != cc.shape[1] or cc.shape[0] < 1:
        raise ValueError("imse_batch: cov_cand must be a non-empty square matrix")
    nc = cc.shape[0]
    if rc.ndim != 2 or rc.shape[1] != nc or rc.shape[0] < 1:
        raise ValueError("imse_batch: cov_ref_cand must be (nr, nc) with nr >= 1")
    if not (noise_variance >= 0 and np.isfinite(noise_variance)):
        raise ValueError("imse_batch: noise_variance must be >= 0")
    nr = rc.shape[0]
    indices, reduction, first = [], [], None
    for step in range(batch):
        denom = np.diag(cc) + noise_variance
        pos = denom > 0
        score = np.zeros(nc)
        score[pos] = np.sum(rc[:, pos] ** 2, axis=0) / denom[pos] / nr
        jb = int(np.argmax(score))                  # first index on a tie
        if step == 0:
            first = score.copy()
        indices.append(jb)
        reduction.append(float(score[jb]))
        if denom[jb] > 0:
            s = 1.0 / np.sqrt(denom[jb])
            uc = cc[:, jb] * s
            ur = rc[:, jb] * s
            cc -= np.outer(uc, uc)
            rc -= np.outer(ur, uc)
    return ImseResult(indices, reduction, first if first is not None else np.zeros(nc))
