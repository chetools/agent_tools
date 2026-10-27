//! Greedy batch selection by the integrated mean squared error (IMSE) of a Gaussian process.
//!
//! Source: the sequential-design heuristic of Seo, Wallat, Graepel & Obermayer, "Gaussian process
//! regression: active data selection and test point rejection" (IJCNN 2000) and the IMSE criterion
//! of Sacks, Welch, Mitchell & Wynn, "Design and analysis of computer experiments", Statistical
//! Science 4 (1989); the rank-one update of the posterior covariance after a hypothetical
//! observation is the Gaussian conditioning identity (Rasmussen & Williams, "Gaussian Processes for
//! Machine Learning", 2006, Appendix A.2); the covariance does not depend on the measured value. Ported from chetools/claude
//! `sequential-design/solver/src/lib.rs` (`greedy_batch`, rule IMSE; same authors), generalised so
//! that the candidate points and the reference points where the error is averaged may differ.
//! Python twin: `library/python/agent_tools/stats/imse_batch.py`.
//!
//! # Method
//! Let `C` be the posterior covariance of the function values. Observing the candidate `j` with
//! noise variance `s2` reduces the covariance to `C - C[:,j] C[j,:] / (C[j,j] + s2)`. The average
//! posterior variance over the `nr` reference points drops by
//! `score_j = sum_r C[r,j]^2 / (C[j,j] + s2) / nr`. One step picks the candidate with the largest
//! score (the first one on a tie), then applies the rank-one downdate for it (the measured value is not
//! needed). Repeating `batch` times gives a batch whose points are chosen jointly (greedily), without
//! refitting the GP. A candidate can be picked again (a replicate) when its remaining variance
//! is still the best use of an experiment.
//!
//! Only the covariances among the candidates and between reference points and candidates are
//! needed; the reference-reference block is not. Cost: `O(batch (nc^2 + nr nc))`.

/// Result of [`imse_batch`].
#[derive(Clone, Debug)]
pub struct ImseResult {
    /// Chosen candidate indices, in selection order (`batch` entries, repeats possible).
    pub indices: Vec<usize>,
    /// For each pick, the drop of the average posterior variance over the reference points caused
    /// by that pick given the earlier ones (units of the covariance matrices). Their sum is the
    /// total drop for the batch.
    pub reduction: Vec<f64>,
    /// The score (drop of the average variance if only that candidate were added) of every
    /// candidate at the first step, length `nc` (same units as `reduction`).
    pub first_scores: Vec<f64>,
}

/// Choose `batch` of the `nc` candidates by greedy IMSE reduction.
///
/// * `cov_cand`: posterior covariance between the candidates, `nc x nc` row-major, symmetric
///   (units `u`, e.g. `y^2`); only its diagonal and columns are read.
/// * `cov_ref_cand`: posterior covariance between the `nr` reference points (rows) and the
///   candidates (columns), `nr x nc` row-major, same units. When the reference points are the
///   candidates, pass the same matrix as `cov_cand`.
/// * `noise_variance`: variance of one new observation (same units, `>= 0`).
/// * `batch`: number of picks (any value, also larger than `nc`).
///
/// Both matrices are copied, not modified. Panics on inconsistent shapes, `nc = 0` or `nr = 0`, a
/// negative or non-finite `noise_variance`. A candidate with `C[j,j] + noise_variance <= 0` (fully
/// known and noise-free) gets score 0, and if such a candidate is picked no downdate is applied
/// (all scores are 0 from then on).
pub fn imse_batch(cov_cand: &[f64], cov_ref_cand: &[f64], nc: usize, nr: usize, noise_variance: f64, batch: usize) -> ImseResult {
    assert!(nc >= 1 && nr >= 1, "imse_batch: need at least one candidate and one reference point");
    assert_eq!(cov_cand.len(), nc * nc, "imse_batch: cov_cand must be nc*nc");
    assert_eq!(cov_ref_cand.len(), nr * nc, "imse_batch: cov_ref_cand must be nr*nc");
    assert!(noise_variance >= 0.0 && noise_variance.is_finite(), "imse_batch: noise_variance must be >= 0");
    let mut cc = cov_cand.to_vec();
    let mut rc = cov_ref_cand.to_vec();
    let mut indices = Vec::with_capacity(batch);
    let mut reduction = Vec::with_capacity(batch);
    let mut first_scores = Vec::new();
    let mut score = vec![0.0; nc];
    let mut uc = vec![0.0; nc];
    let mut ur = vec![0.0; nr];
    for step in 0..batch {
        for j in 0..nc {
            let denom = cc[j * nc + j] + noise_variance;
            score[j] = if denom > 0.0 {
                let mut s = 0.0;
                for r in 0..nr {
                    s += rc[r * nc + j] * rc[r * nc + j];
                }
                s / denom / nr as f64
            } else {
                0.0
            };
        }
        let mut jb = 0;
        for j in 1..nc {
            if score[j] > score[jb] {
                jb = j;
            }
        }
        if step == 0 {
            first_scores = score.clone();
        }
        indices.push(jb);
        reduction.push(score[jb]);
        let denom = cc[jb * nc + jb] + noise_variance;
        if denom > 0.0 {
            let s = 1.0 / denom.sqrt();
            for i in 0..nc {
                uc[i] = cc[i * nc + jb] * s;
            }
            for r in 0..nr {
                ur[r] = rc[r * nc + jb] * s;
            }
            for i in 0..nc {
                for k in 0..nc {
                    cc[i * nc + k] -= uc[i] * uc[k];
                }
            }
            for r in 0..nr {
                for k in 0..nc {
                    rc[r * nc + k] -= ur[r] * uc[k];
                }
            }
        }
    }
    ImseResult { indices, reduction, first_scores }
}
