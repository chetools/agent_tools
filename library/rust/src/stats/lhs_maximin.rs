//! Maximin Latin hypercube design: best of `tries` random Latin hypercubes.
//!
//! Sources: McKay, Beckman & Conover, "A comparison of three methods for selecting values of input
//! variables in the analysis of output from a computer code", Technometrics 21 (1979) 239-245
//! (Latin hypercube sampling); Johnson, Moore & Ylvisaker, "Minimax and maximin distance designs",
//! J. Stat. Plann. Inference 26 (1990) 131-148, and Morris & Mitchell, "Exploratory designs for
//! computational experiments", J. Stat. Plann. Inference 43 (1995) 381-402 (the maximin criterion
//! for space-filling designs; the search here is the plain best-of-k random search, not their
//! simulated-annealing search). Ported from chetools/claude `sequential-design/solver/src/lib.rs`
//! (`lhs_maximin`, own code).
//!
//! Algorithm. The unit range of every input is cut into `n` equal slices. One candidate design
//! draws, for each dimension in turn, a random permutation of the slices (`SplitMix64::permutation`)
//! and then, for each point in turn, a uniform position inside its slice (one `uniform()` per
//! point and dimension). So every candidate has exactly one point in each slice of each input
//! (the Latin property), at a random, not centred, position in the slice: x = (slice + u) / n,
//! u in [0, 1). Of `tries` independent candidates the one whose closest pair of points (Euclidean
//! distance in the unit cube) is farthest apart is kept; on an exact tie the earlier candidate
//! wins. The result is deterministic given the generator state, and bit-identical to the Python
//! twin `library/python/agent_tools/stats/lhs_maximin.py` (the same random draws in the same order,
//! squared distances summed over dimensions in the same order).
//!
//! Draw order per candidate: for dimension 0..d: `permutation(n)` (n-1 uniforms), then n uniforms
//! for the positions. All coordinates and distances are dimensionless (unit hypercube).

use crate::random::splitmix64::SplitMix64;

/// Result of [`lhs_maximin`].
#[derive(Debug, Clone)]
pub struct LhsResult {
    /// True when a design was produced; false when an input was invalid (see `message`).
    pub success: bool,
    /// Human-readable status ("ok" or the reason for failure).
    pub message: String,
    /// Design points, row-major `n x d`: `points[k * d + i]` is coordinate `i` of point `k`.
    /// Dimensionless, in [0, 1). Empty on failure.
    pub points: Vec<f64>,
    /// Number of points.
    pub n: usize,
    /// Number of dimensions.
    pub d: usize,
    /// Smallest pairwise Euclidean distance of the returned design (dimensionless, unit cube).
    /// For `n == 1` it is 1.0 by convention (there is no pair). NaN on failure.
    pub min_distance: f64,
    /// Zero-based index of the candidate that was kept (the first one reaching the best distance).
    pub best_try: usize,
    /// Number of candidate designs generated and evaluated (equals the `tries` argument on success).
    pub n_candidates: usize,
}

/// One random Latin hypercube of `n` points in `d` dimensions, row-major `n x d`, coordinates in
/// [0, 1). Each input has exactly one point in each of the `n` slices `[j/n, (j+1)/n)`, at a
/// uniformly random position inside its slice. Draws `d * (2n - 1)` uniforms (for `n >= 1`).
/// `n` or `d` equal to 0 gives an empty vector.
pub fn lhs_random(rng: &mut SplitMix64, n: usize, d: usize) -> Vec<f64> {
    let mut x = vec![0.0; n * d];
    for i in 0..d {
        let perm = rng.permutation(n);
        for k in 0..n {
            x[k * d + i] = (perm[k] as f64 + rng.uniform()) / n as f64;
        }
    }
    x
}

/// Smallest pairwise Euclidean distance of `n` points in `d` dimensions (row-major `n x d`, any
/// consistent unit; the result has the same unit). Squared differences are summed over the
/// dimensions in order. Returns 1.0 for `n <= 1` (the source's convention for a single point).
pub fn min_pairwise_distance(x: &[f64], n: usize, d: usize) -> f64 {
    let mut dmin = if n > 1 { f64::INFINITY } else { 1.0 };
    for a in 0..n {
        for b in a + 1..n {
            let mut s = 0.0;
            for c in 0..d {
                let t = x[a * d + c] - x[b * d + c];
                s += t * t;
            }
            dmin = dmin.min(s.sqrt());
        }
    }
    dmin
}

/// Maximin Latin hypercube of `n` points in `d` dimensions: the best (largest minimum pairwise
/// distance) of `tries` random Latin hypercubes drawn from `rng`.
///
/// Inputs: `n >= 1` points (dimensionless count), `d >= 1` dimensions, `tries >= 1` candidates
/// (the source's default is 50), `rng` a SplitMix64 generator (the source uses
/// `SplitMix64::from_seed_stream(seed, stream)`). Output points are in [0, 1)^d at random
/// positions within their slices; scale to physical ranges as `lo + x * (hi - lo)`.
///
/// Cost: `tries * (d * 2n + n^2 d / 2)` operations. Because candidates are drawn one after
/// another from the same stream, a run with more `tries` starts with the candidates of a run with
/// fewer, so the achieved minimum distance never decreases when `tries` grows (same generator state).
///
/// Failure (`success == false`, empty points): `n`, `d` or `tries` equal to 0.
pub fn lhs_maximin(rng: &mut SplitMix64, n: usize, d: usize, tries: usize) -> LhsResult {
    if n == 0 || d == 0 || tries == 0 {
        return LhsResult {
            success: false,
            message: format!("n, d and tries must be at least 1 (got n={n}, d={d}, tries={tries})"),
            points: Vec::new(),
            n,
            d,
            min_distance: f64::NAN,
            best_try: 0,
            n_candidates: 0,
        };
    }
    let mut best = Vec::new();
    let mut best_dmin = -1.0;
    let mut best_try = 0;
    for t in 0..tries {
        let x = lhs_random(rng, n, d);
        let dmin = min_pairwise_distance(&x, n, d);
        if dmin > best_dmin {
            best = x;
            best_dmin = dmin;
            best_try = t;
        }
    }
    LhsResult {
        success: true,
        message: "ok".to_string(),
        points: best,
        n,
        d,
        min_distance: best_dmin,
        best_try,
        n_candidates: tries,
    }
}
