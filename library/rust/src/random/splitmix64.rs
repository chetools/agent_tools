//! SplitMix64 generator with uniform, normal (Box-Muller) and Fisher-Yates shuffle helpers.
//!
//! Source: Steele, Lea & Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014;
//! constants as in Vigna's public-domain `splitmix64.c` (prng.di.unimi.it). Ported from
//! chetools/claude `sequential-design/solver/src/lib.rs` (`Rng`). Stream is bit-identical to the
//! Python twin `library/python/agent_tools/random/splitmix64.py` (integers exactly; `uniform`
//! exactly; `normal` to libm rounding, about 1e-15 relative).
//!
//! All outputs are dimensionless. Seeds are explicit; there is no global state.

use std::f64::consts::PI;

/// SplitMix64 pseudo-random generator (64-bit state). Not cryptographically secure.
pub struct SplitMix64 {
    s: u64,
}

impl SplitMix64 {
    /// Generator whose state is exactly `state` (the published algorithm: `state` = seed).
    /// Any `u64` is valid.
    pub fn new(state: u64) -> SplitMix64 {
        SplitMix64 { s: state }
    }

    /// Generator for `(seed, stream)`: state `seed * 1000003 + stream` (wrapping), so that
    /// independent uses of one seed get separate streams. This is the construction used by the
    /// source project.
    pub fn from_seed_stream(seed: u64, stream: u64) -> SplitMix64 {
        SplitMix64 { s: seed.wrapping_mul(1000003).wrapping_add(stream) }
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform on [0, 1): the top 53 bits of `next_u64` times 2^-53 (never returns 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (2.0f64).powi(-53)
    }

    /// Standard normal (mean 0, variance 1) by Box-Muller; uses two uniforms per call and
    /// discards the second variate of the pair.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Random order of `0..n` (Fisher-Yates shuffle, one uniform per swap, `n >= 0`).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.uniform() * (i + 1) as f64) as usize;
            p.swap(i, j);
        }
        p
    }
}
