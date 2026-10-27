//! Mulberry32: a small 32-bit generator, bit-identical in Rust, Python and JavaScript.
//!
//! Source: Tommy Ettinger's public-domain `mulberry32` (JavaScript, `Math.imul` version, 2017).
//! Ported from chetools/claude `heat-exchanger-network/solver/src/lib.rs` (`Mulberry32`).
//! Python twin: `library/python/agent_tools/random/mulberry32.py`.
//!
//! Outputs are dimensionless. Seeds are explicit; there is no global state. Period 2^32; not
//! cryptographically secure and of modest statistical quality (fine for annealing moves and
//! test fixtures).

/// Mulberry32 pseudo-random generator (32-bit state).
pub struct Mulberry32 {
    a: u32,
}

impl Mulberry32 {
    /// Generator with state `seed` (any `u32`; the Python version masks to 32 bits).
    pub fn new(seed: u32) -> Self {
        Mulberry32 { a: seed }
    }

    /// Next 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        self.a = self.a.wrapping_add(0x6D2B79F5);
        let mut t = self.a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    /// Uniform on [0, 1): `next_u32 / 2^32` (exact in f64, never returns 1).
    pub fn random(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0
    }

    /// Integer in `0..n` as `next_u32 % n` (slightly biased unless `n` divides 2^32).
    /// `n` must be in `1..=u32::MAX`; `n = 0` panics.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n >= 1 && n <= u32::MAX as usize, "mulberry32::below: n must be in 1..=u32::MAX");
        (self.next_u32() % n as u32) as usize
    }
}
