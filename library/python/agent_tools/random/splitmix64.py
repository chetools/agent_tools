"""SplitMix64 with uniform, normal (Box-Muller) and Fisher-Yates helpers (Python twin of Rust).

Source: Steele, Lea & Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014;
constants as in Vigna's public-domain splitmix64.c (prng.di.unimi.it). Ported from chetools/claude
`sequential-design/python/sdl.py` (`Rng`). Rust twin: library/rust/src/random/splitmix64.rs.
Integer outputs and `uniform` are bit-identical to Rust; `normal` agrees to libm rounding
(about 1e-15 relative). All outputs are dimensionless; seeds are explicit; no global state.
"""
import math

M64 = (1 << 64) - 1


class SplitMix64:
    """SplitMix64 generator (64-bit state). Not cryptographically secure."""

    def __init__(self, state):
        """State is exactly `state` (the published algorithm: state = seed), masked to 64 bits."""
        self.s = state & M64

    @classmethod
    def from_seed_stream(cls, seed, stream=0):
        """Generator with state seed * 1000003 + stream (mod 2^64): separate streams per seed."""
        return cls((seed * 1000003 + stream) & M64)

    def next_u64(self):
        """Next 64-bit output (Python int)."""
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uniform(self):
        """Uniform on [0, 1): top 53 bits of next_u64 times 2^-53 (never returns 1)."""
        return (self.next_u64() >> 11) * 2.0 ** -53

    def normal(self):
        """Standard normal by Box-Muller; two uniforms per call, second variate discarded."""
        u1, u2 = self.uniform(), self.uniform()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

    def permutation(self, n):
        """Random order of 0..n-1 (Fisher-Yates, one uniform per swap)."""
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(self.uniform() * (i + 1))
            p[i], p[j] = p[j], p[i]
        return p
