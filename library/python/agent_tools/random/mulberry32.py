"""Mulberry32: small 32-bit generator, bit-identical in Python, Rust and JavaScript.

Source: Tommy Ettinger's public-domain mulberry32 (JavaScript, Math.imul version, 2017); ported
from chetools/claude `heat-exchanger-network/python/hen.py`. Rust twin:
library/rust/src/random/mulberry32.rs. Outputs are dimensionless; seeds are explicit; no global
state. Period 2^32, modest statistical quality, not cryptographically secure.
"""

M32 = 0xFFFFFFFF


class Mulberry32:
    """Mulberry32 generator (32-bit state)."""

    def __init__(self, seed):
        """State is `seed` masked to 32 bits."""
        self.a = seed & M32

    def next_u32(self):
        """Next 32-bit output (Python int)."""
        self.a = (self.a + 0x6D2B79F5) & M32
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return (t ^ (t >> 14)) & M32

    def random(self):
        """Uniform on [0, 1): next_u32 / 2^32 (exact, never returns 1)."""
        return self.next_u32() / 4294967296.0

    def below(self, n):
        """Integer in 0..n-1 as next_u32 % n (slightly biased unless n divides 2^32); 1 <= n <= 2^32 - 1."""
        if not 1 <= n <= 0xFFFFFFFF:
            raise ValueError("below: n must be in 1..2**32-1")
        return self.next_u32() % n
