# Area guide: random

Seeded generators whose streams are identical in Rust, Python and JavaScript, for reproducible Monte Carlo, shuffling and annealing in standalone pages. The guidance is general practice; the catalog records what each routine has been tested on and wins on disagreement.

## Scope

Pseudo-random generators with explicit seeds and no global state, and the samplers built directly on them. Statistical distributions and fitting belong in `stats`. Nothing here is cryptographically secure.

## Choosing a routine

| Situation | Routine | Established equivalent | Notes |
|---|---|---|---|
| Reproducible uniforms, normals, permutations, same stream in Rust and Python | `splitmix64` | none (NumPy generators cannot be reproduced in Rust) | 64-bit state. `new(state)` is the published algorithm; `from_seed_stream(seed, stream)` gives independent streams per seed. |
| Same stream in Rust, Python and JavaScript (browser code), small state | `mulberry32` | Ettinger's `mulberry32` | 32-bit state, period 2^32. Matches the JavaScript `Math.imul` version exactly. |

## Pitfalls

- Streams differ between languages: check that both sides use the same constructor (`new(state)` vs `from_seed_stream`), the same seed type (Mulberry32 masks the seed to 32 bits), and consume the same number of draws in the same order. `normal()` uses two uniforms per variate.
- `below(n)` for Mulberry32 is `next_u32 % n`, slightly biased unless `n` divides 2^32; use `random()` for a uniform float when bias matters.
- Do not compare `normal()` bit for bit across languages: it agrees to libm rounding (about 1e-15 relative), while integer outputs and `uniform()` are exact.
- Mulberry32 has modest statistical quality and a short period; use SplitMix64 for large Monte Carlo runs.
