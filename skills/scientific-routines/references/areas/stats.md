# Area guide: stats

Regression and curve fitting with uncertainties, distributions, and design of experiments. The guidance is general practice; each routine's catalog entry records what it has actually been tested on and wins on disagreement.

## Scope

Statistical estimation, fitting and experimental design. Seeded generators and low-level sampling belong in `random`; linear-algebra kernels (Cholesky, LU) in `linalg`.

## Choosing a routine

| Situation | Method / routine | Established equivalent | Notes |
|---|---|---|---|
| Space-filling initial design in [0,1)^d, reproducible and identical in Rust and Python | `lhs_maximin` | `scipy.stats.qmc.LatinHypercube` | Best of k random Latin hypercubes, maximin distance criterion. Random position inside each slice. Weak in high dimension (d = 10, n = 50: min distance about 0.63 at k = 50); pass a `SplitMix64`. |

## Pitfalls

- A Latin hypercube guarantees one point per slice in each single input, not good coverage of the joint space: check the minimum distance (or discrepancy) of the design you get.
- Scale unit-cube designs to physical ranges yourself; if inputs have very different scales, space-filling distance in the unit cube is the meaningful measure, not distance in physical units.
