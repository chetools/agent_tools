# Changelog

All notable changes to the routines in this library. Versions follow semantic versioning.

## Unreleased

- New area `gpu`: WebGPU (WGSL) kernels in `f32` with JS host wrappers, float64 Python references and a headless-Chromium test harness (`make verify`). Rule 7 gains the `f32` exception with per-routine tolerances.
- Added, all `experimental`, version 0.1.0: `multigrid_poisson_cc`, `workgroup_reduce`, `histogram_range`, `marching_cubes`, `particle_advect_rk2` (extracted from chetools/claude `cylinder-cavity-3d`).
- Draft: agent instructions, skill, catalog schema and templates. No routines yet.
- Scope broadened from numerical solvers to general scientific and engineering routines: `numerics/` renamed `library/`, skill renamed `scientific-routines`, area guides added.
