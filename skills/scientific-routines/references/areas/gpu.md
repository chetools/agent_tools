# Area guide: gpu

The guidance here is general practice unless marked otherwise. Each routine's catalog entry records what it has actually been tested on; where the two disagree, the catalog wins.

## Scope

WebGPU compute kernels (WGSL) for browser pages, each with a JS host wrapper in `library/js/` and a float64 Python reference in `library/python/gpu/`. Kernels run in `f32`; every routine states its tolerance against the reference (AGENTS.md section 7). Rendering code (shaders that draw, colour maps) does not belong here. A CPU algorithm without a GPU kernel belongs in its own area.

## Choosing a routine

| Situation | Method / routine | Established equivalent (e.g. SciPy, NumPy) | Notes |
|---|---|---|---|
| Poisson or pressure solve on a uniform 3D box grid (x, y zero-flux walls; z periodic or zero-flux walls) | `multigrid_poisson_cc`: V-cycle multigrid, red-black Gauss-Seidel smoother | direct sparse solve; pyamg or HYPRE geometric multigrid | N power of two >= 4, NZ power of two >= 1. Reports residual RMS per cycle, about 0.14 reduction per V-cycle in 3D. Constant null space removed. |
| Sum, max, min or max-abs of an f32 GPU array of any length | `workgroup_reduce`: two passes, 1024 elements per workgroup | `numpy.sum`, `max`, `min`; `math.fsum` | Sum order differs from numpy: compare relative to sum(\|x\|). max/min/maxabs are exact. |
| Colour-scale range robust to outliers and masked regions | `histogram_range`: max \|x\|, sqrt-scale histogram, percentile, optional mask | `numpy.histogram` plus `numpy.quantile(method="inverted_cdf")` | hi is the upper edge of the percentile's bin (error at most 2*max/nbins). One huge outlier coarsens the bulk. |
| Triangle mesh of an isosurface of a gridded 3D scalar field | `marching_cubes`: classic 256-case table, one thread per cell, atomic append, indirect draw args | `skimage.measure.marching_cubes` | Triangle order is nondeterministic. Ambiguous cells resolve differently from skimage's default. Open at the grid boundary. Check `overflow` after every run. |
| Advect many massless particles through a gridded 3D velocity field | `particle_advect_rk2`: RK2 midpoint, trilinear between cell centres with odd-mirror wall ghost | `scipy.integrate.solve_ivp` with `RegularGridInterpolator` | Out-of-domain particles are clamped and flagged, not re-seeded. Keep a substep at 0.5 h or less (`substepCount`). |

## Pitfalls

- **Only SwiftShader so far.** All tolerances were measured on the SwiftShader software adapter in headless Chromium. Hardware GPUs may fuse multiply-adds differently; re-run `GPU_ADAPTER=default node library/tests/gpu/run.mjs` on the target hardware before relying on the tolerances there.
- **Silent zeros.** A missing bind group entry or a compilation error can leave output buffers at zero. The wrappers use `gpu_util.js`, which throws on compilation and validation errors; keep that when calling kernels by hand.
- **Multigrid:** the problem is singular (Neumann/periodic); a nonzero mean of b stalls the residual unless mean removal is on (the default). The f32 residual floor is about 1e-6 of \|b\|, so a tight `rtol` may never be met.
- **Reductions:** a sum close to zero has a large error relative to itself; judge it relative to sum(\|x\|). WGSL has no infinity literal, so max/min start from ±3.4028234e38.
- **Marching cubes:** if `overflow` is set the stored triangles are an arbitrary subset; raise `maxTris` to at least the reported count.
- **Particle advection:** the velocity layout is [c][k][j][i]; swapping nx and nz does not raise an error. Compare positions in periodic z with wrap-around (z = 0 and z = Lz are one point).
