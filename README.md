# agent_tools

> **Status: DRAFT.** Conventions and structure only. No routines have been added or verified yet.

A shared, verified library of the code scientists and engineers use again and again, plus the instructions AI coding agents need to use it and extend it. Any agent that can read a git repository can use it, including Claude, Codex, Cursor, Gemini CLI and Muse.

## Why this exists

Agents can rewrite an ODE integrator, a spline, a regression fit or a property correlation from scratch each time. The bugs, however, live in the details: step-size control, boundary conditions, numerical stability, units, and data sources. This repo holds each routine **once**, checked against reference results, so every project uses the same tested code. It matters most for standalone pages and WASM builds, where importing a whole library (SciPy, NumPy) is too heavy.

## Scope

Any reusable routine a scientist or engineer would otherwise rewrite. Areas grow as routines are added. The list below shows the kinds of areas expected; it is not a commitment.

| Area id | Examples |
|---|---|
| `ode` | Explicit and implicit integrators, event detection, dense output |
| `roots` | Scalar root bracketing, nonlinear systems, continuation |
| `optimize` | Unconstrained, bounded and constrained minimization, least squares |
| `linalg` | Dense and sparse solvers, factorizations, eigenvalues |
| `interpolate` | Splines, monotone interpolation, multidimensional gridded data |
| `quadrature` | Adaptive integration, Gaussian rules |
| `pde` | Finite difference and finite volume stencils, method of lines |
| `signal` | FFT, filtering, smoothing, spectral estimates |
| `stats` | Regression and curve fitting with uncertainties, distributions, design of experiments |
| `random` | Seeded generators, sampling, Monte Carlo helpers |
| `units` | Unit conversion and dimensional checks |
| `properties` | Physical property correlations (thermodynamic, transport), with data provenance |
| `io` | Reading and writing common scientific data formats |

New areas are added through the same pull-request process as routines (see `AGENTS.md`).

## Where to start

| You are… | Read |
|---|---|
| An AI agent working in this repo or using it | [`AGENTS.md`](AGENTS.md) |
| An agent that supports Agent Skills (`SKILL.md`) | [`skills/scientific-routines/SKILL.md`](skills/scientific-routines/SKILL.md) |
| Looking for a routine | [`library/catalog.toml`](library/catalog.toml) |
| Adding a routine or an area | [`skills/scientific-routines/references/promotion-checklist.md`](skills/scientific-routines/references/promotion-checklist.md) |
| Setting up another repo to use this library | [`templates/AGENTS-snippet.md`](templates/AGENTS-snippet.md) |

## Planned layout

The directories below are created as the first routines are added.

```
agent_tools/
├── AGENTS.md                   # rules for every agent (Claude reads it via CLAUDE.md)
├── CLAUDE.md                   # pointer to AGENTS.md
├── skills/
│   └── scientific-routines/    # Agent Skills format: SKILL.md + references/
│       └── references/areas/   # one selection guide per area
├── library/
│   ├── catalog.toml            # machine-readable index of every routine
│   ├── rust/                   # Rust crate, one module per area (f64, wasm32-friendly)
│   ├── python/                 # readable reference implementation of each routine (also the test oracle)
│   ├── examples/               # one minimal working example per routine
│   └── tests/
│       ├── problems/           # benchmark and test-case definitions
│       └── reference_data/     # stored reference outputs (verification needs no network)
├── library-candidates/         # routines written elsewhere, awaiting promotion
└── templates/                  # text to paste into other repos' AGENTS.md / CLAUDE.md
```

Prebuilt `.wasm` modules with JS bindings are planned as release assets, so agents without a Rust toolchain can still use the routines.

## Open items

- License: not yet chosen. Routines ported from other code (SciPy is BSD-3-Clause; other sources vary) must keep their original notices. Check each source's terms before porting. Property data must cite its source and that source's terms.
- `make verify`: not yet written. It is described in `AGENTS.md` as the target behaviour.
