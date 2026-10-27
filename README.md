# agent_tools

> **Status: DRAFT.** Conventions and structure only. No routines have been added or verified yet.

A shared, verified library of numerical routines for ODE integration, root solving and optimization, plus the instructions AI coding agents need to use it and extend it. Any agent that can read a git repository can use it, including Claude, Codex, Cursor, Gemini CLI and Muse.

## Why this exists

Numerical routines such as SciPy's `solve_ivp`, `root` and `minimize` are often needed inside standalone pages and WASM builds, where importing a whole library is too heavy. Agents can rewrite these routines from scratch each time. The bugs, however, live in the details: step-size control, error estimation, Jacobian reuse and Newton convergence tests. This repo holds each routine **once**, checked against reference results, so every project uses the same tested code.

## Where to start

| You are… | Read |
|---|---|
| An AI agent working in this repo or using it | [`AGENTS.md`](AGENTS.md) |
| An agent that supports Agent Skills (`SKILL.md`) | [`skills/numerical-routines/SKILL.md`](skills/numerical-routines/SKILL.md) |
| Looking for a routine | [`numerics/catalog.toml`](numerics/catalog.toml) |
| Adding a routine | [`skills/numerical-routines/references/promotion-checklist.md`](skills/numerical-routines/references/promotion-checklist.md) |
| Setting up another repo to use this library | [`templates/AGENTS-snippet.md`](templates/AGENTS-snippet.md) |

## Planned layout

The directories below are created as the first routines are added.

```
agent_tools/
├── AGENTS.md                 # rules for every agent (Claude reads it via CLAUDE.md)
├── CLAUDE.md                 # pointer to AGENTS.md
├── skills/
│   └── numerical-routines/   # Agent Skills format: SKILL.md + references/
├── numerics/
│   ├── catalog.toml          # machine-readable index of every routine
│   ├── rust/                 # Rust crate (f64, closure-based, wasm32-friendly)
│   ├── python/               # readable reference twin of each routine (also the test oracle)
│   ├── examples/             # one minimal working example per routine
│   └── tests/
│       ├── problems/         # benchmark problem definitions
│       └── reference_data/   # stored reference outputs (verification needs no SciPy or network)
├── library-candidates/       # routines written elsewhere, awaiting promotion
└── templates/                # text to paste into other repos' AGENTS.md / CLAUDE.md
```

Prebuilt `.wasm` modules with JS bindings are planned as release assets, so agents without a Rust toolchain can still use the routines.

## Open items

- License: not yet chosen. Routines ported from SciPy (BSD-3-Clause) or from Hairer–Wanner Fortran codes must keep their original notices. Check each source's terms before porting.
- `make verify`: not yet written. It is described in `AGENTS.md` as the target behaviour.
