# Snippet for other repositories

Paste the section below into the `AGENTS.md` of any repository where agents write scientific or engineering code, for example chetools/claude. For Claude Code, also add a `CLAUDE.md` containing the line `@AGENTS.md`.

---

## Reusable scientific routines: use chetools/agent_tools

This project uses the shared library at https://github.com/chetools/agent_tools.

- Before writing a routine another scientific or engineering project could reuse (solvers, interpolation, quadrature, signal processing, statistics and fitting, units, property correlations, data formats), read `library/catalog.toml` in that repository and follow its `AGENTS.md`.
- If a routine there fits, use it (Rust crate by git dependency, prebuilt wasm module, or Python reference). Do not rewrite it, and do not patch a copy inside this project.
- If no routine fits and the one you need is general, write it for this task, then promote it to chetools/agent_tools by pull request, following its promotion checklist.
- If you cannot push to chetools/agent_tools from this session, put the routine and a filled-in `library-candidate.md` in `library-candidates/<routine-name>/` in this repository, and say so in your final output.
- Problem-specific code (residuals for one model, tuned initial guesses, one-off plots) stays in this project.
