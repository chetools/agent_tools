# Snippet for other repositories

Paste the section below into the `AGENTS.md` (or `CLAUDE.md`) of any repository where agents write numerical code, for example chetools/claude.

---

## Numerical routines: use chetools/agent_tools

This project uses the shared numerical library at https://github.com/chetools/agent_tools.

- Before writing an ODE integrator, root solver, optimizer, or small dense linear solver, read `numerics/catalog.toml` in that repository and follow its `AGENTS.md`.
- If a routine there fits, use it (Rust crate by git dependency, prebuilt wasm module, or Python twin). Do not rewrite it, and do not patch a copy inside this project.
- If no routine fits and the one you need is general, write it for this task, then promote it to chetools/agent_tools by pull request, following its promotion checklist.
- If you cannot push to chetools/agent_tools from this session, put the routine and a filled-in `library-candidate.md` in `library-candidates/<routine-name>/` in this repository, and say so in your final output.
- Problem-specific code (residuals for one model, tuned initial guesses) stays in this project.
