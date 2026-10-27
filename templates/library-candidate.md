# Library candidate: <routine name>

Fill this in when a routine is written in another project but cannot be promoted to chetools/agent_tools from the current session. Save it as `library-candidates/<routine-name>/library-candidate.md` next to the code.

- **Routine:** <name, e.g. Brent's method for scalar roots>
- **Area / problem class:** <ode | roots | optimize | linalg> / <e.g. scalar_root>
- **Written for:** <project and page, e.g. chetools/claude, packed-bed-runaway>
- **Files:** <paths to Rust and/or Python code in this folder>
- **Algorithm source and license:** <reference; license if code was ported>
- **What was tested:** <problems, tolerances, comparison target such as SciPy version and method>
- **Results:** <error norms, evaluation counts; or "not yet tested">
- **Known limitations:** <anything the next agent should know>
- **Still to do for promotion:** <items from the promotion checklist not yet done>
- **Why it could not be promoted here:** <e.g. no push access to chetools/agent_tools, no network>
