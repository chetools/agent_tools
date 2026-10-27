# Library candidate: <routine name>

Fill this in when a routine is written in another project but cannot be promoted to chetools/agent_tools from the current session. Save it as `library-candidates/<routine-name>/library-candidate.md` next to the code.

- **Routine:** <name, e.g. Brent's method for scalar roots>
- **Area / problem class:** <area id from the README table, or a proposed new one> / <e.g. scalar_root, monotone_interpolation>
- **Written for:** <project and page, e.g. chetools/claude, packed-bed-runaway>
- **Files:** <paths to Rust and/or Python code in this folder>
- **Algorithm or data source and license:** <reference; license if code or data was ported>
- **What was tested:** <test cases, tolerances, comparison target such as SciPy version and method, or published data>
- **Results:** <error norms, evaluation counts; or "not yet tested">
- **Known limitations:** <anything the next agent should know>
- **Still to do for promotion:** <items from the promotion checklist not yet done>
- **Why it could not be promoted here:** <e.g. no push access to chetools/agent_tools, no network>
