# Area guide: process

Process-engineering calculations: heat-exchanger design equations, equipment cost correlations and heat-integration targeting. The guidance is general practice; the catalog records what each routine has been tested on and wins on disagreement.

## Scope

Routines specific to chemical-process design that are not physical property correlations (`properties`), root finders (`roots`) or optimisers (`optimize`). Network synthesis by LP/MILP (SciPy or HiGHS dependent) is out of scope for this crate; targeting without a solver is in.

## Choosing a routine

| Situation | Routine | Established equivalent | Notes |
|---|---|---|---|
| Driving force of a counter-current exchanger, with gradients, ΔT1 may equal ΔT2 | `lmtd` | none (plain formula divides by zero at ΔT1 = ΔT2) | Series branch for \|ΔT1-ΔT2\| < 1e-4 mean; NaN for non-positive input. Derivatives are 5e-9 off just above the switch. |
| Purchase or bare-module cost of a shell-and-tube exchanger from its area, with dC/dA | `hx_capital_cost` | BioSTEAM, IDAES SSLW (Seider), eqcost/CAPCOST (Turton) | Costs in USD of the correlation's index year (Seider CE 567, Turton CEPCI 397); escalate yourself. Seider coefficients unchecked against the book. |
| Annualising a capital cost | `hx_capital_cost::capital_recovery_factor` | any annuity formula | Stable for tiny rates. |
| Minimum hot and cold utility, pinch temperature, composite and grand composite curves for given streams | `pinch_targets` | problem table by hand; heat-cascade LP | One hot and one cold utility; no utility temperatures or forbidden matches. |

## Pitfalls

- Mixed units: area in m² for every routine here (Seider's fit is in ft², converted inside); Seider pressure in psig, Turton in barg.
- Cost outside the fitted area range: both correlations clamp below the lower limit (flat cost, zero slope, `clamped_below`) and extrapolate above (`above_range`). An optimiser that shrinks an exchanger below the limit sees no cost gradient there.
- Temperature scale of `pinch_targets`: hot streams keep their temperatures and cold streams are shifted up by the full ΔTmin, so `pinch_hot` is the hot-stream pinch temperature (cold side: minus ΔTmin). Use `t_shifted` for the textbook (±ΔTmin/2) scale.
- A zero-flow boundary at the first or last interval means a threshold problem, not a pinch.
