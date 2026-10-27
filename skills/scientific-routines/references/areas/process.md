# Area guide: process

Process-engineering calculations: heat-exchanger design equations, equipment cost correlations and heat-integration targeting. The guidance is general practice; the catalog records what each routine has been tested on and wins on disagreement.

## Scope

Routines specific to chemical-process design that are not physical property correlations (`properties`), root finders (`roots`) or optimisers (`optimize`). Network synthesis by LP/MILP (SciPy or HiGHS dependent) is out of scope for this crate; targeting without a solver is in.

## Choosing a routine

| Situation | Routine | Established equivalent | Notes |
|---|---|---|---|
| Driving force of a counter-current exchanger, with gradients, ΔT1 may equal ΔT2 | `lmtd` | none (plain formula divides by zero at ΔT1 = ΔT2) | Series branch for \|ΔT1-ΔT2\| < 1e-4 mean; NaN for non-positive input. Derivatives are accurate to about 3e-12 relative over \|d\|/m from 1e-9 to 2 (log1p form in the closed branch). |
| Purchase or bare-module cost of a shell-and-tube exchanger from its area, with dC/dA | `hx_capital_cost` | BioSTEAM, IDAES SSLW (Seider), eqcost/CAPCOST (Turton) | Costs in USD of the correlation's index year (Seider CE 567, Turton CEPCI 397); escalate yourself. Seider coefficients unchecked against the book; see the provenance note below. |
| Annualising a capital cost | `hx_capital_cost::capital_recovery_factor` | any annuity formula | Stable for tiny rates. |
| Minimum hot and cold utility, pinch temperature, composite and grand composite curves for given streams | `pinch_targets` | problem table by hand; heat-cascade LP | One hot and one cold utility; no utility temperatures or forbidden matches. |

## Pitfalls

- Mixed units: area in m² for every routine here (Seider's fit is in ft², converted inside); Seider pressure in psig, Turton in barg.
- Cost outside the fitted area range: both correlations clamp below the lower limit (flat cost, zero slope, `clamped_below`) and extrapolate above (`above_range`). An optimiser that shrinks an exchanger below the limit sees no cost gradient there.
- Temperature scale of `pinch_targets`: hot streams keep their temperatures and cold streams are shifted up by the full ΔTmin, so `pinch_hot` is the hot-stream pinch temperature (cold side: minus ΔTmin). Use `t_shifted` for the textbook (±ΔTmin/2) scale.
- A zero-flow boundary at the first or last interval means a threshold problem, not a pinch.
- Seider pressure factor below 100 psig (the stated validity range starts there): values are extrapolated unchanged; `seider_pressure_below_range` flags it.

## Open items for the maintainer: `hx_capital_cost` coefficient provenance

Found in the PR review; no coefficient was changed and no book was consulted.

- Floating-head intercepts: BioSTEAM 12.0310 (CE 567) and IDAES SSLW 11.9052 (CE 500) differ by ln(567/500) = 0.1258, so the costs agree to 4.9e-5 after the cost-index shift (tested). This is not a contradiction.
- Fixed-head slope: 0.9228 (BioSTEAM; a Penn lecture snippet agrees) against 0.8228 (IDAES), giving costs 39-61 % apart. Unconfirmed which is right; the IDAES value is probably a typo.
- U-tube intercepts differ by 0.1658, more than the index shift 0.1258 (4.1 % cost difference). Unresolved.
- Turton K, C and B are confirmed only by a third-party worked example (eqcost), not by the book.
