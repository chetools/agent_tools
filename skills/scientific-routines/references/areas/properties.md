# Area guide: properties

Physical property and transport correlations with data provenance. The guidance is general practice; the catalog records what each routine has been tested on and wins on disagreement.

## Scope

Closed-form correlations and tabulated property data (thermodynamic, transport, porous-media), SI units, with the source, valid range and out-of-range behaviour documented. Numerical solvers belong in `numerical-solvers.md`; unit conversion in `units`.

## Choosing a routine

| Situation | Routine | Established equivalent | Notes |
|---|---|---|---|
| Pressure drop of a packed bed of particles, laminar to turbulent | `porous_permeability::ergun_gradient` | none | Spheres/granular, superficial velocity. Viscous = inertial at Re_p = 150/1.75 = 85.7. |
| Permeability of a granular bed from porosity and specific surface | `porous_permeability::kozeny_carman` | none | Pass `a_s` (6/d spheres, 4/d long cylinders). K = 5 default; K = 150/36 matches Ergun's viscous term for spheres. |
| Kozeny constant from a measured permeability | `porous_permeability::fit_kozeny_constant` | `scipy.optimize.curve_fit` | Exact inversion per measurement; for noisy data average the results or fit separately. |
| Transverse permeability of a square or hexagonal cylinder (fibre) array, dense | `gebart_square`, `gebart_hexagonal` | none | Lubrication model; valid near close packing. |
| Transverse permeability of a square cylinder array, dilute | `sangani_acrivos_square` | none | Series truncated at O(phi^4); first two terms are Hasimoto. |

## Pitfalls

- NaN means the input is outside the mathematical domain (porosity not in (0,1), solid fraction above the close-packing limit, negative velocity in Ergun). NaN is not returned for a poor-accuracy regime: check the recommended range in the docs yourself (Gebart for dilute arrays, Sangani-Acrivos for dense arrays, Ergun outside granular beds).
- Use SI units everywhere (m, m^2, Pa s, kg/m^3, m/s); `d` is the particle or cylinder diameter, not the radius.
- The Kozeny-Carman `a_s` is surface per SOLID volume, not per bed volume.
