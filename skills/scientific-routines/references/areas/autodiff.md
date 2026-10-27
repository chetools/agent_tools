# Area guide: autodiff

The guidance here is general practice unless marked otherwise. Each routine's catalog entry records what it has actually been tested on; where the two disagree, the catalog wins.

## Scope

Derivatives of user code by automatic differentiation: forward-mode dual numbers and the `Real` trait that lets numerical routines be written once for `f64` and for derivative-carrying numbers. Finite-difference helpers and symbolic differentiation are not here.

## Choosing a routine

| Situation | Method / routine | Established equivalent (e.g. SciPy, NumPy) | Notes |
|---|---|---|---|
| Derivative of a scalar function of one or a few inputs, or sensitivities through a solver written generically over `Real` | `dual` (forward mode, `Dual<f64>`) | `jax.jvp`, autograd forward mode (orientation only) | One extra evaluation per input direction. Many inputs, one output: forward mode is costly; no reverse mode exists in the library. |
| Second derivative of a scalar function | `dual` with nested duals, `Dual<Dual<f64>>` | `jax.jvp` of `jax.jvp` | Seed `Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0))`; the second derivative is `result.d.d`. |
| Sensitivity of a linear solve `A(p) x = b(p)` | `lu_dense` (linalg) with Dual entries | `-A^{-1}(dA x - db)` | The pivot is chosen on values, so the derivative is exact for that pivot sequence. |

## Pitfalls

Symptoms and fixes specific to this area. General ones are in `../troubleshooting.md`.

- Derivative is NaN or infinite at `sqrt(0)`, `ln(0)` or a division by zero: the function itself is singular there; move the evaluation point.
- Derivative of `abs` at exactly 0 is the right-hand derivative (the argument's derivative is passed through); do not rely on it at kinks.
- Control flow that depends on values (`if x > 0`) is differentiated along the branch taken; derivatives of the branch condition are not seen.
- Comparing AD with finite differences: use central differences with step about 1e-5 (1e-8 for first derivatives of well-scaled functions in f64); a smaller step is dominated by round-off.
