"""Classical fixed-step Runge-Kutta (RK4) step with its exact step Jacobian d y1 / d y0, for floats
and `agent_tools.autodiff.dual.Dual`.

Source: classical RK4 (Hairer, Norsett and Wanner, "Solving Ordinary Differential Equations I",
2nd ed. (Springer 1993), section II.1); the step Jacobian is the derivative of the discrete map,
obtained by the chain rule through the stage equations (the variational equation discretised by
the same four stages; "internal differentiation"). Ported from chetools/muse
`twin-cstr/solver/src/model.rs` (`rk4_step`) and `twin-cstr/python/twin_mhe.py` (`_rk4_batch`),
where it linearised a 4-state CSTR model for an extended Kalman filter / moving-horizon estimator.
Here the system is any n, f and its Jacobian are caller functions, the Jacobian is dense row-major.
Rust twin: library/rust/src/ode/rk4_step_jacobian.rs (same algorithm, same operation order).

Stages for y' = f(t, y), step h:

    k1 = f(t, y)            k2 = f(t + h/2, y + h/2 k1)
    k3 = f(t + h/2, y + h/2 k2)    k4 = f(t + h, y + h k3)        y1 = y + h/6 (k1 + 2 k2 + 2 k3 + k4)

With Gi = df/dy at stage i and A2 = I + h/2 G1, A3 = I + h/2 G2 A2, A4 = I + h G3 A3:

    Phi = I + h/6 (G1 + 2 G2 A2 + 2 G3 A3 + G4 A4)

This is exactly d y1 / d y of the discrete step (it differs from the flow's sensitivity matrix by the
O(h^5) local truncation error). Over several substeps the factors multiply: Phi = Phi_m ... Phi_1.

Pure Python: y is a list of floats or Duals; matrices are flat row-major lists (as in lu_dense).
t and h are plain floats.
"""
import math
from dataclasses import dataclass


@dataclass
class Rk4StepResult:
    """Result of `rk4_step_jacobian` (and, without phi, of `rk4_step`).

    y1: state after the step, list of n floats/Duals [units of y] (on failure a copy of y0);
    phi: step Jacobian d y1_i / d y0_k at phi[i*n + k], flat row-major n*n [dimensionless when the
    components of y share units, else units of y_i / y_k] (on failure the identity; None for rk4_step);
    n: number of components; success / message: "success" or the reason the arguments were rejected;
    nfev: calls of f (4 n_sub, 0 on failure); njac: calls of jac (4 n_sub, 0 on failure).
    """
    y1: list
    phi: list
    n: int
    success: bool = False
    message: str = ""
    nfev: int = 0
    njac: int = 0


def _check(n, t0, h, n_sub):
    if n == 0:
        return "empty initial state"
    if n_sub < 1:
        return "n_sub must be >= 1"
    if not (math.isfinite(t0) and math.isfinite(h)):
        return "t0 and h must be finite"
    return None


def _matmul(a, b, n):
    """a * b for flat row-major n x n matrices."""
    out = [None] * (n * n)
    for i in range(n):
        for k in range(n):
            s = a[i * n] * b[k]
            for j in range(1, n):
                s = s + a[i * n + j] * b[j * n + k]
            out[i * n + k] = s
    return out


def _eye(n):
    return [1.0 if i == k else 0.0 for i in range(n) for k in range(n)]


def rk4_step(f, t0, y0, h, n_sub):
    """One classical RK4 step of size h split into n_sub equal substeps (state only, no Jacobian).

    f(t, y, dydt) writes dy/dt (list of n) [units of y per unit of t]; t0 start time and h the step
    over the whole interval [units of t] (h may be negative); y0 initial state (n >= 1 floats or
    Duals); n_sub >= 1 number of substeps (no default; 1 = a single RK4 step), substep h / n_sub.
    Invalid arguments (n = 0, n_sub < 1, non-finite t0 or h) are reported in the result, never
    raised. Non-finite values from f propagate. State is bit-identical to `rk4_step_jacobian`.
    Returns Rk4StepResult with phi = None.
    """
    n = len(y0)
    msg = _check(n, t0, h, n_sub)
    if msg:
        return Rk4StepResult(list(y0), None, n, False, msg)
    hs = h / n_sub
    y = list(y0)
    k1, k2, k3, k4 = ([0.0] * n for _ in range(4))
    for s in range(n_sub):
        t = t0 + s * hs
        f(t, y, k1)
        f(t + 0.5 * hs, [y[i] + 0.5 * hs * k1[i] for i in range(n)], k2)
        f(t + 0.5 * hs, [y[i] + 0.5 * hs * k2[i] for i in range(n)], k3)
        f(t + hs, [y[i] + hs * k3[i] for i in range(n)], k4)
        y = [y[i] + (hs / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(n)]
    return Rk4StepResult(y, None, n, True, "success", nfev=4 * n_sub)


def rk4_step_jacobian(f, jac, t0, y0, h, n_sub):
    """One classical RK4 step of size h (n_sub equal substeps) and its exact step Jacobian
    Phi = d y1 / d y0, propagated through the stages (see the module docstring).

    f(t, y, dydt) writes dy/dt (list of n). jac(t, y, j) writes the dense Jacobian
    j[i*n + k] = d f_i / d y_k (flat row-major n*n list) [1 / unit of t when components share
    units]; it is called at the four stage states. For Dual states jac must return Duals (same
    code as f). A jac inconsistent with f gives a Phi that is not the derivative of the step.
    t0, h, y0, n_sub as in `rk4_step`. phi is the product of the substep Jacobians.
    Invalid arguments are reported in the result, never raised.
    """
    n = len(y0)
    nn = n * n
    msg = _check(n, t0, h, n_sub)
    if msg:
        return Rk4StepResult(list(y0), _eye(n), n, False, msg)
    eye = _eye(n)
    hs = h / n_sub
    y = list(y0)
    phi = list(eye)
    k1, k2, k3, k4 = ([0.0] * n for _ in range(4))
    g1, g2, g3, g4 = ([0.0] * nn for _ in range(4))
    for s in range(n_sub):
        t = t0 + s * hs
        f(t, y, k1)
        jac(t, y, g1)
        a2 = [eye[i] + 0.5 * hs * g1[i] for i in range(nn)]
        x = [y[i] + 0.5 * hs * k1[i] for i in range(n)]
        f(t + 0.5 * hs, x, k2)
        jac(t + 0.5 * hs, x, g2)
        g2a2 = _matmul(g2, a2, n)
        a3 = [eye[i] + 0.5 * hs * g2a2[i] for i in range(nn)]
        x = [y[i] + 0.5 * hs * k2[i] for i in range(n)]
        f(t + 0.5 * hs, x, k3)
        jac(t + 0.5 * hs, x, g3)
        g3a3 = _matmul(g3, a3, n)
        a4 = [eye[i] + hs * g3a3[i] for i in range(nn)]
        x = [y[i] + hs * k3[i] for i in range(n)]
        f(t + hs, x, k4)
        jac(t + hs, x, g4)
        g4a4 = _matmul(g4, a4, n)
        y = [y[i] + (hs / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(n)]
        astep = [eye[i] + (hs / 6.0) * (g1[i] + 2.0 * g2a2[i] + 2.0 * g3a3[i] + g4a4[i]) for i in range(nn)]
        phi = _matmul(astep, phi, n)
    return Rk4StepResult(y, phi, n, True, "success", nfev=4 * n_sub, njac=4 * n_sub)
