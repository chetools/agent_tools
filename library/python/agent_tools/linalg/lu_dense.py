"""Dense LU factorization with partial pivoting (Python reference of the Rust `lu_dense`).

Source: Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 3.4.1 (Gaussian elimination
with partial pivoting) and Algorithms 3.1.1/3.1.2 (triangular solves). Written from the book.
Rust twin: library/rust/src/linalg/lu_dense.rs. Reference for tests: scipy.linalg.lu_factor,
numpy.linalg.solve.

Entries may be floats or `agent_tools.autodiff.dual.Dual` (the pivot is chosen on plain values,
so a Dual solution carries the exact derivative). Matrices are flat row-major lists of length
n*n, as in Rust (dimensionless here; x has the units of b / A). Pure Python.
"""
import math
from dataclasses import dataclass, field

from agent_tools.autodiff.dual import val


@dataclass
class LuFactor:
    """Result of `lu_factor`.

    lu: packed factors, flat row-major n*n (strict lower = L with unit diagonal, upper = U),
        incomplete if success is False.
    piv: pivot rows as in LAPACK getrf (0-based): at step k row k was swapped with row piv[k].
    n: matrix order.
    success: False if a zero or non-finite pivot was met.
    message: "success" or the column of the failed pivot.
    """
    lu: list
    piv: list
    n: int
    success: bool
    message: str


@dataclass
class LuSolution:
    """Result of `lu_solve` / `solve`: x (list of length n, empty on failure), success, message."""
    x: list = field(default_factory=list)
    success: bool = True
    message: str = "success"


def lu_factor(a, n):
    """LU factorization with partial pivoting of the n x n flat row-major matrix `a`.

    Singularity is reported when the largest candidate pivot in a column is exactly zero or not
    finite; nearly singular matrices are not detected (check the residual). Raises ValueError if
    len(a) != n*n.
    """
    if len(a) != n * n:
        raise ValueError("lu_factor: a must have n*n entries")
    lu = list(a)
    piv = []
    for k in range(n):
        # pivot search on the plain values
        p, best = k, abs(val(lu[k * n + k]))
        for i in range(k + 1, n):
            c = abs(val(lu[i * n + k]))
            if c > best:
                best, p = c, i
        piv.append(p)
        if not (best > 0.0) or not math.isfinite(best):
            return LuFactor(lu, piv, n, False,
                            f"singular matrix: zero or non-finite pivot in column {k}")
        if p != k:
            for j in range(n):
                lu[k * n + j], lu[p * n + j] = lu[p * n + j], lu[k * n + j]
        pivot = lu[k * n + k]
        for i in range(k + 1, n):
            m = lu[i * n + k] / pivot
            lu[i * n + k] = m
            for j in range(k + 1, n):
                lu[i * n + j] = lu[i * n + j] - m * lu[k * n + j]
    return LuFactor(lu, piv, n, True, "success")


def lu_solve(f, b):
    """Solve A x = b with a factorization from `lu_factor`; b has length n.

    Returns success False (and empty x) if the factorization failed.
    """
    n = f.n
    if len(b) != n:
        raise ValueError("lu_solve: b must have n entries")
    if not f.success:
        return LuSolution([], False, f.message)
    x = list(b)
    for k in range(n):
        x[k], x[f.piv[k]] = x[f.piv[k]], x[k]
    for i in range(1, n):
        s = x[i]
        for j in range(i):
            s = s - f.lu[i * n + j] * x[j]
        x[i] = s
    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s = s - f.lu[i * n + j] * x[j]
        x[i] = s / f.lu[i * n + i]
    return LuSolution(x, True, "success")


def solve(a, b, n):
    """Factor and solve A x = b in one call."""
    return lu_solve(lu_factor(a, n), b)
