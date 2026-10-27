"""Cholesky factorization and triangular solves (Python reference of the Rust `cholesky`).

Source: Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 4.2.1 (Cholesky) and
Algorithms 3.1.1/3.1.2 (triangular solves); ported from chetools/claude `sequential-design`.
Rust twin: library/rust/src/linalg/cholesky.rs. Reference for tests: numpy.linalg.cholesky,
scipy.linalg.cho_solve.

Matrices are n x n NumPy arrays (dimensionless here; x has the units of b / A). Right-hand sides
are vectors (n,) or matrices (n, cols). Same loop order as Rust.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CholeskyResult:
    """Result of `cholesky`.

    l: lower-triangular factor (n x n, upper triangle zero) with A = L L^T; incomplete if
       success is False.
    success: False if A is not (numerically) positive definite.
    message: "success" or the column where the pivot was not positive.
    """
    l: np.ndarray
    success: bool
    message: str


def cholesky(a):
    """Cholesky factorization A = L L^T of the square array `a`.

    Only the lower triangle is read; symmetry is not checked. Fails (success False) when a
    diagonal pivot is not strictly positive, or is NaN or infinite (also for semidefinite or badly
    conditioned matrices).
    """
    l = np.array(a, dtype=float)
    if l.ndim != 2 or l.shape[0] != l.shape[1]:
        raise ValueError("cholesky: a must be a square 2-D array")
    n = l.shape[0]
    for j in range(n):
        s = l[j, j]
        for k in range(j):
            s -= l[j, k] * l[j, k]
        if not (s > 0.0 and np.isfinite(s)):
            return CholeskyResult(l, False,
                                  f"matrix is not positive definite: pivot {s:e} in column {j}")
        ljj = np.sqrt(s)
        l[j, j] = ljj
        for i in range(j + 1, n):
            s = l[i, j]
            for k in range(j):
                s -= l[i, k] * l[j, k]
            l[i, j] = s / ljj
        for i in range(j):
            l[i, j] = 0.0
    return CholeskyResult(l, True, "success")


def solve_lower(l, b):
    """Solve L X = B by forward substitution; returns X (new array). L must have a nonzero diagonal."""
    x = np.array(b, dtype=float)
    n = l.shape[0]
    for i in range(n):
        for k in range(i):
            if l[i, k] != 0.0:
                x[i] -= l[i, k] * x[k]
        x[i] /= l[i, i]
    return x


def solve_upper_t(l, b):
    """Solve L^T X = B by back substitution with the lower factor L; returns X (new array)."""
    x = np.array(b, dtype=float)
    n = l.shape[0]
    for i in range(n - 1, -1, -1):
        for k in range(i + 1, n):
            if l[k, i] != 0.0:
                x[i] -= l[k, i] * x[k]
        x[i] /= l[i, i]
    return x


def cholesky_solve(l, b):
    """Solve A X = B given the factor L from `cholesky` (L L^T X = B); returns X."""
    return solve_upper_t(l, solve_lower(l, b))
