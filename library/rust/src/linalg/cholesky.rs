//! Cholesky factorization of a symmetric positive definite matrix, with the triangular solves.
//!
//! Source: Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 4.2.1 (Cholesky,
//! column-oriented) and Algorithms 3.1.1/3.1.2 (triangular solves). Ported from chetools/claude
//! `sequential-design/solver/src/lib.rs` (`cholesky`, `solve_lower`, `solve_upper_t`; same authors).
//! Reference for tests: `numpy.linalg.cholesky`, `scipy.linalg.cho_solve`.
//!
//! Matrices are dense, row-major, `n x n` (dimensionless here; `x` has units of `b / A`).
//! Right-hand sides are the `cols` columns of an `n x cols` row-major matrix.
//! Python twin: `library/python/agent_tools/linalg/cholesky.py`.

/// Result of [`cholesky`].
#[derive(Clone, Debug)]
pub struct CholeskyResult {
    /// Lower-triangular factor L (row-major `n*n`, upper triangle zero) with `A = L L^T`.
    /// Incomplete if `success` is false.
    pub l: Vec<f64>,
    /// False if `A` is not (numerically) positive definite.
    pub success: bool,
    /// Human-readable status (`"success"`, or the column where the pivot was not positive).
    pub message: String,
}

/// Cholesky factorization `A = L L^T` of the `n x n` row-major matrix `a`.
///
/// Only the lower triangle of `a` is read; symmetry is not checked. Fails (`success = false`)
/// when a diagonal pivot is not strictly positive, or is NaN or infinite, which also happens for a
/// semidefinite or badly conditioned matrix. Panics if `a.len() != n*n`.
pub fn cholesky(a: &[f64], n: usize) -> CholeskyResult {
    assert_eq!(a.len(), n * n, "cholesky: a must have n*n entries");
    let mut l = a.to_vec();
    for j in 0..n {
        let mut s = l[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        if !(s > 0.0 && s.is_finite()) {
            return CholeskyResult {
                l,
                success: false,
                message: format!("matrix is not positive definite: pivot {s:e} in column {j}"),
            };
        }
        let ljj = s.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = l[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
        for i in 0..j {
            l[i * n + j] = 0.0;
        }
    }
    CholeskyResult { l, success: true, message: "success".to_string() }
}

/// Solve `L X = B` in place (forward substitution). `l` is lower triangular `n x n` with nonzero
/// diagonal (a zero diagonal gives inf/NaN); `b` is `n x cols` and is overwritten by `X`.
pub fn solve_lower(l: &[f64], n: usize, b: &mut [f64], cols: usize) {
    for i in 0..n {
        for k in 0..i {
            let lik = l[i * n + k];
            if lik != 0.0 {
                let (head, tail) = b.split_at_mut(i * cols);
                let src = &head[k * cols..k * cols + cols];
                for (t, s) in tail[..cols].iter_mut().zip(src) {
                    *t -= lik * s;
                }
            }
        }
        let lii = l[i * n + i];
        for t in &mut b[i * cols..i * cols + cols] {
            *t /= lii;
        }
    }
}

/// Solve `L^T X = B` in place (back substitution with the transpose of the lower factor `l`).
/// Same layout and failure modes as [`solve_lower`].
pub fn solve_upper_t(l: &[f64], n: usize, b: &mut [f64], cols: usize) {
    for i in (0..n).rev() {
        for k in i + 1..n {
            let lki = l[k * n + i];
            if lki != 0.0 {
                let (head, tail) = b.split_at_mut(k * cols);
                for (t, s) in head[i * cols..i * cols + cols].iter_mut().zip(&tail[..cols]) {
                    *t -= lki * s;
                }
            }
        }
        let lii = l[i * n + i];
        for t in &mut b[i * cols..i * cols + cols] {
            *t /= lii;
        }
    }
}

/// Solve `A X = B` in place given the factor `l` from [`cholesky`] (`L L^T X = B`).
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64], cols: usize) {
    solve_lower(l, n, b, cols);
    solve_upper_t(l, n, b, cols);
}
