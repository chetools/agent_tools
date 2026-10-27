//! Dense LU factorization with partial pivoting, and the solve that uses it.
//!
//! Source: Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 3.4.1 (Gaussian elimination
//! with partial pivoting, GEPP) and Algorithms 3.1.1/3.1.2 (triangular solves). Written from the
//! book; no code ported. Reference for tests: `scipy.linalg.lu_factor/lu_solve`, `numpy.linalg.solve`.
//!
//! Generic over [`Real`], so `f64` and [`crate::autodiff::Dual`] entries both work; the pivot is
//! chosen on the plain values (`.val()`), so a Dual solution carries the exact derivative of the
//! solution computed with that pivot sequence.
//!
//! Matrices are dense, row-major, `n x n`, in `Vec`/slices of length `n*n` (dimensionless here;
//! the caller's units pass through: `x` has units of `b / A`). Python twin:
//! `library/python/agent_tools/linalg/lu_dense.py`.

use crate::autodiff::Real;

/// Result of [`lu_factor`].
#[derive(Clone, Debug)]
pub struct LuFactor<T> {
    /// Packed factors, row-major `n*n`: strictly lower part is L (unit diagonal implied), upper
    /// part including the diagonal is U, so that `A[perm] = L U`. Incomplete if `success` is false.
    pub lu: Vec<T>,
    /// Pivot rows, as in LAPACK `getrf` (0-based): at step `k` row `k` was swapped with row `piv[k]`.
    pub piv: Vec<usize>,
    /// Matrix order.
    pub n: usize,
    /// False if a zero or non-finite pivot was met (matrix singular to working precision).
    pub success: bool,
    /// Human-readable status (`"success"`, or the column of the failed pivot).
    pub message: String,
}

/// Result of [`lu_solve`] / [`solve`].
#[derive(Clone, Debug)]
pub struct LuSolution<T> {
    /// Solution vector of length `n`; empty if `success` is false.
    pub x: Vec<T>,
    /// False if the factorization failed.
    pub success: bool,
    /// Human-readable status.
    pub message: String,
}

/// LU factorization with partial pivoting (row interchanges) of the `n x n` row-major matrix `a`.
///
/// Singularity is reported when the largest candidate pivot in a column is exactly zero or not
/// finite; a nearly singular matrix is NOT detected (no condition estimate is made, so check
/// the residual of the solution when conditioning is doubtful). Cost 2n^3/3 flops.
/// Panics if `a.len() != n*n`.
pub fn lu_factor<T: Real>(a: &[T], n: usize) -> LuFactor<T> {
    assert_eq!(a.len(), n * n, "lu_factor: a must have n*n entries");
    let mut lu = a.to_vec();
    let mut piv = Vec::with_capacity(n);
    for k in 0..n {
        // pivot search on the plain values
        let mut p = k;
        let mut best = lu[k * n + k].val().abs();
        for i in k + 1..n {
            let c = lu[i * n + k].val().abs();
            if c > best {
                best = c;
                p = i;
            }
        }
        piv.push(p);
        if !(best > 0.0) || !best.is_finite() {
            return LuFactor {
                lu,
                piv,
                n,
                success: false,
                message: format!("singular matrix: zero or non-finite pivot in column {k}"),
            };
        }
        if p != k {
            for j in 0..n {
                lu.swap(k * n + j, p * n + j);
            }
        }
        let pivot = lu[k * n + k];
        for i in k + 1..n {
            let m = lu[i * n + k] / pivot;
            lu[i * n + k] = m;
            for j in k + 1..n {
                lu[i * n + j] = lu[i * n + j] - m * lu[k * n + j];
            }
        }
    }
    LuFactor { lu, piv, n, success: true, message: "success".to_string() }
}

/// Solve `A x = b` with a factorization from [`lu_factor`]. `b` has length `n`.
/// Returns `success = false` (and empty `x`) if the factorization failed.
/// Panics if `b.len() != n`.
pub fn lu_solve<T: Real>(f: &LuFactor<T>, b: &[T]) -> LuSolution<T> {
    let n = f.n;
    assert_eq!(b.len(), n, "lu_solve: b must have n entries");
    if !f.success {
        return LuSolution { x: Vec::new(), success: false, message: f.message.clone() };
    }
    let mut x = b.to_vec();
    for k in 0..n {
        x.swap(k, f.piv[k]);
    }
    for i in 1..n {
        let mut s = x[i];
        for j in 0..i {
            s = s - f.lu[i * n + j] * x[j];
        }
        x[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s = s - f.lu[i * n + j] * x[j];
        }
        x[i] = s / f.lu[i * n + i];
    }
    LuSolution { x, success: true, message: "success".to_string() }
}

/// Factor and solve `A x = b` in one call (see [`lu_factor`], [`lu_solve`]).
pub fn solve<T: Real>(a: &[T], b: &[T], n: usize) -> LuSolution<T> {
    lu_solve(&lu_factor(a, n), b)
}
