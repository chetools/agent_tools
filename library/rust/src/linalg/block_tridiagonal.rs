//! Block-tridiagonal LU (block Thomas algorithm) with a runtime block size, factor once, solve many.
//!
//! Source: Golub & Van Loan, "Matrix Computations", 4th ed., Section 4.5.1 (block tridiagonal
//! systems; block LU without pivoting across blocks), and Press et al., "Numerical Recipes",
//! 3rd ed., Section 2.4 (Thomas algorithm, the scalar case). Generalised from the 2x2-block
//! solver `Factor` of chetools/claude `packed-bed-runaway`; the diagonal-block solves use
//! [`crate::linalg::lu_dense`] (the source inverted each 2x2 block explicitly).
//! Reference for tests: `numpy.linalg.solve` and `scipy.linalg.solve_banded`.
//!
//! The matrix has `nb` block rows of `m x m` blocks:
//!
//! ```text
//! [ D_0  U_0                 ]
//! [ L_0  D_1  U_1            ]
//! [      L_1  D_2  ...       ]
//! [           ...  D_{nb-1}  ]
//! ```
//!
//! `lower[k]` is block (k+1, k) and `upper[k]` is block (k, k+1), so each has `nb-1` blocks.
//! Blocks are row-major `m*m` slices, stored back to back in one flat slice.
//!
//! Algorithm (block LU, `M_0 = D_0`):
//! `C_i = M_i^-1 U_i`, `M_{i+1} = D_{i+1} - L_i C_i`; solve: `y_0 = M_0^-1 b_0`,
//! `y_i = M_i^-1 (b_i - L_{i-1} y_{i-1})`, then `x_{nb-1} = y_{nb-1}`, `x_i = y_i - C_i x_{i+1}`.
//! Every `M_i^-1` is applied through a dense LU of `M_i` (partial pivoting).
//!
//! Pivoting and stability. Pivoting is WITHIN each diagonal block `M_i` only (row swaps inside the
//! block). There is NO pivoting across blocks: the block order is fixed. The method is
//! therefore reliable when the matrix is block diagonally dominant in the sense
//! `||M_i^-1 L_{i-1}|| + ||M_i^-1 U_i|| < 1` (any consistent norm), which holds for example for
//! strictly block-row-diagonally-dominant matrices, symmetric positive definite matrices, and
//! M-matrices such as discretised diffusion operators. Failure modes: (1) a diagonal block `M_i`
//! that is exactly singular makes the factorization fail loudly (`success = false`, the message
//! names the block), even if the whole matrix is nonsingular (for example `[[0,1],[1,0]]` with
//! `m = 1`); (2) a nearly singular `M_i` is NOT detected and silently inflates `C_i`, losing
//! accuracy; [`BlockTridiagFactor::growth`] reports the largest `||C_i||_inf` as a warning
//! indicator (values far above 1 mean the no-cross-block-pivoting assumption is shaky; check the
//! residual). Cost: O(nb m^3) to factor, O(nb m^2) per right-hand side.
//!
//! Generic over [`Real`], so [`crate::autodiff::Dual`] entries propagate derivatives (pivots are
//! chosen on plain values, as in `lu_dense`). Python twin:
//! `library/python/agent_tools/linalg/block_tridiagonal.py`.

use crate::autodiff::Real;
use crate::linalg::lu_dense::{lu_factor, lu_solve, LuFactor};

/// Factorization of a block-tridiagonal matrix, from [`block_tridiag_factor`].
#[derive(Clone, Debug)]
pub struct BlockTridiagFactor<T> {
    /// Block size `m` (rows and columns of each block).
    pub m: usize,
    /// Number of block rows `nb`; the matrix order is `m * nb`.
    pub nb: usize,
    /// Dense LU (partial pivoting) of each eliminated diagonal block `M_i`; fewer than `nb`
    /// entries if `success` is false.
    pub blocks: Vec<LuFactor<T>>,
    /// Modified upper blocks `C_i = M_i^-1 U_i`, `(nb-1)` blocks of `m*m`, row-major, flat.
    pub cp: Vec<T>,
    /// Copy of the lower blocks `L_i`, `(nb-1)` blocks of `m*m`, row-major, flat.
    pub lower: Vec<T>,
    /// Growth indicator: `max_i ||C_i||_inf` (plain values; dimensionless; 0 if `nb == 1`).
    pub growth: f64,
    /// False if a diagonal block was singular (exactly zero or non-finite pivot).
    pub success: bool,
    /// `"success"`, or the index of the failing block and the reason.
    pub message: String,
}

/// Result of [`block_tridiag_solve`].
#[derive(Clone, Debug)]
pub struct BlockTridiagSolution<T> {
    /// Solution of length `m * nb` (units of `b / A`); empty if `success` is false.
    pub x: Vec<T>,
    /// False if the factorization failed.
    pub success: bool,
    /// Human-readable status.
    pub message: String,
}

/// `out = a * b` for row-major `m x m` blocks.
fn matmul<T: Real>(a: &[T], b: &[T], m: usize) -> Vec<T> {
    let mut out = vec![T::cst(0.0); m * m];
    for i in 0..m {
        for k in 0..m {
            let aik = a[i * m + k];
            for j in 0..m {
                out[i * m + j] = out[i * m + j] + aik * b[k * m + j];
            }
        }
    }
    out
}

/// Factor a block-tridiagonal matrix of `nb` block rows with `m x m` blocks (see the module
/// header for the layout). `lower` and `upper` have `(nb-1)*m*m` entries, `diag` has `nb*m*m`.
/// Entries are dimensionless here (the caller's units pass through). Valid for `m >= 1`,
/// `nb >= 1`. Panics on `m == 0`, `nb == 0` or a wrong slice length. Returns `success = false`
/// if a diagonal block `M_i` is exactly singular; see the module header for when that happens.
pub fn block_tridiag_factor<T: Real>(
    lower: &[T],
    diag: &[T],
    upper: &[T],
    m: usize,
    nb: usize,
) -> BlockTridiagFactor<T> {
    assert!(m >= 1 && nb >= 1, "block_tridiag_factor: m and nb must be >= 1");
    let mm = m * m;
    assert_eq!(diag.len(), nb * mm, "block_tridiag_factor: diag must have nb*m*m entries");
    assert_eq!(lower.len(), (nb - 1) * mm, "block_tridiag_factor: lower must have (nb-1)*m*m entries");
    assert_eq!(upper.len(), (nb - 1) * mm, "block_tridiag_factor: upper must have (nb-1)*m*m entries");
    let mut f = BlockTridiagFactor {
        m,
        nb,
        blocks: Vec::with_capacity(nb),
        cp: Vec::with_capacity((nb - 1) * mm),
        lower: lower.to_vec(),
        growth: 0.0,
        success: true,
        message: "success".to_string(),
    };
    for i in 0..nb {
        let mut mi = diag[i * mm..(i + 1) * mm].to_vec();
        if i > 0 {
            // M_i = D_i - L_{i-1} C_{i-1}
            let c = &f.cp[(i - 1) * mm..i * mm];
            let prod = matmul(&lower[(i - 1) * mm..i * mm], c, m);
            for k in 0..mm {
                mi[k] = mi[k] - prod[k];
            }
        }
        let lu = lu_factor(&mi, m);
        if !lu.success {
            f.success = false;
            f.message = format!("diagonal block {i}: {}", lu.message);
            return f;
        }
        if i + 1 < nb {
            // C_i = M_i^-1 U_i, column by column
            let u = &upper[i * mm..(i + 1) * mm];
            let mut c = vec![T::cst(0.0); mm];
            let mut gnorm = 0.0f64;
            for j in 0..m {
                let col: Vec<T> = (0..m).map(|r| u[r * m + j]).collect();
                let s = lu_solve(&lu, &col);
                for r in 0..m {
                    c[r * m + j] = s.x[r];
                }
            }
            for r in 0..m {
                let row: f64 = (0..m).map(|j| c[r * m + j].val().abs()).sum();
                gnorm = gnorm.max(row);
            }
            f.growth = f.growth.max(gnorm);
            f.cp.extend_from_slice(&c);
        }
        f.blocks.push(lu);
    }
    f
}

/// Solve `A x = b` with a factorization from [`block_tridiag_factor`]; `b` has `m*nb` entries.
/// Call repeatedly for many right-hand sides; the factorization is not modified.
/// Returns `success = false` (empty `x`) if the factorization failed. Panics on a wrong length.
pub fn block_tridiag_solve<T: Real>(f: &BlockTridiagFactor<T>, b: &[T]) -> BlockTridiagSolution<T> {
    let (m, nb) = (f.m, f.nb);
    let mm = m * m;
    assert_eq!(b.len(), m * nb, "block_tridiag_solve: b must have m*nb entries");
    if !f.success {
        return BlockTridiagSolution { x: Vec::new(), success: false, message: f.message.clone() };
    }
    let mut x = b.to_vec();
    // forward sweep: y_i = M_i^-1 (b_i - L_{i-1} y_{i-1})
    for i in 0..nb {
        let mut v: Vec<T> = x[i * m..(i + 1) * m].to_vec();
        if i > 0 {
            let l = &f.lower[(i - 1) * mm..i * mm];
            for r in 0..m {
                let mut s = v[r];
                for k in 0..m {
                    s = s - l[r * m + k] * x[(i - 1) * m + k];
                }
                v[r] = s;
            }
        }
        let y = lu_solve(&f.blocks[i], &v).x;
        x[i * m..(i + 1) * m].copy_from_slice(&y);
    }
    // back substitution: x_i = y_i - C_i x_{i+1}
    for i in (0..nb.saturating_sub(1)).rev() {
        let c = &f.cp[i * mm..(i + 1) * mm];
        for r in 0..m {
            let mut s = x[i * m + r];
            for k in 0..m {
                s = s - c[r * m + k] * x[(i + 1) * m + k];
            }
            x[i * m + r] = s;
        }
    }
    BlockTridiagSolution { x, success: true, message: "success".to_string() }
}
