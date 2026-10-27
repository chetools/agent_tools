"""Block-tridiagonal LU (block Thomas) with a runtime block size (Python reference of the Rust
`block_tridiagonal`).

Source: Golub & Van Loan, "Matrix Computations", 4th ed., Section 4.5.1 (block tridiagonal
systems; block LU without pivoting across blocks), and Press et al., "Numerical Recipes", 3rd
ed., Section 2.4 (Thomas algorithm, the scalar case). Generalised from the 2x2-block solver
`Factor` of chetools/claude `packed-bed-runaway`; the diagonal-block solves use
agent_tools.linalg.lu_dense. Rust twin: library/rust/src/linalg/block_tridiagonal.rs.
Reference for tests: numpy.linalg.solve, scipy.linalg.solve_banded.

The matrix has nb block rows of m x m blocks: diagonal D_0..D_{nb-1}, lower L_k = block (k+1, k)
and upper U_k = block (k, k+1), k = 0..nb-2. Blocks are row-major flat lists of m*m, stored back
to back in one flat list (diag: nb*m*m entries; lower, upper: (nb-1)*m*m). Entries are floats or
`Dual`; they are dimensionless here (x has the units of b / A).

Algorithm (M_0 = D_0): C_i = M_i^-1 U_i, M_{i+1} = D_{i+1} - L_i C_i; solve y_0 = M_0^-1 b_0,
y_i = M_i^-1 (b_i - L_{i-1} y_{i-1}), then x_{nb-1} = y_{nb-1}, x_i = y_i - C_i x_{i+1}. Every
M_i^-1 is applied through a dense LU of M_i (partial pivoting).

Pivoting and stability: pivoting is WITHIN each diagonal block M_i only. There is NO pivoting
across blocks. The method is reliable for block diagonally dominant matrices,
||M_i^-1 L_{i-1}|| + ||M_i^-1 U_i|| < 1 (also SPD matrices and M-matrices such as discretised
diffusion). Failure modes: an exactly singular M_i fails loudly (success False, message names the
block) even when the whole matrix is nonsingular, e.g. [[0,1],[1,0]] with m = 1; a nearly
singular M_i is NOT detected and silently inflates C_i. `growth` = max_i ||C_i||_inf is the
warning indicator (far above 1: check the residual). Cost O(nb m^3) to factor, O(nb m^2) per
right-hand side. Pure Python.
"""
from dataclasses import dataclass, field

from agent_tools.autodiff.dual import val
from agent_tools.linalg.lu_dense import lu_factor, lu_solve


@dataclass
class BlockTridiagFactor:
    """Result of `block_tridiag_factor`.

    m: block size; nb: number of block rows (matrix order m*nb).
    blocks: dense LU (lu_dense.LuFactor, partial pivoting) of each eliminated diagonal block M_i
        (fewer than nb if success is False).
    cp: modified upper blocks C_i = M_i^-1 U_i, flat (nb-1)*m*m.
    lower: copy of the lower blocks, flat (nb-1)*m*m.
    growth: max_i ||C_i||_inf on plain values (dimensionless; 0 if nb == 1).
    success: False if a diagonal block was singular. message: "success" or the failing block.
    """
    m: int
    nb: int
    blocks: list
    cp: list
    lower: list
    growth: float
    success: bool
    message: str


@dataclass
class BlockTridiagSolution:
    """Result of `block_tridiag_solve`: x (length m*nb, empty on failure), success, message."""
    x: list = field(default_factory=list)
    success: bool = True
    message: str = "success"


def _matmul(a, b, m):
    out = [0.0] * (m * m)
    for i in range(m):
        for k in range(m):
            aik = a[i * m + k]
            for j in range(m):
                out[i * m + j] = out[i * m + j] + aik * b[k * m + j]
    return out


def block_tridiag_factor(lower, diag, upper, m, nb):
    """Factor a block-tridiagonal matrix (layout in the module docstring).

    Valid for m >= 1, nb >= 1; raises ValueError otherwise or on a wrong list length. Returns
    success False if a diagonal block M_i is exactly singular (zero or non-finite pivot).
    """
    if m < 1 or nb < 1:
        raise ValueError("block_tridiag_factor: m and nb must be >= 1")
    mm = m * m
    if len(diag) != nb * mm or len(lower) != (nb - 1) * mm or len(upper) != (nb - 1) * mm:
        raise ValueError("block_tridiag_factor: diag needs nb*m*m entries, lower/upper (nb-1)*m*m")
    f = BlockTridiagFactor(m, nb, [], [], list(lower), 0.0, True, "success")
    for i in range(nb):
        mi = list(diag[i * mm:(i + 1) * mm])
        if i > 0:
            # M_i = D_i - L_{i-1} C_{i-1}
            prod = _matmul(lower[(i - 1) * mm:i * mm], f.cp[(i - 1) * mm:i * mm], m)
            mi = [a - p for a, p in zip(mi, prod)]
        lu = lu_factor(mi, m)
        if not lu.success:
            f.success = False
            f.message = f"diagonal block {i}: {lu.message}"
            return f
        if i + 1 < nb:
            # C_i = M_i^-1 U_i, column by column
            u = upper[i * mm:(i + 1) * mm]
            c = [0.0] * mm
            for j in range(m):
                s = lu_solve(lu, [u[r * m + j] for r in range(m)])
                for r in range(m):
                    c[r * m + j] = s.x[r]
            gnorm = max(sum(abs(val(c[r * m + j])) for j in range(m)) for r in range(m))
            f.growth = max(f.growth, gnorm)
            f.cp.extend(c)
        f.blocks.append(lu)
    return f


def block_tridiag_solve(f, b):
    """Solve A x = b with a factorization from `block_tridiag_factor`; b has m*nb entries.

    Call repeatedly for many right-hand sides; f is not modified. Returns success False (empty
    x) if the factorization failed. Raises ValueError on a wrong length.
    """
    m, nb = f.m, f.nb
    mm = m * m
    if len(b) != m * nb:
        raise ValueError("block_tridiag_solve: b must have m*nb entries")
    if not f.success:
        return BlockTridiagSolution([], False, f.message)
    x = list(b)
    # forward sweep: y_i = M_i^-1 (b_i - L_{i-1} y_{i-1})
    for i in range(nb):
        v = x[i * m:(i + 1) * m]
        if i > 0:
            l = f.lower[(i - 1) * mm:i * mm]
            for r in range(m):
                s = v[r]
                for k in range(m):
                    s = s - l[r * m + k] * x[(i - 1) * m + k]
                v[r] = s
        x[i * m:(i + 1) * m] = lu_solve(f.blocks[i], v).x
    # back substitution: x_i = y_i - C_i x_{i+1}
    for i in range(nb - 2, -1, -1):
        c = f.cp[i * mm:(i + 1) * mm]
        for r in range(m):
            s = x[i * m + r]
            for k in range(m):
                s = s - c[r * m + k] * x[(i + 1) * m + k]
            x[i * m + r] = s
    return BlockTridiagSolution(x, True, "success")
