"""Forward-mode dual numbers (Python reference of the Rust `Real` trait and `Dual`).

Source: Griewank & Walther, "Evaluating Derivatives", 2nd ed. (SIAM 2008), ch. 3 (forward mode);
developed in chetools/claude `packed-bed-runaway` (the Rust method `tan` is `deriv` here).
Rust twin: library/rust/src/autodiff/dual.rs. Pure Python (`math` only).

Everything is dimensionless as far as this module is concerned: `Dual(v, d)` is v + d*eps with
eps**2 = 0, where d is the derivative of v with respect to the input seeded with d = 1.

The module-level functions `exp, ln, sqrt, sin, cos, powi, absval` accept a float or a Dual
(whose parts may themselves be Duals, which gives second derivatives). `val` and `deriv` mirror
`Real::val` and `Real::deriv`. Domains as for the real functions: ln needs x > 0, sqrt x >= 0
(infinite derivative at 0); out-of-domain inputs raise the `math` ValueError.
"""
import math


class Dual:
    """Dual number v + d*eps. Operators accept floats on either side (a float is a constant)."""

    __slots__ = ("v", "d")

    def __init__(self, v, d=0.0):
        self.v = v
        self.d = d

    def __repr__(self):
        return f"Dual({self.v!r}, {self.d!r})"

    @staticmethod
    def _lift(o):
        return o if isinstance(o, Dual) else Dual(o, 0.0)

    def __add__(self, o):
        o = Dual._lift(o)
        return Dual(self.v + o.v, self.d + o.d)

    __radd__ = __add__

    def __sub__(self, o):
        o = Dual._lift(o)
        return Dual(self.v - o.v, self.d - o.d)

    def __rsub__(self, o):
        return Dual._lift(o) - self

    def __mul__(self, o):
        o = Dual._lift(o)
        # product rule
        return Dual(self.v * o.v, self.d * o.v + self.v * o.d)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Dual._lift(o)
        # quotient rule
        q = self.v / o.v
        return Dual(q, (self.d - q * o.d) / o.v)

    def __rtruediv__(self, o):
        return Dual._lift(o) / self

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __pow__(self, n):
        """Integer power only (same as `powi`)."""
        if not isinstance(n, int):
            raise TypeError("Dual ** n supports integer n only")
        return powi(self, n)


def val(x):
    """Plain float value of x, dropping all derivative parts."""
    while isinstance(x, Dual):
        x = x.v
    return x


def deriv(x):
    """First-order derivative part of x as a plain float (0.0 for a float)."""
    return val(x.d) if isinstance(x, Dual) else 0.0


def exp(x):
    """Exponential."""
    if isinstance(x, Dual):
        e = exp(x.v)
        return Dual(e, x.d * e)
    return math.exp(x)


def ln(x):
    """Natural logarithm (x > 0)."""
    if isinstance(x, Dual):
        return Dual(ln(x.v), x.d / x.v)
    return math.log(x)


def sqrt(x):
    """Square root (x >= 0; derivative infinite at 0)."""
    if isinstance(x, Dual):
        s = sqrt(x.v)
        return Dual(s, x.d / (2.0 * s))
    return math.sqrt(x)


def sin(x):
    """Sine (radians)."""
    if isinstance(x, Dual):
        return Dual(sin(x.v), x.d * cos(x.v))
    return math.sin(x)


def cos(x):
    """Cosine (radians)."""
    if isinstance(x, Dual):
        return Dual(cos(x.v), -(x.d * sin(x.v)))
    return math.cos(x)


def powi(x, n):
    """Integer power x**n. powi(x, 0) is 1 with derivative 0 for every x (also 0)."""
    if isinstance(x, Dual):
        if n == 0:
            return Dual(1.0, 0.0)
        return Dual(powi(x.v, n), x.d * (float(n) * powi(x.v, n - 1)))
    return float(x) ** n


def absval(x):
    """Absolute value. At exactly 0 the derivative of the argument is passed through unchanged."""
    if isinstance(x, Dual):
        return -x if val(x.v) < 0.0 else x
    return abs(x)
