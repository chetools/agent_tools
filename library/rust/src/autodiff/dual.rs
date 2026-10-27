//! `Real` trait and forward-mode dual numbers.
//!
//! Source: Griewank & Walther, "Evaluating Derivatives", 2nd ed. (SIAM 2008), ch. 3 (forward mode);
//! the `Real`/`Dual` design was developed in chetools/claude `packed-bed-runaway/solver/src/lib.rs`
//! (same authors; the method `tan` is renamed `deriv` here). No third-party code.
//!
//! All quantities are dimensionless as far as this module is concerned: a dual number `v + d*eps`
//! carries a value `v` and the derivative `d` of that value with respect to whatever input was
//! seeded with derivative 1, in the (inverse) units of that input.
//! Python twin: `library/python/agent_tools/autodiff/dual.py`.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A number that numerical code can be written against once: plain `f64`, or a [`Dual`] that
/// also carries a derivative.
///
/// Valid inputs follow the real-number function: `ln` needs a positive argument, `sqrt` a
/// non-negative one (a Dual `sqrt(0)` has an infinite derivative). Out-of-domain inputs give
/// NaN or infinity as `f64` does; nothing is checked.
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// A constant `v` (its derivative is zero).
    fn cst(v: f64) -> Self;
    /// The plain value, dropping all derivative parts.
    fn val(self) -> f64;
    /// The first-order derivative part as a plain number (0 for `f64`). Intended for step-size
    /// control, so that sensitivities are kept as accurate as the solution itself.
    fn deriv(self) -> f64;
    /// Exponential.
    fn exp(self) -> Self;
    /// Natural logarithm (argument > 0).
    fn ln(self) -> Self;
    /// Square root (argument >= 0; derivative infinite at 0).
    fn sqrt(self) -> Self;
    /// Sine (argument in radians).
    fn sin(self) -> Self;
    /// Cosine (argument in radians).
    fn cos(self) -> Self;
    /// Integer power. `powi(0)` is 1 with derivative 0 for every argument (also 0).
    fn powi(self, n: i32) -> Self;
    /// Absolute value. At exactly 0 the derivative of the argument is passed through
    /// unchanged (the right-hand derivative); there is no sub-gradient treatment.
    fn abs(self) -> Self;
}

impl Real for f64 {
    #[inline(always)]
    fn cst(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn val(self) -> f64 {
        self
    }
    #[inline(always)]
    fn deriv(self) -> f64 {
        0.0
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

/// Forward-mode dual number `v + d*eps` with `eps^2 = 0`.
///
/// Arithmetic applies the chain rule, so `d` is the exact derivative of `v` with respect to
/// the input seeded with `d = 1` (build the seed with `Dual::new(x, 1.0)`; constants with
/// `Real::cst`). `T` may itself be a `Dual`, which gives second derivatives: for
/// `Dual<Dual<f64>>` seeded as `Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0))`, the
/// second derivative is `result.d.d`.
#[derive(Clone, Copy, Debug)]
pub struct Dual<T> {
    /// Value part.
    pub v: T,
    /// Derivative part.
    pub d: T,
}

impl<T: Real> Dual<T> {
    /// Build `v + d*eps`.
    pub fn new(v: T, d: T) -> Self {
        Dual { v, d }
    }
}

impl<T: Real> Add for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Dual { v: self.v + o.v, d: self.d + o.d }
    }
}
impl<T: Real> Sub for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Dual { v: self.v - o.v, d: self.d - o.d }
    }
}
impl<T: Real> Mul for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        // product rule
        Dual { v: self.v * o.v, d: self.d * o.v + self.v * o.d }
    }
}
impl<T: Real> Div for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        // quotient rule
        let q = self.v / o.v;
        Dual { v: q, d: (self.d - q * o.d) / o.v }
    }
}
impl<T: Real> Neg for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Dual { v: -self.v, d: -self.d }
    }
}

impl<T: Real> Real for Dual<T> {
    #[inline(always)]
    fn cst(v: f64) -> Self {
        Dual { v: T::cst(v), d: T::cst(0.0) }
    }
    #[inline(always)]
    fn val(self) -> f64 {
        self.v.val()
    }
    #[inline(always)]
    fn deriv(self) -> f64 {
        self.d.val()
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        Dual { v: e, d: self.d * e }
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Dual { v: self.v.ln(), d: self.d / self.v }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        Dual { v: s, d: self.d / (T::cst(2.0) * s) }
    }
    #[inline(always)]
    fn sin(self) -> Self {
        Dual { v: self.v.sin(), d: self.d * self.v.cos() }
    }
    #[inline(always)]
    fn cos(self) -> Self {
        Dual { v: self.v.cos(), d: -(self.d * self.v.sin()) }
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::cst(1.0);
        }
        Dual { v: self.v.powi(n), d: self.d * (T::cst(n as f64) * self.v.powi(n - 1)) }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        if self.v.val() < 0.0 {
            -self
        } else {
            self
        }
    }
}
