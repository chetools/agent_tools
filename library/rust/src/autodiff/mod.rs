//! Automatic differentiation: the `Real` trait and forward-mode `Dual` numbers.

pub mod dual;

pub use dual::{Dual, Real};
