//! agent_tools: shared scientific and engineering routines.
//!
//! One module per area (ids as in the repository README). No runtime
//! dependencies, no unsafe code, no FFI glue.
#![forbid(unsafe_code)]

pub mod autodiff;
pub mod linalg;
pub mod random;
pub mod interpolate;
