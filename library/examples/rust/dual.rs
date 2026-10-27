//! Example: first and second derivative of f(x) = exp(sin x) sqrt(x) + |cos x| / x^2 + ln x
//! at x = 1.9 with forward-mode dual numbers. Prints one JSON object (same keys as the Python example).
use agent_tools::autodiff::{Dual, Real};

/// Written once, generic over the number type.
fn f<T: Real>(x: T) -> T {
    x.sin().exp() * x.sqrt() + x.cos().abs() / x.powi(2) + x.ln()
}

fn main() {
    let x = 1.9_f64;
    let value = f(x);
    // first derivative: seed d = 1
    let first = f(Dual::new(x, 1.0));
    // second derivative: dual of dual
    let second = f(Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0)));
    println!(
        "{{\"x\":{:?},\"f\":{:?},\"df\":{:?},\"d2f\":{:?},\"f_dual\":{:?},\"deriv_of_f64\":{:?}}}",
        x,
        value,
        first.d,
        second.d.d,
        first.v,
        value.deriv()
    );
}
