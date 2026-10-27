//! Example: Cholesky factor and solve of a 3x3 SPD system, and a non-SPD matrix.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::linalg::cholesky::{cholesky, cholesky_solve};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let mut x = [1.0, 2.0, 3.0];
    let r = cholesky(&a, 3);
    cholesky_solve(&r.l, 3, &mut x, 1);
    let bad = cholesky(&[1.0, 2.0, 2.0, 1.0], 2);
    println!(
        "{{\"l\":{},\"x\":{},\"success\":{},\"bad_success\":{}}}",
        list(&r.l),
        list(&x),
        r.success,
        bad.success
    );
}
