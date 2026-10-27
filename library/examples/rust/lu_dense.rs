//! Example: solve a 4x4 system by LU with partial pivoting, and report a singular matrix.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::linalg::lu_dense::{lu_factor, lu_solve};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let a = [
        0.0, 2.0, 1.0, -1.0, //
        3.0, -1.0, 2.0, 0.5, //
        1.0, 1.0, -4.0, 2.0, //
        2.0, 0.0, 1.0, 5.0,
    ];
    let b = [1.0, 2.0, 3.0, 4.0];
    let f = lu_factor(&a, 4);
    let s = lu_solve(&f, &b);
    let sing = lu_factor(&[1.0, 2.0, 2.0, 4.0], 2);
    println!(
        "{{\"x\":{},\"piv\":{:?},\"success\":{},\"singular_success\":{},\"singular_message\":{:?}}}",
        list(&s.x),
        f.piv,
        s.success,
        sing.success,
        sing.message
    );
}
