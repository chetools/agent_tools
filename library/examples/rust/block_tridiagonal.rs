//! Example: solve a block-tridiagonal system (m = 2, nb = 6) by block Thomas, solve a second
//! right-hand side with the same factorization, and report a singular first diagonal block.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::linalg::block_tridiagonal::{block_tridiag_factor, block_tridiag_solve};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let (m, nb) = (2usize, 6usize);
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut diag = Vec::new();
    for i in 0..nb {
        let t = i as f64;
        diag.extend_from_slice(&[4.0 + 0.1 * t, 0.5, -0.3, 5.0 - 0.2 * t]);
        if i + 1 < nb {
            lower.extend_from_slice(&[0.5, 0.1 * t, -0.2, 0.7]);
            upper.extend_from_slice(&[-0.6, 0.2, 0.1 * t - 0.3, 0.4]);
        }
    }
    let b1: Vec<f64> = (0..m * nb).map(|i| (i as f64 + 1.0).sin()).collect();
    let b2: Vec<f64> = (0..m * nb).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let f = block_tridiag_factor(&lower, &diag, &upper, m, nb);
    let s1 = block_tridiag_solve(&f, &b1);
    let s2 = block_tridiag_solve(&f, &b2);
    // [[0,1],[1,0]] with m = 1: nonsingular, but the first diagonal block is zero
    let bad = block_tridiag_factor(&[1.0], &[0.0, 0.0], &[1.0], 1, 2);
    println!(
        "{{\"x1\":{},\"x2\":{},\"growth\":{:?},\"success\":{},\"bad_success\":{},\"bad_message\":{:?}}}",
        list(&s1.x),
        list(&s2.x),
        f.growth,
        f.success,
        bad.success,
        bad.message
    );
}
