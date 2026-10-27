//! Example: LMTD and partial derivatives for a normal, a near-equal and an invalid case.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::process::lmtd::lmtd;

fn list(v: (f64, f64, f64)) -> String {
    format!("[{:?},{:?},{:?}]", v.0, v.1, v.2)
}

fn main() {
    let bad = lmtd(-1.0, 5.0);
    println!(
        "{{\"normal\":{},\"near_equal\":{},\"invalid_is_nan\":{}}}",
        list(lmtd(30.0, 10.0)),
        list(lmtd(20.0, 20.00001)),
        bad.0.is_nan()
    );
}
