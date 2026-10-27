//! Example: max/min of the Hermite interpolant of sin(t) on coarse knots, evaluator values, a tie
//! and an invalid input. Prints one JSON object (same keys as the Python example).
use agent_tools::interpolate::cubic_hermite::cubic_hermite;
use agent_tools::interpolate::hermite_extremum::{hermite_extremum, Extremum};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let t = [0.0, 0.9, 1.7, 2.6, 3.9, 4.6, 5.9, 7.2];
    let y: Vec<f64> = t.iter().map(|x: &f64| x.sin()).collect();
    let dy: Vec<f64> = t.iter().map(|x: &f64| x.cos()).collect();
    let mx = hermite_extremum(&t, &y, &dy, Extremum::Max);
    let mn = hermite_extremum(&t, &y, &dy, Extremum::Min);
    let ev: Vec<f64> = [-0.5, 1.0, 3.0, 8.0].iter().map(|&x| cubic_hermite(&t, &y, &dy, x)).collect();
    let tie = hermite_extremum(&[0.0, 1.0, 2.0], &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], Extremum::Max);
    let bad = hermite_extremum(&[0.0, 1.0, 1.0], &[0.0, 1.0, 2.0], &[1.0, 1.0, 1.0], Extremum::Max);
    println!(
        "{{\"max_t\":{:?},\"max_value\":{:?},\"max_interval\":{},\"min_t\":{:?},\"min_value\":{:?},\"min_interval\":{},\"eval\":{},\"tie_t\":{:?},\"tie_interval\":{},\"bad_success\":{}}}",
        mx.t, mx.value, mx.interval, mn.t, mn.value, mn.interval, list(&ev), tie.t, tie.interval, bad.success
    );
}
