//! Example: pinch targets of Kemp's four-stream problem (dT_min = 10 K) and an invalid stream.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::process::pinch_targets::{pinch_targets, Stream};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let s = |a, b, c| Stream { t_supply: a, t_target: b, cp: c };
    let hot = [s(170.0, 60.0, 3.0), s(150.0, 30.0, 1.5)];
    let cold = [s(20.0, 135.0, 2.0), s(80.0, 140.0, 4.0)];
    let r = pinch_targets(&hot, &cold, 10.0).unwrap();
    let bad = pinch_targets(&[s(10.0, 50.0, 1.0)], &[], 10.0);
    println!(
        "{{\"qh_min\":{:?},\"qc_min\":{:?},\"pinch_hot\":{:?},\"t_hot_scale\":{},\"heat_flow\":{},\"hot_composite_h\":{},\"cold_composite_h\":{},\"bad_rejected\":{}}}",
        r.qh_min,
        r.qc_min,
        r.pinch_hot.unwrap(),
        list(&r.t_hot_scale),
        list(&r.heat_flow),
        list(&r.hot_composite.h),
        list(&r.cold_composite.h),
        bad.is_err()
    );
}
