//! Example: box-constrained convex QP, minimize 0.5 x.H.x + g.x subject to lo <= x <= hi, on nine cases of
//! library/tests/reference_data/qp_box.json (mixed active set, infinite bounds, unconstrained, degenerate zero
//! multiplier, a fixed variable, cond(H) = 1e8 with n = 30, a random n = 20 problem, all bounds active, and a start at a
//! bound with zero gradient), default options.
//! Prints one JSON object (same keys as the Python example): per case x, f, active, multipliers, success, nit, nfact.
use agent_tools::optimize::qp_box::{qp_box, QpBoxOptions};

const CASES: [&str; 9] = [
    "random_n6", "infinite_bounds_n6", "unconstrained_inf_n10", "degenerate_zero_multiplier_n5",
    "fixed_variable_n5", "cond1e8_n30", "sweep_n20_cond1e+02", "all_active_n5", "start_at_bound_zero_gradient_n4",
];

fn list<T: std::fmt::Debug>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let path = format!("{}/../tests/reference_data/qp_box.json", env!("CARGO_MANIFEST_DIR"));
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let vec = |v: &serde_json::Value, null: f64| -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(null)).collect()
    };
    let mut out = Vec::new();
    for name in CASES {
        let c = &doc["cases"][name];
        let (h, g) = (vec(&c["H"], 0.0), vec(&c["g"], 0.0));
        let (lo, hi) = (vec(&c["lo"], f64::NEG_INFINITY), vec(&c["hi"], f64::INFINITY));
        let r = qp_box(&h, &g, &lo, &hi, &QpBoxOptions::default());
        out.push(format!(
            "\"{}\":{{\"x\":{},\"f\":{:?},\"active\":{},\"multipliers\":{},\"success\":{},\"nit\":{},\"nfact\":{}}}",
            name, list(&r.x), r.f, list(&r.active), list(&r.multipliers), r.success, r.nit, r.nfact
        ));
    }
    println!("{{{}}}", out.join(","));
}
