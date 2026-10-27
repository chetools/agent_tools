//! Tests for process::lmtd against 50-digit mpmath reference data.
use agent_tools::process::lmtd::lmtd;

/// L relative error bound (measured worst case 5e-13, just above the switch).
const TOL_L: f64 = 1e-11;
/// Derivative absolute bound away from the switch (measured worst case 5e-12).
const TOL_D: f64 = 1e-11;
/// Sweeps: L and both derivatives, relative to the 50-digit value.
const TOL_SWEEP: f64 = 5e-12;
/// Central finite-difference check, absolute (step 1e-4 on values ~20).
const TOL_FD: f64 = 5e-7;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/lmtd.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn run(group: &str) {
    let r = load();
    let (mut el, mut ed) = (0f64, 0f64);
    for c in r["cases"][group].as_array().unwrap() {
        let f = |k: &str| c[k].as_f64().unwrap();
        let (a, b) = (f("a"), f("b"));
        let (l, la, lb) = lmtd(a, b);
        let e_l = (l - f("L")).abs() / f("L");
        let e_d = (la - f("dLda")).abs().max((lb - f("dLdb")).abs());
        assert!(e_l < TOL_L && e_d < TOL_D, "{group} a={a} b={b}: eL={e_l:e} eD={e_d:e}");
        el = el.max(e_l);
        ed = ed.max(e_d);
    }
    println!("VERIFY lmtd {group} max_rel_err_L={el:e} max_abs_err_dL={ed:e}");
}

#[test]
fn ordinary_and_extreme_ratios() {
    run("ordinary_and_extreme");
}

#[test]
fn switch_region() {
    run("switch_region");
}

/// Review finding: derivatives were 8.2e-9 off just above the series switch (cancellation in ln(a/b)).
fn run_sweep(group: &str) {
    let r = load();
    let cases = r["cases"][group].as_array().unwrap();
    assert!(cases.len() >= 200);
    let (mut el, mut ed) = (0f64, 0f64);
    for c in cases {
        let f = |k: &str| c[k].as_f64().unwrap();
        let (a, b) = (f("a"), f("b"));
        let (l, la, lb) = lmtd(a, b);
        let e_l = (l - f("L")).abs() / f("L");
        let e_d = ((la - f("dLda")) / f("dLda")).abs().max(((lb - f("dLdb")) / f("dLdb")).abs());
        assert!(e_l < TOL_SWEEP && e_d < TOL_SWEEP, "{group} a={a} b={b}: eL={e_l:e} eD={e_d:e}");
        el = el.max(e_l);
        ed = ed.max(e_d);
    }
    println!("VERIFY lmtd {group} n={} max_rel_err_L={el:e} max_rel_err_dL={ed:e}", cases.len());
}

#[test]
fn sweep_rel_diff() {
    run_sweep("sweep_rel_diff");
}

#[test]
fn sweep_ratio() {
    run_sweep("sweep_ratio");
}

#[test]
fn derivatives_vs_finite_differences_both_sides_of_switch() {
    let mut worst = 0f64;
    // relative differences below and above the 1e-4 switch, on a = 20(1+rel/2), b = 20(1-rel/2)
    for rel in [0.0, 1e-6, 5e-5, 9.9e-5, 1.01e-4, 2e-4, 1e-3, 0.5, 1.5] {
        let (a, b) = (20.0 * (1.0 + rel / 2.0), 20.0 * (1.0 - rel / 2.0));
        let (_, la, lb) = lmtd(a, b);
        let h = 1e-4;
        let fa = (lmtd(a + h, b).0 - lmtd(a - h, b).0) / (2.0 * h);
        let fb = (lmtd(a, b + h).0 - lmtd(a, b - h).0) / (2.0 * h);
        worst = worst.max((la - fa).abs()).max((lb - fb).abs());
        assert!((la - fa).abs() < TOL_FD && (lb - fb).abs() < TOL_FD, "rel={rel}");
    }
    println!("VERIFY lmtd fd_both_sides max_abs_diff={worst:e}");
}

#[test]
fn invalid_inputs_give_nan() {
    for (a, b) in [(-1.0, 5.0), (5.0, 0.0), (f64::NAN, 1.0), (f64::INFINITY, 1.0)] {
        let (l, la, lb) = lmtd(a, b);
        assert!(l.is_nan() && la.is_nan() && lb.is_nan(), "{a} {b}");
    }
    let (l1, _, _) = lmtd(30.0, 10.0);
    let (l2, _, _) = lmtd(10.0, 30.0);
    assert_eq!(l1, l2);
    println!("VERIFY lmtd invalid_inputs nan_cases=4");
}
