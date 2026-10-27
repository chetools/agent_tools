//! Tests for autodiff::dual: AD derivatives vs analytic (reference data) and vs central differences.
use agent_tools::autodiff::{Dual, Real};

/// AD vs hand-derived closed forms, error scaled by max(|ref|, 1).
const TOL_ANALYTIC: f64 = 1e-12;
/// AD vs central finite differences (h = 1e-5: truncation ~1e-10, round-off ~1e-11).
const TOL_FD: f64 = 1e-6;
const H: f64 = 1e-5;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/dual.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

fn eval<T: Real>(name: &str, x: T) -> T {
    match name {
        "exp_sin" => x.sin().exp(),
        "sqrt_ln" => x.sqrt() * x.ln(),
        "powi_poly" => x.powi(5) - T::cst(3.0) * x.powi(-2),
        "abs_cos" => x * x.cos().abs(),
        "rational" => (x * x + T::cst(1.0)) / (x * x * x + T::cst(2.0)),
        "arrhenius" => (x / (T::cst(1.0) + x / T::cst(20.0))).exp(),
        _ => panic!("unknown function {name}"),
    }
}

fn first(name: &str, x: f64) -> Dual<f64> {
    eval(name, Dual::new(x, 1.0))
}

#[test]
fn derivatives_vs_analytic_and_finite_differences() {
    let r = load();
    for (name, c) in r["cases"].as_object().unwrap() {
        let (xs, fs, dfs, d2fs) = (vec(&c["x"]), vec(&c["f"]), vec(&c["df"]), vec(&c["d2f"]));
        let (mut e_f, mut e_d1, mut e_d2, mut e_fd1, mut e_fd2) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for i in 0..xs.len() {
            let x = xs[i];
            let a = first(name, x);
            let n = eval(name, Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0)));
            e_f = e_f.max(err(a.v, fs[i]));
            e_d1 = e_d1.max(err(a.d, dfs[i]));
            e_d2 = e_d2.max(err(n.d.d, d2fs[i]));
            // central finite differences of the f64 function and of the AD first derivative
            let fd1 = (eval(name, x + H) - eval(name, x - H)) / (2.0 * H);
            let fd2 = (first(name, x + H).d - first(name, x - H).d) / (2.0 * H);
            e_fd1 = e_fd1.max(err(a.d, fd1));
            e_fd2 = e_fd2.max(err(n.d.d, fd2));
            assert_eq!(n.v.v, a.v, "nested value part must equal plain value");
            assert_eq!(n.d.v, a.d, "nested first derivative must equal first-order dual");
        }
        println!("VERIFY dual {name} max_err_value_vs_analytic={e_f:e}");
        println!("VERIFY dual {name} max_err_d1_vs_analytic={e_d1:e}");
        println!("VERIFY dual {name} max_err_d2_nested_vs_analytic={e_d2:e}");
        println!("VERIFY dual {name} max_err_d1_vs_central_fd={e_fd1:e}");
        println!("VERIFY dual {name} max_err_d2_vs_central_fd={e_fd2:e}");
        assert!(e_f < TOL_ANALYTIC && e_d1 < TOL_ANALYTIC && e_d2 < TOL_ANALYTIC, "{name} analytic");
        assert!(e_fd1 < TOL_FD && e_fd2 < TOL_FD, "{name} finite differences");
    }
}

#[test]
fn edge_cases() {
    // abs: derivative sign follows the sign of the argument
    let neg = Dual::new(-2.0, 1.0).abs();
    assert_eq!((neg.v, neg.d), (2.0, -1.0));
    let pos = Dual::new(2.0, 1.0).abs();
    assert_eq!((pos.v, pos.d), (2.0, 1.0));
    // powi(0) is the constant 1, also at 0 (no 0 * inf)
    let z = Dual::new(0.0, 1.0).powi(0);
    assert_eq!((z.v, z.d), (1.0, 0.0));
    // deriv()/val() on f64 and Dual, and constants carry zero derivative
    assert_eq!(3.0f64.deriv(), 0.0);
    let d = Dual::new(3.0, 5.0);
    assert_eq!((d.val(), d.deriv()), (3.0, 5.0));
    let c: Dual<f64> = Real::cst(4.0);
    assert_eq!((c.val(), c.deriv()), (4.0, 0.0));
    // nested: val() and deriv() see through to the f64 parts
    let n = Dual::new(Dual::new(1.0, 2.0), Dual::new(3.0, 4.0));
    assert_eq!((n.val(), n.deriv()), (1.0, 3.0));
    println!("VERIFY dual edge_cases checks_passed=1");
}
