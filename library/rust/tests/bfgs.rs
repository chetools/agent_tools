//! Tests for optimize::bfgs against exact minimizers and scipy.optimize.minimize(method="BFGS") reference data.
use agent_tools::optimize::bfgs::{bfgs, BfgsOptions};

/// Computed minimizer vs the exact one: max |x - x*| (the stopping test max|g| < 1e-10 bounds it by
/// 1e-10 / lambda_min, at most about 3e-10 for these cases; the margin covers rounding).
const TOL_X_EXACT: f64 = 1e-8;
/// Computed f vs the exact f* (all zero except the quadratic): absolute.
const TOL_F_EXACT: f64 = 1e-15;
/// Same, for quadratic8_cond1e3 (entries of A up to 1e3, so f is a cancellation of terms about 1e3 times larger).
const TOL_F_EXACT_QUAD8: f64 = 1e-13;
/// Computed minimizer vs SciPy's: max |x - x_scipy| (both within TOL_X_EXACT of the exact one).
const TOL_X_SCIPY: f64 = 1e-8;
/// Regression guard on cost: our nfev may be at most this multiple of SciPy's nfev (SciPy uses a
/// Wolfe line search with interpolation, ours is plain halving).
const NFEV_FACTOR: f64 = 3.0;
/// Looser cost guard for the ill-conditioned quadratic8_cond1e3 case (measured ratio 5: 80 vs 16).
const NFEV_FACTOR_QUAD8: f64 = 8.0;
const GTOL: f64 = 1e-10;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/bfgs.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// Objective of a reference case: returns f and writes the gradient.
fn objective(c: &serde_json::Value) -> Box<dyn FnMut(&[f64], &mut [f64]) -> f64> {
    let p = &c["params"];
    match c["kind"].as_str().unwrap() {
        "quadratic" => {
            let (a, b) = (vec(&p["A"]), vec(&p["b"]));
            let n = b.len();
            Box::new(move |x, g| {
                let mut f = 0.0;
                for i in 0..n {
                    let ax: f64 = (0..n).map(|j| a[i * n + j] * x[j]).sum();
                    g[i] = ax - b[i];
                    f += 0.5 * x[i] * ax - b[i] * x[i];
                }
                f
            })
        }
        "ext_rosenbrock" => {
            let a = p["a"].as_f64().unwrap();
            Box::new(move |x, g| {
                let mut f = 0.0;
                for i in 0..x.len() / 2 {
                    let (u, v) = (x[2 * i], x[2 * i + 1]);
                    let t = v - u * u;
                    f += a * t * t + (1.0 - u) * (1.0 - u);
                    g[2 * i] = -4.0 * a * u * t - 2.0 * (1.0 - u);
                    g[2 * i + 1] = 2.0 * a * t;
                }
                f
            })
        }
        k => panic!("unknown kind {k}"),
    }
}

#[test]
fn against_exact_and_scipy() {
    let r = load();
    for (name, c) in r["cases"].as_object().unwrap() {
        let (x0, xe, fe) = (vec(&c["x0"]), vec(&c["x_exact"]), c["f_exact"].as_f64().unwrap());
        let s = &c["scipy"];
        let res = bfgs(objective(c), &x0, &BfgsOptions { max_iter: 1000, gtol: GTOL });
        assert!(res.success, "{name}: {}", res.message);
        let ex = maxdiff(&res.x, &xe);
        let ef = (res.f - fe).abs();
        let es = maxdiff(&res.x, &vec(&s["x"]));
        let (sn, sf) = (s["nit"].as_u64().unwrap(), s["nfev"].as_u64().unwrap());
        println!(
            "VERIFY bfgs rust_{name} max_err_x_vs_exact={ex:e} abs_err_f_vs_exact={ef:e} max_diff_x_vs_scipy={es:e} grad_norm={:e} nit={} nfev={} scipy_nit={sn} scipy_nfev={sf}",
            res.grad_norm, res.nit, res.nfev
        );
        assert!(res.grad_norm < GTOL);
        let tol_f = if name == "quadratic8_cond1e3" { TOL_F_EXACT_QUAD8 } else { TOL_F_EXACT };
        assert!(ex < TOL_X_EXACT && ef < tol_f && es < TOL_X_SCIPY, "{name}");
        let factor = if name == "quadratic8_cond1e3" { NFEV_FACTOR_QUAD8 } else { NFEV_FACTOR };
        assert!((res.nfev as f64) <= factor * sf as f64, "{name}: nfev {} vs scipy {sf}", res.nfev);
    }
}

#[test]
fn diagonal_quadratic_minimizer() {
    // f = (x-3)^2 / 2 + (y+1)^2 * 2: minimizer (3, -1), found to gtol = 1e-10.
    let res = bfgs(
        |x, g| {
            g[0] = x[0] - 3.0;
            g[1] = 4.0 * (x[1] + 1.0);
            0.5 * (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2)
        },
        &[0.0, 0.0],
        &BfgsOptions { max_iter: 50, gtol: GTOL },
    );
    assert!(res.success);
    let e = maxdiff(&res.x, &[3.0, -1.0]);
    println!("VERIFY bfgs rust_diag_quadratic max_err_x_vs_exact={e:e} nit={} nfev={}", res.nit, res.nfev);
    assert!(e < 1e-9);
}

#[test]
fn failures_are_reported() {
    let o = BfgsOptions::default();
    assert_eq!((o.max_iter, o.gtol), (200, 1e-5));
    let ros = |x: &[f64], g: &mut [f64]| {
        g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    };
    // iteration limit
    let r = bfgs(ros, &[-1.2, 1.0], &BfgsOptions { max_iter: 5, gtol: 1e-10 });
    assert!(!r.success && r.nit == 5 && r.message.contains("max_iter"), "{}", r.message);
    // wrong gradient sign: every direction is uphill, the line search must give up
    let r = bfgs(|x, g| { g[0] = -2.0 * x[0]; x[0] * x[0] }, &[1.0], &o);
    assert!(!r.success && r.message.contains("line search failed"), "{}", r.message);
    // non-finite start
    let r = bfgs(|_, g| { g[0] = 0.0; f64::NAN }, &[1.0], &o);
    assert!(!r.success && r.message.contains("non-finite"), "{}", r.message);
    // f = +inf outside x > 0: infeasible trial points are rejected, the minimum of x - ln x is x = 1
    let r = bfgs(
        |x, g| {
            if x[0] <= 0.0 {
                return f64::INFINITY;
            }
            g[0] = 1.0 - 1.0 / x[0];
            x[0] - x[0].ln()
        },
        &[0.1],
        &BfgsOptions { max_iter: 200, gtol: 1e-8 },
    );
    assert!(r.success && (r.x[0] - 1.0).abs() < 1e-7, "{} {:?}", r.message, r.x);
    println!("VERIFY bfgs rust_failure_modes max_iter,line_search,nonfinite_x0,infinite_f_rejected=ok");
}

#[test]
fn stall_stops_when_gtol_is_below_rounding() {
    // gtol far below what rounding allows: the old code took no-progress steps until max_iter
    let r = load();
    let c = &r["cases"]["quadratic5"];
    let res = bfgs(objective(c), &vec(&c["x0"]), &BfgsOptions { max_iter: 5000, gtol: 1e-20 });
    println!("VERIFY bfgs rust_stall success={} nit={} nfev={} grad_norm={:e} message='{}'", res.success, res.nit, res.nfev, res.grad_norm, res.message);
    assert!(!res.success && res.message.contains("no progress"), "{}", res.message);
    assert!(res.nfev < 500 && res.nit < 200, "nit {} nfev {}", res.nit, res.nfev);
    assert!(maxdiff(&res.x, &vec(&c["x_exact"])) < 1e-9 && res.grad_norm < 1e-9);
}

#[test]
fn invalid_options_fail_immediately() {
    let mut calls = 0usize;
    let cases = [
        (BfgsOptions { max_iter: 200, gtol: 0.0 }, "gtol"),
        (BfgsOptions { max_iter: 200, gtol: -1e-5 }, "gtol"),
        (BfgsOptions { max_iter: 200, gtol: f64::NAN }, "gtol"),
        (BfgsOptions { max_iter: 200, gtol: f64::INFINITY }, "gtol"),
        (BfgsOptions { max_iter: 0, gtol: 1e-5 }, "max_iter"),
    ];
    for (o, word) in &cases {
        let r = bfgs(|x, g| { calls += 1; g[0] = 2.0 * x[0]; x[0] * x[0] }, &[1.0], o);
        assert!(!r.success && r.nfev == 0 && r.nit == 0 && r.message.contains(word), "{:?}: {}", o, r.message);
        assert_eq!(r.x, vec![1.0]);
    }
    assert_eq!(calls, 0);
    println!("VERIFY bfgs rust_invalid_options gtol_not_positive_finite,max_iter_zero=fail_with_nfev_0");
}
