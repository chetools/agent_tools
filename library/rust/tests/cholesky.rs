//! Tests for linalg::cholesky against numpy.linalg.cholesky / scipy.linalg.cho_solve reference data.
use agent_tools::linalg::cholesky::{cholesky, cholesky_solve, solve_lower, solve_upper_t};

/// L vs NumPy, relative to max|L|; for ill-conditioned matrices the bound is widened to
/// TOL_X_FACTOR * eps * cond (the two codes differ in operation order, errors scale with cond).
const TOL_L: f64 = 1e-12;
/// Solution vs SciPy within TOL_X_FACTOR * eps * cond (relative to max|x|).
const TOL_X_FACTOR: f64 = 10.0;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/cholesky.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn maxabs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}
fn rel_err(x: &[f64], r: &[f64]) -> f64 {
    let d: Vec<f64> = x.iter().zip(r).map(|(a, b)| a - b).collect();
    maxabs(&d) / maxabs(r)
}

fn run_group(r: &serde_json::Value, group: &str) {
    let eps = f64::EPSILON;
    for c in r["cases"][group].as_array().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let (a, b, lr, xr, cond) = (vec(&c["A"]), vec(&c["b"]), vec(&c["L"]), vec(&c["x"]), c["cond"].as_f64().unwrap());
        let f = cholesky(&a, n);
        assert!(f.success, "{group} n={n}: {}", f.message);
        let el = rel_err(&f.l, &lr);
        let mut x = b.clone();
        cholesky_solve(&f.l, n, &mut x, 1);
        let ex = rel_err(&x, &xr);
        // L L^T reproduces A
        let mut ea = 0f64;
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| f.l[i * n + k] * f.l[j * n + k]).sum();
                ea = ea.max((s - a[i * n + j]).abs());
            }
        }
        let ea = ea / maxabs(&a);
        println!("VERIFY cholesky {group}_n{n} cond={cond:e} rel_err_L_vs_numpy={el:e} rel_err_x_vs_scipy={ex:e} rel_err_LLt_vs_A={ea:e}");
        assert!(el < TOL_L.max(TOL_X_FACTOR * eps * cond), "{group} n={n} L err {el:e}");
        assert!(ex < TOL_X_FACTOR * eps * cond.max(1.0), "{group} n={n} x err {ex:e}");
    }
}

#[test]
fn spd_matrices() {
    run_group(&load(), "spd");
}

#[test]
fn hilbert_matrices() {
    // ill-conditioned (cond up to ~1.6e13): L agrees with NumPy within the cond-scaled bound
    run_group(&load(), "hilbert");
}

#[test]
fn non_spd_reported() {
    let r = load();
    for c in r["cases"]["not_spd"].as_array().unwrap() {
        let f = cholesky(&vec(&c["A"]), c["n"].as_u64().unwrap() as usize);
        assert!(!f.success, "{}", c["name"]);
        assert!(f.message.contains("not positive definite"));
    }
    assert!(!cholesky(&[f64::NAN], 1).success);
    assert!(!cholesky(&[f64::INFINITY], 1).success);
    println!("VERIFY cholesky not_spd cases_reported={}", r["cases"]["not_spd"].as_array().unwrap().len() + 1);
}

#[test]
fn triangular_solves_multiple_rhs() {
    // L = [[2,0],[1,3]]; B has 2 columns
    let l = [2.0, 0.0, 1.0, 3.0];
    let mut b = [2.0, 4.0, 7.0, 5.0]; // rows: [2,4], [7,5]
    solve_lower(&l, 2, &mut b, 2);
    assert_eq!(b, [1.0, 2.0, 2.0, 1.0]);
    // L^T = [[2,1],[0,3]]: solve L^T X = [[3,5],[3,3]] -> x2=[1,1], x1=[(3-1)/2,(5-1)/2]=[1,2]
    let mut c = [3.0, 5.0, 3.0, 3.0];
    solve_upper_t(&l, 2, &mut c, 2);
    assert_eq!(c, [1.0, 2.0, 1.0, 1.0]);
    println!("VERIFY cholesky triangular_multi_rhs exact_match=1");
}
