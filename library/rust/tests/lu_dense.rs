//! Tests for linalg::lu_dense against scipy.linalg.lu_factor/lu_solve reference data.
use agent_tools::autodiff::{Dual, Real};
use agent_tools::linalg::lu_dense::{lu_factor, lu_solve, solve};

/// Well-conditioned random matrices (n = 1..50): solution vs SciPy, relative to max|x|.
const TOL_WELL: f64 = 1e-12;
/// LU factors vs SciPy (n <= 10), relative to max|LU|.
const TOL_FACTORS: f64 = 1e-12;
/// Ill-conditioned matrices: solution vs SciPy within TOL_ILL_FACTOR * eps * cond.
const TOL_ILL_FACTOR: f64 = 10.0;
/// Backward error ||Ax-b||_inf / (||A||_inf ||x||_inf + ||b||_inf).
const TOL_BACKWARD: f64 = 1e-14;
/// Dual solution derivative vs analytic -A^-1 (dA x - db), relative to max|dx|.
const TOL_DUAL: f64 = 1e-11;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/lu_dense.json", env!("CARGO_MANIFEST_DIR"));
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
fn backward(a: &[f64], x: &[f64], b: &[f64], n: usize) -> f64 {
    let mut res = 0.0f64;
    let mut an = 0.0f64;
    for i in 0..n {
        let mut s = -b[i];
        let mut row = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
            row += a[i * n + j].abs();
        }
        res = res.max(s.abs());
        an = an.max(row);
    }
    res / (an * maxabs(x) + maxabs(b))
}

#[test]
fn well_conditioned_n1_to_50() {
    let r = load();
    let (mut ex, mut ef, mut eb, mut cmax) = (0f64, 0f64, 0f64, 0f64);
    for c in r["cases"]["well"].as_array().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let (a, b, xr) = (vec(&c["A"]), vec(&c["b"]), vec(&c["x"]));
        let f = lu_factor(&a, n);
        assert!(f.success);
        let s = lu_solve(&f, &b);
        ex = ex.max(rel_err(&s.x, &xr));
        eb = eb.max(backward(&a, &s.x, &b, n));
        cmax = cmax.max(c["cond"].as_f64().unwrap());
        if let Some(lu) = c.get("lu") {
            let piv: Vec<usize> = c["piv"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(f.piv, piv, "pivot sequence n={n}");
            ef = ef.max(rel_err(&f.lu, &vec(lu)));
        }
    }
    println!("VERIFY lu_dense well max_rel_err_x_vs_scipy={ex:e}");
    println!("VERIFY lu_dense well max_rel_err_lu_vs_scipy_n<=10={ef:e}");
    println!("VERIFY lu_dense well max_backward_error={eb:e}");
    println!("VERIFY lu_dense well max_cond={cmax:e}");
    assert!(ex < TOL_WELL && ef < TOL_FACTORS && eb < TOL_BACKWARD);
}

#[test]
fn ill_conditioned() {
    let r = load();
    let eps = f64::EPSILON;
    for target in [1e4, 1e8, 1e12] {
        let (mut worst, mut eb, mut ratio) = (0f64, 0f64, 0f64);
        for c in r["cases"]["ill"].as_array().unwrap() {
            if c["target_cond"].as_f64().unwrap() != target {
                continue;
            }
            let n = c["n"].as_u64().unwrap() as usize;
            let (a, b, xr, cond) = (vec(&c["A"]), vec(&c["b"]), vec(&c["x"]), c["cond"].as_f64().unwrap());
            let s = solve(&a, &b, n);
            assert!(s.success);
            let e = rel_err(&s.x, &xr);
            worst = worst.max(e);
            ratio = ratio.max(e / (eps * cond));
            eb = eb.max(backward(&a, &s.x, &b, n));
            assert!(e < TOL_ILL_FACTOR * eps * cond, "n={n} cond={cond:e} err={e:e}");
        }
        println!("VERIFY lu_dense ill_cond_{target:e} max_rel_err_x_vs_scipy={worst:e}");
        println!("VERIFY lu_dense ill_cond_{target:e} max_err_over_eps_cond={ratio:e}");
        println!("VERIFY lu_dense ill_cond_{target:e} max_backward_error={eb:e}");
        assert!(eb < TOL_BACKWARD);
    }
}

#[test]
fn singular_is_reported() {
    let r = load();
    for c in r["cases"]["singular"].as_array().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let f = lu_factor(&vec(&c["A"]), n);
        assert!(!f.success, "{}", c["name"]);
        assert!(f.message.contains("singular"));
        let s = lu_solve(&f, &vec![1.0; n]);
        assert!(!s.success && s.x.is_empty());
    }
    let f = lu_factor(&[1.0, f64::NAN, 0.0, 1.0], 2);
    assert!(!f.success);
    println!("VERIFY lu_dense singular cases_reported={}", r["cases"]["singular"].as_array().unwrap().len() + 1);
}

#[test]
fn dual_solution_derivative() {
    let r = load();
    for c in r["cases"]["param"].as_array().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let p = Dual::new(c["p0"].as_f64().unwrap(), 1.0);
        let (a0, a1, b0, b1) = (vec(&c["A0"]), vec(&c["A1"]), vec(&c["b0"]), vec(&c["b1"]));
        let a: Vec<Dual<f64>> = (0..n * n).map(|k| Dual::cst(a0[k]) + Dual::cst(a1[k]) * p).collect();
        let b: Vec<Dual<f64>> = (0..n).map(|k| Dual::cst(b0[k]) + Dual::cst(b1[k]) * p).collect();
        let s = solve(&a, &b, n);
        assert!(s.success);
        let x: Vec<f64> = s.x.iter().map(|d| d.v).collect();
        let dx: Vec<f64> = s.x.iter().map(|d| d.d).collect();
        let (ex, edx) = (rel_err(&x, &vec(&c["x"])), rel_err(&dx, &vec(&c["dx"])));
        println!("VERIFY lu_dense dual_n{n} rel_err_x={ex:e}");
        println!("VERIFY lu_dense dual_n{n} rel_err_dx_vs_analytic={edx:e}");
        assert!(ex < TOL_WELL && edx < TOL_DUAL);
        // deriv() of each entry agrees with the .d field
        assert_eq!(s.x[0].deriv(), dx[0]);
    }
}
