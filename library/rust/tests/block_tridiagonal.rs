//! Tests for linalg::block_tridiagonal against numpy / scipy.solve_banded / classical Thomas reference data.
use agent_tools::autodiff::{Dual, Real};
use agent_tools::linalg::block_tridiagonal::{block_tridiag_factor, block_tridiag_solve};

/// Dominant systems: solution vs numpy.linalg.solve, relative to max|x|.
const TOL_DOMINANT: f64 = 1e-12;
/// Weak-dominance / zero-diagonal-block systems: error <= factor * eps * cond.
const TOL_COND_FACTOR: f64 = 10.0;
/// Backward error ||Ax-b||_inf / (||A||_inf ||x||_inf + ||b||_inf).
const TOL_BACKWARD: f64 = 1e-14;
/// m = 1 vs the classical Thomas algorithm, relative to max|x|.
const TOL_THOMAS: f64 = 1e-13;
/// Dual solution derivative vs analytic A^-1 (b' - A' x), relative to max|dx|.
const TOL_DUAL: f64 = 1e-11;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/block_tridiagonal.json", env!("CARGO_MANIFEST_DIR"));
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
/// Backward error from the block structure directly (independent of the solver's code path).
fn backward(l: &[f64], d: &[f64], u: &[f64], x: &[f64], b: &[f64], m: usize, nb: usize) -> f64 {
    let mm = m * m;
    let (mut res, mut an) = (0f64, 0f64);
    for i in 0..nb {
        for r in 0..m {
            let (mut s, mut row) = (-b[i * m + r], 0.0);
            for c in 0..m {
                let a = d[i * mm + r * m + c];
                s += a * x[i * m + c];
                row += a.abs();
                if i > 0 {
                    let a = l[(i - 1) * mm + r * m + c];
                    s += a * x[(i - 1) * m + c];
                    row += a.abs();
                }
                if i + 1 < nb {
                    let a = u[i * mm + r * m + c];
                    s += a * x[(i + 1) * m + c];
                    row += a.abs();
                }
            }
            res = res.max(s.abs());
            an = an.max(row);
        }
    }
    res / (an * maxabs(x) + maxabs(b))
}

/// Factor once, solve all right-hand sides; returns (max rel err vs reference, max backward error, growth).
fn run_case(c: &serde_json::Value) -> (f64, f64, f64) {
    let (m, nb) = (c["m"].as_u64().unwrap() as usize, c["nb"].as_u64().unwrap() as usize);
    let (l, d, u) = (vec(&c["lower"]), vec(&c["diag"]), vec(&c["upper"]));
    let f = block_tridiag_factor(&l, &d, &u, m, nb);
    assert!(f.success, "{}: {}", c["name"], f.message);
    let (mut ex, mut eb) = (0f64, 0f64);
    for (b, xr) in c["b"].as_array().unwrap().iter().zip(c["x"].as_array().unwrap()) {
        let (b, xr) = (vec(b), vec(xr));
        let s = block_tridiag_solve(&f, &b);
        assert!(s.success);
        ex = ex.max(rel_err(&s.x, &xr));
        eb = eb.max(backward(&l, &d, &u, &s.x, &b, m, nb));
        if let Some(xt) = c.get("x_thomas") {
            let idx = c["b"].as_array().unwrap().iter().position(|v| vec(v) == b).unwrap();
            let e = rel_err(&s.x, &vec(&xt[idx]));
            assert!(e < TOL_THOMAS, "{} vs Thomas: {e:e}", c["name"]);
        }
    }
    (ex, eb, f.growth)
}

#[test]
fn dominant_systems_and_thomas() {
    let r = load();
    for c in r["cases"]["dominant"].as_array().unwrap() {
        let (ex, eb, g) = run_case(c);
        let name = c["name"].as_str().unwrap();
        println!("VERIFY block_tridiagonal {name} max_rel_err_x_vs_numpy={ex:e}");
        println!("VERIFY block_tridiagonal {name} max_backward_error={eb:e}");
        println!("VERIFY block_tridiagonal {name} growth_max_norm_C={g:e}");
        assert!(ex < TOL_DOMINANT && eb < TOL_BACKWARD && g < 1.0);
        if c.get("x_thomas").is_some() {
            println!("VERIFY block_tridiagonal {name} compared_with_classical_thomas=1");
        }
    }
}

#[test]
fn hard_weak_dominance_and_in_block_pivoting() {
    let r = load();
    let eps = f64::EPSILON;
    for key in ["weak", "pivot"] {
        for c in r["cases"][key].as_array().unwrap() {
            let (ex, eb, g) = run_case(c);
            let name = c["name"].as_str().unwrap();
            let cond = c["cond"].as_f64().unwrap();
            println!("VERIFY block_tridiagonal {name} max_rel_err_x_vs_numpy={ex:e}");
            println!("VERIFY block_tridiagonal {name} err_over_eps_cond={:e}", ex / (eps * cond));
            println!("VERIFY block_tridiagonal {name} max_backward_error={eb:e}");
            println!("VERIFY block_tridiagonal {name} growth_max_norm_C={g:e}");
            assert!(ex < TOL_COND_FACTOR * eps * cond && eb < TOL_BACKWARD, "{name}");
        }
    }
}

#[test]
fn singular_diagonal_block_is_reported() {
    let r = load();
    for c in r["cases"]["fail"].as_array().unwrap() {
        let (m, nb) = (c["m"].as_u64().unwrap() as usize, c["nb"].as_u64().unwrap() as usize);
        let f = block_tridiag_factor(&vec(&c["lower"]), &vec(&c["diag"]), &vec(&c["upper"]), m, nb);
        assert!(!f.success);
        let want = format!("diagonal block {}", c["fail_block"].as_u64().unwrap());
        assert!(f.message.contains(&want) && f.message.contains("singular"), "{}", f.message);
        let s = block_tridiag_solve(&f, &vec![1.0; m * nb]);
        assert!(!s.success && s.x.is_empty());
    }
    println!("VERIFY block_tridiagonal fail cases_reported={}", r["cases"]["fail"].as_array().unwrap().len());
}

#[test]
fn dual_solution_derivative() {
    let r = load();
    for c in r["cases"]["param"].as_array().unwrap() {
        let (m, nb) = (c["m"].as_u64().unwrap() as usize, c["nb"].as_u64().unwrap() as usize);
        let p = Dual::new(c["p0"].as_f64().unwrap(), 1.0);
        let lin = |k0: &str, k1: &str| -> Vec<Dual<f64>> {
            vec_pair(&c[k0], &c[k1]).into_iter().map(|(a, b)| Dual::cst(a) + Dual::cst(b) * p).collect()
        };
        let (l, d, u, b) = (lin("L0", "L1"), lin("D0", "D1"), lin("U0", "U1"), lin("b0", "b1"));
        let f = block_tridiag_factor(&l, &d, &u, m, nb);
        assert!(f.success);
        let s = block_tridiag_solve(&f, &b);
        let x: Vec<f64> = s.x.iter().map(|z| z.v).collect();
        let dx: Vec<f64> = s.x.iter().map(|z| z.d).collect();
        let (ex, edx) = (rel_err(&x, &vec(&c["x"])), rel_err(&dx, &vec(&c["dx"])));
        println!("VERIFY block_tridiagonal dual_m{m}_nb{nb} rel_err_x={ex:e}");
        println!("VERIFY block_tridiagonal dual_m{m}_nb{nb} rel_err_dx_vs_analytic={edx:e}");
        assert!(ex < TOL_DOMINANT && edx < TOL_DUAL);
    }
}

fn vec_pair(a: &serde_json::Value, b: &serde_json::Value) -> Vec<(f64, f64)> {
    vec(a).into_iter().zip(vec(b)).collect()
}
