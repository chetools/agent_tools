//! Tests for stats::imse_batch against brute-force greedy re-conditioning (dense NumPy, generator), and the
//! rank-one downdate against re-conditioning with the library GP. Tolerances as in the Python tests.
use agent_tools::stats::gp_matern52::{gp_build, GpModel};
use agent_tools::stats::imse_batch::imse_batch;
use serde_json::Value;

const TOL_RED: f64 = 1e-7; // reductions vs brute force, relative to the largest first score
const TOL_FIRST: f64 = 1e-5; // first-step scores vs brute force (cancellation in the reference)
const TOL_DOWNDATE: f64 = 1e-7; // downdated covariance vs re-conditioned, relative to max|C|

fn load() -> Value {
    let p = format!("{}/../tests/reference_data/imse_batch.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn model(r: &Value) -> GpModel {
    let g = &r["cases"]["gp"];
    gp_build(&vecf(&g["x"]), 2, &vecf(&g["y"]), &vecf(&g["ell"]), g["signal_sd"].as_f64().unwrap(), g["noise_sd"].as_f64().unwrap(), None).unwrap()
}

#[test]
fn vs_brute_force() {
    let r = load();
    let gp = model(&r);
    for name in ["same_grid_8x8", "grid_vs_random_ref"] {
        let c = &r["cases"][name];
        let (cand, refp) = (vecf(&c["candidates"]), vecf(&c["reference"]));
        let (nc, nr) = (cand.len() / 2, refp.len() / 2);
        let b = c["batch"].as_u64().unwrap() as usize;
        let res = imse_batch(&gp.posterior_cov(&cand, &cand), &gp.posterior_cov(&refp, &cand), nc, nr, gp.noise_sd * gp.noise_sd, b);
        let idx: Vec<usize> = c["indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let fs = vecf(&c["first_scores"]);
        let s = fs.iter().fold(0.0f64, |m, v| m.max(*v));
        let er = res.reduction.iter().zip(vecf(&c["reduction"])).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / s;
        let ef = res.first_scores.iter().zip(&fs).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / s;
        println!("VERIFY imse_batch rust_{name} indices_equal={} rel_err_reduction={er:e} rel_err_first_scores={ef:e}", res.indices == idx);
        assert!(res.indices == idx && er < TOL_RED && ef < TOL_FIRST, "{name}");
    }
}

#[test]
fn downdate_equals_reconditioning() {
    let r = load();
    let gp = model(&r);
    let g = vecf(&r["cases"]["same_grid_8x8"]["candidates"]);
    let m = g.len() / 2;
    let c = gp.posterior_cov(&g, &g);
    let res = imse_batch(&c, &c, m, m, gp.noise_sd * gp.noise_sd, 3);
    let mut x3 = gp.x.clone();
    let mut y3 = vecf(&r["cases"]["gp"]["y"]);
    for &j in &res.indices {
        x3.extend_from_slice(&g[2 * j..2 * j + 2]);
        y3.push(0.0);
    }
    let gp3 = gp_build(&x3, 2, &y3, &gp.length_scales, gp.signal_sd, gp.noise_sd, Some((gp.y_mean, gp.y_scale))).unwrap();
    let c3 = gp3.posterior_cov(&g, &g);
    let cmax = c.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    // downdate written out independently of imse_batch's internals
    let mut cd = c.clone();
    for &j in &res.indices {
        let s = (cd[j * m + j] + gp.noise_sd * gp.noise_sd).sqrt();
        let u: Vec<f64> = (0..m).map(|i| cd[i * m + j] / s).collect();
        for i in 0..m {
            for k in 0..m {
                cd[i * m + k] -= u[i] * u[k];
            }
        }
    }
    let err = cd.iter().zip(&c3).fold(0.0f64, |a, (p, q)| a.max((p - q).abs())) / cmax;
    let total: f64 = (0..m).map(|i| c[i * m + i] - c3[i * m + i]).sum::<f64>() / m as f64;
    let es = (res.reduction.iter().sum::<f64>() - total).abs() / total;
    println!("VERIFY imse_batch rust_downdate_vs_reconditioning rel_err={err:e} sum_reduction_vs_total_drop_rel={es:e}");
    assert!(err < TOL_DOWNDATE && es < TOL_DOWNDATE);
}

#[test]
fn edge_cases() {
    let c = [0.0, 0.0, 0.0, 1.0];
    let res = imse_batch(&c, &c, 2, 2, 0.0, 3);
    assert!(res.first_scores[0] == 0.0 && res.indices[0] == 1);
    let res = imse_batch(&[1.0], &[1.0], 1, 1, 0.5, 2);
    assert!(res.indices == vec![0, 0] && (res.reduction[0] - 1.0 / 1.5).abs() < 1e-15);
    println!("VERIFY imse_batch rust_edge_cases zero_variance,replicates=ok");
}
