//! Tests for stats::gp_matern52 against reference data from an independent NumPy/SciPy implementation
//! (Bessel-function Matern kernel, dense likelihood with finite-difference gradients, L-BFGS-B fits, dense
//! predictions). Tolerances as in the Python tests.
use agent_tools::stats::gp_matern52::{gp_build, gp_fit, matern52, nll_grad, to_theta, GpOptions};
use serde_json::Value;

const TOL_KERNEL: f64 = 1e-13; // relative to sf2
const TOL_NLL: f64 = 1e-9; // absolute, smooth2d
const TOL_NLL_HARD: f64 = 1e-8; // absolute, hard5d (cond(K) ~ 1e8)
const TOL_GRAD: f64 = 1e-7; // relative to max(1, max|g|)
const TOL_GRAD_HARD: f64 = 1e-5; // hard5d: the FD reference is noisy
const TOL_FIT_NLL: f64 = 1e-7; // library NLL may exceed the best L-BFGS-B NLL by at most this
const TOL_FIT_THETA: f64 = 1e-3; // max |theta - theta_ref|, well-determined cases
const TOL_PRED: f64 = 1e-9; // relative to max(1, max|value|)
const TOL_INTERP: f64 = 1e-5; // |mean - y| at the data, noise-free
const TOL_INTERP_VAR: f64 = 1e-6; // variance at the data / signal_sd^2

fn load() -> Value {
    let p = format!("{}/../tests/reference_data/gp_matern52.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn maxabs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}
fn scale(v: &[f64]) -> f64 {
    v.iter().fold(1.0f64, |m, x| m.max(x.abs()))
}

#[test]
fn kernel_vs_bessel() {
    let c = &load()["cases"]["kernel"];
    let (ell, sf2) = (vecf(&c["ell"]), c["sf2"].as_f64().unwrap());
    let a = c["a"].as_array().unwrap();
    let b = c["b"].as_array().unwrap();
    let kref = vecf(&c["k"]);
    let mut err = 0.0f64;
    for i in 0..a.len() {
        err = err.max((matern52(&vecf(&a[i]), &vecf(&b[i]), &ell, sf2) - kref[i]).abs() / sf2);
    }
    let e0 = (matern52(&vecf(&a[0]), &vecf(&a[0]), &ell, sf2) - c["k_zero_distance"].as_f64().unwrap()).abs();
    println!("VERIFY gp_matern52 rust_kernel max_rel_err_vs_bessel={err:e} zero_distance_err={e0:e}");
    assert!(err < TOL_KERNEL && e0 == 0.0);
}

#[test]
fn nll_and_gradient_vs_reference() {
    let r = load();
    for (name, tn, tg) in [("smooth2d", TOL_NLL, TOL_GRAD), ("hard5d", TOL_NLL_HARD, TOL_GRAD_HARD)] {
        let c = &r["cases"][name];
        let d = c["d"].as_u64().unwrap() as usize;
        let x = vecf(&c["x"]);
        let (ym, ys) = (c["y_mean"].as_f64().unwrap(), c["y_scale"].as_f64().unwrap());
        let yz: Vec<f64> = vecf(&c["y"]).iter().map(|v| (v - ym) / ys).collect();
        let (mut en, mut eg) = (0.0f64, 0.0f64);
        for t in c["theta_checks"].as_array().unwrap() {
            let th = vecf(&t["theta"]);
            let mut g = vec![0.0; th.len()];
            let f = nll_grad(&x, d, &yz, &th, t["fixed_noise_var"].as_f64(), &mut g);
            en = en.max((f - t["nll_std"].as_f64().unwrap()).abs());
            eg = eg.max(maxabs(&g, &vecf(&t["grad_fd"])) / scale(&g));
        }
        println!("VERIFY gp_matern52 rust_{name} max_abs_err_nll={en:e} max_rel_err_grad_vs_fd={eg:e}");
        assert!(en < tn && eg < tg, "{name}");
    }
}

#[test]
fn gradient_central_differences_both_sides() {
    let c = &load()["cases"]["smooth2d"];
    let x = vecf(&c["x"]);
    let y = vecf(&c["y"]);
    let (ym, ys) = (c["y_mean"].as_f64().unwrap(), c["y_scale"].as_f64().unwrap());
    let yz: Vec<f64> = y.iter().map(|v| (v - ym) / ys).collect();
    let mut worst = 0.0f64;
    let cases: [(Vec<f64>, Option<f64>); 3] = [
        (vec![-2.5, -2.0, 0.1, -3.0], None),
        (vec![1.0, 1.2, 0.7, -0.5], None),
        (vec![-1.0, -0.3, 0.5], Some(1e-3)),
    ];
    for (th, fv) in cases.iter() {
        let p = th.len();
        let mut g = vec![0.0; p];
        let mut tmp = vec![0.0; p];
        nll_grad(&x, 2, &yz, th, *fv, &mut g);
        let mut fd = vec![0.0; p];
        for k in 0..p {
            let h = 1e-6;
            let (mut tp, mut tm) = (th.clone(), th.clone());
            tp[k] += h;
            tm[k] -= h;
            fd[k] = (nll_grad(&x, 2, &yz, &tp, *fv, &mut tmp) - nll_grad(&x, 2, &yz, &tm, *fv, &mut tmp)) / (2.0 * h);
        }
        worst = worst.max(maxabs(&g, &fd) / scale(&fd));
    }
    let (lo, hi) = ([-4.0, -1.0], [2.0, 3.0]);
    let z = [-3.0, 2.5];
    let (_, dth) = to_theta(&z, &lo, &hi);
    let zp: Vec<f64> = z.iter().map(|v| v + 1e-6).collect();
    let zm: Vec<f64> = z.iter().map(|v| v - 1e-6).collect();
    let (tp, tm) = (to_theta(&zp, &lo, &hi).0, to_theta(&zm, &lo, &hi).0);
    let et = (0..2).fold(0.0f64, |m, k| m.max((dth[k] - (tp[k] - tm[k]) / 2e-6).abs()));
    println!("VERIFY gp_matern52 rust_gradient_fd max_rel_err={worst:e} transform_deriv_err={et:e}");
    assert!(worst < 1e-6 && et < 1e-9);
}

#[test]
fn fit_vs_scipy_lbfgsb() {
    let r = load();
    let cases = &r["cases"];
    for name in ["smooth2d", "smooth2d_fixed_noise", "hard5d"] {
        let c = if name == "hard5d" { &cases["hard5d"] } else { &cases["smooth2d"] };
        let (rf, noise) = match name {
            "smooth2d" => (c, None),
            "smooth2d_fixed_noise" => (&c["fixed_noise"], c["fixed_noise"]["noise_sd"].as_f64()),
            _ => (c, c["noise_sd"].as_f64()),
        };
        let d = c["d"].as_u64().unwrap() as usize;
        let (x, y) = (vecf(&c["x"]), vecf(&c["y"]));
        let res = gp_fit(&x, d, &y, &GpOptions { noise_sd: noise, ..GpOptions::default() }).unwrap();
        let m = &res.model;
        let diff = res.nll_std - rf["best"]["nll_std"].as_f64().unwrap();
        let mut th: Vec<f64> = m.length_scales.iter().map(|v| v.ln()).collect();
        th.push((m.signal_sd / m.y_scale).ln());
        if noise.is_none() {
            th.push((m.noise_sd / m.y_scale).ln());
        }
        let dth = maxabs(&th, &vecf(&rf["best"]["theta"]));
        let lnfev: u64 = rf["lbfgsb_fits"].as_array().unwrap().iter().map(|f| f["nfev"].as_u64().unwrap()).sum();
        println!(
            "VERIFY gp_matern52 rust_fit_{name} nll_minus_lbfgsb_best={diff:e} max_abs_diff_theta={dth:e} nfev={} nit={} success={} lbfgsb_nfev={lnfev}",
            res.nfev, res.nit, res.success
        );
        assert!(diff < TOL_FIT_NLL, "{name}");
        if name != "hard5d" {
            assert!(res.success && dth < TOL_FIT_THETA, "{name}");
        }
    }
}

#[test]
fn predict_vs_dense_solve_and_interpolation() {
    let r = load();
    let c = &r["cases"]["smooth2d"];
    let f = &c["fitted"];
    let (x, y) = (vecf(&c["x"]), vecf(&c["y"]));
    let gp = gp_build(&x, 2, &y, &vecf(&f["ell"]), f["signal_sd"].as_f64().unwrap(), f["noise_sd"].as_f64().unwrap(), None).unwrap();
    let g = vecf(&c["test_points"]);
    let (mean, var) = gp.predict(&g);
    let cov = gp.posterior_cov(&g[..8], &g[6..]);
    let (pm, pv, pc) = (vecf(&c["pred_mean"]), vecf(&c["pred_var"]), vecf(&c["cov_ab"]));
    let em = maxabs(&mean, &pm) / scale(&pm);
    let ev = maxabs(&var, &pv) / scale(&pv);
    let ec = maxabs(&cov, &pc) / scale(&pc);
    let el = (gp.log_marginal_likelihood - f["log_marginal_likelihood"].as_f64().unwrap()).abs();
    println!("VERIFY gp_matern52 rust_predict_smooth2d rel_err_mean={em:e} rel_err_var={ev:e} rel_err_cov={ec:e} abs_err_lml={el:e}");
    assert!(em.max(ev).max(ec) < TOL_PRED && el < 1e-6);

    let c1 = &r["cases"]["interp1d"];
    let (x1, y1) = (vecf(&c1["x"]), vecf(&c1["y"]));
    let (ell, sd) = (vecf(&c1["ell"]), c1["signal_sd"].as_f64().unwrap());
    let gp1 = gp_build(&x1, 1, &y1, &ell, sd, c1["noise_sd"].as_f64().unwrap(), None).unwrap();
    let (m1, v1) = gp1.predict(&vecf(&c1["test_points"]));
    let e1 = (maxabs(&m1, &vecf(&c1["pred_mean"])) / scale(&vecf(&c1["pred_mean"])))
        .max(maxabs(&v1, &vecf(&c1["pred_var"])) / scale(&vecf(&c1["pred_var"])));
    println!("VERIFY gp_matern52 rust_predict_interp1d rel_err_mean_var={e1:e}");
    assert!(e1 < TOL_PRED);
    let gp0 = gp_build(&x1, 1, &y1, &ell, sd, 0.0, None).unwrap();
    let (m0, v0) = gp0.predict(&x1);
    let ei = maxabs(&m0, &y1);
    let evi = v0.iter().fold(0.0f64, |m, v| m.max(*v)) / (sd * sd);
    println!("VERIFY gp_matern52 rust_interpolation max_abs_err_mean_at_data={ei:e} max_rel_var_at_data={evi:e}");
    assert!(ei < TOL_INTERP && evi < TOL_INTERP_VAR);
}

#[test]
fn options_and_failures() {
    let o = GpOptions::default();
    assert!(o.noise_sd.is_none() && o.length_starts == vec![0.1, 0.3, 1.0] && o.n_random_starts == 0);
    assert!((o.log_length_bounds.0 - 0.02f64.ln()).abs() < 1e-15 && o.bfgs.gtol == 1e-5);
    let c = &load()["cases"]["smooth2d"];
    let (x, y) = (vecf(&c["x"]), vecf(&c["y"]));
    let oo = GpOptions { length_starts: vec![], n_random_starts: 3, seed: 9, ..GpOptions::default() };
    let a = gp_fit(&x, 2, &y, &oo).unwrap();
    let b = gp_fit(&x, 2, &y, &oo).unwrap();
    assert_eq!(a.start_nll, b.start_nll);
    let mut x2 = x.clone();
    x2.extend_from_slice(&x[..2]);
    let mut y2 = y.clone();
    y2.push(y[0]);
    assert!(gp_build(&x2, 2, &y2, &[1.0, 1.0], 1.0, 0.0, Some((0.0, 1e-6))).is_err());
    println!("VERIFY gp_matern52 rust_options_failures defaults,seeded_restarts,singular_build=ok");
}
