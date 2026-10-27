//! Tests for stats::rtd_analysis against reference data from library/tests/generators/rtd_analysis.py
//! (SciPy gamma / quad / linregress / curve_fit, closed-closed Laplace-pole series) and exact solutions.
use agent_tools::stats::rtd_analysis::*;
use serde_json::Value;

/// Moments of a truncated record vs the independent SciPy computation (same algorithm), relative.
const TOL_MOMENTS_VS_SCIPY: f64 = 1e-9;
/// Exponential record: tail extrapolation is exact up to the trapezoid error, relative.
const TOL_EXP_TAIL: f64 = 1e-5;
/// Variance of an early-truncated tanks record with tail extrapolation vs the exact value, relative
/// (the gamma tail is only approximately exponential).
const TOL_TANKS_VAR_TAIL: f64 = 5e-2;
/// Same for dispersion records (tails are nearly exponential), relative.
const TOL_DISP_VAR_TAIL: f64 = 1e-2;
/// Moment fits on exact distribution moments recover the parameter, relative.
const TOL_MOMENT_INVERSION: f64 = 1e-9;
/// Closed-closed: reference moments come from a pole series accurate to about 1e-7.
const TOL_CC_SERIES: f64 = 1e-5;
/// Least squares vs curve_fit (relative in parameter and tau).
const TOL_LSQ_VS_CURVEFIT: f64 = 1e-5;
/// Least squares on noiseless data vs the true parameters, relative.
const TOL_LSQ_EXACT: f64 = 1e-6;
/// Least squares on 3 % noise vs the true parameters, relative.
const TOL_LSQ_NOISY: f64 = 5e-2;
/// ln Gamma vs scipy.special.gammaln, relative.
const TOL_LNGAMMA: f64 = 1e-13;

fn load() -> Value {
    let p = format!("{}/../tests/reference_data/rtd_analysis.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn ln_gamma_vs_scipy() {
    let r = load();
    let g = &r["cases"]["ln_gamma"];
    let mut worst = 0f64;
    for (x, y) in vec(&g["x"]).iter().zip(vec(&g["ref"])) {
        worst = worst.max(((ln_gamma(*x) - y) / y.abs().max(1.0)).abs());
    }
    println!("VERIFY rtd_analysis ln_gamma max_rel_err_vs_scipy_gammaln={worst:e}");
    assert!(worst < TOL_LNGAMMA);
}

#[test]
fn moments_of_truncated_records() {
    let r = load();
    for d in r["cases"]["datasets"].as_array().unwrap() {
        let name = d["name"].as_str().unwrap();
        let (t, c) = (vec(&d["t"]), vec(&d["C"]));
        let rm = &d["ref_moments"];
        let with = rtd_moments(&t, &c, 0.0, true);
        let without = rtd_moments(&t, &c, 0.0, false);
        let e_ref = f(&rm["mean_tail"]);
        let (ev, ev0) = (rel(with.variance, f(&rm["var_tail"])), rel(without.variance, f(&rm["var_notail"])));
        let em = rel(with.mean, e_ref);
        println!("VERIFY rtd_analysis {name} moments_rel_err_vs_scipy mean={em:e} var_tail={ev:e} var_notail={ev0:e}");
        assert!(em < TOL_MOMENTS_VS_SCIPY && ev < TOL_MOMENTS_VS_SCIPY && ev0 < TOL_MOMENTS_VS_SCIPY, "{name}");
        match (rm["lam"].as_f64(), with.tail_decay_time) {
            (Some(a), Some(b)) => assert!(rel(b, a) < 1e-9, "{name} lam"),
            (None, None) => {}
            other => panic!("{name}: tail decay time mismatch {other:?}"),
        }
        let (xm, xv) = (f(&d["exact_mean"]), f(&d["exact_variance"]));
        let (err_t, err_n) = (rel(with.variance, xv), rel(without.variance, xv));
        println!(
            "VERIFY rtd_analysis {name} var_rel_err_vs_exact with_tail={err_t:e} without_tail={err_n:e} mean_with_tail={:e}",
            rel(with.mean, xm)
        );
        match d["kind"].as_str().unwrap() {
            "tanks" if name.starts_with("cstr") => assert!(err_t < TOL_EXP_TAIL && rel(with.mean, xm) < TOL_EXP_TAIL, "{name}"),
            "tanks" if d["noise_rel_peak"].as_f64().unwrap() == 0.0 => assert!(err_t < TOL_TANKS_VAR_TAIL && err_t < err_n, "{name}"),
            "oo" | "cc" if d["noise_rel_peak"].as_f64().unwrap() == 0.0 => assert!(err_t < TOL_DISP_VAR_TAIL && err_t < err_n, "{name}"),
            _ => {
                // noisy record: negative samples in the tail disable the extrapolation
                assert!(with.tail_decay_time.is_none(), "{name}");
            }
        }
        // E integrates to 1 including the tail, so over the record alone it is below 1 (trapezoid
        // overshoot on convex decay is far smaller than 1e-6 here)
        let area: f64 = (1..t.len()).map(|i| 0.5 * (with.e[i] + with.e[i - 1]) * (t[i] - t[i - 1])).sum();
        assert!(area < 1.0 + 1e-6 && area > 0.5, "{name} area {area}");
    }
}

#[test]
fn exponential_with_pulse_correction() {
    // CSTR (lam = 40 s) driven by a square pulse of width 10 s: response is known in closed form.
    let (lam, w) = (40.0f64, 10.0f64);
    let t: Vec<f64> = (0..40001).map(|i| i as f64 * 0.01).collect();
    let c: Vec<f64> = t
        .iter()
        .map(|&x| if x < w { (1.0 - (-x / lam).exp()) / w } else { (-x / lam).exp() * ((w / lam).exp() - 1.0) / w })
        .collect();
    let m = rtd_moments(&t, &c, w, true);
    let (em, ev) = (rel(m.mean, lam), rel(m.variance, lam * lam));
    println!("VERIFY rtd_analysis pulse_cstr mean_rel_err={em:e} var_rel_err={ev:e}");
    assert!(em < 1e-4 && ev < 1e-4);
    let raw = rtd_moments(&t, &c, 0.0, true);
    assert!((raw.mean - lam - w / 2.0).abs() < 1e-2 * lam);
}

#[test]
fn two_plane_moment_difference_is_exact() {
    // Advection-diffusion Green's function at two planes (source test): D_L recovered, any end conditions.
    let (u, d, x1, x2) = (0.3f64, 0.05f64, 20.0f64, 60.0f64);
    let n = 400001;
    let t: Vec<f64> = (0..n).map(|i| 1e-3 + (2000.0 - 1e-3) * i as f64 / (n - 1) as f64).collect();
    let sig = |x: f64| -> Vec<f64> {
        t.iter().map(|&ti| (-(x - u * ti).powi(2) / (4.0 * d * ti)).exp() / (4.0 * std::f64::consts::PI * d * ti).sqrt()).collect()
    };
    let (m1, m2) = (rtd_moments(&t, &sig(x1), 0.0, false), rtd_moments(&t, &sig(x2), 0.0, false));
    let p = fit_dispersion_planes(m1.mean, m1.variance, m2.mean, m2.variance, x2 - x1);
    let (eu, ed) = (rel(p.u, u), rel(p.d_l, d));
    println!("VERIFY rtd_analysis two_plane u_rel_err={eu:e} D_L_rel_err={ed:e}");
    assert!(eu < 1e-3 && ed < 1e-3);
}

#[test]
fn moment_inversions_on_exact_moments() {
    let r = load();
    for d in r["cases"]["datasets"].as_array().unwrap() {
        let name = d["name"].as_str().unwrap();
        let (kind, truth, tau) = (d["kind"].as_str().unwrap(), f(&d["param"]), f(&d["tau"]));
        if d["noise_rel_peak"].as_f64().unwrap() != 0.0 {
            continue;
        }
        let (xm, xv) = (f(&d["exact_mean"]), f(&d["exact_variance"]));
        let (fit, tol) = match kind {
            "tanks" => (fit_moments(xm, xv, RtdModel::Tanks, BoundaryCondition::OpenOpen), TOL_MOMENT_INVERSION),
            "oo" => (fit_moments(xm, xv, RtdModel::Dispersion, BoundaryCondition::OpenOpen), TOL_MOMENT_INVERSION),
            _ => (fit_moments(xm, xv, RtdModel::Dispersion, BoundaryCondition::ClosedClosed), TOL_CC_SERIES),
        };
        assert!(fit.success);
        let ep = rel(fit.param, truth);
        println!("VERIFY rtd_analysis {name} moment_inversion_exact param_rel_err={ep:e}");
        assert!(ep < tol, "{name}");
        if kind == "oo" {
            assert!(rel(fit.tau, tau) < TOL_MOMENT_INVERSION, "{name} tau");
        }
        // forward formula
        if kind != "tanks" {
            let bc = if kind == "oo" { BoundaryCondition::OpenOpen } else { BoundaryCondition::ClosedClosed };
            let e = rel(sigma2_dispersion(truth, bc) * tau * tau, xv);
            println!("VERIFY rtd_analysis {name} sigma2_formula_vs_quad rel_err={e:e}");
            assert!(e < if kind == "oo" { TOL_MOMENT_INVERSION } else { TOL_CC_SERIES });
        }
    }
}

#[test]
fn moment_fits_on_truncated_records() {
    // Parameters recovered from moments of the truncated record (with tail), relative tolerances.
    let r = load();
    for d in r["cases"]["datasets"].as_array().unwrap() {
        if d["noise_rel_peak"].as_f64().unwrap() != 0.0 {
            continue;
        }
        let name = d["name"].as_str().unwrap();
        let m = rtd_moments(&vec(&d["t"]), &vec(&d["C"]), 0.0, true);
        let (kind, truth) = (d["kind"].as_str().unwrap(), f(&d["param"]));
        let fit = match kind {
            "tanks" => fit_moments(m.mean, m.variance, RtdModel::Tanks, BoundaryCondition::OpenOpen),
            "oo" => fit_moments(m.mean, m.variance, RtdModel::Dispersion, BoundaryCondition::OpenOpen),
            _ => fit_moments(m.mean, m.variance, RtdModel::Dispersion, BoundaryCondition::ClosedClosed),
        };
        let e = rel(fit.param, truth);
        println!("VERIFY rtd_analysis {name} moment_fit_param_rel_err_vs_truth={e:e}");
        assert!(fit.success && e < 0.06, "{name}");
    }
}

#[test]
fn least_squares_vs_curve_fit_and_truth() {
    let r = load();
    for d in r["cases"]["datasets"].as_array().unwrap() {
        let Some(cf) = d.get("curve_fit") else { continue };
        let name = d["name"].as_str().unwrap();
        let noisy = d["noise_rel_peak"].as_f64().unwrap() != 0.0;
        let model = if d["kind"] == "tanks" { RtdModel::Tanks } else { RtdModel::Dispersion };
        let (t, e) = (vec(&d["t"]), vec(&d["E"]));
        let (truth, tau) = (f(&d["param"]), f(&d["tau"]));
        let free = fit_lsq(&t, &e, model, None);
        let fixed = fit_lsq(&t, &e, model, Some(tau));
        let (ep, et) = (rel(free.param, f(&cf["p"])), rel(free.tau, f(&cf["tau"])));
        let ef = rel(fixed.param, f(&cf["p_fixed_tau"]));
        let tr = rel(free.param, truth).max(rel(free.tau, tau));
        println!(
            "VERIFY rtd_analysis {name} lsq_vs_curve_fit param_rel={ep:e} tau_rel={et:e} fixed_tau_param_rel={ef:e} vs_truth_rel={tr:e} nfev_free={} nfev_fixed={}",
            free.nfev, fixed.nfev
        );
        assert!(free.success && fixed.success, "{name}: {} {}", free.message, fixed.message);
        if noisy {
            assert!(ep < TOL_LSQ_VS_CURVEFIT && et < TOL_LSQ_VS_CURVEFIT && ef < TOL_LSQ_VS_CURVEFIT, "{name}");
            assert!(tr < TOL_LSQ_NOISY, "{name}");
            assert!((free.sse - f(&cf["sse"])).abs() < 1e-6 * f(&cf["sse"]), "{name} sse");
        } else {
            assert!(tr < TOL_LSQ_EXACT, "{name}");
        }
    }
}

#[test]
fn noisy_records_moments_fail_but_lsq_works() {
    // Hard case: 3 % noise. Moments are biased high (clipped noise in the tail); lsq is not.
    let r = load();
    for d in r["cases"]["datasets"].as_array().unwrap() {
        if d["noise_rel_peak"].as_f64().unwrap() == 0.0 {
            continue;
        }
        let name = d["name"].as_str().unwrap();
        let m = rtd_moments(&vec(&d["t"]), &vec(&d["C"]), 0.0, true);
        let model = if d["kind"] == "tanks" { RtdModel::Tanks } else { RtdModel::Dispersion };
        let mf = fit_moments(m.mean, m.variance, model, BoundaryCondition::OpenOpen);
        let lf = fit_lsq(&vec(&d["t"]), &vec(&d["E"]), model, None);
        let truth = f(&d["param"]);
        let (em, el) = (rel(mf.param, truth), rel(lf.param, truth));
        println!("VERIFY rtd_analysis {name} param_rel_err_vs_truth moments={em:e} lsq={el:e}");
        assert!(el < em, "{name}: lsq should beat moments on noisy data");
    }
}

#[test]
fn failure_modes() {
    // tail_fit: too short, non-positive, growing
    assert!(tail_fit(&[0.0, 1.0, 2.0], &[1.0, 0.5, 0.2], 0.25).is_none());
    let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mut c: Vec<f64> = t.iter().map(|x| (-x / 3.0).exp()).collect();
    assert!((tail_fit(&t, &c, 0.25).unwrap() - 3.0).abs() < 1e-9);
    c[19] = 0.0;
    assert!(tail_fit(&t, &c, 0.25).is_none());
    let g: Vec<f64> = t.iter().map(|x| (x / 3.0).exp()).collect();
    assert!(tail_fit(&t, &g, 0.25).is_none());
    // moment fits
    assert!(!fit_moments(10.0, 0.0, RtdModel::Tanks, BoundaryCondition::OpenOpen).success);
    assert!(!fit_moments(10.0, f64::NAN, RtdModel::Tanks, BoundaryCondition::OpenOpen).success);
    assert!(!fit_moments(10.0, 100.0, RtdModel::Dispersion, BoundaryCondition::ClosedClosed).success);
    // lsq: peak narrower than the search range allows (Pe = 2e4 > 1e4) -> flagged, not silent
    let tt: Vec<f64> = (0..4001).map(|i| 0.8 + 0.4 * i as f64 / 4000.0).collect();
    let ee: Vec<f64> = tt.iter().map(|&x| e_dispersion(x, 2e4)).collect();
    let l = fit_lsq(&tt, &ee, RtdModel::Dispersion, Some(1.0));
    assert!(!l.success && l.message.contains("edge"));
    // lsq: no area
    assert!(!fit_lsq(&[0.0, 1.0], &[0.0, 0.0], RtdModel::Tanks, None).success);
    println!("VERIFY rtd_analysis failure_modes all_ok=1");
}
