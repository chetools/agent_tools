//! Tests for interpolate::hermite_extremum and interpolate::cubic_hermite against reference data
//! (exact cubic roots; SciPy CubicHermiteSpline + bounded minimize_scalar) and hand-derived edge cases.
use agent_tools::autodiff::dual::{Dual, Real};
use agent_tools::interpolate::cubic_hermite::cubic_hermite;
use agent_tools::interpolate::hermite_extremum::{hermite_extremum, Extremum};

/// Exact cubic: value and location of the extremum (abs; values are O(1..10)).
const TOL_CUBIC_VALUE: f64 = 1e-13;
const TOL_CUBIC_T: f64 = 1e-12;
/// Sine / Gaussian vs SciPy dense+refine: the value is flat at an extremum so it agrees far better
/// than the location, which is limited by minimize_scalar (xatol 1e-13, effectively ~1e-8).
const TOL_REF_VALUE: f64 = 1e-12;
const TOL_REF_T: f64 = 1e-7;
/// Evaluator vs CubicHermiteSpline, absolute.
const TOL_EVAL: f64 = 1e-13;
/// Observed order of the Gaussian peak-value error between the finest spacings (4th order expected).
const ORDER_RANGE: (f64, f64) = (3.7, 4.3);
/// Dual (AD) derivative of the peak value vs central finite differences (h = 1e-6), absolute.
const TOL_AD: f64 = 1e-7;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/hermite_extremum.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn f(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn exact_cubic() {
    let r = load();
    let c = &r["cases"]["cubic"];
    let (t, y, dy) = (vec(&c["t"]), vec(&c["y"]), vec(&c["dy"]));
    for (name, kind) in [("max", Extremum::Max), ("min", Extremum::Min)] {
        let e = hermite_extremum(&t, &y, &dy, kind);
        assert!(e.success);
        let (et, ev) = ((e.t - f(&c[name]["t"])).abs(), (e.value - f(&c[name]["value"])).abs());
        println!("VERIFY hermite_extremum cubic_{name} err_t={et:e} err_value={ev:e} interval={}", e.interval);
        assert!(et < TOL_CUBIC_T && ev < TOL_CUBIC_VALUE, "{name}: {et:e} {ev:e}");
    }
    // the interpolant reproduces the cubic
    let co = vec(&c["coef"]);
    let mut worst = 0f64;
    for i in 0..=400 {
        let x = 3.6 * i as f64 / 400.0;
        let p = ((co[0] * x + co[1]) * x + co[2]) * x + co[3];
        worst = worst.max((cubic_hermite(&t, &y, &dy, x) - p).abs());
    }
    println!("VERIFY hermite_extremum cubic_reproduce max_abs_err={worst:e}");
    assert!(worst < TOL_CUBIC_VALUE);
}

#[test]
fn sine_vs_scipy() {
    let r = load();
    let c = &r["cases"]["sine"];
    let (t, y, dy) = (vec(&c["t"]), vec(&c["y"]), vec(&c["dy"]));
    for (name, kind) in [("max", Extremum::Max), ("min", Extremum::Min)] {
        let e = hermite_extremum(&t, &y, &dy, kind);
        let (et, ev) = ((e.t - f(&c[name]["t"])).abs(), (e.value - f(&c[name]["value"])).abs());
        println!("VERIFY hermite_extremum sine_{name} err_t={et:e} err_value={ev:e} interval={}", e.interval);
        assert!(et < TOL_REF_T && ev < TOL_REF_VALUE);
        assert_eq!(e.interval as u64, c[name]["interval"].as_u64().unwrap());
    }
}

#[test]
fn gaussian_convergence() {
    let r = load();
    let c = &r["cases"]["gaussian"];
    let (t0, truth) = (f(&c["t0"]), f(&c["true_value"]));
    let mut errs = Vec::new();
    for run in c["runs"].as_array().unwrap() {
        let (t, y, dy) = (vec(&run["t"]), vec(&run["y"]), vec(&run["dy"]));
        let e = hermite_extremum(&t, &y, &dy, Extremum::Max);
        let (h, n) = (f(&run["h"]), run["n"].as_u64().unwrap());
        let (rt, rv) = (f(&run["ref"]["t"]), f(&run["ref"]["value"]));
        let (d_t, d_v) = ((e.t - rt).abs(), (e.value - rv).abs());
        let (err_v, err_t) = ((truth - e.value).abs(), (e.t - t0).abs());
        println!(
            "VERIFY hermite_extremum gaussian_n{n} h={h:.5e} err_value_vs_true={err_v:e} err_t_vs_true={err_t:e} diff_value_vs_scipy={d_v:e} diff_t_vs_scipy={d_t:e}"
        );
        assert!(d_t < TOL_REF_T && d_v < TOL_REF_VALUE);
        errs.push((h, err_v));
    }
    let k = errs.len();
    for w in errs.windows(2).skip(k - 3) {
        let order = (w[0].1 / w[1].1).ln() / (w[0].0 / w[1].0).ln();
        println!("VERIFY hermite_extremum gaussian_order h={:.5e}->{:.5e} observed_order={order:.3}", w[0].0, w[1].0);
        assert!(order > ORDER_RANGE.0 && order < ORDER_RANGE.1, "order {order}");
    }
}

#[test]
fn evaluator_vs_scipy() {
    let r = load();
    let c = &r["cases"]["evaluator"];
    let (t, y, dy, x, rf) = (vec(&c["t"]), vec(&c["y"]), vec(&c["dy"]), vec(&c["x"]), vec(&c["ref"]));
    let worst = x.iter().zip(&rf).fold(0f64, |m, (&xi, &ri)| m.max((cubic_hermite(&t, &y, &dy, xi) - ri).abs()));
    println!("VERIFY cubic_hermite evaluator max_abs_err_vs_scipy={worst:e} npoints={}", x.len());
    assert!(worst < TOL_EVAL);
}

#[test]
fn edge_cases() {
    let mx = |t: &[f64], y: &[f64], dy: &[f64]| hermite_extremum(t, y, dy, Extremum::Max);
    let mn = |t: &[f64], y: &[f64], dy: &[f64]| hermite_extremum(t, y, dy, Extremum::Min);
    // monotone increasing: max at last knot (interval n-2), min at first knot
    let (t, y, dy) = ([0.0, 1.0, 3.0], [0.0, 1.0, 4.0], [1.0, 1.0, 2.0]);
    let (a, b) = (mx(&t, &y, &dy), mn(&t, &y, &dy));
    assert!(a.success && a.t == 3.0 && a.value == 4.0 && a.interval == 1);
    assert!(b.t == 0.0 && b.value == 0.0 && b.interval == 0);
    // peak exactly at an interior knot -> interval = that knot's index
    let a = mx(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, -1.0]);
    assert!(a.t == 1.0 && a.value == 1.0 && a.interval == 1);
    // tie: flat data returns the leftmost knot; two equal symmetric peaks return the left one
    let a = mx(&[0.0, 1.0, 2.0], &[1.0, 1.0, 1.0], &[0.0; 3]);
    assert!(a.t == 0.0 && a.interval == 0);
    let a = mx(&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 0.0, 1.0, 0.0], &[1.0, 0.0, -1.0, 0.0, -1.0]);
    assert!(a.t == 1.0 && a.interval == 1);
    // inflection with zero slope (y = s^3 on [-1, 1]) is not an extremum: max is the right knot
    let a = mx(&[-1.0, 1.0], &[-1.0, 1.0], &[3.0, 3.0]);
    assert!(a.t == 1.0 && a.value == 1.0);
    // linear data (a = 0, b = 0): no stationary point
    let a = mx(&[0.0, 2.0], &[1.0, 5.0], &[2.0, 2.0]);
    assert!(a.t == 2.0 && a.value == 5.0);
    // quadratic piece (derivative linear): y = -(t-0.3)^2 on [0,1]: max at 0.3, value 0
    let a = mx(&[0.0, 1.0], &[-0.09, -0.49], &[0.6, -1.4]);
    assert!((a.t - 0.3).abs() < 1e-15 && a.value.abs() < 1e-15 && a.interval == 0);
    // n = 2 minimum of an upward parabola
    let a = mn(&[0.0, 1.0], &[0.09, 0.49], &[-0.6, 1.4]);
    assert!((a.t - 0.3).abs() < 1e-15 && a.value.abs() < 1e-15);
    // invalid input is reported, never panics
    for (t, y, dy) in [
        (vec![0.0], vec![0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0], vec![0.0, 0.0]),
        (vec![0.0, 1.0, 1.0], vec![0.0; 3], vec![0.0; 3]),
        (vec![1.0, 0.0], vec![0.0; 2], vec![0.0; 2]),
        (vec![0.0, 1.0], vec![f64::NAN, 0.0], vec![0.0; 2]),
        (vec![0.0, f64::INFINITY], vec![0.0; 2], vec![0.0; 2]),
    ] {
        let e = mx(&t, &y, &dy);
        assert!(!e.success && e.t.is_nan() && e.value.is_nan() && !e.message.is_empty());
    }
    println!("VERIFY hermite_extremum edge_cases checks_passed=1");
}

/// Peak value of a Gaussian profile as a function of its centre mu, with AD through the samples.
fn peak_of(mu: Dual<f64>, t: &[f64]) -> Dual<f64> {
    let s = 0.05;
    let (y, dy): (Vec<Dual<f64>>, Vec<Dual<f64>>) = t
        .iter()
        .map(|&ti| {
            let d = Dual::new(ti, 0.0) - mu;
            let g = (-(d * d) / Dual::new(2.0 * s * s, 0.0)).exp();
            (g, -d / Dual::new(s * s, 0.0) * g)
        })
        .unzip();
    hermite_extremum(t, &y, &dy, Extremum::Max).value
}

#[test]
fn dual_peak_derivative_vs_finite_differences() {
    // peak height of exp(-(t-mu)^2/(2 s^2)) sampled at mu-dependent values: d(peak)/d(mu) via Dual
    // vs central differences of the plain-f64 peak on both sides of mu = 0.5137.
    let t: Vec<f64> = (0..21).map(|i| 0.3 + 0.0225 * i as f64).collect();
    for mu in [0.5137, 0.5137 + 0.004] {
        let d = peak_of(Dual::new(mu, 1.0), &t);
        let hh = 1e-6;
        let (p, m) = (peak_of(Dual::new(mu + hh, 0.0), &t).v, peak_of(Dual::new(mu - hh, 0.0), &t).v);
        let fd = (p - m) / (2.0 * hh);
        println!("VERIFY hermite_extremum ad_mu={mu} dual_dpeak={:e} fd_dpeak={fd:e} abs_diff={:e}", d.d, (d.d - fd).abs());
        assert!((d.d - fd).abs() < TOL_AD, "ad {} fd {}", d.d, fd);
    }
}
