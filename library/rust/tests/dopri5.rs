//! Tests for ode::dopri5: step-by-step comparison with SciPy RK45, exact solutions and invariants,
//! DOP853 reference data, Arenstorf and a stiffness study (step-size limitation), failure reporting.
use agent_tools::ode::dopri5::{dopri5, Dopri5Options, Dopri5Result};

/// Accepted step times vs SciPy RK45 (same rtol, atol): absolute difference allowed, in time units.
/// Round-off differences in the error norm (different summation order) are fed back through the
/// controller and grow to about 1e-8; the step counts must nevertheless agree exactly.
const TOL_T_SEQ: f64 = 1e-6;
/// Final state vs SciPy RK45, relative to max |y|.
const TOL_Y_VS_RK45: f64 = 1e-6;
/// Achieved final error (max-norm relative to max |y_ref|) may exceed rtol by at most this factor
/// (local error control, global error accumulates; Arenstorf and Kepler are the worst cases).
const TOL_ERR_OVER_RTOL: f64 = 1e3;

/// The same for Arenstorf's orbit: the close approaches to the moon amplify the local errors (a
/// periodic orbit has no damping), so the final error is thousands of times rtol.
const TOL_ERR_OVER_RTOL_ARENSTORF: f64 = 2e4;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/dopri5.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
/// max_i |a_i - b_i| / max_j |b_j|
fn relmax(a: &[f64], b: &[f64]) -> f64 {
    let s = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / s
}

const MU: f64 = 0.012277471;
fn arenstorf(_: f64, y: &[f64], d: &mut [f64]) {
    let mh = 1.0 - MU;
    let d1 = ((y[0] + MU).powi(2) + y[1] * y[1]).powf(1.5);
    let d2 = ((y[0] - mh).powi(2) + y[1] * y[1]).powf(1.5);
    d[0] = y[2];
    d[1] = y[3];
    d[2] = y[0] + 2.0 * y[3] - mh * (y[0] + MU) / d1 - MU * (y[0] - mh) / d2;
    d[3] = y[1] - 2.0 * y[2] - mh * y[1] / d1 - MU * y[1] / d2;
}
fn lotka(_: f64, y: &[f64], d: &mut [f64]) {
    d[0] = 1.0 * y[0] - 0.5 * y[0] * y[1];
    d[1] = 0.2 * y[0] * y[1] - 0.6 * y[1];
}
fn ham(y: &[f64]) -> f64 {
    0.2 * y[0] - 0.6 * y[0].ln() + 0.5 * y[1] - 1.0 * y[1].ln()
}
fn vdp1(_: f64, y: &[f64], d: &mut [f64]) {
    d[0] = y[1];
    d[1] = (1.0 - y[0] * y[0]) * y[1] - y[0];
}
fn oscillator(_: f64, y: &[f64], d: &mut [f64]) {
    d[0] = -0.1 * y[0] + 2.0 * y[1];
    d[1] = -2.0 * y[0] - 0.1 * y[1];
}
fn kepler(_: f64, y: &[f64], d: &mut [f64]) {
    let r3 = (y[0] * y[0] + y[1] * y[1]).powf(1.5);
    d[0] = y[2];
    d[1] = y[3];
    d[2] = -y[0] / r3;
    d[3] = -y[1] / r3;
}
const ECC: f64 = 0.6;
/// Analytic Kepler orbit from perihelion (a = 1, mu = 1): Kepler's equation M = E - e sin E by Newton.
fn kepler_exact(t: f64) -> [f64; 4] {
    let mut e = t;
    for _ in 0..100 {
        e -= (e - ECC * e.sin() - t) / (1.0 - ECC * e.cos());
    }
    let s = (1.0 - ECC * ECC).sqrt();
    let r = 1.0 - ECC * e.cos();
    [e.cos() - ECC, s * e.sin(), -e.sin() / r, s * e.cos() / r]
}
type Rhs = fn(f64, &[f64], &mut [f64]);
fn problem(name: &str) -> Rhs {
    match name {
        "arenstorf" => arenstorf,
        "lotka" => lotka,
        "vdp1" => vdp1,
        "oscillator" => oscillator,
        "kepler" => kepler,
        _ => panic!("unknown problem {name}"),
    }
}
const NAMES: [&str; 5] = ["arenstorf", "lotka", "vdp1", "oscillator", "kepler"];

fn run(name: &str, c: &serde_json::Value, rtol: f64, atol: f64) -> Dopri5Result {
    let o = Dopri5Options { rtol, atol, ..Default::default() };
    dopri5(problem(name), (0.0, c["t_end"].as_f64().unwrap()), &vec(&c["y0"]), &o)
}

#[test]
fn step_sequence_vs_scipy_rk45() {
    let data = load();
    for name in NAMES {
        let c = &data["cases"][name];
        for (key, r) in c["rk45"].as_object().unwrap() {
            let (rtol, atol) = (r["rtol"].as_f64().unwrap(), r["atol"].as_f64().unwrap());
            let run = run(name, c, rtol, atol);
            assert!(run.success, "{}", run.message);
            let ts = vec(&r["t"]);
            let nsc = r["nsteps"].as_u64().unwrap() as usize;
            let ey = relmax(run.y_final(), &vec(&r["y_final"]));
            let m = ts.len().min(run.t.len());
            let et = (0..m).map(|i| (run.t[i] - ts[i]).abs()).fold(0.0, f64::max);
            println!(
                "VERIFY dopri5 {name}_rtol{key} nsteps={} scipy_nsteps={nsc} nfev={} scipy_nfev={} nrejected={} max_dt_vs_scipy={et:e} y_final_vs_scipy={ey:e}",
                run.nsteps, run.nfev, r["nfev"], run.nrejected
            );
            assert_eq!(run.nsteps, nsc, "{name} {key}");
            assert_eq!(run.nfev as u64, r["nfev"].as_u64().unwrap(), "{name} {key}");
            assert!(et < TOL_T_SEQ && ey < TOL_Y_VS_RK45, "{et:e} {ey:e}");
        }
    }
}

#[test]
fn exact_solutions_and_invariant() {
    let data = load();
    for rtol in [1e-3, 1e-6, 1e-9] {
        let atol = 1e-3 * rtol;
        // damped rotation, exact exp(-0.1 t) (cos 2t, -sin 2t)
        let c = &data["cases"]["oscillator"];
        let te = c["t_end"].as_f64().unwrap();
        let r = run("oscillator", c, rtol, atol);
        let ex = [(-0.1 * te).exp() * (2.0 * te).cos(), -(-0.1 * te).exp() * (2.0 * te).sin()];
        let e_osc = relmax(r.y_final(), &ex);
        // Kepler, two periods
        let c = &data["cases"]["kepler"];
        let te = c["t_end"].as_f64().unwrap();
        let rk = run("kepler", c, rtol, atol);
        let e_kep = relmax(rk.y_final(), &kepler_exact(te));
        // Lotka-Volterra conserved quantity, drift of H after t = 30 relative to H0 (|H0| ~ 1)
        let c = &data["cases"]["lotka"];
        let rl = run("lotka", c, rtol, atol);
        let h0 = c["H0"].as_f64().unwrap();
        let drift = (ham(rl.y_final()) - h0).abs() / h0.abs();
        println!(
            "VERIFY dopri5 exact_rtol{rtol:e} oscillator_err={e_osc:e} err/rtol={:e} kepler_err={e_kep:e} err/rtol={:e} lotka_H_drift={drift:e} drift/rtol={:e} nsteps={}/{}/{}",
            e_osc / rtol, e_kep / rtol, drift / rtol, r.nsteps, rk.nsteps, rl.nsteps
        );
        assert!(e_osc < TOL_ERR_OVER_RTOL * rtol && e_kep < TOL_ERR_OVER_RTOL * rtol && drift < TOL_ERR_OVER_RTOL * rtol);
    }
}

#[test]
fn dop853_reference_and_arenstorf() {
    let data = load();
    let mut last_steps = 0;
    for rtol in [1e-3, 1e-6, 1e-9] {
        for name in NAMES {
            let c = &data["cases"][name];
            let r = run(name, c, rtol, 1e-3 * rtol);
            assert!(r.success, "{}", r.message);
            let e = relmax(r.y_final(), &vec(&c["y_ref"]));
            println!("VERIFY dopri5 {name}_rtol{rtol:e} err_vs_dop853={e:e} err/rtol={:e} nsteps={} nrejected={} dop853_ref_nfev(rtol 1e-13)={}", e / rtol, r.nsteps, r.nrejected, c["dop853_nfev"]);
            // Arenstorf at rtol 1e-3 is too loose to follow the orbit (error near 1); only 1e-6 and 1e-9 are asserted
            let factor = if name == "arenstorf" { TOL_ERR_OVER_RTOL_ARENSTORF } else { TOL_ERR_OVER_RTOL };
            if !(name == "arenstorf" && rtol == 1e-3) {
                assert!(e < factor * rtol, "{name} {rtol:e}: {e:e}");
            }
            if name == "arenstorf" {
                assert!(r.nsteps > last_steps, "nsteps must grow as rtol tightens");
                last_steps = r.nsteps;
            }
        }
    }
}

#[test]
fn stiffness_limits_the_step() {
    let data = load();
    let s = &data["cases"]["stiffness"];
    let te = s["t_end"].as_f64().unwrap();
    let mut prev = 0;
    for row in s["rows"].as_array().unwrap() {
        let lam = row["lam"].as_f64().unwrap();
        let o = Dopri5Options { rtol: 1e-6, atol: 1e-9, ..Default::default() };
        let r = dopri5(|t, y, d| d[0] = -lam * (y[0] - t.cos()), (0.0, te), &[0.0], &o);
        assert!(r.success, "{}", r.message);
        let e = (r.y_final()[0] - row["y_ref"].as_f64().unwrap()).abs();
        println!(
            "VERIFY dopri5 stiff_lam{lam:e} nsteps={} scipy_nsteps={} nrejected={} abs_err_vs_radau={e:e} steps*h_stability={:e}",
            r.nsteps, row["rk45_nsteps"], r.nrejected, r.nsteps as f64 / (lam * te)
        );
        assert_eq!(r.nsteps as u64, row["rk45_nsteps"].as_u64().unwrap());
        assert!(r.nsteps > prev && e < 1e-4);
        prev = r.nsteps;
    }
    // a step budget that is too small fails cleanly with the partial trajectory
    let lam = 1e4;
    let o = Dopri5Options { max_steps: 500, ..Default::default() };
    let r = dopri5(|t, y, d| d[0] = -lam * (y[0] - t.cos()), (0.0, te), &[0.0], &o);
    println!("VERIFY dopri5 stiff_budget success={} message={:?} t_reached={:e}", r.success, r.message, r.t.last().unwrap());
    assert!(!r.success && r.message.contains("max_steps") && *r.t.last().unwrap() < te);
}

#[test]
fn options_and_time_dependence() {
    // y' = cos t is time dependent only through the stage times: exact sin t
    let f = |t: f64, _: &[f64], d: &mut [f64]| d[0] = t.cos();
    let r = dopri5(f, (0.0, 5.0), &[0.0], &Dopri5Options { rtol: 1e-10, atol: 1e-13, ..Default::default() });
    assert!(r.success && (r.y_final()[0] - 5f64.sin()).abs() < 1e-8);
    assert_eq!(*r.t.last().unwrap(), 5.0);
    // hmax and h0
    let o = Dopri5Options { hmax: Some(0.01), ..Default::default() };
    let r = dopri5(f, (0.0, 1.0), &[0.0], &o);
    assert!(r.success && r.nsteps >= 100);
    let o = Dopri5Options { h0: Some(1e-3), ..Default::default() };
    let r = dopri5(f, (0.0, 1.0), &[0.0], &o);
    assert!(r.success && r.t[1] == 1e-3 && r.nfev == 1 + 6 * (r.nsteps + r.nrejected));
    // nfev bookkeeping with the automatic first step
    let r = dopri5(f, (0.0, 1.0), &[0.0], &Dopri5Options::default());
    assert_eq!(r.nfev, 2 + 6 * (r.nsteps + r.nrejected));
    // per-component atol: y2 ~ 1e-9 throughout
    let g = |_: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -y[0];
        d[1] = y[0] * 1e-9 - 10.0 * y[1];
    };
    let exact = 1e-9 / 9.0 * ((-1.0f64).exp() - (-10.0f64).exp());
    let loose = Dopri5Options { rtol: 1e-6, atol: 1e-6, ..Default::default() };
    let tight = Dopri5Options { rtol: 1e-6, atol_vec: vec![1e-6, 1e-16], ..Default::default() };
    let e = |o: &Dopri5Options| {
        let r = dopri5(g, (0.0, 1.0), &[1.0, 0.0], o);
        assert!(r.success);
        ((r.y_final()[1] - exact).abs() / exact, r.nsteps)
    };
    let ((el, nl), (et, nt)) = (e(&loose), e(&tight));
    println!("VERIFY dopri5 atol_vec err_loose={el:e} nsteps={nl} err_per_component={et:e} nsteps={nt}");
    assert!(et < 1e-4 && et < el);
}

#[test]
fn failures_are_reported() {
    let f = |_: f64, y: &[f64], d: &mut [f64]| d[0] = y[0] * y[0];
    let o = Dopri5Options::default();
    // finite-time blow-up y' = y^2, y(0) = 1 at t = 1: must fail, not hang
    let r = dopri5(f, (0.0, 2.0), &[1.0], &o);
    println!("VERIFY dopri5 blowup success={} message={:?} t_reached={:e}", r.success, r.message, r.t.last().unwrap());
    assert!(!r.success && !r.message.is_empty() && *r.t.last().unwrap() < 2.0);
    assert!(!dopri5(f, (1.0, 1.0), &[1.0], &o).success);
    assert!(!dopri5(f, (0.0, 1.0), &[], &o).success);
    assert!(!dopri5(f, (0.0, 1.0), &[1.0], &Dopri5Options { rtol: 0.0, ..Default::default() }).success);
    assert!(!dopri5(f, (0.0, 1.0), &[1.0], &Dopri5Options { atol_vec: vec![1e-8, 1e-8], ..Default::default() }).success);
    assert!(!dopri5(f, (0.0, 1.0), &[1.0], &Dopri5Options { h0: Some(-1.0), ..Default::default() }).success);
    let few = Dopri5Options { max_steps: 5, ..Default::default() };
    let r = dopri5(|_, y, d| d[0] = -y[0], (0.0, 100.0), &[1.0], &few);
    assert!(!r.success && r.message.contains("max_steps"));
    // f = 0 and y0 = 0 with atol = 0: zero scale, automatic first step impossible; h0 fixes it
    let zero = |_: f64, _: &[f64], d: &mut [f64]| d[0] = 0.0;
    let z = Dopri5Options { atol: 0.0, ..Default::default() };
    assert!(!dopri5(zero, (0.0, 1.0), &[0.0], &z).success);
    // f = 0 with the default atol: first-step heuristic must not divide by zero
    let r = dopri5(zero, (0.0, 1.0), &[1.0], &Dopri5Options::default());
    assert!(r.success && r.y_final()[0] == 1.0, "{}", r.message);
}

#[test]
fn nan_rhs_is_not_accepted() {
    // f turns NaN for t > 0.5: the NaN error estimate must reject the step, so the run fails
    // instead of returning success with a NaN state.
    let f = |t: f64, y: &[f64], d: &mut [f64]| d[0] = if t > 0.5 { f64::NAN } else { -y[0] };
    let r = dopri5(f, (0.0, 1.0), &[1.0], &Dopri5Options::default());
    println!("VERIFY dopri5 nan_rhs success={} message={:?}", r.success, r.message);
    assert!(!r.success);
    assert!(r.y.iter().all(|v| v.is_finite()));
}
