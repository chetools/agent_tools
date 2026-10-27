//! Tests for ode::rosenbrock23: exact solutions, SciPy Radau reference data (Robertson, Van der Pol
//! mu = 1000), dual-number sensitivities vs finite differences and vs exact/independent values.
use agent_tools::autodiff::{Dual, Real};
use agent_tools::ode::rosenbrock23::{rosenbrock23, Rosenbrock23Options};

/// Achieved error may exceed the requested rtol by at most this factor (the tolerance controls the local
/// error of a 2nd-order method, so the global error is larger). Used for the exact-solution and SciPy cases.
const TOL_ERR_OVER_RTOL: f64 = 500.0;
/// AD derivative vs exact -t exp(-k t), relative, at rtol 1e-8.
const TOL_AD_EXACT: f64 = 1e-4;
/// AD derivative vs central FD of the same integrator (same tolerances), relative; FD noise ~ rtol/h.
const TOL_AD_FD: f64 = 1e-3;
/// AD Robertson dy/dk1 vs independent forward-sensitivity reference, relative to max |dy|, at rtol 1e-7.
const TOL_AD_ROB: f64 = 1e-4;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/rosenbrock23.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn measure(y: &[f64], r: &[f64], atol: f64) -> f64 {
    y.iter().zip(r).map(|(a, b)| (a - b).abs() / (b.abs() + atol)).fold(0.0, f64::max)
}

fn rob_f<T: Real>(k1: T, y: &[T], d: &mut [T]) {
    let c = T::cst;
    d[0] = -(k1 * y[0]) + c(1e4) * y[1] * y[2];
    d[1] = k1 * y[0] - c(1e4) * y[1] * y[2] - c(3e7) * y[1] * y[1];
    d[2] = c(3e7) * y[1] * y[1];
}
fn rob_j<T: Real>(k1: T, y: &[T], j: &mut [T]) {
    let c = T::cst;
    let z = c(0.0);
    j.copy_from_slice(&[
        -k1, c(1e4) * y[2], c(1e4) * y[1],
        k1, -(c(1e4) * y[2]) - c(6e7) * y[1], -(c(1e4) * y[1]),
        z, c(6e7) * y[1], z,
    ]);
}
const MU: f64 = 1000.0;
fn vdp_f(_t: f64, y: &[f64], d: &mut [f64]) {
    d[0] = y[1];
    d[1] = MU * (1.0 - y[0] * y[0]) * y[1] - y[0];
}
fn vdp_j(_t: f64, y: &[f64], j: &mut [f64]) {
    j.copy_from_slice(&[0.0, 1.0, -2.0 * MU * y[0] * y[1] - 1.0, MU * (1.0 - y[0] * y[0])]);
}
fn run_rob(t_end: f64, rtol: f64, atol: f64) -> agent_tools::ode::rosenbrock23::Rosenbrock23Result<f64> {
    let o = Rosenbrock23Options { rtol, atol, h0: Some(1e-6), ..Default::default() };
    rosenbrock23(|_, y, d| rob_f(0.04, y, d), |_, y, j| rob_j(0.04, y, j), None, (0.0, t_end), &[1.0, 0.0, 0.0], &o)
}
fn run_vdp(t_end: f64, rtol: f64, atol: f64) -> agent_tools::ode::rosenbrock23::Rosenbrock23Result<f64> {
    let o = Rosenbrock23Options { rtol, atol, h0: Some(1e-3), ..Default::default() };
    rosenbrock23(vdp_f, vdp_j, None, (0.0, t_end), &[2.0, 0.0], &o)
}

#[test]
fn exponential_decay_vs_exact() {
    let k = 2.5;
    for rtol in [1e-4, 1e-6, 1e-8] {
        let o = Rosenbrock23Options { rtol, atol: 1e-14, ..Default::default() };
        let r = rosenbrock23(|_, y, d| d[0] = -k * y[0], |_, _, j| j[0] = -k, None, (0.0, 1.0), &[1.0], &o);
        assert!(r.success, "{}", r.message);
        let e = (r.y_final()[0] - (-k * 1.0f64).exp()).abs() / (-k * 1.0f64).exp();
        println!("VERIFY rosenbrock23 decay_rtol{rtol:e} rel_err_vs_exact={e:e} nsteps={} nrejected={}", r.nsteps, r.nrejected);
        // FSAL: stored derivative at the end is f(t1, y1); trajectory ends exactly at t1
        assert_eq!(*r.t.last().unwrap(), 1.0);
        assert!((r.dydt[r.dydt.len() - 1] + k * r.y_final()[0]).abs() < 1e-15);
        assert_eq!(r.nfev, 1 + 2 * (r.nsteps + r.nrejected));
        assert_eq!(r.nlu, r.nsteps + r.nrejected);
        assert_eq!(r.njac, r.nsteps); // Jacobian reused after a rejection
        assert!(e < TOL_ERR_OVER_RTOL * rtol, "{e:e}");
    }
}

#[test]
fn nonautonomous_vs_exact() {
    // y' = -lam (y - cos t), exact: (lam^2 cos t + lam sin t)/(1+lam^2) + C exp(-lam t)
    let lam = 1000.0;
    let exact = |t: f64| {
        let c = 1.0 - lam * lam / (1.0 + lam * lam);
        (lam * lam * t.cos() + lam * t.sin()) / (1.0 + lam * lam) + c * (-lam * t).exp()
    };
    let f = |t: f64, y: &[f64], d: &mut [f64]| d[0] = -lam * (y[0] - t.cos());
    let j = |_: f64, _: &[f64], jm: &mut [f64]| jm[0] = -lam;
    let mut g = |t: f64, _: &[f64], o: &mut [f64]| o[0] = -lam * t.sin();
    let base = Rosenbrock23Options { rtol: 1e-8, atol: 1e-12, h0: Some(1e-6), ..Default::default() };
    let r_an = rosenbrock23(f, j, Some(&mut g), (0.0, 3.0), &[1.0], &base);
    let r_fd = rosenbrock23(f, j, None, (0.0, 3.0), &[1.0], &Rosenbrock23Options { fd_dfdt: true, ..base.clone() });
    let r_no = rosenbrock23(f, j, None, (0.0, 3.0), &[1.0], &base);
    assert!(r_an.success && r_fd.success && r_no.success);
    let ex = exact(3.0);
    let (e_an, e_fd, e_no) = (
        (r_an.y_final()[0] - ex).abs() / ex.abs(),
        (r_fd.y_final()[0] - ex).abs() / ex.abs(),
        (r_no.y_final()[0] - ex).abs() / ex.abs(),
    );
    println!("VERIFY rosenbrock23 nonautonomous rel_err_analytic_dfdt={e_an:e} fd_dfdt={e_fd:e} zero_dfdt={e_no:e}");
    println!("VERIFY rosenbrock23 nonautonomous nsteps analytic={} fd={} zero={}", r_an.nsteps, r_fd.nsteps, r_no.nsteps);
    assert!(e_an < 1e-6 && e_fd < 1e-6);
}

#[test]
fn robertson_vs_radau() {
    let r = load();
    let atol_factor = 1e-6; // atol = atol_factor * rtol, as in the generator
    for refc in r["cases"]["robertson"]["refs"].as_array().unwrap() {
        let te = refc["t_end"].as_f64().unwrap();
        let rtol = 1e-6;
        let run = run_rob(te, rtol, atol_factor * rtol);
        assert!(run.success, "{}", run.message);
        let e = measure(run.y_final(), &vec(&refc["y"]), atol_factor * rtol);
        println!("VERIFY rosenbrock23 robertson_t{te:e} err_vs_radau={e:e} nsteps={} nrejected={} nfev={} nlu={}", run.nsteps, run.nrejected, run.nfev, run.nlu);
        assert!(e < TOL_ERR_OVER_RTOL * rtol, "{e:e}");
        let s: f64 = run.y_final().iter().sum();
        assert!((s - 1.0).abs() < 1e-8, "mass balance {s}");
    }
}

#[test]
fn van_der_pol_and_tolerance_study() {
    let r = load();
    let refs = r["cases"]["vdp1000"]["refs"].as_array().unwrap();
    for refc in refs {
        let te = refc["t_end"].as_f64().unwrap();
        let rtol = 1e-7;
        let run = run_vdp(te, rtol, 1e-3 * rtol);
        assert!(run.success, "{}", run.message);
        let e = measure(run.y_final(), &vec(&refc["y"]), 1e-3 * rtol);
        println!("VERIFY rosenbrock23 vdp1000_t{te:e}_rtol1e-7 err_vs_radau={e:e} nsteps={}", run.nsteps);
        assert!(e < TOL_ERR_OVER_RTOL * rtol, "{e:e}");
    }
    // tolerance study at the last end time of each problem, against the stored Radau/BDF counts
    for (name, atol_f) in [("robertson", 1e-6), ("vdp1000", 1e-3)] {
        let last = r["cases"][name]["refs"].as_array().unwrap().last().unwrap();
        let (te, yref) = (last["t_end"].as_f64().unwrap(), vec(&last["y"]));
        for row in r["cases"]["tolerance_study"][name].as_array().unwrap() {
            let rtol = row["rtol"].as_f64().unwrap();
            let atol = atol_f * rtol;
            assert!((row["atol"].as_f64().unwrap() - atol).abs() <= 1e-12 * atol);
            let run = if name == "robertson" { run_rob(te, rtol, atol) } else { run_vdp(te, rtol, atol) };
            assert!(run.success);
            let e = measure(run.y_final(), &yref, atol);
            let (rad, bdf) = (&row["Radau"], &row["BDF"]);
            println!(
                "VERIFY rosenbrock23 study_{name}_rtol{rtol:e} err={e:e} err/rtol={:e} nfev={} nsteps={} nrejected={} nlu={} njac={} | Radau err={:e} nfev={} nsteps={} | BDF err={:e} nfev={} nsteps={}",
                e / rtol, run.nfev, run.nsteps, run.nrejected, run.nlu, run.njac,
                rad["err"].as_f64().unwrap(), rad["nfev"], rad["nsteps"],
                bdf["err"].as_f64().unwrap(), bdf["nfev"], bdf["nsteps"]
            );
            assert!(e < TOL_ERR_OVER_RTOL * rtol, "{name} rtol {rtol:e}: err {e:e}");
        }
    }
}

fn decay_dual(k: Dual<f64>, rtol: f64, t_end: f64) -> Dual<f64> {
    let o = Rosenbrock23Options { rtol, atol: 1e-14, ..Default::default() };
    let r = rosenbrock23(
        |_, y: &[Dual<f64>], d: &mut [Dual<f64>]| d[0] = -(k * y[0]),
        |_, _, j: &mut [Dual<f64>]| j[0] = -k,
        None,
        (0.0, t_end),
        &[Dual::cst(1.0)],
        &o,
    );
    assert!(r.success);
    r.y_final()[0]
}

#[test]
fn ad_sensitivity_decay_and_robertson() {
    // (d) linear decay: dy/dk at t_end vs exact -t exp(-k t) and vs central FD of the integrator
    let (k, te) = (2.5, 3.0);
    for rtol in [1e-6, 1e-8] {
        let y = decay_dual(Dual::new(k, 1.0), rtol, te);
        let exact = -te * (-k * te).exp();
        let h = 1e-3 * k;
        let fd = (decay_dual(Dual::cst(k + h), rtol, te).v - decay_dual(Dual::cst(k - h), rtol, te).v) / (2.0 * h);
        let (e_ex, e_fd) = ((y.d - exact).abs() / exact.abs(), (y.d - fd).abs() / exact.abs());
        // the primal value must not change when derivatives are carried
        let plain = decay_dual(Dual::cst(k), rtol, te).v;
        println!("VERIFY rosenbrock23 ad_decay_rtol{rtol:e} rel_err_vs_exact={e_ex:e} rel_diff_vs_fd={e_fd:e} value_changed_by_ad={:e}", (y.v - plain).abs());
        if rtol == 1e-8 {
            assert!(e_ex < TOL_AD_EXACT && e_fd < TOL_AD_FD);
        }
    }
    // Robertson dy/dk1 at t = 40 vs independent forward-sensitivity reference (Radau 1e-12)
    let r = load();
    let sref = vec(&r["cases"]["robertson_sens"]["dy_dk1"]);
    let rtol = 1e-7;
    let o = Rosenbrock23Options { rtol, atol: 1e-15, h0: Some(1e-6), ..Default::default() };
    let go = |k1: Dual<f64>| {
        rosenbrock23(
            |_, y: &[Dual<f64>], d: &mut [Dual<f64>]| rob_f(k1, y, d),
            |_, y: &[Dual<f64>], j: &mut [Dual<f64>]| rob_j(k1, y, j),
            None,
            (0.0, 40.0),
            &[Dual::cst(1.0), Dual::cst(0.0), Dual::cst(0.0)],
            &o,
        )
    };
    let run = go(Dual::new(0.04, 1.0));
    assert!(run.success, "{}", run.message);
    let dy: Vec<f64> = run.y_final().iter().map(|d| d.d).collect();
    let smax = sref.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let e_ref = dy.iter().zip(&sref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / smax;
    // central FD of the same integrator, same tolerances (relative step 1e-3)
    let h = 0.04 * 1e-3;
    let (yp, ym) = (go(Dual::cst(0.04 + h)), go(Dual::cst(0.04 - h)));
    let e_fd = (0..3).map(|i| ((yp.y_final()[i].v - ym.y_final()[i].v) / (2.0 * h) - dy[i]).abs()).fold(0.0, f64::max) / smax;
    println!("VERIFY rosenbrock23 ad_robertson_dk1 rel_err_vs_forward_sens={e_ref:e} rel_diff_vs_fd={e_fd:e} nsteps_dual={} nrejected_dual={}", run.nsteps, run.nrejected);
    assert!(e_ref < TOL_AD_ROB && e_fd < TOL_AD_FD, "{e_ref:e} {e_fd:e}");
}

#[test]
fn failures_are_reported() {
    let f = |_: f64, y: &[f64], d: &mut [f64]| d[0] = y[0] * y[0];
    let j = |_: f64, y: &[f64], jm: &mut [f64]| jm[0] = 2.0 * y[0];
    let o = Rosenbrock23Options::default();
    // finite-time blow-up y' = y^2, y(0) = 1 at t = 1: must fail, not hang
    let r = rosenbrock23(f, j, None, (0.0, 2.0), &[1.0], &o);
    assert!(!r.success && !r.message.is_empty());
    println!("VERIFY rosenbrock23 blowup message={:?} t_reached={:e}", r.message, r.t.last().unwrap());
    assert!(!rosenbrock23(f, j, None, (1.0, 1.0), &[1.0], &o).success);
    assert!(!rosenbrock23(f, j, None, (0.0, 1.0), &[], &o).success);
    let few = Rosenbrock23Options { max_steps: 5, ..Default::default() };
    let r = rosenbrock23(|_, y, d| d[0] = -y[0], |_, _, jm| jm[0] = -1.0, None, (0.0, 100.0), &[1.0], &few);
    assert!(!r.success && r.message.contains("max_steps"));
    let bad = Rosenbrock23Options { atol_vec: vec![1e-8, 1e-8], ..Default::default() };
    assert!(!rosenbrock23(f, j, None, (0.0, 1.0), &[1.0], &bad).success);
}

#[test]
fn nan_rhs_is_not_accepted() {
    // f turns NaN for t > 0.5: the NaN error estimate must reject the step, so the run fails
    // instead of returning success with a NaN state.
    let f = |t: f64, y: &[f64], d: &mut [f64]| d[0] = if t > 0.5 { f64::NAN } else { -y[0] };
    let j = |_: f64, _: &[f64], jm: &mut [f64]| jm[0] = -1.0;
    let r = rosenbrock23(f, j, None, (0.0, 1.0), &[1.0], &Rosenbrock23Options::default());
    println!("VERIFY rosenbrock23 nan_rhs success={} message={:?}", r.success, r.message);
    assert!(!r.success);
    assert!(r.y.iter().all(|v| v.is_finite()));
}

#[test]
fn per_component_atol_is_used() {
    // y2 is ~1e-9 throughout: a loose scalar atol ignores it, a per-component atol resolves it
    let f = |_: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -y[0];
        d[1] = y[0] * 1e-9 - 10.0 * y[1];
    };
    let j = |_: f64, _: &[f64], jm: &mut [f64]| jm.copy_from_slice(&[-1.0, 0.0, 1e-9, -10.0]);
    let loose = Rosenbrock23Options { rtol: 1e-6, atol: 1e-6, ..Default::default() };
    let tight = Rosenbrock23Options { rtol: 1e-6, atol_vec: vec![1e-6, 1e-16], ..Default::default() };
    let exact = 1e-9 / 9.0 * ((-1.0f64).exp() - (-10.0f64).exp());
    let e = |o: &Rosenbrock23Options| {
        let r = rosenbrock23(f, j, None, (0.0, 1.0), &[1.0, 0.0], o);
        assert!(r.success);
        ((r.y_final()[1] - exact).abs() / exact, r.nsteps)
    };
    let (el, nl) = e(&loose);
    let (et, nt) = e(&tight);
    println!("VERIFY rosenbrock23 atol_vec err_loose={el:e} nsteps={nl} err_per_component={et:e} nsteps={nt}");
    assert!(et < 1e-4 && et < el);
}
