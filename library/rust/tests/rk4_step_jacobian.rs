//! Tests for ode::rk4_step_jacobian: linear system (degree-4 Taylor polynomial of exp(A h)), Van der Pol and
//! Lorenz step Jacobians against complex-step reference data, central finite differences and dual numbers,
//! fourth-order convergence of the state and of the product of step Jacobians against DOP853 flow data,
//! and the CSTR setting of the source's RK4 parity test.
use agent_tools::autodiff::{Dual, Real};
use agent_tools::ode::rk4_step_jacobian::{rk4_step, rk4_step_jacobian};

/// Linear case: step and Jacobian equal the degree-4 Taylor polynomial, relative to max |entry|.
const TOL_LINEAR: f64 = 1e-13;
/// Non-linear points: Rust step and Jacobian vs reference data (complex-step NumPy RK4), relative to max |phi|.
const TOL_REF: f64 = 1e-11;
/// Variational Jacobian vs central finite differences of the step (relative step 1e-5), relative to max |phi|.
const TOL_FD: f64 = 1e-6;
/// Variational Jacobian vs dual-number derivative of the step (same arithmetic, different order), relative to max |phi|.
const TOL_DUAL: f64 = 1e-12;
/// Observed convergence order must lie within this distance of 4.
const TOL_ORDER: f64 = 0.15;
/// CSTR parity: RK4 (h = 1 s) vs DOP853, relative to |y|. This is truncation error of the chosen state (1.3e-7
/// measured); the source's own bar (1e-9) was for its state at the operating point and is not reachable here.
const TOL_PARITY_Y: f64 = 1e-6;
/// CSTR parity: step Jacobian vs flow sensitivity matrix, relative to max |entry| (truncation, not rounding).
const TOL_PARITY_PHI: f64 = 1e-5;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/rk4_step_jacobian.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn rel_max(a: &[f64], b: &[f64]) -> f64 {
    let s = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / s
}

fn vdp_f<T: Real>(mu: f64, y: &[T], d: &mut [T]) {
    d[0] = y[1];
    d[1] = T::cst(mu) * (T::cst(1.0) - y[0] * y[0]) * y[1] - y[0];
}
fn vdp_j<T: Real>(mu: f64, y: &[T], j: &mut [T]) {
    j[0] = T::cst(0.0);
    j[1] = T::cst(1.0);
    j[2] = -(T::cst(2.0 * mu) * y[0] * y[1]) - T::cst(1.0);
    j[3] = T::cst(mu) * (T::cst(1.0) - y[0] * y[0]);
}
fn lor_f<T: Real>(y: &[T], d: &mut [T]) {
    let c = T::cst;
    d[0] = c(10.0) * (y[1] - y[0]);
    d[1] = y[0] * (c(28.0) - y[2]) - y[1];
    d[2] = y[0] * y[1] - c(8.0 / 3.0) * y[2];
}
fn lor_j<T: Real>(y: &[T], j: &mut [T]) {
    let c = T::cst;
    j.copy_from_slice(&[c(-10.0), c(10.0), c(0.0), c(28.0) - y[2], c(-1.0), -y[0], y[1], y[0], c(-8.0 / 3.0)]);
}

/// Step Jacobian by dual numbers: column j seeds y0_j (state-only step, no Jacobian closure).
fn dual_phi(f: impl Fn(f64, &[Dual<f64>], &mut [Dual<f64>]), y0: &[f64], h: f64, n_sub: usize) -> Vec<f64> {
    let n = y0.len();
    let mut phi = vec![0.0; n * n];
    for j in 0..n {
        let y: Vec<Dual<f64>> = (0..n).map(|i| Dual::new(y0[i], if i == j { 1.0 } else { 0.0 })).collect();
        let r = rk4_step(&f, 0.0, &y, h, n_sub);
        for i in 0..n {
            phi[i * n + j] = r.y1[i].d;
        }
    }
    phi
}
/// Step Jacobian by central finite differences of the step.
fn fd_phi(f: impl Fn(f64, &[f64], &mut [f64]), y0: &[f64], h: f64, n_sub: usize) -> Vec<f64> {
    let n = y0.len();
    let mut phi = vec![0.0; n * n];
    for j in 0..n {
        let d = 1e-5 * y0[j].abs().max(1.0);
        let (mut yp, mut ym) = (y0.to_vec(), y0.to_vec());
        yp[j] += d;
        ym[j] -= d;
        let (a, b) = (rk4_step(&f, 0.0, &yp, h, n_sub).y1, rk4_step(&f, 0.0, &ym, h, n_sub).y1);
        for i in 0..n {
            phi[i * n + j] = (a[i] - b[i]) / (2.0 * d);
        }
    }
    phi
}

fn check_points(name: &str, pts: &[serde_json::Value], f: impl Fn(f64, &[f64], &mut [f64]) + Copy,
                jf: impl Fn(f64, &[f64], &mut [f64]) + Copy, fd: impl Fn(f64, &[Dual<f64>], &mut [Dual<f64>]) + Copy) {
    for p in pts {
        let (y0, h, m) = (vec(&p["y0"]), p["h"].as_f64().unwrap(), p["n_sub"].as_u64().unwrap() as usize);
        let tag = p["tag"].as_str().unwrap();
        let r = rk4_step_jacobian(f, jf, 0.0, &y0, h, m);
        assert!(r.success && r.nfev == 4 * m && r.njac == 4 * m);
        let s = rk4_step(f, 0.0, &y0, h, m);
        assert_eq!(s.y1, r.y1, "rk4_step must be bit-identical to the state part");
        let (ey, ep) = (rel_max(&r.y1, &vec(&p["y1"])), rel_max(&r.phi, &vec(&p["phi"])));
        let (efd, edu) = (rel_max(&fd_phi(f, &y0, h, m), &r.phi), rel_max(&dual_phi(fd, &y0, h, m), &r.phi));
        println!("VERIFY rk4_step_jacobian rust_{name}_{tag}_nsub{m} y_vs_ref={ey:e} phi_vs_ref={ep:e} phi_vs_fd={efd:e} phi_vs_dual={edu:e}");
        assert!(ey < TOL_REF && ep < TOL_REF && efd < TOL_FD && edu < TOL_DUAL);
    }
}

#[test]
fn linear_system_is_taylor_polynomial() {
    let d = load();
    let c = &d["cases"]["linear"];
    let (a, y0) = (vec(&c["a"]), vec(&c["y0"]));
    for s in c["steps"].as_array().unwrap() {
        let (h, m) = (s["h"].as_f64().unwrap(), s["n_sub"].as_u64().unwrap() as usize);
        let r = rk4_step_jacobian(
            |_, y: &[f64], d: &mut [f64]| {
                for i in 0..3 {
                    d[i] = (0..3).map(|k| a[i * 3 + k] * y[k]).sum();
                }
            },
            |_, _: &[f64], j: &mut [f64]| j.copy_from_slice(&a),
            0.0, &y0, h, m,
        );
        let (ey, ep) = (rel_max(&r.y1, &vec(&s["y1"])), rel_max(&r.phi, &vec(&s["phi"])));
        println!("VERIFY rk4_step_jacobian rust_linear_nsub{m} y_vs_taylor={ey:e} phi_vs_taylor={ep:e}");
        assert!(ey < TOL_LINEAR && ep < TOL_LINEAR);
    }
}

#[test]
fn van_der_pol_step_jacobian() {
    let d = load();
    let c = &d["cases"]["vdp"];
    let mu = c["mu"].as_f64().unwrap();
    check_points("vdp", c["points"].as_array().unwrap(), move |_, y, d| vdp_f(mu, y, d), move |_, y, j| vdp_j(mu, y, j),
                 move |_, y, d| vdp_f(mu, y, d));
}

#[test]
fn lorenz_step_jacobian() {
    let d = load();
    check_points("lorenz", d["cases"]["lorenz"]["points"].as_array().unwrap(), |_, y, d| lor_f(y, d), |_, y, j| lor_j(y, j),
                 |_, y, d| lor_f(y, d));
}

#[test]
fn global_error_is_fourth_order() {
    let d = load();
    let c = &d["cases"]["convergence"];
    let (y0, t_end, mu) = (vec(&c["y0"]), c["T"].as_f64().unwrap(), c["mu"].as_f64().unwrap());
    let (yref, pref) = (vec(&c["y_T"]), vec(&c["phi_T"]));
    let mut prev: Option<(f64, f64)> = None;
    for m in [10usize, 20, 40, 80, 160, 320] {
        let r = rk4_step_jacobian(|_, y, d| vdp_f(mu, y, d), |_, y, j| vdp_j(mu, y, j), 0.0, &y0, t_end, m);
        let (ey, ep) = (rel_max(&r.y1, &yref), rel_max(&r.phi, &pref));
        match prev {
            Some((py, pp)) => {
                let (oy, op) = ((py / ey).log2(), (pp / ep).log2());
                println!("VERIFY rk4_step_jacobian rust_convergence_nsub{m} err_y={ey:e} err_phi={ep:e} order_y={oy:.3} order_phi={op:.3}");
                if m >= 40 {
                    assert!((oy - 4.0).abs() < TOL_ORDER && (op - 4.0).abs() < TOL_ORDER, "orders {oy} {op}");
                }
            }
            None => println!("VERIFY rk4_step_jacobian rust_convergence_nsub{m} err_y={ey:e} err_phi={ep:e}"),
        }
        prev = Some((ey, ep));
    }
}

#[test]
fn cstr_parity_case() {
    let d = load();
    let c = &d["cases"]["parity"];
    let p = &c["cstr"];
    let g = |k: &str| p[k].as_f64().unwrap();
    let (qv, c0, t0, tc, k0, er, dh, ua) = (g("qv"), g("c0"), g("t0"), g("tc"), g("k0"), g("er"), g("dh"), g("ua"));
    let f = move |_: f64, y: &[f64], d: &mut [f64]| {
        let k = k0 * (-er / y[1]).exp();
        d[0] = qv * (c0 - y[0]) - k * y[0];
        d[1] = qv * (t0 - y[1]) + dh * k * y[0] + ua * (tc - y[1]);
    };
    let jf = move |_: f64, y: &[f64], j: &mut [f64]| {
        let k = k0 * (-er / y[1]).exp();
        let dk = k * er / (y[1] * y[1]);
        j.copy_from_slice(&[-qv - k, -dk * y[0], dh * k, -qv - ua + dh * dk * y[0]]);
    };
    let (y0, dt, m) = (vec(&c["y0"]), c["dt"].as_f64().unwrap(), c["n_sub"].as_u64().unwrap() as usize);
    let r = rk4_step_jacobian(f, jf, 0.0, &y0, dt, m);
    let ey = r.y1.iter().zip(vec(&c["y_ref"])).fold(0.0f64, |a, (x, y)| a.max((x - y).abs() / y.abs()));
    let ep = rel_max(&r.phi, &vec(&c["phi_ref"]));
    let ec = rel_max(&r.phi, &vec(&c["phi_complex_step"]));
    println!("VERIFY rk4_step_jacobian rust_cstr_parity y_rel_vs_dop853={ey:e} phi_vs_flow_sensitivity={ep:e} phi_vs_complex_step={ec:e}");
    assert!(ey < TOL_PARITY_Y && ep < TOL_PARITY_PHI && ec < TOL_REF);
}

#[test]
fn invalid_arguments_are_reported() {
    let f = |_: f64, y: &[f64], d: &mut [f64]| d[0] = -y[0];
    let jf = |_: f64, _: &[f64], j: &mut [f64]| j[0] = -1.0;
    for r in [
        rk4_step_jacobian(f, jf, 0.0, &[], 1.0, 1),
        rk4_step_jacobian(f, jf, 0.0, &[1.0], 1.0, 0),
        rk4_step_jacobian(f, jf, 0.0, &[1.0], f64::NAN, 1),
        rk4_step_jacobian(f, jf, f64::INFINITY, &[1.0], 1.0, 1),
    ] {
        assert!(!r.success && !r.message.is_empty() && r.nfev == 0);
    }
    assert!(!rk4_step(f, 0.0, &[1.0], 1.0, 0).success);
    // backward step undoes a forward step to O(h^5) for a smooth problem
    let fw = rk4_step(f, 0.0, &[1.0], 0.1, 4);
    let bw = rk4_step(f, 0.1, &fw.y1, -0.1, 4);
    println!("VERIFY rk4_step_jacobian rust_backward_roundtrip err={:e}", (bw.y1[0] - 1.0).abs());
    assert!((bw.y1[0] - 1.0).abs() < 1e-8);
}
