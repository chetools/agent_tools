//! Example: Robertson's stiff kinetics to t = 40 with dual numbers: y(40) and dy(40)/dk1 (k1 = 0.04).
//! Prints one JSON object (same keys as the Python example).
use agent_tools::autodiff::{Dual, Real};
use agent_tools::ode::rosenbrock23::{rosenbrock23, Rosenbrock23Options};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let k1 = Dual::new(0.04, 1.0);
    let c = Dual::<f64>::cst;
    let f = |_: f64, y: &[Dual<f64>], d: &mut [Dual<f64>]| {
        d[0] = -(k1 * y[0]) + c(1e4) * y[1] * y[2];
        d[1] = k1 * y[0] - c(1e4) * y[1] * y[2] - c(3e7) * y[1] * y[1];
        d[2] = c(3e7) * y[1] * y[1];
    };
    let jac = |_: f64, y: &[Dual<f64>], j: &mut [Dual<f64>]| {
        let z = c(0.0);
        j.copy_from_slice(&[
            -k1, c(1e4) * y[2], c(1e4) * y[1],
            k1, -(c(1e4) * y[2]) - c(6e7) * y[1], -(c(1e4) * y[1]),
            z, c(6e7) * y[1], z,
        ]);
    };
    let o = Rosenbrock23Options { rtol: 1e-5, atol: 1e-9, h0: Some(1e-6), ..Default::default() };
    let r = rosenbrock23(f, jac, None, (0.0, 40.0), &[c(1.0), c(0.0), c(0.0)], &o);
    let y: Vec<f64> = r.y_final().iter().map(|d| d.v).collect();
    let dy: Vec<f64> = r.y_final().iter().map(|d| d.d).collect();
    let mid = r.y_at(r.nsteps / 2);
    println!(
        "{{\"success\":{},\"message\":{:?},\"nsteps\":{},\"nrejected\":{},\"nfev\":{},\"njac\":{},\"nlu\":{},\"t_mid\":{:?},\"y_mid\":{},\"t_final\":{:?},\"y\":{},\"dy_dk1\":{}}}",
        r.success, r.message, r.nsteps, r.nrejected, r.nfev, r.njac, r.nlu, r.t[r.nsteps / 2],
        list(&mid.iter().map(|d| d.v).collect::<Vec<_>>()), r.t[r.nsteps], list(&y), list(&dy)
    );
}
