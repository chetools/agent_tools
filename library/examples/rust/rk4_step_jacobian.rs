//! Example: generic exothermic CSTR (C in mol/L, T in K, t in s), one RK4 step over dt = 10 s with 10
//! substeps: y1 and the step Jacobian, plus column 0 of the Jacobian by dual numbers (seed dC0).
//! Prints one JSON object (same keys as the Python example).
use agent_tools::autodiff::{Dual, Real};
use agent_tools::ode::rk4_step_jacobian::{rk4_step, rk4_step_jacobian};

const QV: f64 = 1.0 / 60.0;
const C0: f64 = 1.0;
const T0: f64 = 350.0;
const TC: f64 = 300.0;
const K0: f64 = 1.2e9;
const ER: f64 = 8750.0;
const DH: f64 = 209.0;
const UA: f64 = 2.09 / 60.0;

fn rhs<T: Real>(_t: f64, y: &[T], d: &mut [T]) {
    let c = T::cst;
    let k = c(K0) * (-(c(ER) / y[1])).exp();
    d[0] = c(QV) * (c(C0) - y[0]) - k * y[0];
    d[1] = c(QV) * (c(T0) - y[1]) + c(DH) * k * y[0] + c(UA) * (c(TC) - y[1]);
}
fn jacobian<T: Real>(_t: f64, y: &[T], j: &mut [T]) {
    let c = T::cst;
    let k = c(K0) * (-(c(ER) / y[1])).exp();
    let dk = k * c(ER) / (y[1] * y[1]);
    j[0] = -c(QV) - k;
    j[1] = -(dk * y[0]);
    j[2] = c(DH) * k;
    j[3] = -c(QV) - c(UA) + c(DH) * dk * y[0];
}

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let y0 = [0.5, 355.0];
    let r = rk4_step_jacobian(rhs::<f64>, jacobian::<f64>, 0.0, &y0, 10.0, 10);
    let yd = [Dual::new(0.5, 1.0), Dual::new(355.0, 0.0)];
    let rd = rk4_step(rhs::<Dual<f64>>, 0.0, &yd, 10.0, 10);
    println!(
        "{{\"success\":{},\"message\":{:?},\"nfev\":{},\"njac\":{},\"y1\":{},\"phi\":{},\"dual_col0\":{}}}",
        r.success, r.message, r.nfev, r.njac, list(&r.y1), list(&r.phi),
        list(&rd.y1.iter().map(|d| d.d).collect::<Vec<_>>())
    );
}
