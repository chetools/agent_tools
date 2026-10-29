//! Example: Arenstorf's periodic orbit (restricted three-body problem, Hairer-Norsett-Wanner I, II.0)
//! over one period at rtol 1e-6, atol 1e-9. Prints one JSON object (same keys as the Python example).
use agent_tools::ode::dopri5::{dopri5, Dopri5Options};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let m = 0.012277471; // moon mass fraction, dimensionless
    let mh = 1.0 - m;
    let f = |_: f64, y: &[f64], d: &mut [f64]| {
        let d1 = ((y[0] + m).powi(2) + y[1] * y[1]).powf(1.5);
        let d2 = ((y[0] - mh).powi(2) + y[1] * y[1]).powf(1.5);
        d[0] = y[2];
        d[1] = y[3];
        d[2] = y[0] + 2.0 * y[3] - mh * (y[0] + m) / d1 - m * (y[0] - mh) / d2;
        d[3] = y[1] - 2.0 * y[2] - mh * y[1] / d1 - m * y[1] / d2;
    };
    let period = 17.0652165601579625588917206249;
    let y0 = [0.994, 0.0, 0.0, -2.00158510637908252240537862224];
    let o = Dopri5Options { rtol: 1e-6, atol: 1e-9, ..Default::default() };
    let r = dopri5(f, (0.0, period), &y0, &o);
    let mid = r.nsteps / 2;
    println!(
        "{{\"success\":{},\"message\":\"{}\",\"nsteps\":{},\"nrejected\":{},\"nfev\":{},\"t_mid\":{:?},\"y_mid\":{},\"t_final\":{:?},\"y\":{}}}",
        r.success, r.message, r.nsteps, r.nrejected, r.nfev, r.t[mid], list(r.y_at(mid)), r.t[r.nsteps], list(r.y_final())
    );
}
