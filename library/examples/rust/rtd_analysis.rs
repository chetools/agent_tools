//! Example: RTD analysis of two synthetic records truncated early (tanks N = 4, tau = 60 s;
//! open-open dispersion Pe = 12, L/u = 80 s): moments with tail, moment fits, least-squares fits.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::stats::rtd_analysis::{
    e_dispersion, e_tanks, fit_lsq, fit_moments, rtd_moments, BoundaryCondition, RtdModel,
};

fn main() {
    let (n, dt) = (481usize, 0.5f64);
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let c: Vec<f64> = t.iter().map(|&x| 3.0 * e_tanks(x / 60.0, 4.0) / 60.0).collect();
    let m = rtd_moments(&t, &c, 0.0, true);
    let oc = BoundaryCondition::OpenOpen;
    let nt = fit_moments(m.mean, m.variance, RtdModel::Tanks, oc);
    let oo = fit_moments(m.mean, m.variance, RtdModel::Dispersion, oc);
    let cc = fit_moments(m.mean, m.variance, RtdModel::Dispersion, BoundaryCondition::ClosedClosed);
    let lt = fit_lsq(&t, &m.e, RtdModel::Tanks, None);
    let t2: Vec<f64> = (0..801).map(|i| i as f64 * 0.5).collect();
    let e2: Vec<f64> = t2.iter().map(|&x| e_dispersion(x / 80.0, 12.0) / 80.0).collect();
    let ld = fit_lsq(&t2, &e2, RtdModel::Dispersion, None);
    let e_sum: f64 = m.e.iter().sum::<f64>() * dt;
    println!(
        "{{\"mean\":{:?},\"variance\":{:?},\"lam\":{:?},\"e_sum_dt\":{:?},\"N_moments\":{:?},\"Pe_oo\":{:?},\"tau_oo\":{:?},\"Pe_cc\":{:?},\"N_lsq\":{:?},\"tau_lsq\":{:?},\"sse_lsq\":{:?},\"nfev_lsq\":{},\"ok_lsq\":{},\"Pe_lsq\":{:?},\"tau_disp_lsq\":{:?},\"nfev_disp\":{},\"ok_disp\":{}}}",
        m.mean, m.variance, m.tail_decay_time.unwrap(), e_sum, nt.param, oo.param, oo.tau, cc.param,
        lt.param, lt.tau, lt.sse, lt.nfev, lt.success, ld.param, ld.tau, ld.nfev, ld.success
    );
}
