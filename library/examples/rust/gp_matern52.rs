//! Example: fit a Matern-5/2 ARD GP to 15 noisy samples of f(x) = sin(6 x1) + x2^2 on [0,1]^2
//! (inputs from SplitMix64 seed 11, noise 0.05 N(0,1)), with the default starts plus 2 random
//! starts (seed 5); predict at 4 points and give the posterior covariance between 3 of them.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::random::splitmix64::SplitMix64;
use agent_tools::stats::gp_matern52::{gp_fit, GpOptions};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let mut rng = SplitMix64::new(11);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..15 {
        let (a, b) = (rng.uniform(), rng.uniform());
        x.extend([a, b]);
        y.push((6.0 * a).sin() + b * b + 0.05 * rng.normal());
    }
    let opt = GpOptions { n_random_starts: 2, seed: 5, ..GpOptions::default() };
    let r = gp_fit(&x, 2, &y, &opt).expect("fit failed");
    let g = [0.1, 0.2, 0.5, 0.5, 0.9, 0.3, 0.33, 0.77];
    let (mean, var) = r.model.predict(&g);
    let cov = r.model.posterior_cov(&g[..6], &g[..6]);
    println!(
        "{{\"length_scales\":{},\"signal_sd\":{:?},\"noise_sd\":{:?},\"nll_std\":{:?},\"log_marginal_likelihood\":{:?},\"start_nll\":{},\"best_start\":{},\"success\":{},\"nfev\":{},\"nit\":{},\"mean\":{},\"var\":{},\"cov\":{}}}",
        list(&r.model.length_scales),
        r.model.signal_sd,
        r.model.noise_sd,
        r.nll_std,
        r.model.log_marginal_likelihood,
        list(&r.start_nll),
        r.best_start,
        r.success,
        r.nfev,
        r.nit,
        list(&mean),
        list(&var),
        list(&cov)
    );
}
