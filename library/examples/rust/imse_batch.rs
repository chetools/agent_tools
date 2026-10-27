//! Example: greedy IMSE batches for a GP with fixed hyperparameters (length scales 0.3, 0.5,
//! signal_sd 1, noise_sd 0.05) conditioned on 6 samples of f(x) = sin(6 x1) + x2^2 (inputs from
//! SplitMix64 seed 3). Case "same": candidates = reference = 6 x 6 grid on [0,1]^2, batch 4.
//! Case "other_ref": the same candidates, 25 reference points from SplitMix64 seed 4, batch 3.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::random::splitmix64::SplitMix64;
use agent_tools::stats::gp_matern52::gp_build;
use agent_tools::stats::imse_batch::imse_batch;

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let mut rng = SplitMix64::new(3);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..6 {
        let (a, b) = (rng.uniform(), rng.uniform());
        x.extend([a, b]);
        y.push((6.0 * a).sin() + b * b);
    }
    let gp = gp_build(&x, 2, &y, &[0.3, 0.5], 1.0, 0.05, None).expect("build failed");
    let mut cand = Vec::new();
    for j in 0..6 {
        for i in 0..6 {
            cand.extend([i as f64 / 5.0, j as f64 / 5.0]);
        }
    }
    let mut rr = SplitMix64::new(4);
    let refp: Vec<f64> = (0..50).map(|_| rr.uniform()).collect();
    let c_cc = gp.posterior_cov(&cand, &cand);
    let mut parts = Vec::new();
    for (name, r_pts, b) in [("same", &cand, 4usize), ("other_ref", &refp, 3usize)] {
        let nr = r_pts.len() / 2;
        let res = imse_batch(&c_cc, &gp.posterior_cov(r_pts, &cand), 36, nr, gp.noise_sd * gp.noise_sd, b);
        let idx: Vec<String> = res.indices.iter().map(|i| i.to_string()).collect();
        parts.push(format!(
            "\"{name}\":{{\"indices\":[{}],\"reduction\":{},\"first_scores\":{}}}",
            idx.join(","),
            list(&res.reduction),
            list(&res.first_scores)
        ));
    }
    println!("{{{}}}", parts.join(","));
}
