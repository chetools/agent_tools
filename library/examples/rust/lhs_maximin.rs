//! Example: maximin Latin hypercube, 12 points in 3 dimensions, 50 candidates, seed 42 stream 0
//! (SplitMix64::from_seed_stream), and the hard case n = 50, d = 10 with seed 7.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::random::splitmix64::SplitMix64;
use agent_tools::stats::lhs_maximin::lhs_maximin;

fn fmt(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let a = lhs_maximin(&mut SplitMix64::from_seed_stream(42, 0), 12, 3, 50);
    let b = lhs_maximin(&mut SplitMix64::from_seed_stream(7, 0), 50, 10, 50);
    println!(
        "{{\"small_points\":{},\"small_min_distance\":{:?},\"small_best_try\":{},\"hard_points\":{},\"hard_min_distance\":{:?},\"hard_best_try\":{}}}",
        fmt(&a.points), a.min_distance, a.best_try, fmt(&b.points), b.min_distance, b.best_try
    );
}
