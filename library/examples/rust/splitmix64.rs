//! Example: SplitMix64 stream for state 42: integers, uniforms, normals, a shuffle.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::random::splitmix64::SplitMix64;

fn main() {
    let mut r = SplitMix64::new(42);
    let u64s: Vec<u64> = (0..5).map(|_| r.next_u64()).collect();
    let uniforms: Vec<f64> = (0..5).map(|_| r.uniform()).collect();
    let normals: Vec<f64> = (0..5).map(|_| r.normal()).collect();
    let perm = r.permutation(10);
    let fmt = |v: &[f64]| format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","));
    println!(
        "{{\"u64\":{:?},\"uniform\":{},\"normal\":{},\"permutation\":{:?}}}",
        u64s,
        fmt(&uniforms),
        fmt(&normals),
        perm
    );
}
