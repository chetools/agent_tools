//! Example: Mulberry32 stream for seed 1: integers, uniforms, bounded integers.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::random::mulberry32::Mulberry32;

fn main() {
    let mut r = Mulberry32::new(1);
    let u32s: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
    let uniforms: Vec<f64> = (0..4).map(|_| r.random()).collect();
    let below: Vec<usize> = (0..6).map(|_| r.below(10)).collect();
    let fmt = |v: &[f64]| format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","));
    println!("{{\"u32\":{:?},\"random\":{},\"below_10\":{:?}}}", u32s, fmt(&uniforms), below);
}
