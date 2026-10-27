//! Tests for random::mulberry32: streams equal to Ettinger's JavaScript (reference data computed
//! in node), plus statistics.
use agent_tools::random::mulberry32::Mulberry32;

const TOL_MEAN_UNIFORM: f64 = 5e-3;
const TOL_VAR_UNIFORM: f64 = 2e-3;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/mulberry32.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn streams_are_bit_identical() {
    let r = load();
    let mut n = 0;
    for (seed, c) in r["cases"]["streams"].as_object().unwrap() {
        let s: u32 = seed.parse().unwrap();
        let (mut a, mut b, mut d, mut e) = (Mulberry32::new(s), Mulberry32::new(s), Mulberry32::new(s), Mulberry32::new(s));
        for i in 0..c["u32"].as_array().unwrap().len() {
            assert_eq!(a.next_u32() as u64, c["u32"][i].as_u64().unwrap(), "seed {seed}");
            assert_eq!(b.random(), c["random"][i].as_f64().unwrap());
            assert_eq!(d.below(10) as u64, c["below_10"][i].as_u64().unwrap());
            assert_eq!(e.below(1000) as u64, c["below_1000"][i].as_u64().unwrap());
            n += 1;
        }
    }
    // existing project check: heat-exchanger-network checks.rs::mulberry32_matches_javascript
    let mut g = Mulberry32::new(1);
    let v: Vec<u32> = (0..4).map(|_| g.next_u32()).collect();
    assert_eq!(v, vec![2693262067, 11749833, 2265367787, 4213581821]);
    println!("VERIFY mulberry32 streams mismatches_vs_node=0 values_checked={n}");
}

#[test]
fn statistics() {
    let mut g = Mulberry32::new(2024);
    let u: Vec<f64> = (0..100_000).map(|_| g.random()).collect();
    assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));
    let m = u.iter().sum::<f64>() / u.len() as f64;
    let v = u.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / u.len() as f64;
    println!("VERIFY mulberry32 uniform_1e5 mean_minus_half={:e} var_minus_1_12={:e}", m - 0.5, v - 1.0 / 12.0);
    assert!((m - 0.5).abs() < TOL_MEAN_UNIFORM && (v - 1.0 / 12.0).abs() < TOL_VAR_UNIFORM);
}
