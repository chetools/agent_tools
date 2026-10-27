//! Tests for random::splitmix64: bit-exact streams from the published algorithm, plus statistics.
use agent_tools::random::splitmix64::SplitMix64;

/// Box-Muller normals vs Python/libm: relative rounding difference.
const TOL_NORMAL: f64 = 1e-14;
/// Statistical sanity with N = 1e5 samples (fixed seed, so deterministic): about 5-8 standard errors.
const TOL_MEAN_UNIFORM: f64 = 5e-3;
const TOL_VAR_UNIFORM: f64 = 2e-3;
const TOL_MEAN_NORMAL: f64 = 2e-2;
const TOL_VAR_NORMAL: f64 = 3e-2;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/splitmix64.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn streams_are_bit_identical() {
    let r = load();
    let mut n_checked = 0;
    for (seed, c) in r["cases"]["streams"].as_object().unwrap() {
        let mut g = SplitMix64::new(seed.parse::<u64>().unwrap());
        let mut h = SplitMix64::new(seed.parse::<u64>().unwrap());
        for (u, f) in c["u64"].as_array().unwrap().iter().zip(c["uniform"].as_array().unwrap()) {
            assert_eq!(g.next_u64(), u.as_u64().unwrap(), "seed {seed}");
            assert_eq!(h.uniform(), f.as_f64().unwrap(), "seed {seed}");
            n_checked += 1;
        }
    }
    let ss = &r["cases"]["seed_stream"];
    let mut g = SplitMix64::from_seed_stream(ss["seed"].as_u64().unwrap(), ss["stream"].as_u64().unwrap());
    for u in ss["u64"].as_array().unwrap() {
        assert_eq!(g.next_u64(), u.as_u64().unwrap());
    }
    println!("VERIFY splitmix64 streams u64_and_uniform_mismatches=0 values_checked={n_checked}");
}

#[test]
fn normals_and_permutations() {
    let r = load();
    let mut worst = 0f64;
    for (seed, v) in r["cases"]["normal"].as_object().unwrap() {
        let mut g = SplitMix64::new(seed.parse::<u64>().unwrap());
        for x in v.as_array().unwrap() {
            let (a, b) = (g.normal(), x.as_f64().unwrap());
            worst = worst.max((a - b).abs() / b.abs().max(1e-300));
        }
    }
    println!("VERIFY splitmix64 normal max_rel_diff_vs_reference={worst:e}");
    assert!(worst < TOL_NORMAL);
    let mut n_perm = 0;
    for (key, v) in r["cases"]["permutation"].as_object().unwrap() {
        let (s, n) = key.split_once(':').unwrap();
        let p = SplitMix64::new(s.parse().unwrap()).permutation(n.parse().unwrap());
        let want: Vec<usize> = v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(p, want, "{key}");
        n_perm += 1;
    }
    assert!(SplitMix64::new(1).permutation(0).is_empty());
    println!("VERIFY splitmix64 permutation mismatches=0 cases={n_perm}");
}

#[test]
fn statistics() {
    let n = 100_000;
    let mut g = SplitMix64::new(2024);
    let u: Vec<f64> = (0..n).map(|_| g.uniform()).collect();
    assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));
    let (m, v) = mean_var(&u);
    println!("VERIFY splitmix64 uniform_1e5 mean_minus_half={:e} var_minus_1_12={:e}", m - 0.5, v - 1.0 / 12.0);
    assert!((m - 0.5).abs() < TOL_MEAN_UNIFORM && (v - 1.0 / 12.0).abs() < TOL_VAR_UNIFORM);
    let z: Vec<f64> = (0..n).map(|_| g.normal()).collect();
    let (m, v) = mean_var(&z);
    println!("VERIFY splitmix64 normal_1e5 mean={m:e} var_minus_1={:e}", v - 1.0);
    assert!(m.abs() < TOL_MEAN_NORMAL && (v - 1.0).abs() < TOL_VAR_NORMAL);
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    (m, x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / x.len() as f64)
}
