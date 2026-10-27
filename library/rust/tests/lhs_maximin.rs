//! Tests for stats::lhs_maximin: Latin property, bit-exact designs vs an independent implementation,
//! monotone improvement with the number of candidates, comparison with scipy.stats.qmc.LatinHypercube.
use agent_tools::random::splitmix64::SplitMix64;
use agent_tools::stats::lhs_maximin::{lhs_maximin, lhs_random, min_pairwise_distance};

/// Minimum distance recomputed by scipy pdist vs ours (summation order only).
const TOL_DMIN: f64 = 1e-14;
/// Fair test: mean min distance of ours (tries = 1) vs scipy plain LHS, same distribution:
/// accept a difference up to this many standard errors of the difference (200 seeds each).
const Z_FAIR: f64 = 4.0;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/lhs_maximin.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    (m, x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (x.len() as f64 - 1.0))
}

/// Exactly one point in each of the n slices of every input, all coordinates in [0, 1).
fn is_latin(x: &[f64], n: usize, d: usize) -> bool {
    if x.len() != n * d || x.iter().any(|&v| !(0.0..1.0).contains(&v)) {
        return false;
    }
    (0..d).all(|i| {
        let mut seen = vec![0usize; n];
        for k in 0..n {
            seen[(x[k * d + i] * n as f64).floor() as usize] += 1;
        }
        seen.iter().all(|&c| c == 1)
    })
}

#[test]
fn latin_property_many_cases() {
    let mut cases = 0;
    let mut bad = 0;
    for n in [1usize, 2, 3, 5, 8, 13, 20, 33, 50, 100] {
        for d in [1usize, 2, 3, 6, 10] {
            for seed in 0..6u64 {
                let r = lhs_maximin(&mut SplitMix64::from_seed_stream(seed, 0), n, d, 10);
                assert!(r.success && r.n == n && r.d == d && r.n_candidates == 10);
                if !is_latin(&r.points, n, d) {
                    bad += 1;
                }
                // reported distance equals the recomputed one
                assert_eq!(r.min_distance, min_pairwise_distance(&r.points, n, d));
                // a plain candidate is Latin too
                if !is_latin(&lhs_random(&mut SplitMix64::from_seed_stream(seed, 1), n, d), n, d) {
                    bad += 1;
                }
                cases += 2;
            }
        }
    }
    println!("VERIFY lhs_maximin latin cases={cases} violations={bad}");
    assert_eq!(bad, 0);
}

#[test]
fn designs_match_independent_implementation() {
    let r = load();
    let mut worst = 0f64;
    let mut mismatches = 0;
    for c in r["cases"]["designs"].as_array().unwrap() {
        let (n, d) = (c["n"].as_u64().unwrap() as usize, c["d"].as_u64().unwrap() as usize);
        let res = lhs_maximin(
            &mut SplitMix64::from_seed_stream(c["seed"].as_u64().unwrap(), 0),
            n,
            d,
            c["tries"].as_u64().unwrap() as usize,
        );
        let want = vec_f64(&c["points"]);
        mismatches += res.points.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(res.best_try as u64, c["best_try"].as_u64().unwrap(), "n={n} d={d}");
        let wd = c["min_distance"].as_f64().unwrap();
        worst = worst.max((res.min_distance - wd).abs() / wd);
        if n == 50 && d == 10 {
            println!("VERIFY lhs_maximin hard_n50_d10 min_distance={:.6} best_try={}", res.min_distance, res.best_try);
        }
    }
    println!("VERIFY lhs_maximin designs coordinate_mismatches={mismatches} max_rel_dmin_diff_vs_scipy_pdist={worst:e}");
    assert_eq!(mismatches, 0);
    assert!(worst < TOL_DMIN);
}

#[test]
fn improves_monotonically_with_tries() {
    let ks = [1usize, 2, 5, 10, 20, 50, 100, 200];
    let mut means = vec![0.0; ks.len()];
    let seeds = 50u64;
    for seed in 0..seeds {
        let mut prev = -1.0;
        for (j, &k) in ks.iter().enumerate() {
            let r = lhs_maximin(&mut SplitMix64::from_seed_stream(seed, 0), 30, 5, k);
            assert!(r.min_distance >= prev, "seed {seed} tries {k}");
            prev = r.min_distance;
            means[j] += r.min_distance / seeds as f64;
        }
    }
    for w in means.windows(2) {
        assert!(w[1] > w[0]);
    }
    let txt: Vec<String> = ks.iter().zip(&means).map(|(k, m)| format!("k{k}={m:.4}")).collect();
    println!("VERIFY lhs_maximin monotone n30_d5 nondecreasing_every_seed=1 mean_min_distance {}", txt.join(" "));
}

#[test]
fn seed_study_vs_scipy() {
    let r = load();
    let mut worst_ours = 0f64;
    for (key, c) in r["cases"]["seed_study"].as_object().unwrap() {
        let (n, d) = (c["n"].as_u64().unwrap() as usize, c["d"].as_u64().unwrap() as usize);
        let nseeds = c["ours_1"].as_array().unwrap().len();
        let (w1, w50) = (vec_f64(&c["ours_1"]), vec_f64(&c["ours_50"]));
        let (sp, cd) = (vec_f64(&c["scipy_plain"]), vec_f64(&c["scipy_random_cd"]));
        let (mut o1, mut o50) = (Vec::new(), Vec::new());
        for s in 0..nseeds {
            o1.push(lhs_maximin(&mut SplitMix64::from_seed_stream(s as u64, 0), n, d, 1).min_distance);
            o50.push(lhs_maximin(&mut SplitMix64::from_seed_stream(s as u64, 0), n, d, 50).min_distance);
        }
        for (a, b) in o1.iter().zip(&w1).chain(o50.iter().zip(&w50)) {
            worst_ours = worst_ours.max((a - b).abs() / b);
        }
        let (m1, v1) = mean_var(&o1);
        let (m50, v50) = mean_var(&o50);
        let (msp, vsp) = mean_var(&sp);
        let (mcd, _) = mean_var(&cd);
        let z = (m1 - msp) / (v1 / nseeds as f64 + vsp / nseeds as f64).sqrt();
        println!(
            "VERIFY lhs_maximin study_{key} mean_dmin ours_tries1={m1:.4} ours_tries50={m50:.4} sd50={:.4} scipy_plain={msp:.4} scipy_random_cd={mcd:.4} z_ours1_vs_scipy_plain={z:.2} nseeds={nseeds}",
            v50.sqrt()
        );
        // fair: same distribution as one scipy plain LHS
        assert!(z.abs() < Z_FAIR, "{key}: z = {z}");
        // fair: best of 50 candidates beats a single candidate
        assert!(m50 > msp, "{key}");
        // comparison with scipy optimization='random-cd' (a discrepancy criterion, not maximin) is reported only
    }
    println!("VERIFY lhs_maximin study max_rel_diff_vs_independent_impl={worst_ours:e}");
    assert_eq!(worst_ours, 0.0);
}

#[test]
fn invalid_input_and_single_point() {
    for (n, d, t) in [(0, 3, 5), (4, 0, 5), (4, 3, 0)] {
        let r = lhs_maximin(&mut SplitMix64::new(1), n, d, t);
        assert!(!r.success && r.points.is_empty() && r.min_distance.is_nan() && !r.message.is_empty());
    }
    let r = lhs_maximin(&mut SplitMix64::new(1), 1, 4, 3);
    assert!(r.success && r.min_distance == 1.0 && r.points.len() == 4);
    println!("VERIFY lhs_maximin edge invalid_inputs_fail=1 single_point_dmin=1");
}
