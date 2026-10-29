//! Tests for optimize::qp_box: KKT certificates, exact / enumerated / SciPy references, the muse fixture, failure modes.
use agent_tools::linalg::cholesky::{cholesky, cholesky_solve};
use agent_tools::optimize::qp_box::{qp_box, QpBoxOptions, QpBoxResult};
use agent_tools::random::splitmix64::SplitMix64;

// KKT certificate, relative to scale = 1 + max|g| + max|H| sum|x| (a bound on |Hx| and |g|):
/// Stationarity: max over free i of |(Hx+g)_i| / scale (measured <= 3e-17).
const TOL_STAT: f64 = 1e-13;
/// Multiplier sign: a bound-active variable's q-based multiplier >= -TOL_SIGN scale.
const TOL_SIGN: f64 = 1e-13;
/// Reported multipliers vs the ones recomputed from q = Hx + g, relative to scale.
const TOL_MULT_REPORT: f64 = 1e-12;
// Feasibility (lo <= x <= hi) and complementarity (active variable exactly on its bound, free one with multiplier 0)
// are tested exactly (tolerance 0).
/// max |x - x_ref| (exact / enumerated); measured <= 2.5e-13 (cond 1e8, n = 30).
const TOL_X_REF: f64 = 1e-9;
/// max |multiplier - reference| relative to 1 + max reference multiplier; measured <= 5e-10 (cond 1e8).
const TOL_MULT_REF: f64 = 1e-8;
/// |f - f_ref| relative to 1 + |x|^T |H| |x| + |g|^T |x| (rounding of f itself).
const TOL_F_REF: f64 = 1e-13;
/// max |x - x_scipy| (L-BFGS-B ftol 1e-15, gtol 1e-12); measured <= 3.3e-7.
const TOL_X_SCIPY: f64 = 1e-6;
/// Same for cond(H) = 1e8 (L-BFGS-B is accurate to about 5e-6 there).
const TOL_X_SCIPY_ILL: f64 = 1e-4;
/// The muse test's own tolerance against its SciPy fixture (measured 5.7e-8).
const TOL_X_MUSE: f64 = 1e-6;
/// "Zero multiplier" cases: |multiplier| <= this (g carries the rounding of 1e-10 data).
const TOL_ZERO_MULT: f64 = 1e-8;
/// A reference variable counts as "at a bound" when within this of it.
const BOUND_TIE: f64 = 1e-9;

fn load(name: &str) -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn vec(v: &serde_json::Value, null: f64) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(null)).collect()
}
fn ivec(v: &serde_json::Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}
fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

struct Problem {
    n: usize,
    h: Vec<f64>,
    g: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}
fn problem(c: &serde_json::Value) -> Problem {
    let g = vec(&c["g"], 0.0);
    Problem {
        n: g.len(),
        h: vec(&c["H"], 0.0),
        lo: vec(&c["lo"], f64::NEG_INFINITY),
        hi: vec(&c["hi"], f64::INFINITY),
        g,
    }
}
fn solve(p: &Problem) -> QpBoxResult {
    qp_box(&p.h, &p.g, &p.lo, &p.hi, &QpBoxOptions::default())
}
fn matvec(p: &Problem, x: &[f64]) -> Vec<f64> {
    (0..p.n).map(|i| (0..p.n).map(|j| p.h[i * p.n + j] * x[j]).sum::<f64>()).collect()
}

/// Max-norm residuals of the KKT conditions of `r`, recomputed with the full H (the routine reads the lower triangle).
struct Cert {
    stat: f64,
    sign: f64,
    feas: f64,
    compl: f64,
    mult_report: f64,
}
fn certificate(p: &Problem, r: &QpBoxResult) -> Cert {
    let n = p.n;
    let q: Vec<f64> = matvec(p, &r.x).iter().zip(&p.g).map(|(a, b)| a + b).collect();
    let hmax = p.h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let gmax = p.g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let scale = 1.0 + gmax + hmax * r.x.iter().map(|v| v.abs()).sum::<f64>();
    let mut c = Cert { stat: 0.0, sign: 0.0, feas: 0.0, compl: 0.0, mult_report: 0.0 };
    for i in 0..n {
        let (a, x) = (r.active[i], r.x[i]);
        c.feas = c.feas.max(p.lo[i] - x).max(x - p.hi[i]);
        let fixed = p.lo[i] == p.hi[i];
        match a {
            0 => {
                c.stat = c.stat.max(q[i].abs() / scale);
                c.compl = c.compl.max(r.multipliers[i].abs());
                c.mult_report = c.mult_report.max(r.multipliers[i].abs() / scale);
            }
            _ => {
                let bound = if a == -1 { p.lo[i] } else { p.hi[i] };
                c.compl = c.compl.max((x - bound).abs());
                let m = if a == -1 { q[i] } else { -q[i] };
                if !fixed {
                    c.sign = c.sign.max(-m / scale);
                }
                let expect = if fixed { q[i].abs() } else { m.max(0.0) };
                c.mult_report = c.mult_report.max((r.multipliers[i] - expect).abs() / scale);
            }
        }
    }
    c
}
fn check_certificate(p: &Problem, r: &QpBoxResult, label: &str) -> Cert {
    assert!(r.success, "{label}: {}", r.message);
    let c = certificate(p, r);
    assert!(
        c.stat <= TOL_STAT && c.sign <= TOL_SIGN && c.feas <= 0.0 && c.compl <= 0.0 && c.mult_report <= TOL_MULT_REPORT,
        "{label}: stat {:e} sign {:e} feas {:e} compl {:e} mult_report {:e}",
        c.stat, c.sign, c.feas, c.compl, c.mult_report
    );
    assert!(r.multipliers.iter().all(|&m| m >= 0.0));
    c
}
fn f_scale(p: &Problem, x: &[f64]) -> f64 {
    let mut s = 1.0;
    for i in 0..p.n {
        s += p.g[i].abs() * x[i].abs();
        for j in 0..p.n {
            s += x[i].abs() * p.h[i * p.n + j].abs() * x[j].abs();
        }
    }
    s
}

fn active_agrees(r: &QpBoxResult, ref_active: &[i64], ref_mult: &[f64], x_ref: &[f64], p: &Problem) -> bool {
    for i in 0..p.n {
        let at_bound = (x_ref[i] - p.lo[i]).abs() <= BOUND_TIE || (x_ref[i] - p.hi[i]).abs() <= BOUND_TIE;
        if ref_active[i] == 0 && !at_bound {
            if r.active[i] != 0 {
                return false;
            }
        } else if ref_mult[i] > TOL_ZERO_MULT && p.lo[i] != p.hi[i] && r.active[i] as i64 != ref_active[i] {
            return false;
        }
    }
    true
}

#[test]
fn reference_cases() {
    let doc = load("qp_box");
    for (name, c) in doc["cases"].as_object().unwrap() {
        let p = problem(c);
        let r = solve(&p);
        let cert = check_certificate(&p, &r, name);
        let x_ref = vec(&c["x_ref"], 0.0);
        let ex = maxdiff(&r.x, &x_ref);
        assert!(ex <= TOL_X_REF, "{name}: {ex:e}");
        let refd = if c.get("exact").is_some() { Some(&c["exact"]) } else { c.get("enum") };
        let (mut em, mut ef) = (f64::NAN, f64::NAN);
        if let Some(rf) = refd {
            let rm = vec(&rf["multipliers"], 0.0);
            let rmax = rm.iter().fold(0.0f64, |m, v| m.max(*v));
            em = maxdiff(&r.multipliers, &rm) / (1.0 + rmax);
            ef = (r.f - rf["f"].as_f64().unwrap()).abs() / f_scale(&p, &r.x);
            assert!(em <= TOL_MULT_REF && ef <= TOL_F_REF, "{name}: mult {em:e} f {ef:e}");
            assert!(active_agrees(&r, &ivec(&rf["active"]), &rm, &x_ref, &p), "{name}: {:?} vs {:?}", r.active, rf["active"]);
        } else {
            assert!(r.active.iter().all(|&a| a == 0), "{name}");
        }
        let sc = c["scipy"].clone();
        let es = maxdiff(&r.x, &vec(&sc["x"], 0.0));
        let tol_s = if c["kind"] == "ill_conditioned" { TOL_X_SCIPY_ILL } else { TOL_X_SCIPY };
        assert!(es <= tol_s, "{name}: {es:e}");
        println!(
            "VERIFY qp_box rust_{name} n={} x_ref={} max_err_x_vs_ref={ex:.2e} rel_err_mult={em:.2e} rel_err_f={ef:.2e} kkt_stat={:.1e} kkt_sign={:.1e} kkt_feas={:.1e} diff_x_vs_scipy={es:.2e} nactive={} nit={} nfact={} scipy_nit={}",
            p.n, c["x_ref_source"].as_str().unwrap(), cert.stat, cert.sign, cert.feas,
            r.active.iter().filter(|&&a| a != 0).count(), r.nit, r.nfact, sc["nit"]
        );
    }
}

#[test]
fn unconstrained_equals_direct_cholesky_solve() {
    let doc = load("qp_box");
    for name in ["unconstrained_inf_n10", "unconstrained_wide_n10"] {
        let p = problem(&doc["cases"][name]);
        let r = solve(&p);
        let ch = cholesky(&p.h, p.n);
        let mut xd: Vec<f64> = p.g.iter().map(|v| -v).collect();
        cholesky_solve(&ch.l, p.n, &mut xd, 1);
        let d = maxdiff(&r.x, &xd);
        let xmax = xd.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(r.success && r.active.iter().all(|&a| a == 0) && r.multipliers.iter().all(|&m| m == 0.0));
        assert!(d <= 1e-13 * (1.0 + xmax), "{d:e}");
        // one step to the minimizer, one pass to certify (no new solve): probe + 1 free-set factorization
        assert_eq!((r.nit, r.nfact), (2, 2));
        println!("VERIFY qp_box rust_{name}_vs_cholesky max_diff_x={d:.2e} nit={} nfact={}", r.nit, r.nfact);
    }
}

#[test]
fn all_bounds_active() {
    let doc = load("qp_box");
    let p = problem(&doc["cases"]["all_active_n5"]);
    let r = solve(&p);
    check_certificate(&p, &r, "all_active");
    assert!(r.active.iter().all(|&a| a != 0) && r.multipliers.iter().all(|&m| m > 0.0));
    for i in 0..p.n {
        assert_eq!(r.x[i], if r.active[i] == -1 { p.lo[i] } else { p.hi[i] });
        assert_eq!(r.active[i] as f64, -p.g[i].signum());
    }
    let mmin = r.multipliers.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    println!("VERIFY qp_box rust_all_active nactive=5 min_multiplier={mmin:.2e} nit={} nfact={}", r.nit, r.nfact);
}

#[test]
fn degenerate_zero_multiplier() {
    let doc = load("qp_box");
    let c = &doc["cases"]["degenerate_zero_multiplier_n5"];
    let p = problem(c);
    let r = solve(&p);
    check_certificate(&p, &r, "degenerate");
    let xk = vec(&c["constructed"]["x"], 0.0);
    let dx = maxdiff(&r.x, &xk);
    assert!(dx <= 1e-9);
    let q: Vec<f64> = matvec(&p, &r.x).iter().zip(&p.g).map(|(a, b)| a + b).collect();
    assert!(q[0].abs() <= TOL_ZERO_MULT && q[1].abs() <= TOL_ZERO_MULT);
    assert!((r.x[0] - p.lo[0]).abs() <= 1e-9 && (r.x[1] - p.hi[1]).abs() <= 1e-9);
    assert!(r.active[3] == -1 && r.active[4] == 1);
    assert!((r.multipliers[3] - 1.5).abs() < 1e-8 && (r.multipliers[4] - 0.7).abs() < 1e-8);
    // start point 0 is on three bounds with gradient 0: converges at once, bounds stay active with multiplier 0
    let p2 = problem(&doc["cases"]["start_at_bound_zero_gradient_n4"]);
    let r2 = solve(&p2);
    check_certificate(&p2, &r2, "start_at_bound");
    assert_eq!(r2.active, vec![-1, 0, -1, 1]);
    assert!(r2.multipliers.iter().all(|&m| m == 0.0) && r2.x.iter().all(|&v| v == 0.0));
    assert_eq!((r2.nit, r2.nfact), (1, 2));
    println!(
        "VERIFY qp_box rust_degenerate constructed_max_err_x={dx:.2e} zero_mult_residuals={:.1e},{:.1e} nit={} start_at_bound_nit={} start_at_bound_active={:?}",
        q[0].abs(), q[1].abs(), r.nit, r2.nit, r2.active
    );
}

#[test]
fn fixed_variable_is_never_released() {
    let doc = load("qp_box");
    let p = problem(&doc["cases"]["fixed_variable_n5"]);
    let r = solve(&p);
    check_certificate(&p, &r, "fixed");
    assert!(r.x[1] == 0.3 && r.x[3] == -0.2 && r.active[1] != 0 && r.active[3] != 0);
    let q: Vec<f64> = matvec(&p, &r.x).iter().zip(&p.g).map(|(a, b)| a + b).collect();
    for i in [1, 3] {
        assert_eq!(r.active[i] == -1, q[i] >= 0.0);
    }
    assert!(r.nit <= 3, "{}", r.nit); // the muse source releases a wrong-signed fixed variable and re-adds it (6 iterations)
    println!("VERIFY qp_box rust_fixed_variable nit={} nfact={} active={:?}", r.nit, r.nfact, r.active);
}

#[test]
fn ill_conditioned_free_set_does_not_stall() {
    // Regression (found by the Python random sweep): n = 2, cond(H) = 6e6, free minimizer at |x| ~ 2e2. The residual of the
    // Cholesky solve is ~ eps cond |x| ~ 1e-7 > tol_step = 1e-10, so a re-solve after the full step never reaches
    // max|p| <= tol_step: the muse algorithm ran into its cap (4 n) and reported failure. Reference: the exact 2 x 2
    // solution of the same doubles in rational arithmetic (Python fractions.Fraction), rounded to f64.
    let h = [49189.72355913, -122006.2183222, -122006.2183222, 302621.52422222];
    let g = [-127.21531714, 931.73530249];
    let lo = [f64::NEG_INFINITY; 2];
    let hi = [f64::INFINITY, 0.37999599];
    let r = qp_box(&h, &g, &lo, &hi, &QpBoxOptions::default());
    assert!(r.success, "{}", r.message);
    let x_exact = [-213.69305675621672, -86.15650700607154];
    let dx = maxdiff(&r.x, &x_exact);
    assert!(dx <= 1e-6 && r.active == vec![0, 0]);
    assert_eq!((r.nit, r.nfact), (2, 2));
    println!("VERIFY qp_box rust_ill_conditioned_free_set_does_not_stall cond=6e6 max_err_x_vs_exact_rational={dx:.2e} nit={} nfact={}", r.nit, r.nfact);
}

#[test]
fn muse_fixture() {
    // chetools/muse twin-cstr/solver/tests/fixtures/qp.json: 26 random SPD box QPs n = 4..124 (SciPy L-BFGS-B reference)
    let doc = load("qp_box_muse_fixture");
    let (mut worst, mut nit_max, mut count) = (0.0f64, 0usize, 0);
    let mut nit124 = Vec::new();
    for (name, c) in doc["cases"].as_object().unwrap() {
        let p = problem(c);
        let r = solve(&p);
        check_certificate(&p, &r, name);
        let d = maxdiff(&r.x, &vec(&c["x_ref"], 0.0));
        assert!(d <= TOL_X_MUSE, "{name}: {d:e}");
        worst = worst.max(d);
        nit_max = nit_max.max(r.nit);
        if p.n == 124 {
            nit124.push(r.nit);
        }
        count += 1;
    }
    assert_eq!(count, 26);
    println!("VERIFY qp_box rust_muse_fixture cases={count} max_abs_dev_vs_scipy={worst:.2e} max_nit={nit_max} nit_n124={nit124:?}");
}

#[test]
fn random_problems_up_to_n50_certificate() {
    // H = B diag(d) B^T + diag(d_min) from Gaussian B (not orthogonal; conditioning is whatever it is), n up to 50.
    let mut rng = SplitMix64::new(20261029);
    let (mut worst, mut count, mut nit50) = (0.0f64, 0, Vec::new());
    for n in [2usize, 5, 10, 20, 30, 40, 50] {
        for spread in [0.0f64, 2.0, 4.0, 6.0] {
            let b: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
            let d: Vec<f64> = (0..n).map(|k| 10f64.powf(spread * k as f64 / (n - 1) as f64)).collect();
            let mut h = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let s: f64 = (0..n).map(|k| b[i * n + k] * d[k] * b[j * n + k]).sum();
                    h[i * n + j] = s;
                    h[j * n + i] = s;
                }
                h[i * n + i] += 1.0;
            }
            let hd = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
            let g: Vec<f64> = (0..n).map(|_| 2.0 * hd.sqrt() * rng.normal()).collect();
            let lo: Vec<f64> = (0..n).map(|_| -(0.05 + 0.95 * rng.uniform())).collect();
            let hi: Vec<f64> = (0..n).map(|_| 0.05 + 0.95 * rng.uniform()).collect();
            let p = Problem { n, h, g, lo, hi };
            let r = solve(&p);
            let c = check_certificate(&p, &r, &format!("n{n}_spread{spread}"));
            assert!(r.nit <= 50.max(20 * n));
            worst = worst.max(c.stat);
            if n == 50 {
                nit50.push(r.nit);
            }
            count += 1;
        }
    }
    println!("VERIFY qp_box rust_random_n50 problems={count} all_kkt_ok=1 max_kkt_stat={worst:.1e} nit_at_n50={nit50:?}");
}

// (n, H row-major, g, hi) with lo = 0: nondegenerate, well-conditioned, found by hill climbing on the pass count
const CAP_REPRODUCERS: [(usize, &[f64], &[f64], f64); 3] = [
    (3, &[1.0472399095265337, 0.9421865156436513, -1.5521828673786728, 0.9421865156436513, 1.179456712389007, -1.4812687162226472, -1.5521828673786728, -1.4812687162226472, 2.8891061401248], &[-0.2768828882302678, -0.4946297289368097, -0.4182782400244921], 1.602283819497114),
    (4, &[1.3802835278185264, -0.7516270309602238, -0.21645844042208112, -0.4591834757852036, -0.7516270309602238, 1.1844470243729093, 0.8437979841461524, 0.6266504515422855, -0.21645844042208112, 0.8437979841461524, 4.03710395165706, -0.9178952259079698, -0.4591834757852036, 0.6266504515422855, -0.9178952259079698, 1.3951420125652543], &[-0.4292886726390505, -1.4503185691175424, 0.31810988987637895, -1.9702710108575923], 1.579564953358756),
    (5, &[0.8868805615949854, -0.14703752442530216, -0.2680469265695435, -0.1016354121942829, -0.4192624030048251, -0.14703752442530216, 0.9936361398860136, 0.5205418672786014, -0.12310454567891946, -1.084242870466544, -0.2680469265695435, 0.5205418672786014, 1.2024044824646984, 0.5703495039483478, -0.8522460281354195, -0.1016354121942829, -0.12310454567891946, 0.5703495039483478, 1.6003587461216677, 0.23891391513417226, -0.4192624030048251, -1.084242870466544, -0.8522460281354195, 0.23891391513417226, 1.9895783536657685], &[-0.10766510778311858, -0.3708280984436486, -0.8334775274509005, -0.9713629640398181, -0.10279924143925213], 1.618494052698689),
];

#[test]
fn default_cap_reproducers() {
    // PR review: the old default cap 4 n failed on these ordinary problems (releases count as passes)
    let mut out = Vec::new();
    for (n, h, g, hi) in CAP_REPRODUCERS {
        let p = Problem { n, h: h.to_vec(), g: g.to_vec(), lo: vec![0.0; n], hi: vec![hi; n] };
        let r = solve(&p);
        assert!(r.success && r.nit > 4 * n && r.nit <= 50.max(20 * n), "n={n} nit={}", r.nit);
        let c = check_certificate(&p, &r, &format!("cap_n{n}"));
        let capped = qp_box(&p.h, &p.g, &p.lo, &p.hi, &QpBoxOptions { max_iter: Some(4 * n), ..QpBoxOptions::default() });
        assert!(!capped.success);
        out.push((n, r.nit, c.stat));
    }
    println!("VERIFY qp_box rust_default_cap_reproducers (n,nit,kkt_stat)={out:?} old_cap=4n_fails=ok");
}

#[test]
fn failures_and_options() {
    let o = QpBoxOptions::default();
    assert_eq!((o.max_iter, o.tol_step, o.tol_dual), (None, 1e-10, 1e-9));
    let h = [2.0, 0.5, 0.5, 1.0];
    let (g, lo, hi) = ([1.0, -2.0], [-1.0, -1.0], [1.0, 1.0]);
    let ok = qp_box(&h, &g, &lo, &hi, &o);
    assert!(ok.success && ok.message == "converged");
    // semidefinite and indefinite H: reported, x = clamp(0), nothing else done
    for hb in [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 3.0, 3.0, 1.0]] {
        let r = qp_box(&hb, &g, &lo, &hi, &o);
        assert!(!r.success && r.message.contains("not positive definite") && r.nit == 0 && r.nfact == 1);
        assert_eq!(r.x, vec![0.0, 0.0]);
        assert!(r.f.is_finite());
    }
    let r = qp_box(&[1.0, 1.0, 1.0, 1.0], &g, &[0.5, -1.0], &[1.0, 1.0], &o); // clamp(0) respects the bounds
    assert_eq!(r.x, vec![0.5, 0.0]);
    // invalid input: nfact 0, f NaN
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    let bad: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = vec![
        (h.to_vec(), g.to_vec(), vec![1.0, 0.0], vec![0.0, 1.0]),
        (h.to_vec(), g.to_vec(), vec![nan, 0.0], hi.to_vec()),
        (h.to_vec(), g.to_vec(), lo.to_vec(), vec![1.0, nan]),
        (h.to_vec(), g.to_vec(), vec![inf, 0.0], vec![inf, 1.0]),
        (h.to_vec(), g.to_vec(), lo.to_vec(), vec![-inf, 1.0]),
        (vec![nan, 0.0, 0.0, 1.0], g.to_vec(), lo.to_vec(), hi.to_vec()),
        (h.to_vec(), vec![inf, 0.0], lo.to_vec(), hi.to_vec()),
        (h.to_vec(), g.to_vec(), vec![-1.0], hi.to_vec()),
        (vec![1.0; 6], g.to_vec(), lo.to_vec(), hi.to_vec()),
    ];
    for (k, (hh, gg, l, u)) in bad.iter().enumerate() {
        let r = qp_box(hh, gg, l, u, &o);
        assert!(!r.success && r.message.contains("invalid input") && r.nfact == 0 && r.nit == 0 && r.f.is_nan(), "{k}: {}", r.message);
    }
    // invalid options
    let opts = [
        QpBoxOptions { max_iter: Some(0), ..o.clone() },
        QpBoxOptions { tol_step: -1.0, ..o.clone() },
        QpBoxOptions { tol_step: nan, ..o.clone() },
        QpBoxOptions { tol_dual: inf, ..o.clone() },
        QpBoxOptions { tol_dual: -1e-9, ..o.clone() },
    ];
    for op in &opts {
        let r = qp_box(&h, &g, &lo, &hi, op);
        assert!(!r.success && r.message.contains("invalid option") && r.nfact == 0);
    }
    // iteration cap: returns the feasible last iterate and says so
    let doc = load("qp_box");
    let p = problem(&doc["cases"]["random_n6"]);
    let r = qp_box(&p.h, &p.g, &p.lo, &p.hi, &QpBoxOptions { max_iter: Some(2), ..o.clone() });
    assert!(!r.success && r.message.contains("max_iter") && r.nit == 2);
    assert!((0..p.n).all(|i| r.x[i] >= p.lo[i] && r.x[i] <= p.hi[i]) && r.f.is_finite());
    let full = solve(&p);
    assert!(r.f > full.f);
    let with = |m: usize| qp_box(&p.h, &p.g, &p.lo, &p.hi, &QpBoxOptions { max_iter: Some(m), ..o.clone() });
    assert!(with(full.nit).success && !with(full.nit - 1).success);
    // only the lower triangle of H is read
    let mut hj = p.h.clone();
    for i in 0..p.n {
        for j in i + 1..p.n {
            hj[i * p.n + j] = 1e6;
        }
    }
    assert_eq!(qp_box(&hj, &p.g, &p.lo, &p.hi, &o).x, full.x);
    // a looser tol_dual keeps a slightly wrong-signed bound: x_0 stays at its lower bound
    assert_eq!(qp_box(&[1.0], &[-1e-6], &[0.0], &[1.0], &o).x[0], 1e-6);
    let r = qp_box(&[1.0], &[-1e-6], &[0.0], &[1.0], &QpBoxOptions { tol_dual: 1e-5, ..o.clone() });
    assert!(r.success && r.x[0] == 0.0 && r.active[0] == -1 && r.multipliers[0] == 0.0);
    println!("VERIFY qp_box rust_failure_modes semidefinite,indefinite,invalid_input(9),invalid_option(5),max_iter,lower_triangle_only,tol_dual=ok");
}
