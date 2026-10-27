//! Tests for properties::porous_permeability: formulas vs independently re-implemented reference
//! data (no published table values; see the generator), NaN policy, asymptotes, Ergun limits.
use agent_tools::properties::porous_permeability::*;

const RTOL_FORMULA: f64 = 1e-12;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/porous_permeability.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn f(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn rel(a: f64, b: f64) -> f64 {
    if b != 0.0 { (a - b).abs() / b.abs() } else { a.abs() }
}

#[test]
fn formulas_vs_reference() {
    let c = &load()["cases"];
    let mut w = [0.0f64; 5];
    for r in c["kozeny_carman"].as_array().unwrap() {
        w[0] = w[0].max(rel(kozeny_carman(f(&r["eps"]), f(&r["a_s"]), f(&r["K"])), f(&r["k"])));
    }
    for r in c["ergun"].as_array().unwrap() {
        let g = ergun_gradient(f(&r["u"]), f(&r["eps"]), f(&r["d"]), f(&r["mu"]), f(&r["rho"]), ERGUN_A, ERGUN_B);
        w[1] = w[1].max(rel(g, f(&r["grad"])));
    }
    let fns: [(&str, fn(f64, f64) -> f64); 3] =
        [("gebart_square", gebart_square), ("gebart_hexagonal", gebart_hexagonal), ("sangani_acrivos_square", sangani_acrivos_square)];
    for (i, (name, func)) in fns.iter().enumerate() {
        for r in c[*name].as_array().unwrap() {
            w[2 + i] = w[2 + i].max(rel(func(f(&r["eps"]), f(&r["d"])), f(&r["k"])));
        }
    }
    for (n, v) in ["kozeny_carman", "ergun", "gebart_square", "gebart_hexagonal", "sangani_acrivos_square"].iter().zip(w) {
        println!("VERIFY porous_permeability {n} rs_max_rel_err_vs_reference={v:e}");
        assert!(v < RTOL_FORMULA, "{n}");
    }
}

#[test]
fn nan_outside_domain() {
    let c = &load()["cases"]["nan_inputs"];
    for r in c["kozeny_carman"].as_array().unwrap() {
        assert!(kozeny_carman(f(&r[0]), f(&r[1]), f(&r[2])).is_nan());
    }
    for r in c["fit_kozeny_constant"].as_array().unwrap() {
        assert!(fit_kozeny_constant(f(&r[0]), f(&r[1]), f(&r[2])).is_nan());
    }
    for r in c["ergun_gradient"].as_array().unwrap() {
        assert!(ergun_gradient(f(&r[0]), f(&r[1]), f(&r[2]), f(&r[3]), f(&r[4]), ERGUN_A, ERGUN_B).is_nan());
    }
    for (name, func) in [("gebart_square", gebart_square as fn(f64, f64) -> f64), ("gebart_hexagonal", gebart_hexagonal), ("sangani_acrivos_square", sangani_acrivos_square)] {
        for r in c[name].as_array().unwrap() {
            assert!(func(f(&r[0]), f(&r[1])).is_nan(), "{name}");
        }
    }
    assert_eq!(gebart_square(1.0 - PHI_MAX_SQUARE, 1e-3), 0.0);
    println!("VERIFY porous_permeability nan_policy rs_cases_checked=ok");
}

#[test]
fn limits() {
    // Gebart near close packing
    let mut errs = [0.0f64; 2];
    for (i, delta) in [1e-5, 1e-3].into_iter().enumerate() {
        let phi = PHI_MAX_SQUARE - delta;
        let lim = 16.0 / (9.0 * std::f64::consts::PI * 2f64.sqrt()) * (delta / (2.0 * PHI_MAX_SQUARE)).powf(2.5);
        let e = (gebart_square(1.0 - phi, 2.0) / lim - 1.0).abs();
        println!("VERIFY porous_permeability gebart_asymptote rs_square@{delta:e}={e:e}");
        errs[i] = e;
    }
    assert!(errs[0] < 1e-4 && errs[0] < errs[1] / 50.0);
    // dilute SA -> Hasimoto
    let phi = 1e-3;
    let h = 0.25 / (8.0 * phi) * (-(phi as f64).ln() - 1.476 + 2.0 * phi);
    let s = (sangani_acrivos_square(1.0 - phi, 1.0) - h) / (0.25 / (8.0 * phi) * phi * phi);
    println!("VERIFY porous_permeability dilute_sa_vs_hasimoto rs_coeff_phi2={s:.5}");
    assert!((s - (-1.774 + 4.076 * phi)).abs() < 1e-6); // next series term is +4.076 phi
    // Ergun laminar limit and Kozeny-Carman/Blake-Kozeny link
    let (eps, d, mu, rho): (f64, f64, f64, f64) = (0.4, 2e-3, 1e-3, 1000.0);
    let k = eps.powi(3) * d * d / (150.0 * (1.0 - eps).powi(2));
    let u = 1e-4 * mu * (1.0 - eps) / (rho * d);
    let lam = ergun_gradient(u, eps, d, mu, rho, ERGUN_A, ERGUN_B) / (mu * u / k) - 1.0;
    let kc = kozeny_carman(eps, 6.0 / d, 150.0 / 36.0);
    println!("VERIFY porous_permeability ergun_laminar rs_rel={lam:e} kc_blake_kozeny_rel={:e}", (kc / k - 1.0).abs());
    assert!(lam.abs() < 2e-4 && (kc / k - 1.0).abs() < 1e-14);
    // fit round trip
    let kk = kozeny_carman(0.37, 8000.0, 4.7);
    assert!((fit_kozeny_constant(kk, 0.37, 8000.0) / 4.7 - 1.0).abs() < 1e-14);
}
