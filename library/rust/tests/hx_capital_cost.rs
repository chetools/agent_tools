//! Tests for process::hx_capital_cost: published tables, complex-step reference values, finite differences.
use agent_tools::process::hx_capital_cost::*;

/// Cost vs the independent re-implementation (same coefficients), relative.
const TOL_COST: f64 = 1e-12;
/// dC/dA vs the complex-step derivative, relative to max(|dC/dA|, 1e-3 USD/m^2).
const TOL_DERIV: f64 = 1e-11;
/// Turton eqcost published test case: relative (the source project's bound; eqcost rounds constants).
const TOL_EQCOST: f64 = 3e-3;
/// Central finite difference of the cost, relative, step 1e-4 A.
const TOL_FD: f64 = 1e-6;
/// Floating head, BioSTEAM (CE 567) vs IDAES SSLW (CE 500) after the index shift (measured 4.9e-5), relative.
const TOL_INDEX_SHIFT: f64 = 6e-5;
/// Capital recovery factor vs 50-digit values, relative.
const TOL_CRF: f64 = 1e-13;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/hx_capital_cost.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn mat(s: &str) -> Material {
    match s {
        "CS" => Material::Cs,
        "brass" => Material::Brass,
        "CrMo" => Material::CrMo,
        "SS" => Material::Ss,
        "Monel" => Material::Monel,
        "Ti" => Material::Ti,
        _ => panic!("{s}"),
    }
}
fn typ(s: &str) -> HxType {
    match s {
        "floating_head" => HxType::FloatingHead,
        "fixed_head" => HxType::FixedHead,
        "u_tube" => HxType::UTube,
        _ => panic!("{s}"),
    }
}
fn rel(a: f64, b: f64, floor: f64) -> f64 {
    (a - b).abs() / b.abs().max(floor)
}
fn f(v: &serde_json::Value, k: &str) -> f64 {
    v[k].as_f64().unwrap()
}
fn seider(c: &serde_json::Value, area: f64) -> HxCost {
    let (a, b) = seider_material_coeffs(mat(c["shell"].as_str().unwrap()), mat(c["tube"].as_str().unwrap())).unwrap();
    seider_bare_module_cost(
        area,
        typ(c["type"].as_str().unwrap()),
        a,
        b,
        seider_length_factor(c["tube_ft"].as_u64().unwrap() as u32).unwrap(),
        seider_pressure_factor(f(c, "p_psig")),
        SEIDER_F_BM,
    )
}
fn turton(c: &serde_json::Value, area: f64) -> HxCost {
    turton_bare_module_cost(
        area,
        turton_material_factor(mat(c["material"].as_str().unwrap())).unwrap(),
        turton_pressure_factor(f(c, "p_barg"), TURTON_C),
        TURTON_K,
        TURTON_B,
        TURTON_AMIN_M2,
        TURTON_AMAX_M2,
    )
}

#[test]
fn published_tables() {
    let r = load();
    let p = &r["cases"]["published"];
    for row in p["seider_fm"].as_array().unwrap() {
        let (a, b) = seider_material_coeffs(mat(row[0].as_str().unwrap()), mat(row[1].as_str().unwrap())).unwrap();
        assert_eq!((a, b), (row[2].as_f64().unwrap(), row[3].as_f64().unwrap()));
    }
    assert!(seider_material_coeffs(Material::Ti, Material::Cs).is_none());
    let c: Vec<f64> = p["seider_fp_coeffs"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mut worst = 0f64;
    for psig in [0.0, 100.0, 600.0, 2000.0] {
        let x = psig / 100.0;
        worst = worst.max((seider_pressure_factor(psig) - (c[0] + c[1] * x + c[2] * x * x)).abs());
    }
    for (k, v) in p["seider_fl"].as_object().unwrap() {
        assert_eq!(seider_length_factor(k.parse().unwrap()), Some(v.as_f64().unwrap()));
    }
    assert!(seider_length_factor(13).is_none());
    assert!(turton_material_factor(Material::Brass).is_none());
    println!("VERIFY hx_capital_cost published_tables max_abs_err_FP={worst:e} fm_rows={}", p["seider_fm"].as_array().unwrap().len());
    assert!(worst < 1e-15);
}

#[test]
fn turton_eqcost_case() {
    let r = load();
    let e = &r["cases"]["published"]["turton_eqcost"];
    let fp = turton_pressure_factor(f(e, "p_barg"), TURTON_C);
    let res = turton_bare_module_cost(f(e, "area_m2"), 1.0, fp, TURTON_K, TURTON_B, TURTON_AMIN_M2, TURTON_AMAX_M2);
    let cp0 = 10f64.powf(TURTON_K[0] + TURTON_K[1] * 2.0 + TURTON_K[2] * 4.0);
    let (e_fp, e_cp0, e_cbm) = (rel(fp, f(e, "fp"), 1.0), rel(cp0, f(e, "cp0"), 1.0), rel(res.cost_usd, f(e, "cbm"), 1.0));
    println!("VERIFY hx_capital_cost turton_eqcost rel_err_FP={e_fp:e} rel_err_Cp0={e_cp0:e} rel_err_CBM={e_cbm:e}");
    assert!(e_fp < TOL_EQCOST && e_cp0 < TOL_EQCOST && e_cbm < TOL_EQCOST);
}

#[test]
fn seider_values_and_ranges() {
    let r = load();
    let (mut ec, mut ed) = (0f64, 0f64);
    let cases = r["cases"]["seider_values"].as_array().unwrap();
    for c in cases {
        let area = f(c, "area_m2");
        let res = seider(c, area);
        let (e_c, e_d) = (rel(res.cost_usd, f(c, "cost"), 1.0), rel(res.dcost_da, f(c, "dcost_da"), 1e-3));
        assert!(e_c < TOL_COST && e_d < TOL_DERIV, "{c}: {e_c:e} {e_d:e}");
        assert_eq!(res.clamped_below, c.get("clamped").is_some(), "{c}");
        assert_eq!(res.above_range, area * M2_TO_FT2 > SEIDER_AMAX_FT2, "{c}");
        ec = ec.max(e_c);
        ed = ed.max(e_d);
    }
    println!("VERIFY hx_capital_cost seider_values n={} max_rel_err_cost={ec:e} max_rel_err_dcost={ed:e}", cases.len());
}

#[test]
fn turton_values_and_ranges() {
    let r = load();
    let (mut ec, mut ed) = (0f64, 0f64);
    let cases = r["cases"]["turton_values"].as_array().unwrap();
    for c in cases {
        let area = f(c, "area_m2");
        let res = turton(c, area);
        let (e_c, e_d) = (rel(res.cost_usd, f(c, "cost"), 1.0), rel(res.dcost_da, f(c, "dcost_da"), 1e-3));
        assert!(e_c < TOL_COST && e_d < TOL_DERIV, "{c}: {e_c:e} {e_d:e}");
        assert_eq!(res.clamped_below, c.get("clamped").is_some(), "{c}");
        assert_eq!(res.above_range, area > TURTON_AMAX_M2, "{c}");
        ec = ec.max(e_c);
        ed = ed.max(e_d);
    }
    println!("VERIFY hx_capital_cost turton_values n={} max_rel_err_cost={ec:e} max_rel_err_dcost={ed:e}", cases.len());
}

#[test]
fn derivative_vs_finite_differences_both_sides_of_clamp() {
    let mut worst = 0f64;
    let lo_s = SEIDER_AMIN_FT2 / M2_TO_FT2;
    for typ in [HxType::FloatingHead, HxType::FixedHead, HxType::UTube] {
        let (a, b) = seider_material_coeffs(Material::Cs, Material::Ti).unwrap();
        let cost = |area: f64| seider_bare_module_cost(area, typ, a, b, 1.12, 1.2, SEIDER_F_BM);
        // 0.97 and 1.03 straddle the clamp; 1.0 is skipped (slope discontinuity); 30 and 1115 are further out
        for area in [0.5 * lo_s, 0.97 * lo_s, 1.03 * lo_s, 20.0, 150.0, 1115.0, 2000.0] {
            let h = 1e-4 * area;
            let fd = (cost(area + h).cost_usd - cost(area - h).cost_usd) / (2.0 * h);
            let d = cost(area).dcost_da;
            let e = (d - fd).abs() / fd.abs().max(1.0);
            worst = worst.max(e);
            assert!(e < TOL_FD, "seider {typ:?} {area}: {d} vs {fd}");
        }
        // continuity at the clamp
        let (below, above) = (cost(lo_s * (1.0 - 1e-12)).cost_usd, cost(lo_s).cost_usd);
        assert!(rel(below, above, 1.0) < 1e-9);
    }
    let fp = turton_pressure_factor(20.0, TURTON_C);
    let cost = |area: f64| turton_bare_module_cost(area, 2.73, fp, TURTON_K, TURTON_B, TURTON_AMIN_M2, TURTON_AMAX_M2);
    for area in [5.0, 9.7, 10.3, 100.0, 1000.0, 3000.0] {
        let h = 1e-4 * area;
        let fd = (cost(area + h).cost_usd - cost(area - h).cost_usd) / (2.0 * h);
        let e = (cost(area).dcost_da - fd).abs() / fd.abs().max(1.0);
        worst = worst.max(e);
        assert!(e < TOL_FD, "turton {area}");
    }
    assert!(rel(cost(10.0 * (1.0 - 1e-12)).cost_usd, cost(10.0).cost_usd, 1.0) < 1e-9);
    println!("VERIFY hx_capital_cost fd_both_sides max_rel_diff={worst:e}");
}

#[test]
fn capital_recovery_factor_vs_mpmath() {
    let r = load();
    let mut worst = 0f64;
    for c in r["cases"]["crf"].as_array().unwrap() {
        let e = rel(capital_recovery_factor(f(c, "rate"), f(c, "years")), f(c, "crf"), 1e-300);
        worst = worst.max(e);
        assert!(e < TOL_CRF, "{c}: {e:e}");
    }
    for (i, n) in [(-1.0, 10.0), (0.1, 0.0), (0.1, -3.0), (f64::NAN, 5.0)] {
        assert!(capital_recovery_factor(i, n).is_nan());
    }
    assert!(seider_bare_module_cost(0.0, HxType::UTube, 0.0, 0.0, 1.0, 1.0, 3.17).cost_usd.is_nan());
    assert!(turton_bare_module_cost(-1.0, 1.0, 1.0, TURTON_K, TURTON_B, 10.0, 1000.0).dcost_da.is_nan());
    println!("VERIFY hx_capital_cost crf max_rel_err={worst:e}");
}

#[test]
fn seider_pressure_below_range_flag() {
    assert_eq!(SEIDER_PMIN_PSIG, 100.0);
    let flags: Vec<bool> = [0.0, 99.999, 100.0, 300.0, 2000.0].iter().map(|&p| seider_pressure_below_range(p)).collect();
    assert_eq!(flags, [true, true, false, false, false]);
    assert_eq!(seider_pressure_factor(50.0), 0.9803 + 0.018 * 0.5 + 0.0017 * 0.25); // values unchanged below range
}

/// BioSTEAM 12.0310 (CE 567) vs IDAES SSLW 11.9052 (CE 500): the difference is ln(567/500), so the costs
/// agree after the index shift. (Fixed-head slope and U-tube intercept differences are open items.)
#[test]
fn floating_head_intercepts_differ_by_cost_index_shift() {
    let r = load();
    let idaes = &r["cases"]["published"]["idaes_sslw_cb"];
    let i: Vec<f64> = idaes["floating_head"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let ce = f(idaes, "ce_basis");
    let [c0, c1, c2] = HxType::FloatingHead.seider_cb();
    assert_eq!((c1, c2), (i[1], i[2]));
    let diff = c0 - i[0] - (SEIDER_CE_BASIS / ce).ln();
    let mut worst = 0f64;
    for af in [150.0f64, 1000.0, 12000.0] {
        let lg = af.ln();
        let bio = (c0 + c1 * lg + c2 * lg * lg).exp();
        let ide = (i[0] + i[1] * lg + i[2] * lg * lg).exp() * SEIDER_CE_BASIS / ce;
        worst = worst.max((bio / ide - 1.0).abs());
    }
    println!("VERIFY hx_capital_cost floating_head_index_shift intercept_diff_minus_ln_ratio={diff:e} max_rel_cost_diff={worst:e}");
    assert!(diff.abs() < 1e-4 && worst < TOL_INDEX_SHIFT);
}
