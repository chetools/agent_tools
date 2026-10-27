//! Tests for process::pinch_targets: Kemp four-stream example (published targets), 25 Furman-Sahinidis
//! instances against an independent HiGHS heat-cascade LP and the utility costs published by Letsios et al.
use agent_tools::process::pinch_targets::{composite_curve, pinch_targets, Stream, PinchTargets};

/// Targets vs the published Kemp numbers (exact decimals), absolute [kW].
const TOL_PUBLISHED: f64 = 1e-9;
/// Targets vs HiGHS LP: absolute, scaled by (1 + q) (HiGHS feasibility tolerance ~1e-7).
const TOL_LP: f64 = 1e-6;
/// Utility cost vs published value, relative.
const TOL_COST: f64 = 1e-9;
/// Balance identities of the curves, relative to the largest heat flow.
const TOL_BALANCE: f64 = 1e-12;

fn load() -> serde_json::Value {
    let p = format!("{}/../tests/reference_data/pinch_targets.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn streams(v: &serde_json::Value) -> Vec<Stream> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let f = |i: usize| s[i].as_f64().unwrap();
            Stream { t_supply: f(0), t_target: f(1), cp: f(2) }
        })
        .collect()
}
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let k = xs.partition_point(|&v| v <= x) - 1;
    ys[k] + (ys[k + 1] - ys[k]) * (x - xs[k]) / (xs[k + 1] - xs[k])
}

/// Checks the cascade/curve identities; returns the smallest horizontal gap H_cold(T - dt) - H_hot(T).
fn balance(r: &PinchTargets, dt: f64) -> f64 {
    let scale = r.heat_flow.iter().cloned().fold(0.0, f64::max).max(1.0);
    assert!(r.heat_flow.iter().all(|&f| f >= -TOL_BALANCE * scale), "negative cascaded flow");
    assert!((r.heat_flow[0] - r.qh_min).abs() <= TOL_BALANCE * scale);
    assert!((r.heat_flow[r.heat_flow.len() - 1] - r.qc_min).abs() <= TOL_BALANCE * scale);
    let (hh, hc) = (&r.hot_composite, &r.cold_composite);
    let hot_end = hh.h.last().copied().unwrap_or(0.0);
    let cold_end = hc.h.last().copied().unwrap_or(r.qc_min);
    let big = hot_end.max(cold_end).max(1.0);
    assert!(((cold_end - hot_end) - r.qh_min).abs() <= TOL_BALANCE * big, "cold end - hot end != QH");
    let mut gap = f64::INFINITY;
    let mut pts: Vec<f64> = hh.t.clone();
    pts.extend(hc.t.iter().map(|t| t + dt));
    for t in pts {
        let hhot = if hh.t.is_empty() { 0.0 } else { interp(t, &hh.t, &hh.h) };
        let hcold = if hc.t.is_empty() { r.qc_min } else { interp(t - dt, &hc.t, &hc.h) };
        gap = gap.min(hcold - hhot);
    }
    assert!(gap >= -TOL_BALANCE * big, "composite curves cross: gap {gap:e}");
    gap
}

#[test]
fn kemp_four_stream_published() {
    let r = load();
    let k = &r["cases"]["kemp_four_stream"];
    let (hot, cold, dt) = (streams(&k["hot"]), streams(&k["cold"]), k["dt_min"].as_f64().unwrap());
    let p = &k["published"];
    let res = pinch_targets(&hot, &cold, dt).unwrap();
    let g = |n: &str| p[n].as_f64().unwrap();
    let e_h = (res.qh_min - g("qh_min")).abs();
    let e_c = (res.qc_min - g("qc_min")).abs();
    let pinch = res.pinch_hot.unwrap();
    let e_p = (pinch - g("pinch_hot")).abs().max(((pinch - 0.5 * dt) - g("pinch_shifted")).abs());
    let demand_h: f64 = cold.iter().map(|s| s.cp * (s.t_target - s.t_supply)).sum();
    let demand_c: f64 = hot.iter().map(|s| s.cp * (s.t_supply - s.t_target)).sum();
    let e_d = (demand_h - g("heating_demand")).abs().max((demand_c - g("cooling_demand")).abs());
    let e_r = ((demand_h - res.qh_min) - g("heat_recovery")).abs();
    let gap = balance(&res, dt);
    // at the pinch the composite curves are exactly dt apart
    println!("VERIFY pinch_targets kemp err_QH={e_h:e} err_QC={e_c:e} err_pinch={e_p:e} err_demands={e_d:e} err_recovery={e_r:e} min_curve_gap={gap:e}");
    for e in [e_h, e_c, e_p, e_d, e_r, gap.abs()] {
        assert!(e < TOL_PUBLISHED);
    }
    // grand composite: zero flow at the pinch only (shifted 85), top = QH, bottom = QC
    let zeros: Vec<f64> = res.t_shifted.iter().zip(&res.heat_flow).filter(|(_, &f)| f.abs() < 1e-9).map(|(t, _)| *t).collect();
    assert_eq!(zeros, vec![85.0]);
}

#[test]
fn furman_instances_vs_lp_and_published_cost() {
    let r = load();
    let (mut worst_lp, mut worst_cost, mut worst_gap_pinch) = (0f64, 0f64, 0f64);
    let mut n_pinch = 0;
    let cases = r["cases"]["furman"].as_array().unwrap();
    for c in cases {
        let (hot, cold, dt) = (streams(&c["hot"]), streams(&c["cold"]), c["dt_min"].as_f64().unwrap());
        let res = pinch_targets(&hot, &cold, dt).unwrap();
        let (lh, lc) = (c["lp_qh"].as_f64().unwrap(), c["lp_qc"].as_f64().unwrap());
        let e_lp = ((res.qh_min - lh).abs() / (1.0 + lh)).max((res.qc_min - lc).abs() / (1.0 + lc));
        let ch = c.get("hu_cost").map(|v| v.as_f64().unwrap()).unwrap_or(0.0);
        let cost = res.qh_min * ch + res.qc_min * c["cu_cost"].as_f64().unwrap();
        let publ = c["published_cost"].as_f64().unwrap();
        let e_cost = (cost - publ).abs() / publ.abs().max(1e-300);
        let gap = balance(&res, dt);
        if let Some(p) = res.pinch_hot {
            if res.heat_flow[0] > 0.0 && *res.heat_flow.last().unwrap() > 0.0 && p < res.t_hot_scale[0] {
                // an interior pinch touches the curves
                let big = res.hot_composite.h.last().copied().unwrap().max(1.0);
                worst_gap_pinch = worst_gap_pinch.max(gap.abs() / big);
                n_pinch += 1;
            }
        }
        assert!(e_lp < TOL_LP, "{}: LP {e_lp:e}", c["name"]);
        assert!(e_cost < TOL_COST || publ == 0.0 && cost == 0.0, "{}: cost {e_cost:e}", c["name"]);
        worst_lp = worst_lp.max(e_lp);
        worst_cost = worst_cost.max(e_cost);
    }
    println!("VERIFY pinch_targets furman instances={} max_rel_err_vs_linprog={worst_lp:e} max_rel_err_cost_vs_published={worst_cost:e} interior_pinch_cases={n_pinch} max_rel_curve_gap_at_pinch={worst_gap_pinch:e}", cases.len());
    assert!(worst_gap_pinch < TOL_BALANCE);
}

#[test]
fn composite_curve_and_error_handling() {
    let s = |a, b, c| Stream { t_supply: a, t_target: b, cp: c };
    // two overlapping cold streams: cp 1 over 0-10 and cp 2 over 5-20
    let c = composite_curve(&[s(0.0, 10.0, 1.0), s(5.0, 20.0, 2.0)]);
    assert_eq!(c.t, vec![0.0, 5.0, 10.0, 20.0]);
    assert_eq!(c.h, vec![0.0, 5.0, 20.0, 40.0]);
    assert!(pinch_targets(&[s(10.0, 50.0, 1.0)], &[], 5.0).is_err());
    assert!(pinch_targets(&[], &[s(50.0, 10.0, 1.0)], 5.0).is_err());
    assert!(pinch_targets(&[s(50.0, 10.0, -1.0)], &[], 5.0).is_err());
    assert!(pinch_targets(&[s(50.0, 10.0, 1.0)], &[], -1.0).is_err());
    assert!(pinch_targets(&[], &[], 5.0).is_err());
    // a lone hot stream needs no hot utility; all its heat goes to cold utility
    let r = pinch_targets(&[s(100.0, 40.0, 2.0)], &[], 10.0).unwrap();
    assert_eq!((r.qh_min, r.qc_min), (0.0, 120.0));
    println!("VERIFY pinch_targets edge_cases checks=7");
}
