"""Tests for process.pinch_targets: Kemp four-stream example (published targets), 25 Furman-Sahinidis
instances against an independent HiGHS heat-cascade LP and the published utility costs, and Rust."""
import numpy as np
import pytest
from conftest import load_reference, python_example, rust_example

from agent_tools.process.pinch_targets import composite_curve, pinch_targets

TOL_PUBLISHED = 1e-9   # targets vs the published Kemp numbers, absolute [kW]
TOL_LP = 1e-6          # targets vs HiGHS LP, absolute / (1 + q)
TOL_COST = 1e-9        # utility cost vs published, relative
TOL_BALANCE = 1e-12    # curve/cascade identities, relative to the largest heat flow
TOL_RUST = 1e-12       # Python vs Rust example, relative to the largest value


def balance(r, dt):
    """Assert the cascade/curve identities; return min over T of H_cold(T - dt) - H_hot(T)."""
    flows = np.array(r.heat_flow)
    scale = max(flows.max(), 1.0)
    assert (flows >= -TOL_BALANCE * scale).all()
    assert abs(flows[0] - r.qh_min) <= TOL_BALANCE * scale and abs(flows[-1] - r.qc_min) <= TOL_BALANCE * scale
    (th, hh), (tc, hc) = r.hot_composite, r.cold_composite
    hot_end = hh[-1] if len(hh) else 0.0
    cold_end = hc[-1] if len(hc) else r.qc_min
    big = max(hot_end, cold_end, 1.0)
    assert abs((cold_end - hot_end) - r.qh_min) <= TOL_BALANCE * big      # curves balanced by QH, QC
    pts = list(th) + [t + dt for t in tc]
    gaps = [(np.interp(t - dt, tc, hc) if len(tc) else r.qc_min) - (np.interp(t, th, hh) if len(th) else 0.0)
            for t in pts]
    gap = min(gaps)
    assert gap >= -TOL_BALANCE * big, gap
    return gap


def S(arr):
    return [tuple(s) for s in arr]


def test_kemp_four_stream_published():
    k = load_reference("pinch_targets")["cases"]["kemp_four_stream"]
    hot, cold, dt, p = S(k["hot"]), S(k["cold"]), k["dt_min"], k["published"]
    r = pinch_targets(hot, cold, dt)
    dem_h = sum(c[2] * (c[1] - c[0]) for c in cold)
    dem_c = sum(h[2] * (h[0] - h[1]) for h in hot)
    e_h, e_c = abs(r.qh_min - p["qh_min"]), abs(r.qc_min - p["qc_min"])
    e_p = max(abs(r.pinch_hot - p["pinch_hot"]), abs(r.pinch_hot - dt / 2 - p["pinch_shifted"]))
    e_d = max(abs(dem_h - p["heating_demand"]), abs(dem_c - p["cooling_demand"]))
    e_r = abs((dem_h - r.qh_min) - p["heat_recovery"])
    gap = balance(r, dt)
    print(f"VERIFY pinch_targets py_kemp err_QH={e_h:e} err_QC={e_c:e} err_pinch={e_p:e} "
          f"err_demands={e_d:e} err_recovery={e_r:e} min_curve_gap={gap:e}")
    assert max(e_h, e_c, e_p, e_d, e_r, abs(gap)) < TOL_PUBLISHED
    zeros = [t for t, f in zip(r.t_shifted, r.heat_flow) if abs(f) < 1e-9]
    assert zeros == [85.0]


def test_furman_instances_vs_lp_and_published_cost():
    cases = load_reference("pinch_targets")["cases"]["furman"]
    worst_lp = worst_cost = worst_gap = 0.0
    n_pinch = 0
    for c in cases:
        r = pinch_targets(S(c["hot"]), S(c["cold"]), c["dt_min"])
        e_lp = max(abs(r.qh_min - c["lp_qh"]) / (1 + c["lp_qh"]), abs(r.qc_min - c["lp_qc"]) / (1 + c["lp_qc"]))
        cost = r.qh_min * c.get("hu_cost", 0.0) + r.qc_min * c["cu_cost"]
        pub = c["published_cost"]
        e_cost = abs(cost - pub) / abs(pub) if pub else abs(cost)
        gap = balance(r, c["dt_min"])
        if r.pinch_hot is not None and r.heat_flow[0] > 0 and r.heat_flow[-1] > 0 and r.pinch_hot < r.t_hot_scale[0]:
            worst_gap = max(worst_gap, abs(gap) / max(r.hot_composite[1][-1], 1.0))   # interior pinch touches
            n_pinch += 1
        assert e_lp < TOL_LP and e_cost < TOL_COST, (c["name"], e_lp, e_cost)
        worst_lp, worst_cost = max(worst_lp, e_lp), max(worst_cost, e_cost)
    print(f"VERIFY pinch_targets py_furman instances={len(cases)} max_rel_err_vs_linprog={worst_lp:e} "
          f"max_rel_err_cost_vs_published={worst_cost:e} interior_pinch_cases={n_pinch} "
          f"max_rel_curve_gap_at_pinch={worst_gap:e}")
    assert worst_gap < TOL_BALANCE


def test_composite_curve_and_error_handling():
    t, h = composite_curve([(0.0, 10.0, 1.0), (5.0, 20.0, 2.0)])
    assert t == [0.0, 5.0, 10.0, 20.0] and h == [0.0, 5.0, 20.0, 40.0]
    for args in [([(10.0, 50.0, 1.0)], [], 5.0), ([], [(50.0, 10.0, 1.0)], 5.0),
                 ([(50.0, 10.0, -1.0)], [], 5.0), ([(50.0, 10.0, 1.0)], [], -1.0), ([], [], 5.0)]:
        with pytest.raises(ValueError):
            pinch_targets(*args)
    r = pinch_targets([(100.0, 40.0, 2.0)], [], 10.0)
    assert (r.qh_min, r.qc_min) == (0.0, 120.0)
    print("VERIFY pinch_targets py_edge_cases checks=7")


def test_python_vs_rust_example():
    py, rs = python_example("pinch_targets"), rust_example("pinch_targets")
    assert py["bad_rejected"] and rs["bad_rejected"]
    worst = 0.0
    for key in ("qh_min", "qc_min", "pinch_hot"):
        worst = max(worst, abs(py[key] - rs[key]))
    for key in ("t_hot_scale", "heat_flow", "hot_composite_h", "cold_composite_h"):
        assert len(py[key]) == len(rs[key]), key
        worst = max(worst, float(np.max(np.abs(np.array(py[key]) - np.array(rs[key])))))
    print(f"VERIFY pinch_targets python_vs_rust max_abs_diff={worst:e}")
    assert worst < TOL_RUST * 1e3
