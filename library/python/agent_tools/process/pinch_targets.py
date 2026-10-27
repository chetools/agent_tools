"""Pinch targets: problem-table heat cascade, composite curves and grand composite curve.

Sources: Linnhoff, Townsend, Boland, Hewitt, Thomas, Guy, Marsland, "A User Guide on Process
Integration for the Efficient Use of Energy", IChemE, 1982 (problem table algorithm); Kemp,
"Pinch Analysis and Process Integration", 2nd ed., Butterworth-Heinemann, 2007 (composite and
grand composite curves). Ported from chetools/claude heat-exchanger-network (hen.py
`problem_table`, solver/src/lib.rs `problem_table`, `composite`); the LP/MILP target models of the
source are not part of this routine.

Streams are (t_supply, t_target, cp) tuples: temperatures in one consistent unit (K or deg C; only
differences and shifts are used, so both work) and cp the heat-capacity flowrate m*cp in kW/K
(constant over the temperature range; a stream with a phase change must be split into segments,
and an isothermal stream carries no duty here). Hot streams cool (t_supply > t_target), cold
streams heat (t_supply < t_target).

Temperature scale. Following the source, the hot streams keep their temperatures and the cold
streams are shifted UP by the full dt_min (the "hot scale"). This is the textbook shifted scale
(hot - dt_min/2, cold + dt_min/2) displaced by +dt_min/2; the heat-flow values are identical and
`t_shifted` gives the textbook scale. `pinch_hot` is the pinch temperature of the hot streams; the
cold-stream pinch temperature is pinch_hot - dt_min.

Algorithm (problem table): sort the interval boundaries descending; the net heat released in each
interval is sum(cp_hot * overlap) - sum(cp_cold * overlap); cascade the heat from the top starting
at 0, add the largest deficit as hot utility so no cascaded flow is negative. qh_min is that
deficit, qc_min the final flow. The pinch is where the cascaded flow is zero (|flow| <
1e-6 (1 + qh_min)); only the lowest such boundary is reported (see PinchTargets.heat_flow for
others). If it is the first or last boundary the problem is a threshold problem (no true pinch).
"""
import math
from dataclasses import dataclass, field

#: Zero test of the cascaded heat flow, relative to (1 + qh_min) (dimensionless).
PINCH_ZERO_TOL = 1e-6


@dataclass
class PinchTargets:
    """Result of pinch_targets (kW, temperatures in the input unit)."""
    qh_min: float                 # minimum hot utility [kW]
    qc_min: float                 # minimum cold utility [kW]
    pinch_hot: object             # pinch temperature on the hot streams, or None if no zero flow
    t_hot_scale: list = field(default_factory=list)   # interval boundaries, descending [T]
    t_shifted: list = field(default_factory=list)     # same, textbook shifted scale hot - dt_min/2 [T]
    heat_flow: list = field(default_factory=list)     # cascaded heat flow at each boundary [kW]: the grand composite curve is (heat_flow, t_shifted)
    hot_composite: tuple = ((), ())    # (T ascending, H [kW] from 0 at the cold end)
    cold_composite: tuple = ((), ())   # (T ascending, H [kW]) shifted by qc_min so the curves are balanced


def _check(streams, hot, name):
    for t_s, t_t, cp in streams:
        if not all(math.isfinite(v) for v in (t_s, t_t, cp)) or cp < 0.0:
            raise ValueError(f"{name} stream ({t_s}, {t_t}, {cp}): need finite temperatures and cp >= 0")
        if (hot and t_s < t_t) or (not hot and t_s > t_t):
            raise ValueError(f"{name} stream ({t_s}, {t_t}, {cp}): wrong direction for a {name} stream")


def composite_curve(streams):
    """Composite curve of streams [(t_supply, t_target, cp)]: (T ascending, cumulative H [kW]),
    H = 0 at the lowest temperature. Duplicate temperatures are merged exactly (float equality)."""
    t = sorted({v for s in streams for v in (s[0], s[1])})
    h = [0.0]
    for lo, hi in zip(t, t[1:]):
        cp = sum(s[2] for s in streams if min(s[0], s[1]) <= lo and max(s[0], s[1]) >= hi)
        h.append(h[-1] + cp * (hi - lo))
    return t, h


def pinch_targets(hot, cold, dt_min):
    """Minimum utilities, pinch, composite curves and grand composite curve.

    hot, cold: lists of (t_supply, t_target, cp) (see module docstring); dt_min: minimum approach
    temperature [same unit as T, >= 0]. Raises ValueError for a stream going the wrong way, negative
    cp or dt_min, or no streams at all. Returns a PinchTargets.
    """
    if dt_min < 0.0 or not math.isfinite(dt_min):
        raise ValueError("dt_min must be finite and >= 0")
    _check(hot, True, "hot")
    _check(cold, False, "cold")
    if not hot and not cold:
        raise ValueError("no streams")
    edges = {v for s in hot for v in s[:2]} | {v + dt_min for s in cold for v in s[:2]}
    b = sorted(edges, reverse=True)
    cascade, flow = [0.0], 0.0
    for hi, lo in zip(b, b[1:]):
        net = 0.0
        for t_s, t_t, cp in hot:
            net += cp * max(0.0, min(hi, t_s) - max(lo, t_t))
        for t_s, t_t, cp in cold:
            net -= cp * max(0.0, min(hi, t_t + dt_min) - max(lo, t_s + dt_min))
        flow += net
        cascade.append(flow)
    qh = max(0.0, -min(cascade))
    flows = [c + qh for c in cascade]
    pinch = None
    for t, f in zip(b, flows):
        if abs(f) < PINCH_ZERO_TOL * (1.0 + qh):
            pinch = t
    th, hh = composite_curve(hot) if hot else ([], [])
    tc, hc = composite_curve(cold) if cold else ([], [])
    qc = flows[-1]
    return PinchTargets(qh_min=qh, qc_min=qc, pinch_hot=pinch, t_hot_scale=b,
                        t_shifted=[t - 0.5 * dt_min for t in b], heat_flow=flows,
                        hot_composite=(th, hh), cold_composite=(tc, [h + qc for h in hc]))
