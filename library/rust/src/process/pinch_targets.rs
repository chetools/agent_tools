//! Pinch targets: problem-table heat cascade, composite curves and grand composite curve.
//!
//! Sources: Linnhoff, Townsend, Boland, Hewitt, Thomas, Guy, Marsland, "A User Guide on Process
//! Integration for the Efficient Use of Energy", IChemE, 1982 (problem table algorithm); Kemp,
//! "Pinch Analysis and Process Integration", 2nd ed., Butterworth-Heinemann, 2007 (composite and
//! grand composite curves). Ported from chetools/claude heat-exchanger-network (hen.py
//! `problem_table`, solver/src/lib.rs `problem_table`, `composite`); the LP/MILP target models of the
//! source are not part of this routine.
//!
//! Streams are `Stream { t_supply, t_target, cp }`: temperatures in one consistent unit (K or deg C;
//! only differences and shifts are used, so both work) and `cp` the heat-capacity flowrate m*cp in
//! kW/K (constant over the temperature range; a stream with a phase change must be split into
//! segments, and an isothermal stream carries no duty here). Hot streams cool
//! (`t_supply > t_target`), cold streams heat (`t_supply < t_target`).
//!
//! Temperature scale. Following the source, the hot streams keep their temperatures and the cold
//! streams are shifted UP by the full `dt_min` (the "hot scale"). This is the textbook shifted scale
//! (hot - dt_min/2, cold + dt_min/2) displaced by +dt_min/2; the heat-flow values are identical and
//! `t_shifted` gives the textbook scale. `pinch_hot` is the pinch temperature of the hot streams; the
//! cold-stream pinch temperature is `pinch_hot - dt_min`.
//!
//! Algorithm (problem table): sort the interval boundaries descending; the net heat released in each
//! interval is sum(cp_hot * overlap) - sum(cp_cold * overlap); cascade the heat from the top starting
//! at 0, add the largest deficit as hot utility so no cascaded flow is negative. `qh_min` is that
//! deficit, `qc_min` the final flow. The pinch is where the cascaded flow is zero
//! (|flow| < 1e-6 (1 + qh_min)); only the lowest such boundary is reported (see `heat_flow` for
//! others). If it is the first or last boundary the problem is a threshold problem (no true pinch).

/// Zero test of the cascaded heat flow, relative to (1 + qh_min) (dimensionless).
pub const PINCH_ZERO_TOL: f64 = 1e-6;

/// A stream with constant heat-capacity flowrate.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    /// Supply (inlet) temperature [T].
    pub t_supply: f64,
    /// Target (outlet) temperature [T].
    pub t_target: f64,
    /// Heat-capacity flowrate m*cp [kW/K], >= 0.
    pub cp: f64,
}

/// A piecewise-linear composite curve: temperature `t` ascending [T] and cumulative heat `h` [kW].
#[derive(Clone, Debug, Default)]
pub struct Curve {
    pub t: Vec<f64>,
    pub h: Vec<f64>,
}

/// Result of [`pinch_targets`] (kW; temperatures in the input unit).
#[derive(Clone, Debug)]
pub struct PinchTargets {
    /// Minimum hot utility [kW].
    pub qh_min: f64,
    /// Minimum cold utility [kW].
    pub qc_min: f64,
    /// Pinch temperature on the hot streams, `None` if no cascaded flow is zero.
    pub pinch_hot: Option<f64>,
    /// Interval boundaries, descending [T].
    pub t_hot_scale: Vec<f64>,
    /// Same boundaries on the textbook shifted scale, hot - dt_min/2 [T].
    pub t_shifted: Vec<f64>,
    /// Cascaded heat flow at each boundary [kW]; the grand composite curve is (heat_flow, t_shifted).
    pub heat_flow: Vec<f64>,
    /// Hot composite curve (H = 0 at the cold end).
    pub hot_composite: Curve,
    /// Cold composite curve shifted by `qc_min` so the two curves are balanced.
    pub cold_composite: Curve,
}

fn check(streams: &[Stream], hot: bool, name: &str) -> Result<(), String> {
    for s in streams {
        if !(s.t_supply.is_finite() && s.t_target.is_finite() && s.cp.is_finite()) || s.cp < 0.0 {
            return Err(format!("{name} stream {s:?}: need finite temperatures and cp >= 0"));
        }
        if (hot && s.t_supply < s.t_target) || (!hot && s.t_supply > s.t_target) {
            return Err(format!("{name} stream {s:?}: wrong direction for a {name} stream"));
        }
    }
    Ok(())
}

/// Composite curve of `streams`: T ascending and cumulative H [kW], H = 0 at the lowest
/// temperature. Duplicate temperatures are merged exactly (float equality).
pub fn composite_curve(streams: &[Stream]) -> Curve {
    let mut t: Vec<f64> = streams.iter().flat_map(|s| [s.t_supply, s.t_target]).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.dedup();
    let mut h = vec![0.0];
    for w in t.windows(2) {
        let mut cp = 0.0;
        for s in streams {
            if s.t_supply.min(s.t_target) <= w[0] && s.t_supply.max(s.t_target) >= w[1] {
                cp += s.cp;
            }
        }
        let last = h[h.len() - 1];
        h.push(last + cp * (w[1] - w[0]));
    }
    if t.is_empty() {
        h.clear();
    }
    Curve { t, h }
}

/// Minimum utilities, pinch, composite curves and grand composite curve.
///
/// `hot`, `cold`: streams (see module docs); `dt_min`: minimum approach temperature [same unit as T,
/// >= 0]. Errors (`Err(message)`) for a stream going the wrong way, negative cp or `dt_min`, or no
/// streams at all.
pub fn pinch_targets(hot: &[Stream], cold: &[Stream], dt_min: f64) -> Result<PinchTargets, String> {
    if !(dt_min >= 0.0 && dt_min.is_finite()) {
        return Err("dt_min must be finite and >= 0".into());
    }
    check(hot, true, "hot")?;
    check(cold, false, "cold")?;
    if hot.is_empty() && cold.is_empty() {
        return Err("no streams".into());
    }
    let mut b: Vec<f64> = Vec::new();
    for s in hot {
        b.extend([s.t_supply, s.t_target]);
    }
    for s in cold {
        b.extend([s.t_supply + dt_min, s.t_target + dt_min]);
    }
    b.sort_by(|x, y| y.partial_cmp(x).unwrap());
    b.dedup();
    let mut cascade = vec![0.0];
    let mut flow = 0.0;
    for w in b.windows(2) {
        let (hi, lo) = (w[0], w[1]);
        let mut net = 0.0;
        for s in hot {
            net += s.cp * 0f64.max(hi.min(s.t_supply) - lo.max(s.t_target));
        }
        for s in cold {
            net -= s.cp * 0f64.max(hi.min(s.t_target + dt_min) - lo.max(s.t_supply + dt_min));
        }
        flow += net;
        cascade.push(flow);
    }
    let mn = cascade.iter().cloned().fold(f64::INFINITY, f64::min);
    let qh = 0f64.max(-mn);
    let flows: Vec<f64> = cascade.iter().map(|c| c + qh).collect();
    let mut pinch = None;
    for (t, f) in b.iter().zip(&flows) {
        if f.abs() < PINCH_ZERO_TOL * (1.0 + qh) {
            pinch = Some(*t);
        }
    }
    let qc = flows[flows.len() - 1];
    let hc = composite_curve(hot);
    let mut cc = composite_curve(cold);
    for h in cc.h.iter_mut() {
        *h += qc;
    }
    Ok(PinchTargets {
        qh_min: qh,
        qc_min: qc,
        pinch_hot: pinch,
        t_shifted: b.iter().map(|t| t - 0.5 * dt_min).collect(),
        t_hot_scale: b,
        heat_flow: flows,
        hot_composite: hc,
        cold_composite: cc,
    })
}
