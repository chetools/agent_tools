//! Example: Seider and Turton bare-module cost of a 100 m^2 exchanger, and the capital recovery factor.
//! Prints one JSON object (same keys as the Python example).
use agent_tools::process::hx_capital_cost::*;

fn main() {
    let (a, b) = seider_material_coeffs(Material::Cs, Material::Ti).unwrap();
    let s = seider_bare_module_cost(
        100.0,
        HxType::FloatingHead,
        a,
        b,
        seider_length_factor(12).unwrap(),
        seider_pressure_factor(100.0),
        SEIDER_F_BM,
    );
    let t = turton_bare_module_cost(
        100.0,
        turton_material_factor(Material::Ss).unwrap(),
        turton_pressure_factor(20.0, TURTON_C),
        TURTON_K,
        TURTON_B,
        TURTON_AMIN_M2,
        TURTON_AMAX_M2,
    );
    let small = seider_bare_module_cost(5.0, HxType::UTube, 0.0, 0.0, 1.0, 1.0, SEIDER_F_BM);
    println!(
        "{{\"seider_cost\":{:?},\"seider_dcost_da\":{:?},\"turton_cost\":{:?},\"turton_dcost_da\":{:?},\"small_clamped\":{},\"crf\":{:?}}}",
        s.cost_usd, s.dcost_da, t.cost_usd, t.dcost_da, small.clamped_below, capital_recovery_factor(0.1, 10.0)
    );
}
