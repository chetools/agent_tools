//! Example: permeability correlations for a few inputs (SI units).
//! Prints one JSON object (same keys as the Python example).
use agent_tools::properties::porous_permeability::*;

fn main() {
    let (eps, d) = (0.4, 1e-3);
    let k = kozeny_carman(eps, 6.0 / d, KOZENY_DEFAULT);
    println!(
        "{{\"kozeny_carman\":{:?},\"fit_kozeny_constant\":{:?},\"ergun_gradient\":{:?},\"gebart_square\":{:?},\"gebart_hexagonal\":{:?},\"sangani_acrivos_square\":{:?},\"nan_case_is_nan\":{}}}",
        k,
        fit_kozeny_constant(k, eps, 6.0 / d),
        ergun_gradient(0.01, eps, d, 1e-3, 1000.0, ERGUN_A, ERGUN_B),
        gebart_square(0.3, d),
        gebart_hexagonal(0.2, d),
        sangani_acrivos_square(0.8, d),
        gebart_square(0.1, d).is_nan()
    );
}
