//! Example: BFGS on the 2-D Rosenbrock function from (-1.2, 1) and on the separable extended
//! Rosenbrock function with n = 10 from (-1.2, 1, -1.2, 1, ...), gtol = 1e-10.
//! Prints one JSON object (same keys as the Python example). "xs" holds every point at which
//! `fg` was called, in call order (flat), so the two languages can be compared iterate by iterate.
use agent_tools::optimize::bfgs::{bfgs, BfgsOptions};

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(","))
}

/// Extended Rosenbrock (Moré, Garbow, Hillstrom 1981, problem 21), n even.
fn ext_rosenbrock(x: &[f64], g: &mut [f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..x.len() / 2 {
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        let t = b - a * a;
        let u = 1.0 - a;
        f += 100.0 * (t * t) + u * u;
        g[2 * i] = -400.0 * a * t - 2.0 * u;
        g[2 * i + 1] = 200.0 * t;
    }
    f
}

fn run(n: usize) -> String {
    let x0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    let mut xs: Vec<f64> = Vec::new();
    let opt = BfgsOptions { max_iter: 200, gtol: 1e-10 };
    let r = bfgs(
        |x, g| {
            xs.extend_from_slice(x);
            ext_rosenbrock(x, g)
        },
        &x0,
        &opt,
    );
    format!(
        "{{\"x\":{},\"f\":{:?},\"grad_norm\":{:?},\"success\":{},\"nit\":{},\"nfev\":{},\"xs\":{}}}",
        list(&r.x),
        r.f,
        r.grad_norm,
        r.success,
        r.nit,
        r.nfev,
        list(&xs)
    )
}

fn main() {
    println!("{{\"rosenbrock2\":{},\"ext_rosenbrock10\":{}}}", run(2), run(10));
}
