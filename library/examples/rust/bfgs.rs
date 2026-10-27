//! Example: BFGS on the 2-D Rosenbrock function from (-1.2, 1) and on the separable extended
//! Rosenbrock function with n = 10 from (-1.2, 1, -1.2, 1, ...), gtol = 1e-10.
//! Also the 5-D quadratic f = x.A.x/2 - b.x of the reference data (library/tests/reference_data/bfgs.json),
//! from 0, with the objective summed by plain loops in the same order as the Python example.
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

/// f = x.A.x/2 - b.x with A row-major flat; plain loops in the same operation order as the Python example.
fn quadratic(a: &[f64], b: &[f64], x: &[f64], g: &mut [f64]) -> f64 {
    let n = b.len();
    let mut f = 0.0;
    for i in 0..n {
        let mut ax = 0.0;
        for j in 0..n {
            ax += a[i * n + j] * x[j];
        }
        g[i] = ax - b[i];
        f += 0.5 * x[i] * ax - b[i] * x[i];
    }
    f
}

fn run<O: FnMut(&[f64], &mut [f64]) -> f64>(x0: Vec<f64>, mut objective: O) -> String {
    let mut xs: Vec<f64> = Vec::new();
    let opt = BfgsOptions { max_iter: 200, gtol: 1e-10 };
    let r = bfgs(
        |x, g| {
            xs.extend_from_slice(x);
            objective(x, g)
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

fn rosenbrock_start(n: usize) -> Vec<f64> {
    (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect()
}

fn main() {
    // the quadratic's A and b come from the reference data (the file is also read by the Python example)
    let path = format!("{}/../tests/reference_data/bfgs.json", env!("CARGO_MANIFEST_DIR"));
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let p = &doc["cases"]["quadratic5"]["params"];
    let vec = |v: &serde_json::Value| -> Vec<f64> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
    let (qa, qb) = (vec(&p["A"]), vec(&p["b"]));
    println!(
        "{{\"rosenbrock2\":{},\"ext_rosenbrock10\":{},\"quadratic5\":{}}}",
        run(rosenbrock_start(2), ext_rosenbrock),
        run(rosenbrock_start(10), ext_rosenbrock),
        run(vec![0.0; 5], |x, g| quadratic(&qa, &qb, x, g))
    );
}
