"""Example: first and second derivative of f(x) = exp(sin x) sqrt(x) + |cos x| / x^2 + ln x
at x = 1.9 with forward-mode dual numbers. Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.autodiff.dual import Dual, absval, cos, deriv, exp, ln, powi, sin, sqrt  # noqa: E402


def f(x):
    """Written once; works for float and Dual (also nested)."""
    return exp(sin(x)) * sqrt(x) + absval(cos(x)) / powi(x, 2) + ln(x)


x = 1.9
value = f(x)
first = f(Dual(x, 1.0))
second = f(Dual(Dual(x, 1.0), Dual(1.0, 0.0)))
print(json.dumps({"x": x, "f": value, "df": first.d, "d2f": second.d.d,
                  "f_dual": first.v, "deriv_of_f64": deriv(value)}))
