"""Generator for reference_data/dual.json: analytic derivatives of test functions.

f, f' and f'' are typed from hand-derived closed forms (formulas in the notes below); the tests
compare the AD results with these AND with central finite differences, so a mistake in either
the formula or the AD would show.
Run: python library/tests/generators/dual.py
"""
import math

from _common import write

XS = [0.3, 0.7, 1.0, 1.9, 2.5, 4.0]


def exp_sin(x):
    f = math.exp(math.sin(x)); s, c = math.sin(x), math.cos(x)
    return f, f * c, f * (c * c - s)


def sqrt_ln(x):
    l = math.log(x); r = math.sqrt(x)
    return r * l, l / (2 * r) + 1 / r, -l / (4 * x * r)


def powi_poly(x):
    return x**5 - 3 * x**-2, 5 * x**4 + 6 * x**-3, 20 * x**3 - 18 * x**-4


def abs_cos(x):
    c, s = math.cos(x), math.sin(x); g = math.copysign(1.0, c)
    return x * abs(c), abs(c) - x * s * g, g * (-2 * s - x * c)


def rational(x):
    N = x**2 + 1; D = x**3 + 2
    n1 = -x**4 - 3 * x**2 + 4 * x          # numerator of f'
    n2 = -4 * x**3 - 6 * x + 4
    return N / D, n1 / D**2, n2 / D**2 - 6 * x**2 * n1 / D**3


def arrhenius(x):
    u = 20 * x / (20 + x); u1 = 400 / (20 + x) ** 2; u2 = -800 / (20 + x) ** 3
    f = math.exp(u)
    return f, f * u1, f * (u1 * u1 + u2)


FUNCS = {
    "exp_sin": ("exp(sin x)", exp_sin),
    "sqrt_ln": ("sqrt(x) ln x", sqrt_ln),
    "powi_poly": ("x^5 - 3 x^-2", powi_poly),
    "abs_cos": ("x |cos x|", abs_cos),
    "rational": ("(x^2+1)/(x^3+2)", rational),
    "arrhenius": ("exp(x/(1+x/20))", arrhenius),
}

cases = {}
for name, (expr, fn) in FUNCS.items():
    rows = [fn(x) for x in XS]
    cases[name] = {"expr": expr, "x": XS, "f": [r[0] for r in rows],
                   "df": [r[1] for r in rows], "d2f": [r[2] for r in rows]}

write("dual", "library/tests/generators/dual.py",
      {"points": XS, "arithmetic": "float64, hand-derived closed forms"},
      "Dimensionless. f, f', f'' of each function evaluated from hand-derived closed forms "
      "(no CAS available). Tests also compare AD with central finite differences.", cases)
