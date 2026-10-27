"""Example: permeability correlations for a few inputs (SI units).  Prints one JSON object
(same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.properties import porous_permeability as pp  # noqa: E402

eps, d = 0.4, 1e-3
k = pp.kozeny_carman(eps, 6.0 / d, 5.0)
print(json.dumps({
    "kozeny_carman": float(k),
    "fit_kozeny_constant": float(pp.fit_kozeny_constant(k, eps, 6.0 / d)),
    "ergun_gradient": float(pp.ergun_gradient(0.01, eps, d, 1e-3, 1000.0, 150.0, 1.75)),
    "gebart_square": float(pp.gebart_square(0.3, d)),
    "gebart_hexagonal": float(pp.gebart_hexagonal(0.2, d)),
    "sangani_acrivos_square": float(pp.sangani_acrivos_square(0.8, d)),
    "nan_case_is_nan": bool(pp.gebart_square(0.1, d) != pp.gebart_square(0.1, d)),
}))
