"""Example: Seider and Turton bare-module cost of a 100 m^2 exchanger, and the capital recovery factor.
Prints one JSON object (same keys as the Rust example)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from agent_tools.process.hx_capital_cost import (  # noqa: E402
    capital_recovery_factor, seider_bare_module_cost, seider_length_factor, seider_material_coeffs,
    seider_pressure_factor, turton_bare_module_cost, turton_material_factor, turton_pressure_factor)

a, b = seider_material_coeffs("CS", "Ti")
s = seider_bare_module_cost(100.0, "floating_head", a, b, seider_length_factor(12), seider_pressure_factor(100.0))
t = turton_bare_module_cost(100.0, turton_material_factor("SS"), turton_pressure_factor(20.0))
small = seider_bare_module_cost(5.0, "u_tube", 0.0, 0.0, 1.0, 1.0)
print(json.dumps({"seider_cost": s.cost_usd, "seider_dcost_da": s.dcost_da,
                  "turton_cost": t.cost_usd, "turton_dcost_da": t.dcost_da,
                  "small_clamped": small.clamped_below, "crf": capital_recovery_factor(0.1, 10.0)}))
