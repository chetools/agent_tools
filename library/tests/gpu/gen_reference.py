"""Generate (or check) the stored reference data for the `gpu` area.

    python library/tests/gpu/gen_reference.py           # write library/tests/reference_data/gpu/<id>.json
    python library/tests/gpu/gen_reference.py --check   # regenerate in memory, fail if a stored file differs

Each routine's Python reference module ``library/python/gpu/<id>.py`` provides ``make_reference() -> dict``
(JSON-serialisable, float64 values).  The generator adds a ``_header`` recording the generator, library
versions and date, as AGENTS.md section 5 requires.  ``--check`` compares everything except the date, with a
relative tolerance of 1e-12, so the stored data cannot drift from the reference implementation unnoticed.
"""
import datetime
import importlib
import json
import math
import pathlib
import platform
import sys

import numpy as np

LIB = pathlib.Path(__file__).resolve().parents[2]
OUT = LIB / "tests" / "reference_data" / "gpu"
sys.path.insert(0, str(LIB / "python"))

IDS = sorted(p.stem for p in (LIB / "tests" / "gpu").glob("*.test.js"))
IDS = [i[:-5] if i.endswith(".test") else i for i in IDS]


def header(mod_id):
    return {
        "generator": f"library/python/gpu/{mod_id}.py make_reference()",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "precision": "float64",
        "date": datetime.date.today().isoformat(),
    }


def close(a, b, path="", rtol=1e-12):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys differ {sorted(set(a) ^ set(b))}"
        for k in a:
            r = close(a[k], b[k], f"{path}.{k}", rtol)
            if r:
                return r
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = close(x, y, f"{path}[{i}]", rtol)
            if r:
                return r
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isclose(a, b, rel_tol=rtol, abs_tol=1e-300) or (math.isnan(a) and math.isnan(b)):
            return None
        return f"{path}: {a!r} != {b!r}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def main():
    check = "--check" in sys.argv
    if not IDS:
        sys.exit("no GPU tests found")
    bad = 0
    for mod_id in IDS:
        mod = importlib.import_module(f"gpu.{mod_id}")
        data = mod.make_reference()
        if not isinstance(data, dict) or not data:
            sys.exit(f"{mod_id}: make_reference() returned no data")
        data = json.loads(json.dumps(data))  # normalise tuples, numpy scalars must already be converted
        path = OUT / f"{mod_id}.json"
        if check:
            if not path.exists():
                print(f"FAIL {mod_id}: {path} missing")
                bad += 1
                continue
            stored = json.loads(path.read_text())
            stored.pop("_header", None)
            diff = close(data, stored)
            print(f"{'FAIL' if diff else 'ok  '} {mod_id}: stored reference data {'differs: ' + diff if diff else 'reproduced'}")
            bad += bool(diff)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"_header": header(mod_id), **data}, indent=1) + "\n")
            print(f"wrote {path.relative_to(LIB.parent)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
