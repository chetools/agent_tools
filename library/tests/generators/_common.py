"""Shared helpers for the reference-data generators (not a generator itself)."""
import json
import pathlib
import platform
import subprocess

import numpy
import scipy

DATE = "2026-10-27"
OUT = pathlib.Path(__file__).resolve().parents[1] / "reference_data"


def node_version():
    """Version string of node, or None if node is not installed."""
    try:
        return subprocess.run(["node", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions(**extra):
    v = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
    v.update(extra)
    return v


def write(routine, generator, settings, notes, cases, **extra_versions):
    """Write library/tests/reference_data/<routine>.json with the standard header."""
    doc = {"header": {"generator": generator, "versions": versions(**extra_versions),
                      "settings": settings, "date": DATE, "notes": notes}, "cases": cases}
    path = OUT / f"{routine}.json"
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
