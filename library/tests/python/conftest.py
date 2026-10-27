"""Shared test helpers: reference data, and the Rust and Python examples.

`rust_example(name)` and `python_example(name)` run an example and return the
one JSON object it prints, so a test can compare the two languages.
"""
import functools
import json
import pathlib
import subprocess
import sys

LIB = pathlib.Path(__file__).resolve().parents[2]  # .../library
sys.path.insert(0, str(LIB / "python"))


def load_reference(routine):
    """Return the parsed library/tests/reference_data/<routine>.json."""
    with open(LIB / "tests" / "reference_data" / f"{routine}.json") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _build_rust_examples():
    # Build all examples once per session so each rust_example call only runs a binary.
    subprocess.run(["cargo", "build", "--release", "--quiet", "--examples"],
                   cwd=LIB / "rust", check=True)


def _parse_one_json(proc, what):
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def rust_example(name):
    """Run library/examples/rust/<name>.rs (cargo example) and parse its JSON output."""
    _build_rust_examples()
    proc = subprocess.run(["cargo", "run", "--release", "--quiet", "--example", name],
                          cwd=LIB / "rust", capture_output=True, text=True)
    return _parse_one_json(proc, f"rust example {name}")


def python_example(name):
    """Run library/examples/python/<name>.py and parse its JSON output."""
    proc = subprocess.run([sys.executable, str(LIB / "examples" / "python" / f"{name}.py")],
                          capture_output=True, text=True)
    return _parse_one_json(proc, f"python example {name}")
