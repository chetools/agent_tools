"""Catalog check: library/catalog.toml is well formed and every path it names exists.

An empty catalog passes. A routine with a missing field, a duplicate id, a bad
status, an unknown area or a path that does not exist fails.
"""
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[3]
REQUIRED = ["id", "name", "area", "problem_class", "status", "version", "source", "example"]
PATH_FIELDS = ["rust", "python", "example", "example_python"]
STATUSES = {"experimental", "verified"}


def readme_areas():
    """Area ids from the table in README.md (first column, in backticks)."""
    text = (ROOT / "README.md").read_text()
    return set(re.findall(r"^\| `([a-z_]+)` \|", text, flags=re.M))


def check_catalog(catalog_path=ROOT / "library" / "catalog.toml", root=ROOT, areas=None):
    """Return a list of problems found (empty if the catalog is fine)."""
    areas = readme_areas() if areas is None else areas
    with open(catalog_path, "rb") as f:
        routines = tomllib.load(f).get("routine", [])
    problems, seen = [], set()
    for r in routines:
        rid = r.get("id", "<no id>")
        for field in REQUIRED:
            if not r.get(field):
                problems.append(f"{rid}: missing field '{field}'")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        if not (r.get("rust") or r.get("python")):
            problems.append(f"{rid}: neither 'rust' nor 'python' given")
        if r.get("status") not in STATUSES:
            problems.append(f"{rid}: status {r.get('status')!r} not in {sorted(STATUSES)}")
        if r.get("area") not in areas:
            problems.append(f"{rid}: area {r.get('area')!r} not in the README area table")
        if not re.fullmatch(r"\d+\.\d+\.\d+", str(r.get("version", ""))):
            problems.append(f"{rid}: version {r.get('version')!r} is not major.minor.patch")
        for field in PATH_FIELDS:
            if r.get(field) and not (root / r[field]).is_file():
                problems.append(f"{rid}: {field} path {r[field]} does not exist")
    return problems


def test_catalog():
    assert check_catalog() == []


def test_catalog_check_detects_defects(tmp_path):
    """The checker itself fails on planted defects (it is not a silent no-op)."""
    good = ('[[routine]]\nid="a"\nname="A"\narea="linalg"\nproblem_class="x"\nstatus="experimental"\n'
            'version="0.1.0"\nsource="s"\nexample="ex.txt"\nrust="ex.txt"\n')
    (tmp_path / "ex.txt").write_text("x")

    def run(text):
        p = tmp_path / "cat.toml"
        p.write_text(text)
        return check_catalog(p, tmp_path, {"linalg"})

    assert run("") == []
    assert run(good) == []
    assert run(good.replace('rust="ex.txt"', 'rust="missing.rs"'))
    assert run(good.replace("experimental", "done"))
    assert run(good.replace('area="linalg"', 'area="nope"'))
    assert run(good + good)
    assert run(good.replace('source="s"\n', ""))
