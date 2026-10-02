"""Every third-party import must be declared, and anything the API needs to
boot must be a *core* dependency rather than an extra.

Why this exists. A repo audit found that `langgraph` was declared nowhere and
`torch` only under the `[ml]` extra, while CI installed `.[dev]` alone -- so a
clean clone produced an application that could not start, and CI could not have
caught it because the workflow had never actually run. Everything passed
locally only because this machine's venv had the packages from earlier manual
work.

That is the class of bug a test suite cannot catch by testing behaviour,
because the behaviour is fine *here*. It has to be caught by checking the
manifest against the code, which is what this module does.

The scan is AST-based, so imports inside functions are counted too -- a grep
for a leading `import` misses those and would under-report.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

import pytest

BACKEND = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = BACKEND / "fleetmind"
PYPROJECT = BACKEND / "pyproject.toml"

# import name -> distribution name, where PyPI disagrees with the module name.
IMPORT_TO_DISTRIBUTION = {
    "jwt": "pyjwt",
    "dotenv": "python-dotenv",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "email_validator": "email-validator",
    "pydantic_settings": "pydantic-settings",
}

# Declared on purpose despite never being `import`ed in our own code.
NOT_IMPORTED_BY_DESIGN = {
    "psycopg2-binary",  # SQLAlchemy's DBAPI driver, loaded by URL scheme
    "alembic",  # a CLI we invoke, not a library we import
    "email-validator",  # pulled in at runtime by pydantic's EmailStr
    "uvicorn",  # the ASGI server, invoked from the command line
    "pytest",
    "httpx",
    "ruff",
    "mypy",
}


def _third_party_imports() -> dict[str, list[str]]:
    """Map distribution name -> source locations that import it."""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, list[str]] = {}

    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue  # relative import, i.e. our own package
                names = [node.module]
            else:
                continue

            for name in names:
                root = name.split(".")[0]
                if root in stdlib or root == "fleetmind" or root.startswith("_"):
                    continue
                dist = IMPORT_TO_DISTRIBUTION.get(root, root)
                found.setdefault(dist, []).append(f"{path.relative_to(BACKEND)}:{node.lineno}")
    return found


def _declared() -> tuple[set[str], dict[str, set[str]]]:
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = cfg["project"]

    def names(requirements: list[str]) -> set[str]:
        return {r.split(">=")[0].split("==")[0].split("[")[0].strip() for r in requirements}

    core = names(project.get("dependencies", []))
    extras = {k: names(v) for k, v in project.get("optional-dependencies", {}).items()}
    return core, extras


def test_every_third_party_import_is_declared_somewhere():
    core, extras = _declared()
    declared = set(core)
    for group in extras.values():
        declared |= group

    used = _third_party_imports()
    missing = {dist: sites for dist, sites in used.items() if dist not in declared}

    assert not missing, "imported but declared nowhere in pyproject.toml: " + "; ".join(
        f"{dist} (e.g. {sites[0]})" for dist, sites in sorted(missing.items())
    )


def test_nothing_is_declared_that_is_never_used():
    """A dead declaration is a lie about what the project needs."""
    core, extras = _declared()
    declared = set(core)
    for group in extras.values():
        declared |= group

    used = set(_third_party_imports())
    dead = declared - used - NOT_IMPORTED_BY_DESIGN

    assert not dead, (
        "declared in pyproject.toml but never imported: "
        + ", ".join(sorted(dead))
        + " -- remove it, or add it to NOT_IMPORTED_BY_DESIGN with a reason"
    )


def _module_file(dotted: str) -> pathlib.Path | None:
    """Resolve a fleetmind.* module name to its file, if it is ours."""
    if not dotted.startswith("fleetmind"):
        return None
    rel = pathlib.Path(*dotted.split(".")[1:])
    for candidate in (PACKAGE / rel.with_suffix(".py"), PACKAGE / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _reachable_third_party(entry: str) -> dict[str, list[str]]:
    """Walk the first-party import graph from `entry`, collecting what it needs.

    Static on purpose. The obvious implementation -- import the app and diff
    `sys.modules` -- is order-dependent: by the time it runs, another test
    module has already imported torch, so the diff comes back empty and the
    test silently cannot fail. That is exactly how the original bug survived,
    so this walks the graph instead and gives the same answer in any order.
    """
    stdlib = set(sys.stdlib_module_names)
    seen: set[str] = set()
    needed: dict[str, list[str]] = {}
    queue = [entry]

    while queue:
        dotted = queue.pop()
        if dotted in seen:
            continue
        seen.add(dotted)
        path = _module_file(dotted)
        if path is None:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue
                targets = [node.module]
            else:
                continue

            for name in targets:
                root = name.split(".")[0]
                if root == "fleetmind":
                    queue.append(name)
                elif root not in stdlib and not root.startswith("_"):
                    dist = IMPORT_TO_DISTRIBUTION.get(root, root)
                    needed.setdefault(dist, []).append(f"{path.relative_to(BACKEND)}:{node.lineno}")
    return needed


def test_importing_the_api_needs_only_core_dependencies():
    """The one that would have caught the original bug.

    Anything reachable from the application entry point but declared only
    under an extra means `pip install -e .` yields an app that cannot start.
    """
    needed = _reachable_third_party("fleetmind.api.main")
    core, extras = _declared()

    declared_in_extras: set[str] = set()
    for group in extras.values():
        declared_in_extras |= group

    offenders = {
        dist: sites
        for dist, sites in needed.items()
        if dist not in core and dist in declared_in_extras
    }

    assert not offenders, (
        "importing the API reaches "
        + ", ".join(f"{d} (via {s[0]})" for d, s in sorted(offenders.items()))
        + ", which is declared only under an optional extra. A clean "
        "`pip install -e .` would therefore produce an app that cannot start. "
        "Move it into [project.dependencies]."
    )


def test_the_api_entry_point_really_does_reach_torch_and_langgraph():
    """Guards the guard: if the walk ever stops finding these, it has broken.

    A reachability test that reaches nothing passes vacuously, which is the
    failure mode that let the original bug through in the first place.
    """
    needed = _reachable_third_party("fleetmind.api.main")
    assert "torch" in needed, "import walk found no torch -- the walk is broken"
    assert "langgraph" in needed, "import walk found no langgraph -- the walk is broken"


@pytest.mark.parametrize("required", ["langgraph", "torch"])
def test_the_two_packages_that_caused_the_original_bug_stay_core(required):
    """Named explicitly so a future dependency tidy-up cannot quietly undo this.

    `langgraph` was declared nowhere at all; `torch` sat under `[ml]`. Both are
    reached from the API through replay -> pipeline -> {ml.predict, agents.graph}.
    """
    core, _ = _declared()
    assert required in core
