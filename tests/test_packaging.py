"""Structural guards on the source tree.

`src/scdino` is imported as `scdino.*` from the repository root. Every
directory below it must be a *regular* package, i.e. carry an `__init__.py`.

A directory without one still imports, as a PEP 420 implicit namespace package,
which is why this drifts silently. Namespace packages merge across `sys.path`
instead of raising on a name collision, and several packaging and type-checking
tools skip them, so the mix is worth pinning down.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "scdino"

IGNORED = {"__pycache__"}


def package_dirs() -> list[Path]:
    """Every directory under src/scdino that holds Python modules."""
    out = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_dir():
            continue
        if any(part in IGNORED or part.endswith(".egg-info") for part in path.parts):
            continue
        if any(child.suffix == ".py" for child in path.iterdir()):
            out.append(path)
    return out


ALL_PACKAGES = [SRC, *package_dirs()]


def test_the_source_tree_was_found():
    assert SRC.is_dir(), f"expected a source tree at {SRC}"
    assert len(ALL_PACKAGES) > 5


@pytest.mark.parametrize("package", ALL_PACKAGES, ids=lambda p: p.name)
def test_every_package_has_an_init(package):
    init = package / "__init__.py"
    assert init.is_file(), (
        f"{package.relative_to(SRC.parent.parent)} has no __init__.py, so it is an "
        "implicit namespace package rather than a regular one"
    )


@pytest.mark.parametrize("package", ALL_PACKAGES, ids=lambda p: p.name)
def test_packages_import_as_regular_packages(package):
    """A regular package has a real __file__; a namespace package has None."""
    import importlib

    module_name = ".".join(package.relative_to(SRC.parent).parts)
    module = importlib.import_module(module_name)
    assert getattr(module, "__file__", None) is not None, (
        f"{module_name} imported as a namespace package"
    )
