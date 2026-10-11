"""Every package a baseline's numbers depend on is pinned exactly and installed
at its pin, so a lock refresh cannot move a baseline."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# The baselines' own packages, and ldpc, which runs Belief-Matching's BP.
BASELINE_PACKAGES = ("pymatching", "beliefmatching", "ldpc", "tesseract-decoder")


def _exact_pins() -> dict[str, str]:
    """``name -> version`` of every ``name==version`` requirement or constraint."""
    pyproject = tomllib.loads(PYPROJECT.read_text())
    requirements = (
        pyproject["project"]["dependencies"]
        + pyproject["tool"]["uv"]["constraint-dependencies"]
    )
    return dict(req.split("==", 1) for req in requirements if "==" in req)


@pytest.mark.parametrize("package", BASELINE_PACKAGES)
def test_baseline_package_is_pinned_exactly_and_installed_at_its_pin(
    package: str,
) -> None:
    pins = _exact_pins()
    assert package in pins, f"{package} has no exact pin in {PYPROJECT}"
    assert version(package) == pins[package]
