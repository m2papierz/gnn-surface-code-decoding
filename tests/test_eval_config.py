"""EvalConfig resolves its circuit directory at the boundary."""

from __future__ import annotations

from pathlib import Path

from evaluation.config import EvalConfig


def test_circuit_dir_follows_the_named_operation() -> None:
    """The operation the fields end with decides the circuits a sanity run samples."""
    cfg = EvalConfig.from_fields({"operation": "zz_merge_split"})
    assert cfg.circuit_dir == Path("data/circuits/zz_merge_split")


def test_explicit_circuit_dir_is_kept_as_a_path() -> None:
    cfg = EvalConfig.from_fields(
        {"operation": "zz_merge_split", "circuit_dir": "elsewhere"}
    )
    assert cfg.circuit_dir == Path("elsewhere")
