"""Tests for the operation profile registry.

Covers the frozen OperationProfile record, the memory registration, profile
resolution, the round-trip through discovery/dataset/serialization for every
registered entry, and the grep gate that forbids operation name comparisons
outside the registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from sampling.graph import extract_circuit_metadata
from sampling.profile import (
    MEMORY_PROFILE,
    MetricPolicy,
    OperationProfile,
    registered_operations,
    resolve_profile,
)
from sampling.representation import SPATIAL, SPATIAL_MEMORY


REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUITS_DIR = REPO_ROOT / "data" / "circuits" / "memory"


class TestOperationProfileValidation:
    def test_rejects_invalid_operation_name(self) -> None:
        with pytest.raises(ValueError, match="operation must match"):
            OperationProfile(
                operation="Bad-Name",
                circuit_root=Path("data/circuits/bad"),
                metadata_extractor=extract_circuit_metadata,
                representation_version="spatial",
                observable_names=("obs",),
                metric_policy=MetricPolicy(include_per_round_ler=True),
                run_dir_segment="bad",
            )

    def test_rejects_empty_observable_names(self) -> None:
        with pytest.raises(ValueError, match="observable_names must be non-empty"):
            OperationProfile(
                operation="test_op",
                circuit_root=Path("data/circuits/test"),
                metadata_extractor=extract_circuit_metadata,
                representation_version="spatial",
                observable_names=(),
                metric_policy=MetricPolicy(include_per_round_ler=True),
                run_dir_segment="test",
            )

    def test_rejects_empty_run_dir_segment(self) -> None:
        with pytest.raises(ValueError, match="run_dir_segment must be non-empty"):
            OperationProfile(
                operation="test_op",
                circuit_root=Path("data/circuits/test"),
                metadata_extractor=extract_circuit_metadata,
                representation_version="spatial",
                observable_names=("obs",),
                metric_policy=MetricPolicy(include_per_round_ler=True),
                run_dir_segment="",
            )

    def test_rejects_unknown_representation_version(self) -> None:
        with pytest.raises(ValueError, match="Unknown representation version"):
            OperationProfile(
                operation="test_op",
                circuit_root=Path("data/circuits/test"),
                metadata_extractor=extract_circuit_metadata,
                representation_version="nonexistent_v99",
                observable_names=("obs",),
                metric_policy=MetricPolicy(include_per_round_ler=True),
                run_dir_segment="test",
            )

    def test_rejects_non_callable_extractor(self) -> None:
        with pytest.raises(ValueError, match="metadata_extractor must be callable"):
            OperationProfile(
                operation="test_op",
                circuit_root=Path("data/circuits/test"),
                metadata_extractor="not_callable",  # type: ignore[arg-type]
                representation_version="spatial",
                observable_names=("obs",),
                metric_policy=MetricPolicy(include_per_round_ler=True),
                run_dir_segment="test",
            )

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            MEMORY_PROFILE.operation = "x"  # type: ignore[misc]


class TestMemoryProfile:
    def test_data_contract_matches_spatial_memory(self) -> None:
        assert MEMORY_PROFILE.data_contract == SPATIAL_MEMORY

    def test_data_contract_representation_matches_spatial(self) -> None:
        contract = MEMORY_PROFILE.data_contract
        assert contract.representation == SPATIAL
        assert contract.node_dim == 6
        assert contract.edge_dim == 6
        assert contract.version == "spatial"

    def test_data_contract_labels_match_memory(self) -> None:
        contract = MEMORY_PROFILE.data_contract
        assert contract.num_observables == 1
        assert contract.labels.observable_names == ("logical_observable",)


class TestResolution:
    def test_resolve_memory(self) -> None:
        profile = resolve_profile("memory")
        assert profile is MEMORY_PROFILE

    def test_unknown_raises_listing_registered(self) -> None:
        with pytest.raises(ValueError, match="Unknown operation") as exc_info:
            resolve_profile("nonexistent_op")
        assert "memory" in str(exc_info.value)

    def test_registered_operations_contains_memory(self) -> None:
        ops = registered_operations()
        assert "memory" in ops

    def test_registered_operations_is_frozenset(self) -> None:
        ops = registered_operations()
        assert isinstance(ops, frozenset)


class TestRegistryRoundTrip:
    """For each registered operation, drive it through discovery, dataset
    construction, and result serialization.  An operation that some module
    cannot serve fails here rather than at the first real run.
    """

    def test_discovery(self) -> None:
        """Circuits are discoverable from each profile's circuit_root."""
        from sampling.sampler import settings_from_circuit_dir

        for op in registered_operations():
            profile = resolve_profile(op)
            root = REPO_ROOT / profile.circuit_root
            if not root.is_dir():
                pytest.skip(f"Circuit root {root} not present")
            points = settings_from_circuit_dir(root)
            assert len(points) > 0
            for pt in points:
                assert pt.operation == profile.operation

    def test_dataset_construction(self) -> None:
        """A dataset can be constructed from each profile's contract."""
        from model.dataset import StreamingSurfaceCodeDataset
        from sampling.sampler import settings_from_circuit_dir

        for op in registered_operations():
            profile = resolve_profile(op)
            root = REPO_ROOT / profile.circuit_root
            if not root.is_dir():
                pytest.skip(f"Circuit root {root} not present")
            settings = settings_from_circuit_dir(root)
            contract = profile.data_contract
            ds = StreamingSurfaceCodeDataset(
                settings=settings,
                master_seed=42,
                contract=contract,
            )
            sample = next(iter(ds))
            assert sample.x.shape[1] == contract.node_dim
            assert sample.edge_attr.shape[1] == contract.edge_dim

    def test_result_serialization(self) -> None:
        """A serialized result row carries the operation from the profile."""
        from evaluation.evaluator import (
            DecoderPointResult,
            EvalPointResult,
            _point_to_dict,
        )
        from evaluation.stats import (
            EvalOutcome,
            McNemarResult,
            StoppingDecision,
            WilsonInterval,
        )
        from sampling.experiment import ExperimentKey

        for op in registered_operations():
            profile = resolve_profile(op)
            wi = WilsonInterval(
                lower=0.0, upper=0.1, point=0.05, n_errors=5, n_total=100, alpha=0.05
            )
            dr = DecoderPointResult(
                decoder_name="test",
                n_shots=100,
                n_errors=5,
                ler=0.05,
                ler_interval=wi,
                per_round_ler=0.01,
                per_round_interval=wi,
                correct=np.ones(100, dtype=np.bool_),
            )
            mr = McNemarResult(
                statistic=1.0,
                p_value=0.3,
                n_discordant=10,
                gnn_wins=6,
                baseline_wins=4,
            )
            result = EvalPointResult(
                key=ExperimentKey(
                    operation=profile.operation,
                    distance=3,
                    rounds=3,
                    error_prob=0.01,
                ),
                n_shots_used=100,
                decoder_results={"test": dr},
                mcnemar_results={"baseline": mr},
                stopping=StoppingDecision(
                    action="stop",
                    reason="test",
                    outcome=EvalOutcome.RESOLVED_DIFFERENT,
                    mcnemar=mr,
                ),
                outcome=EvalOutcome.RESOLVED_DIFFERENT,
            )
            row = _point_to_dict(result)
            assert row["operation"] == profile.operation


# Patterns in the registry and experiment identity modules that are allowed to
# name operations.  Everything else in src/ should read a field of the
# resolved profile, not compare an operation string to a literal.
_ALLOWED_FILES: frozenset[str] = frozenset(
    {
        "src/sampling/profile.py",
        "src/sampling/experiment.py",
        "src/sampling/logical_ops.py",
    }
)

# Pattern: if <anything>operation<anything> == "..." or == '<operation>'
# We look for Python comparisons of any identifier containing "operation"
# against a string literal.
_OP_COMPARE_RE = re.compile(
    r"""
    (?:                          # LHS == "literal"
        \boperation\b\s*==\s*["']
    |                            # "literal" == RHS
        ["']\w+["']\s*==\s*\boperation\b
    )
    """,
    re.VERBOSE,
)


class TestGrepGate:
    """No src/ module outside the registry compares an operation name to a
    string literal.  This prevents ``if operation == "memory"`` branches that
    would encode the experiment axis as code paths rather than as data.
    """

    def test_no_operation_literal_comparison_in_src(self) -> None:
        src_dir = REPO_ROOT / "src"
        violations: list[str] = []

        for py_file in sorted(src_dir.rglob("*.py")):
            rel = str(py_file.relative_to(REPO_ROOT))
            if rel in _ALLOWED_FILES:
                continue
            if "__pycache__" in rel:
                continue

            source = py_file.read_text(encoding="utf-8")
            for lineno, line in enumerate(source.splitlines(), start=1):
                if line.lstrip().startswith("#"):
                    continue
                if _OP_COMPARE_RE.search(line):
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

        assert violations == [], (
            "Found operation name literal comparisons outside the registry. "
            "Read a field of the resolved profile instead:\n" + "\n".join(violations)
        )
