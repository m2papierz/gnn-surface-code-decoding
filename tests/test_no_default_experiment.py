"""No default experiment: the operation and its contract are always carried.

Covers the source gate (no default or constant in ``src/`` names an operation
or a representation), the boundaries that reject an omitted operation, the
manifest every committed circuit carries, and the evaluation path end to end:
each mode decodes through ``GNNDecoder`` under the contract of the operation it
evaluates, and a checkpoint trained under another contract is refused at load.
"""

from __future__ import annotations

import ast
import json
import runpy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from decoders import GNNDecoder, load_gnn_checkpoint
from evaluation.config import EvalConfig
from model.decoder import build_model
from sampling import representation
from sampling.experiment import circuit_key, circuit_manifest_path
from sampling.graph import CircuitMetadata
from sampling.profile import registered_operations, resolve_profile
from sampling.representation import DataContract, RepresentationDescriptor
from sampling.sampler import settings_from_circuit_dir
from training.config import TrainConfig
from training.primitives import build_training_state, write_checkpoint


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_gnn.py"

# The one module whose constants are the representation registry's entries.
_REGISTRY_MODULE = SRC_DIR / "sampling" / "representation.py"

# Constants whose text coincides with an experiment name without naming one.
_COINCIDENT_CONSTANTS = frozenset(
    {
        # Schedule-phase vocabulary of the phased one-hot: "memory" is a phase.
        ("src/sampling/graph.py", "PHASE_TYPES"),
    }
)

# Parameters that carry the experiment or its contract: never defaulted.
_CARRIED_PARAMETERS = frozenset(
    {
        "operation",
        "contract",
        "metadata_extractor",
        "node_dim",
        "edge_dim",
        "num_observables",
    }
)

# Calls that resolve an experiment from the registries.
_RESOLVERS = frozenset({"resolve_profile", "resolve_descriptor", "resolve_builder"})


def _experiment_literals() -> frozenset[str]:
    versions = {d.version for d in representation.registered_representations()}
    return registered_operations() | versions


def _descriptor_names() -> frozenset[str]:
    return frozenset(
        name
        for name, value in vars(representation).items()
        if isinstance(value, RepresentationDescriptor | DataContract)
    )


def _names_an_experiment(node: ast.AST) -> bool:
    """Whether an expression resolves to an operation or a representation."""
    literals, descriptors = _experiment_literals(), _descriptor_names()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value in literals:
            return True
        if isinstance(sub, ast.Name) and sub.id in descriptors:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in descriptors:
            return True
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.id if isinstance(func, ast.Name) else None
            if isinstance(func, ast.Attribute):
                name = func.attr
            if name in _RESOLVERS:
                return True
    return False


def _source_modules() -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted(SRC_DIR.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _parameter_defaults(
    func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> Iterator[tuple[str, ast.expr]]:
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    for arg, default in zip(
        positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
    ):
        yield arg.arg, default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if default is not None:
            yield arg.arg, default


def _assigned_values(path: Path, body: list[ast.stmt]) -> Iterator[ast.expr]:
    """Values bound by the assignments of one block, coincident constants aside."""
    module = str(path.relative_to(REPO_ROOT))
    for stmt in body:
        if not isinstance(stmt, ast.Assign | ast.AnnAssign) or stmt.value is None:
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if not any((module, name) in _COINCIDENT_CONSTANTS for name in names):
            yield stmt.value


def _location(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', '?')}"


def _experiment_violations(path: Path, module: ast.Module) -> list[str]:
    """Defaults, field defaults and constants of *module* that name an experiment."""
    scanned: list[ast.expr] = []
    if path != _REGISTRY_MODULE:
        scanned += _assigned_values(path, module.body)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef):
            scanned += _assigned_values(path, node.body)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            scanned += (default for _, default in _parameter_defaults(node))
    return [_location(path, value) for value in scanned if _names_an_experiment(value)]


class TestSourceGate:
    def test_no_default_or_constant_names_an_experiment(self) -> None:
        violations = [
            violation
            for path, module in _source_modules()
            for violation in _experiment_violations(path, module)
        ]
        assert violations == [], "experiment resolved by a default:\n" + "\n".join(
            violations
        )

    def test_carried_parameters_have_no_default(self) -> None:
        """An operation, contract, extractor or model width is always passed."""
        violations = [
            f"{_location(path, default)}: {name}"
            for path, module in _source_modules()
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
            for name, default in _parameter_defaults(node)
            if name in _CARRIED_PARAMETERS
        ]
        assert violations == [], "defaulted carried parameter:\n" + "\n".join(
            violations
        )

    @pytest.mark.parametrize(
        "source",
        [
            "def f(operation='memory'): ...",
            "def f(*, contract=SPATIAL): ...",
            "class C:\n    version: str = 'phased'",
            "DEFAULT = resolve_profile('zz_merge_split')",
            "def f(dim=representation.PHASED.node_dim): ...",
        ],
    )
    def test_gate_detects_each_form(self, source: str) -> None:
        """Each form the gate scans is one it catches."""
        assert _experiment_violations(SRC_DIR / "snippet.py", ast.parse(source))


class TestConfigBoundary:
    @pytest.mark.parametrize("config_cls", [TrainConfig, EvalConfig])
    def test_yaml_without_operation_is_rejected(
        self, config_cls: type[TrainConfig] | type[EvalConfig], tmp_path: Path
    ) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("seed: 7\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no operation") as exc_info:
            config_cls.from_yaml(yaml_path)
        for operation in registered_operations():
            assert operation in str(exc_info.value)


class TestCommittedCircuits:
    @pytest.mark.parametrize("operation", sorted(registered_operations()))
    def test_every_circuit_declares_its_operation(self, operation: str) -> None:
        root = REPO_ROOT / resolve_profile(operation).circuit_root
        circuits = sorted(root.glob("*.stim"))
        assert circuits, f"no committed circuits under {root}"
        for circuit_path in circuits:
            assert circuit_manifest_path(circuit_path).is_file(), circuit_path
            assert circuit_key(circuit_path).operation == operation


def _write_checkpoint(path: Path, operation: str) -> None:
    """An untrained checkpoint written by the trainer for *operation*."""
    contract = resolve_profile(operation).data_contract
    cfg = TrainConfig.from_fields(
        {"operation": operation, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    )
    state = build_training_state(
        cfg,
        contract=contract,
        total_budget=cfg.sample_budget,
        device=torch.device("cpu"),
    )
    write_checkpoint(
        path,
        state,
        samples_consumed=0,
        best_metric=1.0,
        decision_threshold=0.0,
        contract=contract,
        cfg=cfg,
    )


@pytest.fixture(scope="module")
def eval_script() -> dict[str, Any]:
    return runpy.run_path(str(EVAL_SCRIPT))


class TestEvalPath:
    @pytest.mark.parametrize("operation", sorted(registered_operations()))
    def test_sanity_mode_decodes_under_the_operation_contract(
        self, operation: str, eval_script: dict[str, Any], tmp_path: Path
    ) -> None:
        """Fresh shots of every operation decode under that operation's contract.

        The phased checkpoint once met graphs from the spatial builder here and
        failed with a 6-vs-11 node-feature mismatch.
        """
        checkpoint = tmp_path / "best.pt"
        _write_checkpoint(checkpoint, operation)
        output = tmp_path / "sanity.json"
        distance = settings_from_circuit_dir(resolve_profile(operation).circuit_root)[
            0
        ].distance

        eval_script["main"](
            [
                "--sanity",
                "--operation",
                operation,
                "--checkpoint",
                str(checkpoint),
                "--distances",
                str(distance),
                "--shots",
                "64",
                "-o",
                str(output),
            ]
        )

        results = json.loads(output.read_text(encoding="utf-8"))
        assert results
        assert all(r["n_shots"] == 64 and r["distance"] == distance for r in results)

    def test_dry_run_decodes_the_ci_shard(
        self, eval_script: dict[str, Any], tmp_path: Path
    ) -> None:
        output = tmp_path / "dry_run.json"
        eval_script["main"](["--dry-run", "--operation", "memory", "-o", str(output)])
        (point,) = json.loads(output.read_text(encoding="utf-8"))["points"]
        assert point["operation"] == "memory"

    def test_checkpoint_of_another_contract_is_rejected_at_load(
        self, tmp_path: Path
    ) -> None:
        checkpoint = tmp_path / "best.pt"
        _write_checkpoint(checkpoint, "memory")
        with pytest.raises(ValueError, match="Representation mismatch"):
            load_gnn_checkpoint(
                checkpoint,
                contract=resolve_profile("zz_merge_split").data_contract,
                device=torch.device("cpu"),
            )

    def test_checkpoint_restores_under_its_own_contract(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "best.pt"
        _write_checkpoint(checkpoint, "zz_merge_split")
        contract = resolve_profile("zz_merge_split").data_contract
        restored = load_gnn_checkpoint(
            checkpoint, contract=contract, device=torch.device("cpu")
        )
        assert restored.contract == contract
        assert restored.model.encoder.node_dim == contract.node_dim

    def test_decoder_rejects_a_model_of_another_width(self) -> None:
        """A spatial-width model cannot be handed phased graphs."""
        spatial = resolve_profile("memory").data_contract
        phased = resolve_profile("zz_merge_split").data_contract
        model = build_model(
            node_dim=spatial.node_dim,
            edge_dim=spatial.edge_dim,
            hidden_dim=16,
            num_layers=1,
            num_observables=phased.num_observables,
        )
        n_det = 8
        metadata = CircuitMetadata(
            detector_coords=np.zeros((n_det, 3)),
            distance=3,
            rounds=3,
            num_detectors=n_det,
            dem_edge_weights=np.zeros((n_det, n_det)),
        )
        with pytest.raises(ValueError, match="does not match contract 'phased'"):
            GNNDecoder(model, metadata, contract=phased)
