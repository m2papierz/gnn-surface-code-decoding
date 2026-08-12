"""Tests for src/sampling/logical_ops.py - the logical-operation circuit source.

Covers the validation gate (including deliberately broken inputs), metadata
derivation, the no-tqec-import invariant, and end-to-end circuit generation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import stim

from sampling.experiment import circuit_manifest_path
from sampling.logical_ops import (
    CircuitValidationError,
    PhaseWindow,
    compile_block_graph_circuit,
    derive_circuit_metadata,
    generate_and_write_circuit,
    validate_logical_op_circuit,
)


def _memory_circuit_and_metadata(
    k: int = 1, p: float = 0.003
) -> tuple[Any, stim.Circuit, dict[str, Any]]:
    """Compile a tqec memory circuit and derive its metadata."""
    from tqec import Basis, gallery

    bg = gallery.memory(Basis.Z)
    circuit = compile_block_graph_circuit(bg, k=k, error_prob=p)
    metadata = derive_circuit_metadata(bg, circuit, k=k)
    return bg, circuit, metadata


def _cnot_circuit_and_metadata(
    k: int = 1, p: float = 0.003
) -> tuple[Any, stim.Circuit, dict[str, Any]]:
    """Compile a tqec CNOT circuit and derive its metadata."""
    from tqec import Basis, gallery

    bg = gallery.cnot(Basis.Z)
    circuit = compile_block_graph_circuit(bg, k=k, error_prob=p)
    metadata = derive_circuit_metadata(bg, circuit, k=k)
    return bg, circuit, metadata


class TestNoTqecImportOutsideLogicalOps:
    """tqec must not be imported by any src/ module except logical_ops.py."""

    def test_no_tqec_import_in_src(self) -> None:
        src_root = Path(__file__).resolve().parent.parent / "src"
        allowed = src_root / "sampling" / "logical_ops.py"

        violations: list[str] = []
        for py_file in src_root.rglob("*.py"):
            if py_file == allowed:
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "tqec" or alias.name.startswith("tqec."):
                            violations.append(
                                f"{py_file.relative_to(src_root)}:{node.lineno} "
                                f"imports {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "tqec" or node.module.startswith("tqec.")
                    ):
                        violations.append(
                            f"{py_file.relative_to(src_root)}:{node.lineno} "
                            f"imports from {node.module}"
                        )

        assert not violations, (
            "tqec must only be imported by src/sampling/logical_ops.py, "
            "but found:\n" + "\n".join(f"  {v}" for v in violations)
        )


class TestPhaseWindow:
    def test_valid(self) -> None:
        pw = PhaseWindow(name="memory", t_start=0.0, t_end=3.0)
        assert pw.contains_time(0.0)
        assert pw.contains_time(2.9)
        assert not pw.contains_time(3.0)
        assert not pw.contains_time(-0.1)

    def test_invalid_range(self) -> None:
        with pytest.raises(ValueError, match="t_end"):
            PhaseWindow(name="bad", t_start=5.0, t_end=2.0)

    def test_zero_width(self) -> None:
        with pytest.raises(ValueError, match="t_end"):
            PhaseWindow(name="zero", t_start=1.0, t_end=1.0)


class TestCompileBlockGraphCircuit:
    def test_memory_k1(self) -> None:
        _, circuit, _ = _memory_circuit_and_metadata(k=1, p=0.003)
        assert circuit.num_detectors == 24
        assert circuit.num_observables == 1

    def test_memory_k2(self) -> None:
        _, circuit, _ = _memory_circuit_and_metadata(k=2, p=0.003)
        assert circuit.num_detectors == 120
        assert circuit.num_observables == 1

    def test_cnot_k1(self) -> None:
        _, circuit, _ = _cnot_circuit_and_metadata(k=1, p=0.003)
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 2

    def test_invalid_k(self) -> None:
        from tqec import Basis, gallery

        bg = gallery.memory(Basis.Z)
        with pytest.raises(ValueError, match="k must be >= 1"):
            compile_block_graph_circuit(bg, k=0, error_prob=0.003)

    def test_invalid_error_prob(self) -> None:
        from tqec import Basis, gallery

        bg = gallery.memory(Basis.Z)
        with pytest.raises(ValueError, match="error_prob"):
            compile_block_graph_circuit(bg, k=1, error_prob=0.0)


class TestDeriveCircuitMetadata:
    def test_memory_single_phase(self) -> None:
        _, _, metadata = _memory_circuit_and_metadata()
        windows = metadata["phase_windows"]
        assert len(windows) == 1
        assert windows[0].name == "memory"

    def test_memory_no_seam(self) -> None:
        _, _, metadata = _memory_circuit_and_metadata()
        assert metadata["seam_detector_indices"] == []

    def test_memory_single_patch(self) -> None:
        _, _, metadata = _memory_circuit_and_metadata()
        assert metadata["num_blocks"] == 1
        assert set(metadata["patch_ids"]) == {0}

    def test_cnot_has_merge_phases(self) -> None:
        _, _, metadata = _cnot_circuit_and_metadata()
        names = [pw.name for pw in metadata["phase_windows"]]
        assert "merge" in names
        assert "memory" in names

    def test_cnot_has_seam_detectors(self) -> None:
        _, _, metadata = _cnot_circuit_and_metadata()
        assert len(metadata["seam_detector_indices"]) > 0

    def test_cnot_three_blocks(self) -> None:
        _, _, metadata = _cnot_circuit_and_metadata()
        assert metadata["num_blocks"] == 3
        assert sorted(set(metadata["patch_ids"])) == [0, 1, 2]

    def test_phase_windows_partition_timeline(self) -> None:
        _, circuit, metadata = _memory_circuit_and_metadata()
        coords = circuit.get_detector_coordinates()
        windows = metadata["phase_windows"]
        for det_idx in range(circuit.num_detectors):
            t = coords[det_idx][2]
            covered = sum(1 for pw in windows if pw.contains_time(t))
            assert covered == 1, f"detector {det_idx} at t={t} covered {covered} times"

    def test_cnot_phase_windows_partition_timeline(self) -> None:
        _, circuit, metadata = _cnot_circuit_and_metadata()
        coords = circuit.get_detector_coordinates()
        windows = metadata["phase_windows"]
        for det_idx in range(circuit.num_detectors):
            t = coords[det_idx][2]
            covered = sum(1 for pw in windows if pw.contains_time(t))
            assert covered == 1, f"detector {det_idx} at t={t} covered {covered} times"


class TestValidateValidCircuit:
    def test_memory_passes(self) -> None:
        _, circuit, metadata = _memory_circuit_and_metadata()
        prov = validate_logical_op_circuit(
            circuit,
            operation="memory",
            distance=3,
            error_prob=0.003,
            phase_windows=metadata["phase_windows"],
            seam_detector_indices=metadata["seam_detector_indices"],
            patch_ids=metadata["patch_ids"],
            num_blocks=metadata["num_blocks"],
            expected_observable_count=1,
        )
        assert prov["num_detectors"] == 24
        assert prov["num_observables"] == 1
        assert isinstance(prov["tqec_predecomposes"], bool)

    def test_cnot_passes(self) -> None:
        bg, circuit, metadata = _cnot_circuit_and_metadata()
        surfaces = bg.find_correlation_surfaces()
        prov = validate_logical_op_circuit(
            circuit,
            operation="zz_merge_split",
            distance=3,
            error_prob=0.003,
            phase_windows=metadata["phase_windows"],
            seam_detector_indices=metadata["seam_detector_indices"],
            patch_ids=metadata["patch_ids"],
            num_blocks=metadata["num_blocks"],
            expected_observable_count=len(surfaces),
        )
        assert prov["num_detectors"] > 0
        assert prov["num_observables"] == 2


class TestValidateGateBrokenInputs:
    """The gate must refuse deliberately broken inputs."""

    def _valid_args(self) -> tuple[stim.Circuit, dict[str, Any]]:
        _, circuit, metadata = _memory_circuit_and_metadata()
        return circuit, {
            "operation": "memory",
            "distance": 3,
            "error_prob": 0.003,
            "phase_windows": metadata["phase_windows"],
            "seam_detector_indices": metadata["seam_detector_indices"],
            "patch_ids": metadata["patch_ids"],
            "num_blocks": metadata["num_blocks"],
            "expected_observable_count": 1,
        }

    def test_empty_phase_windows(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["phase_windows"] = []
        with pytest.raises(CircuitValidationError, match="no phase windows"):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_detector_outside_all_phase_windows(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["phase_windows"] = [
            PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
        ]
        with pytest.raises(CircuitValidationError, match="not covered"):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_non_contiguous_windows(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["phase_windows"] = [
            PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
            PhaseWindow(name="memory", t_start=2.0, t_end=3.0),
        ]
        with pytest.raises(CircuitValidationError, match="contiguous"):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_merge_with_empty_seam(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["phase_windows"] = [
            PhaseWindow(name="merge", t_start=0.0, t_end=3.0),
        ]
        kwargs["seam_detector_indices"] = []
        with pytest.raises(
            CircuitValidationError,
            match="seam_detector_indices is empty",
        ):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_seam_outside_merge_window(self) -> None:
        from tqec import Basis, gallery

        _, circuit, metadata = _cnot_circuit_and_metadata()
        bg_cnot = gallery.cnot(Basis.Z)
        surfaces = bg_cnot.find_correlation_surfaces()
        merge_windows = [pw for pw in metadata["phase_windows"] if pw.name == "merge"]
        assert merge_windows, "expected merge windows for CNOT"

        memory_windows = [pw for pw in metadata["phase_windows"] if pw.name == "memory"]
        assert memory_windows

        coords = circuit.get_detector_coordinates()
        memory_det = None
        for det_idx in range(circuit.num_detectors):
            t = coords[det_idx][2]
            if any(mw.contains_time(t) for mw in memory_windows):
                memory_det = det_idx
                break
        assert memory_det is not None

        with pytest.raises(CircuitValidationError, match="not inside any merge"):
            validate_logical_op_circuit(
                circuit,
                operation="zz_merge_split",
                distance=3,
                error_prob=0.003,
                phase_windows=metadata["phase_windows"],
                seam_detector_indices=[memory_det],
                patch_ids=metadata["patch_ids"],
                num_blocks=metadata["num_blocks"],
                expected_observable_count=len(surfaces),
            )

    def test_wrong_patch_count(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["num_blocks"] = 5
        with pytest.raises(CircuitValidationError, match="distinct patches"):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_wrong_observable_count(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["expected_observable_count"] = 99
        with pytest.raises(CircuitValidationError, match="observables"):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_patch_ids_wrong_length(self) -> None:
        circuit, kwargs = self._valid_args()
        kwargs["patch_ids"] = [0, 0, 0]
        with pytest.raises(CircuitValidationError, match="patch_ids length"):
            validate_logical_op_circuit(circuit, **kwargs)

    def test_seam_index_out_of_range(self) -> None:
        """Seam index beyond the detector count."""
        from tqec import Basis, gallery

        _, circuit, metadata = _cnot_circuit_and_metadata()
        bg_cnot = gallery.cnot(Basis.Z)
        surfaces = bg_cnot.find_correlation_surfaces()
        with pytest.raises(CircuitValidationError, match="out of range"):
            validate_logical_op_circuit(
                circuit,
                operation="zz_merge_split",
                distance=3,
                error_prob=0.003,
                phase_windows=metadata["phase_windows"],
                seam_detector_indices=[999999],
                patch_ids=metadata["patch_ids"],
                num_blocks=metadata["num_blocks"],
                expected_observable_count=len(surfaces),
            )

    def test_coordinates_checked(self) -> None:
        """A circuit with fewer than 3 coordinate dimensions fails."""
        bad_circuit = stim.Circuit("""
            R 0
            M 0
            DETECTOR(1.0, 2.0) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
        """)
        with pytest.raises(CircuitValidationError, match="dimensions"):
            validate_logical_op_circuit(
                bad_circuit,
                operation="test",
                distance=3,
                error_prob=0.003,
                phase_windows=[PhaseWindow("memory", 0.0, 10.0)],
                seam_detector_indices=[],
                patch_ids=[0],
                num_blocks=1,
                expected_observable_count=1,
            )


class TestGenerateAndWrite:
    def test_writes_circuit_and_manifest(self, tmp_path: Path) -> None:
        from tqec import Basis, gallery

        bg = gallery.memory(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="memory",
            k=1,
            error_prob=0.003,
            output_dir=tmp_path,
            generating_command="pytest",
        )
        assert circuit_path.exists()

        manifest_path = circuit_manifest_path(circuit_path)
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["operation"] == "memory"
        assert manifest["distance"] == 3
        assert manifest["rounds"] == 3
        assert manifest["error_prob"] == 0.003
        assert manifest["k"] == 1
        assert manifest["tqec_version"] == "0.2.0"
        assert manifest["num_detectors"] == 24
        assert manifest["num_observables"] == 1
        assert isinstance(manifest["phase_windows"], list)
        assert isinstance(manifest["seam_detector_indices"], list)
        assert isinstance(manifest["circuit_sha256"], str)

    def test_circuit_sha256_matches(self, tmp_path: Path) -> None:
        import hashlib

        from tqec import Basis, gallery

        bg = gallery.memory(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="memory",
            k=1,
            error_prob=0.003,
            output_dir=tmp_path,
        )
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        actual_sha = hashlib.sha256(circuit_path.read_bytes()).hexdigest()
        assert manifest["circuit_sha256"] == actual_sha

    def test_circuit_loads_and_dem_decomposes(self, tmp_path: Path) -> None:
        from tqec import Basis, gallery

        bg = gallery.memory(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="memory",
            k=2,
            error_prob=0.005,
            output_dir=tmp_path,
        )
        circuit = stim.Circuit.from_file(str(circuit_path))
        dem = circuit.detector_error_model(decompose_errors=True)
        assert dem.num_detectors == circuit.num_detectors

    def test_cnot_end_to_end(self, tmp_path: Path) -> None:
        from tqec import Basis, gallery

        bg = gallery.cnot(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="zz_merge_split",
            k=1,
            error_prob=0.003,
            output_dir=tmp_path,
        )
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        assert manifest["operation"] == "zz_merge_split"
        assert manifest["num_observables"] == 2
        assert len(manifest["phase_windows"]) == 4
        assert len(manifest["seam_detector_indices"]) > 0

    def test_manifest_carries_patch_ids_and_num_blocks(self, tmp_path: Path) -> None:
        from tqec import Basis, gallery

        bg = gallery.cnot(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="zz_merge_split",
            k=1,
            error_prob=0.003,
            output_dir=tmp_path,
        )
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        assert "patch_ids" in manifest
        assert "num_blocks" in manifest
        assert manifest["num_blocks"] == 3
        assert len(manifest["patch_ids"]) == manifest["num_detectors"]
        assert sorted(set(manifest["patch_ids"])) == [0, 1, 2]


class TestLogicalOperationMetadata:
    """Typed metadata record for logical-operation circuits."""

    def test_empty_phase_windows_rejects(self) -> None:
        from sampling.logical_ops import LogicalOperationMetadata

        with pytest.raises(ValueError, match="phase_windows must be non-empty"):
            LogicalOperationMetadata(
                phase_windows=(),
                seam_detector_indices=(),
                patch_ids=(0,),
                num_blocks=1,
                observables=("obs",),
            )

    def test_non_contiguous_windows_rejects(self) -> None:
        from sampling.logical_ops import LogicalOperationMetadata

        with pytest.raises(ValueError, match="contiguous"):
            LogicalOperationMetadata(
                phase_windows=(
                    PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
                    PhaseWindow(name="memory", t_start=2.0, t_end=3.0),
                ),
                seam_detector_indices=(),
                patch_ids=(0,),
                num_blocks=1,
                observables=("obs",),
            )

    def test_empty_observables_rejects(self) -> None:
        from sampling.logical_ops import LogicalOperationMetadata

        with pytest.raises(ValueError, match="observables must be non-empty"):
            LogicalOperationMetadata(
                phase_windows=(PhaseWindow(name="memory", t_start=0.0, t_end=3.0),),
                seam_detector_indices=(),
                patch_ids=(0,),
                num_blocks=1,
                observables=(),
            )

    def test_wrong_num_blocks_rejects(self) -> None:
        from sampling.logical_ops import LogicalOperationMetadata

        with pytest.raises(ValueError, match="distinct values"):
            LogicalOperationMetadata(
                phase_windows=(PhaseWindow(name="memory", t_start=0.0, t_end=3.0),),
                seam_detector_indices=(),
                patch_ids=(0, 0, 0),
                num_blocks=5,
                observables=("obs",),
            )

    def test_negative_seam_index_rejects(self) -> None:
        from sampling.logical_ops import LogicalOperationMetadata

        with pytest.raises(ValueError, match="negative index"):
            LogicalOperationMetadata(
                phase_windows=(PhaseWindow(name="memory", t_start=0.0, t_end=3.0),),
                seam_detector_indices=(-1,),
                patch_ids=(0,),
                num_blocks=1,
                observables=("obs",),
            )


class TestLogicalOperationMetadataValidateAgainstCoords:
    """Coordinate-dependent gate invariant re-assertion."""

    def test_uncovered_detector_raises(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 5.0]], dtype=np.float64)
        meta = LogicalOperationMetadata(
            phase_windows=(PhaseWindow(name="memory", t_start=0.0, t_end=1.0),),
            seam_detector_indices=(),
            patch_ids=(0, 0),
            num_blocks=1,
            observables=("obs",),
        )
        with pytest.raises(ValueError, match="not covered"):
            meta.validate_against_coords(coords)

    def test_seam_outside_merge_raises(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 1.5]], dtype=np.float64)
        meta = LogicalOperationMetadata(
            phase_windows=(
                PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
                PhaseWindow(name="merge", t_start=1.0, t_end=2.0),
            ),
            seam_detector_indices=(0,),
            patch_ids=(0, 0),
            num_blocks=1,
            observables=("obs",),
        )
        with pytest.raises(ValueError, match="not inside any merge"):
            meta.validate_against_coords(coords)

    def test_merge_without_seam_raises(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 1.5]], dtype=np.float64)
        meta = LogicalOperationMetadata(
            phase_windows=(
                PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
                PhaseWindow(name="merge", t_start=1.0, t_end=2.0),
            ),
            seam_detector_indices=(),
            patch_ids=(0, 0),
            num_blocks=1,
            observables=("obs",),
        )
        with pytest.raises(ValueError, match="seam_detector_indices is empty"):
            meta.validate_against_coords(coords)

    def test_seam_index_out_of_range_raises(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 1.5]], dtype=np.float64)
        meta = LogicalOperationMetadata(
            phase_windows=(
                PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
                PhaseWindow(name="merge", t_start=1.0, t_end=2.0),
            ),
            seam_detector_indices=(999,),
            patch_ids=(0, 0),
            num_blocks=1,
            observables=("obs",),
        )
        with pytest.raises(ValueError, match="out of range"):
            meta.validate_against_coords(coords)

    def test_patch_ids_length_mismatch_raises(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array([[0.0, 0.0, 0.5]], dtype=np.float64)
        meta = LogicalOperationMetadata(
            phase_windows=(PhaseWindow(name="memory", t_start=0.0, t_end=1.0),),
            seam_detector_indices=(),
            patch_ids=(0, 0, 0),
            num_blocks=1,
            observables=("obs",),
        )
        with pytest.raises(ValueError, match="patch_ids length"):
            meta.validate_against_coords(coords)

    def test_valid_passes(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 1.5]], dtype=np.float64)
        meta = LogicalOperationMetadata(
            phase_windows=(
                PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
                PhaseWindow(name="merge", t_start=1.0, t_end=2.0),
            ),
            seam_detector_indices=(1,),
            patch_ids=(0, 0),
            num_blocks=1,
            observables=("obs",),
        )
        meta.validate_against_coords(coords)


class TestLogicalOperationMetadataFromManifest:
    """Build LogicalOperationMetadata from a circuit manifest."""

    def test_cnot_roundtrip(self, tmp_path: Path) -> None:
        """Generate a CNOT, read its manifest, reconstruct metadata."""
        import numpy as np
        from tqec import Basis, gallery

        from sampling.logical_ops import LogicalOperationMetadata

        bg = gallery.cnot(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="zz_merge_split",
            k=1,
            error_prob=0.003,
            output_dir=tmp_path,
        )
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        circuit = stim.Circuit.from_file(str(circuit_path))
        coord_dict = circuit.get_detector_coordinates()
        n_det = circuit.num_detectors
        coords = np.zeros((n_det, 3), dtype=np.float64)
        for det_id, c in coord_dict.items():
            coords[det_id, : min(len(c), 3)] = c[:3]

        lo_meta = LogicalOperationMetadata.from_manifest(
            manifest,
            observables=("obs_0", "obs_1"),
            detector_coords=coords,
        )
        assert len(lo_meta.phase_windows) == 4
        assert lo_meta.num_blocks == 3
        assert lo_meta.observables == ("obs_0", "obs_1")
        assert len(lo_meta.patch_ids) == n_det
        assert len(lo_meta.seam_detector_indices) > 0

    def test_observable_count_mismatch_raises(self, tmp_path: Path) -> None:
        """Manifest observable count vs profile observable names mismatch."""
        import numpy as np
        from tqec import Basis, gallery

        from sampling.logical_ops import LogicalOperationMetadata

        bg = gallery.cnot(Basis.Z)
        circuit_path = generate_and_write_circuit(
            bg,
            operation="zz_merge_split",
            k=1,
            error_prob=0.003,
            output_dir=tmp_path,
        )
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        circuit = stim.Circuit.from_file(str(circuit_path))
        coord_dict = circuit.get_detector_coordinates()
        n_det = circuit.num_detectors
        coords = np.zeros((n_det, 3), dtype=np.float64)
        for det_id, c in coord_dict.items():
            coords[det_id, : min(len(c), 3)] = c[:3]

        with pytest.raises(ValueError, match="observables"):
            LogicalOperationMetadata.from_manifest(
                manifest,
                observables=("only_one",),
                detector_coords=coords,
            )


class TestLogicalOperationMetadataToCircuitMetadataFields:
    """Conversion to arrays for CircuitMetadata."""

    def test_memory_single_phase(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array(
            [[0.0, 0.0, 0.0], [2.0, 2.0, 1.0], [4.0, 4.0, 2.0]],
            dtype=np.float64,
        )
        meta = LogicalOperationMetadata(
            phase_windows=(PhaseWindow(name="memory", t_start=0.0, t_end=3.0),),
            seam_detector_indices=(),
            patch_ids=(0, 0, 0),
            num_blocks=1,
            observables=("logical_observable",),
        )
        fields = meta.to_circuit_metadata_fields(coords)
        np.testing.assert_array_equal(fields["phase_ids"], [0, 0, 0])
        assert fields["phase_names"] == ("memory",)
        np.testing.assert_array_equal(fields["patch_ids"], [0, 0, 0])
        np.testing.assert_array_equal(fields["seam_mask"], [False, False, False])

    def test_cnot_multi_phase(self) -> None:
        import numpy as np

        from sampling.logical_ops import LogicalOperationMetadata

        coords = np.array(
            [
                [0.0, 0.0, 0.5],
                [1.0, 1.0, 1.5],
                [2.0, 2.0, 2.5],
                [3.0, 3.0, 3.5],
            ],
            dtype=np.float64,
        )
        meta = LogicalOperationMetadata(
            phase_windows=(
                PhaseWindow(name="memory", t_start=0.0, t_end=1.0),
                PhaseWindow(name="merge", t_start=1.0, t_end=2.0),
                PhaseWindow(name="memory", t_start=2.0, t_end=3.0),
                PhaseWindow(name="memory", t_start=3.0, t_end=4.0),
            ),
            seam_detector_indices=(1,),
            patch_ids=(0, 1, 0, 1),
            num_blocks=2,
            observables=("obs_0", "obs_1"),
        )
        fields = meta.to_circuit_metadata_fields(coords)
        np.testing.assert_array_equal(fields["phase_ids"], [0, 1, 2, 3])
        assert fields["phase_names"] == ("memory", "merge", "memory", "memory")
        np.testing.assert_array_equal(fields["patch_ids"], [0, 1, 0, 1])
        np.testing.assert_array_equal(fields["seam_mask"], [False, True, False, False])


class TestExtractLogicalOpCircuitMetadata:
    """End-to-end: circuit + metadata => CircuitMetadata with logical-op fields."""

    def test_cnot_produces_populated_circuit_metadata(self) -> None:
        from sampling.graph import CircuitMetadata
        from sampling.logical_ops import (
            LogicalOperationMetadata,
            extract_logical_op_circuit_metadata,
        )

        _, circuit, raw_meta = _cnot_circuit_and_metadata()
        lo_meta = LogicalOperationMetadata(
            phase_windows=tuple(raw_meta["phase_windows"]),
            seam_detector_indices=tuple(raw_meta["seam_detector_indices"]),
            patch_ids=tuple(raw_meta["patch_ids"]),
            num_blocks=raw_meta["num_blocks"],
            observables=("obs_0", "obs_1"),
        )
        cm = extract_logical_op_circuit_metadata(
            circuit,
            distance=3,
            rounds=circuit.num_detectors,
            logical_op_metadata=lo_meta,
        )
        assert isinstance(cm, CircuitMetadata)
        assert cm.phase_ids is not None
        assert cm.phase_names is not None
        assert cm.patch_ids is not None
        assert cm.seam_mask is not None
        assert cm.phase_ids.shape == (cm.num_detectors,)
        assert cm.patch_ids.shape == (cm.num_detectors,)
        assert cm.seam_mask.shape == (cm.num_detectors,)
        assert cm.seam_mask.any()


class TestCircuitMetadataLogicalOpFields:
    """CircuitMetadata optional logical-operation fields."""

    def test_none_by_default(self) -> None:
        import numpy as np

        from sampling.graph import CircuitMetadata

        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        cm = CircuitMetadata(
            detector_coords=coords,
            distance=3,
            rounds=3,
            num_detectors=1,
            dem_edge_weights=np.zeros((1, 1), dtype=np.float64),
        )
        assert cm.phase_ids is None
        assert cm.phase_names is None
        assert cm.patch_ids is None
        assert cm.seam_mask is None

    def test_all_set_valid(self) -> None:
        import numpy as np

        from sampling.graph import CircuitMetadata

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        cm = CircuitMetadata(
            detector_coords=coords,
            distance=3,
            rounds=3,
            num_detectors=2,
            dem_edge_weights=np.zeros((2, 2), dtype=np.float64),
            phase_ids=np.array([0, 0], dtype=np.intp),
            phase_names=("memory",),
            patch_ids=np.array([0, 0], dtype=np.intp),
            seam_mask=np.array([False, False], dtype=np.bool_),
        )
        assert cm.phase_ids is not None
        assert cm.phase_names == ("memory",)

    def test_partial_set_raises(self) -> None:
        import numpy as np

        from sampling.graph import CircuitMetadata

        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="all-or-nothing"):
            CircuitMetadata(
                detector_coords=coords,
                distance=3,
                rounds=3,
                num_detectors=1,
                dem_edge_weights=np.zeros((1, 1), dtype=np.float64),
                phase_ids=np.array([0], dtype=np.intp),
            )

    def test_wrong_phase_ids_shape_raises(self) -> None:
        import numpy as np

        from sampling.graph import CircuitMetadata

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phase_ids shape"):
            CircuitMetadata(
                detector_coords=coords,
                distance=3,
                rounds=3,
                num_detectors=2,
                dem_edge_weights=np.zeros((2, 2), dtype=np.float64),
                phase_ids=np.array([0], dtype=np.intp),
                phase_names=("memory",),
                patch_ids=np.array([0, 0], dtype=np.intp),
                seam_mask=np.array([False, False], dtype=np.bool_),
            )

    def test_phase_id_out_of_range_raises(self) -> None:
        import numpy as np

        from sampling.graph import CircuitMetadata

        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phase_ids values"):
            CircuitMetadata(
                detector_coords=coords,
                distance=3,
                rounds=3,
                num_detectors=1,
                dem_edge_weights=np.zeros((1, 1), dtype=np.float64),
                phase_ids=np.array([5], dtype=np.intp),
                phase_names=("memory",),
                patch_ids=np.array([0], dtype=np.intp),
                seam_mask=np.array([False], dtype=np.bool_),
            )
