"""Tests for src/sampling/circuit_metadata.py: layouts, coordinates, extractors."""

from __future__ import annotations

import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import stim

from sampling.circuit_metadata import (
    PhaseWindow,
    assign_phase_ids,
    build_layout,
    read_circuit_metadata,
    read_logical_op_metadata,
)
from sampling.experiment import circuit_key, circuit_manifest_path
from sampling.graph import LogicalOperationLayout, read_detector_coordinates
from sampling.profile import registered_operations, resolve_profile


REPO_ROOT = Path(__file__).resolve().parent.parent
MERGE_SPLIT_CIRCUITS = sorted(
    (REPO_ROOT / "data/circuits/zz_merge_split").glob("*.stim"), key=str
)
MERGE_SPLIT_EVAL_SETS = sorted((REPO_ROOT / "data/eval/zz_merge_split").iterdir())
COMMITTED_CIRCUITS = sorted((REPO_ROOT / "data/circuits").glob("*/*.stim"), key=str)
FROZEN_SETS = sorted(
    [
        *(REPO_ROOT / "data/eval").glob("*/*/"),
        *(REPO_ROOT / "data/ci_shard").glob("*/"),
    ],
    key=str,
)

_WINDOWS = (
    PhaseWindow("memory", 0.0, 1.0),
    PhaseWindow("merge", 1.0, 2.0),
    PhaseWindow("memory", 2.0, 3.0),
)


def _reference_phase_ids(
    windows: tuple[PhaseWindow, ...], times: np.ndarray
) -> list[int]:
    """Brute force: the index of the one window whose half-open span holds t."""
    phase_ids = []
    for t in times:
        holding = [i for i, window in enumerate(windows) if window.contains_time(t)]
        assert len(holding) == 1
        phase_ids.append(holding[0])
    return phase_ids


def _layout(
    *,
    seam_detector_indices: list[int] | None = None,
    patch_ids: list[float] | None = None,
    num_blocks: int = 3,
) -> LogicalOperationLayout:
    """Build a four-detector layout over ``_WINDOWS``, one field replaced."""
    return build_layout(
        _WINDOWS,
        seam_detector_indices=[1]
        if seam_detector_indices is None
        else seam_detector_indices,
        patch_ids=[0, 1, 2, 0] if patch_ids is None else patch_ids,
        num_blocks=num_blocks,
        detector_times=np.array([0.5, 1.5, 2.5, 0.0]),
    )


class TestAssignPhaseIds:
    def test_matches_brute_force_on_random_times(self) -> None:
        rng = np.random.default_rng(0)
        times = np.concatenate([rng.uniform(0.0, 3.0, 500), [0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(
            assign_phase_ids(_WINDOWS, times), _reference_phase_ids(_WINDOWS, times)
        )

    def test_boundaries_are_half_open(self) -> None:
        phase_ids = assign_phase_ids(_WINDOWS, np.array([0.0, 1.0, 2.0, 2.999]))
        np.testing.assert_array_equal(phase_ids, [0, 1, 2, 2])
        assert phase_ids.dtype == np.intp

    @pytest.mark.parametrize("t", [-0.5, 3.0, 7.0, float("nan")])
    def test_uncovered_detector_named_by_index_and_time(self, t: float) -> None:
        times = np.array([0.5, 1.5, t, 2.5])
        with pytest.raises(ValueError, match=rf"detector 2 at t={t} is not covered"):
            assign_phase_ids(_WINDOWS, times)

    def test_gap_between_windows_raises(self) -> None:
        windows = (PhaseWindow("memory", 0.0, 1.0), PhaseWindow("memory", 2.0, 3.0))
        with pytest.raises(ValueError, match="not contiguous"):
            assign_phase_ids(windows, np.array([0.5]))

    def test_no_windows_raises(self) -> None:
        with pytest.raises(ValueError, match="no phase windows"):
            assign_phase_ids((), np.array([0.5]))

    def test_no_detectors_is_empty(self) -> None:
        assert assign_phase_ids(_WINDOWS, np.empty(0)).shape == (0,)


def _declared_coordinates(circuit: stim.Circuit) -> list[list[float]]:
    """Reference: each DETECTOR's arguments in the flattened, shift-free circuit."""
    return [
        instruction.gate_args_copy()
        for instruction in circuit.flattened()
        if isinstance(instruction, stim.CircuitInstruction)
        and instruction.name == "DETECTOR"
    ]


def _frozen_coordinates(frozen_dir: Path) -> np.ndarray:
    """The detector coordinates a frozen eval set or CI shard stored."""
    if (frozen_dir / "data.npz").exists():
        return np.load(frozen_dir / "data.npz")["detector_coords"]
    return np.load(frozen_dir / "detector_coords.npy")


class TestReadDetectorCoordinates:
    def test_reads_shifted_coordinates_in_detector_order(self) -> None:
        circuit = stim.Circuit("""
            M 0 1 2
            DETECTOR(1, 2, 0) rec[-1]
            REPEAT 2 {
                DETECTOR(3, 4, 0) rec[-2]
                SHIFT_COORDS(0, 0, 1)
            }
        """)
        coords = read_detector_coordinates(circuit)
        assert coords.dtype == np.float64
        np.testing.assert_array_equal(coords, [[1, 2, 0], [3, 4, 0], [3, 4, 1]])

    @pytest.mark.parametrize(
        ("declaration", "declared"),
        [
            ("DETECTOR", "0 coordinate dimensions ()"),
            ("DETECTOR(1, 2)", "2 coordinate dimensions (1.0, 2.0)"),
            ("DETECTOR(1, 2, 3, 4)", "4 coordinate dimensions (1.0, 2.0, 3.0, 4.0)"),
        ],
    )
    def test_detector_without_exactly_xyt_raises_naming_it(
        self, declaration: str, declared: str
    ) -> None:
        circuit = stim.Circuit(f"""
            M 0 1 2
            DETECTOR(0, 0, 0) rec[-1]
            {declaration} rec[-2]
            DETECTOR(0, 0, 1) rec[-3]
        """)
        with pytest.raises(
            ValueError,
            match=rf"detector 1 declares {re.escape(declared)}, expected 3: "
            r"\(x, y, t\); 1 of 3 detectors are malformed",
        ):
            read_detector_coordinates(circuit)

    def test_no_detectors_is_empty(self) -> None:
        assert read_detector_coordinates(stim.Circuit("M 0")).shape == (0, 3)

    @pytest.mark.parametrize(
        "circuit_path",
        COMMITTED_CIRCUITS,
        ids=lambda path: f"{path.parent.name}/{path.stem}",
    )
    def test_committed_circuit_reads_what_its_detectors_declare(
        self, circuit_path: Path
    ) -> None:
        circuit = stim.Circuit.from_file(str(circuit_path))
        np.testing.assert_array_equal(
            read_detector_coordinates(circuit),
            np.array(_declared_coordinates(circuit), dtype=np.float64),
            strict=True,
        )

    @pytest.mark.parametrize(
        "frozen_dir",
        FROZEN_SETS,
        ids=lambda path: f"{path.parent.name}/{path.name}",
    )
    def test_frozen_set_stored_its_circuits_coordinates(self, frozen_dir: Path) -> None:
        manifest = json.loads((frozen_dir / "manifest.json").read_text())
        circuit = stim.Circuit.from_file(str(REPO_ROOT / manifest["circuit_file"]))
        np.testing.assert_array_equal(
            read_detector_coordinates(circuit),
            _frozen_coordinates(frozen_dir),
            strict=True,
        )


class TestBuildLayout:
    def test_valid(self) -> None:
        layout = _layout()
        np.testing.assert_array_equal(layout.phase_ids, [0, 1, 2, 0])
        assert layout.phase_names == ("memory", "merge", "memory")
        np.testing.assert_array_equal(layout.patch_ids, [0, 1, 2, 0])
        np.testing.assert_array_equal(layout.seam_mask, [False, True, False, False])

    def test_merge_without_seam_raises(self) -> None:
        with pytest.raises(ValueError, match="seam_detector_indices is empty"):
            _layout(seam_detector_indices=[])

    @pytest.mark.parametrize("index", [4, -1])
    def test_seam_index_out_of_range_raises(self, index: int) -> None:
        with pytest.raises(ValueError, match=rf"index {index} out of range \[0, 4\)"):
            _layout(seam_detector_indices=[index])

    def test_seam_outside_merge_raises(self) -> None:
        with pytest.raises(ValueError, match="seam detector 0 at t=0.5 is not inside"):
            _layout(seam_detector_indices=[1, 0])

    def test_patch_ids_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"patch_ids length \(3\)"):
            _layout(patch_ids=[0, 1, 2])

    @pytest.mark.parametrize(
        ("patch_ids", "num_blocks"),
        [([0, 1, 3, 0], 3), ([0, 1, 2, 0], 4), ([0, -1, 1, 0], 2)],
    )
    def test_patch_ids_not_exactly_the_blocks_raises(
        self, patch_ids: list[int], num_blocks: int
    ) -> None:
        with pytest.raises(ValueError, match="distinct patches"):
            _layout(patch_ids=patch_ids, num_blocks=num_blocks)

    def test_non_integer_indices_raise(self) -> None:
        with pytest.raises(ValueError, match="must hold integers"):
            _layout(patch_ids=[0.0, 1.0, 2.0, 0.5])

    def test_phase_name_outside_vocabulary_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the phase vocabulary"):
            build_layout(
                (PhaseWindow("hold", 0.0, 1.0),),
                seam_detector_indices=[],
                patch_ids=[0],
                num_blocks=1,
                detector_times=np.array([0.5]),
            )


class TestCommittedMergeSplitMetadata:
    @pytest.mark.parametrize(
        "eval_dir", MERGE_SPLIT_EVAL_SETS, ids=lambda path: path.name
    )
    def test_matches_the_layout_frozen_in_its_eval_set(self, eval_dir: Path) -> None:
        """Every frozen eval set stored the layout its circuit was read with."""
        manifest = json.loads((eval_dir / "manifest.json").read_text())
        frozen = np.load(eval_dir / "data.npz")
        metadata = read_logical_op_metadata(REPO_ROOT / manifest["circuit_file"])

        assert metadata.layout is not None
        np.testing.assert_array_equal(
            metadata.detector_coords, frozen["detector_coords"]
        )
        np.testing.assert_array_equal(metadata.layout.phase_ids, frozen["phase_ids"])
        np.testing.assert_array_equal(metadata.layout.patch_ids, frozen["patch_ids"])
        np.testing.assert_array_equal(metadata.layout.seam_mask, frozen["seam_mask"])
        assert metadata.layout.phase_names == tuple(manifest["phase_names"])

    @pytest.mark.parametrize(
        "circuit_path", MERGE_SPLIT_CIRCUITS, ids=lambda path: path.stem
    )
    def test_matches_a_brute_force_read_of_its_manifest(
        self, circuit_path: Path
    ) -> None:
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        windows = tuple(
            PhaseWindow(w["name"], w["t_start"], w["t_end"])
            for w in manifest["phase_windows"]
        )
        key = circuit_key(circuit_path)
        metadata = read_logical_op_metadata(circuit_path)

        assert (metadata.distance, metadata.rounds) == (key.distance, key.rounds)
        assert metadata.layout is not None
        np.testing.assert_array_equal(
            metadata.layout.phase_ids,
            _reference_phase_ids(windows, metadata.detector_coords[:, 2]),
        )
        np.testing.assert_array_equal(metadata.layout.patch_ids, manifest["patch_ids"])
        np.testing.assert_array_equal(
            np.flatnonzero(metadata.layout.seam_mask),
            sorted(set(manifest["seam_detector_indices"])),
        )


def _copy_circuit(circuit_path: Path, destination: Path) -> Path:
    """Copy a committed circuit and its manifest into *destination*."""
    copied = destination / circuit_path.name
    shutil.copyfile(circuit_path, copied)
    shutil.copyfile(circuit_manifest_path(circuit_path), circuit_manifest_path(copied))
    return copied


def _edit_manifest(circuit_path: Path, **fields: object) -> None:
    manifest_path = circuit_manifest_path(circuit_path)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(fields)
    manifest_path.write_text(json.dumps(manifest))


class TestResolutionByIdentity:
    def test_each_circuit_reads_its_own_manifest(self, tmp_path: Path) -> None:
        """Two p variants of one (d, r) each get the layout beside them."""
        first, second = (
            _copy_circuit(path, tmp_path) for path in MERGE_SPLIT_CIRCUITS[:2]
        )
        assert circuit_key(first).distance == circuit_key(second).distance
        manifest = json.loads(circuit_manifest_path(second).read_text())
        seam = manifest["seam_detector_indices"]
        _edit_manifest(second, seam_detector_indices=seam[1:])

        first_layout = read_logical_op_metadata(first).layout
        second_layout = read_logical_op_metadata(second).layout
        assert first_layout is not None and second_layout is not None
        assert np.flatnonzero(first_layout.seam_mask).tolist() == sorted(seam)
        assert np.flatnonzero(second_layout.seam_mask).tolist() == sorted(seam[1:])

    def test_unmanifested_circuit_raises(self, tmp_path: Path) -> None:
        circuit_path = tmp_path / "unlabelled.stim"
        shutil.copyfile(MERGE_SPLIT_CIRCUITS[0], circuit_path)
        with pytest.raises(FileNotFoundError, match="has no manifest"):
            read_logical_op_metadata(circuit_path)

    def test_missing_layout_field_raises_naming_the_key(self, tmp_path: Path) -> None:
        circuit_path = _copy_circuit(MERGE_SPLIT_CIRCUITS[0], tmp_path)
        manifest_path = circuit_manifest_path(circuit_path)
        manifest = json.loads(manifest_path.read_text())
        del manifest["patch_ids"]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(
            ValueError, match=rf"{circuit_key(circuit_path)}: .* \['patch_ids'\]"
        ):
            read_logical_op_metadata(circuit_path)

    def test_uncovered_detector_raises_naming_key_and_detector(
        self, tmp_path: Path
    ) -> None:
        circuit_path = _copy_circuit(MERGE_SPLIT_CIRCUITS[0], tmp_path)
        manifest = json.loads(circuit_manifest_path(circuit_path).read_text())
        windows = manifest["phase_windows"]
        windows[0]["t_start"] = 0.5
        _edit_manifest(circuit_path, phase_windows=windows)
        times = read_circuit_metadata(circuit_path).detector_coords[:, 2]
        first = int(np.flatnonzero(times < 0.5)[0])
        with pytest.raises(
            ValueError,
            match=rf"{circuit_key(circuit_path)}: .*detector {first} at t=0.0 is not",
        ):
            read_logical_op_metadata(circuit_path)


class TestDetectorWithoutCoordinates:
    @pytest.mark.parametrize("operation", sorted(registered_operations()))
    def test_raises_naming_the_detector_and_the_circuit(
        self, operation: str, tmp_path: Path
    ) -> None:
        profile = resolve_profile(operation)
        committed = sorted((REPO_ROOT / profile.circuit_root).glob("*.stim"))[0]
        circuit_path = _copy_circuit(committed, tmp_path)
        text = circuit_path.read_text()
        first = text.index("DETECTOR(")
        circuit_path.write_text(
            text[:first] + "DETECTOR" + text[text.index(")", first) + 1 :]
        )
        with pytest.raises(
            ValueError,
            match=rf"{circuit_key(circuit_path)}: circuit "
            rf"{re.escape(str(circuit_path))}: detector 0 declares 0 coordinate",
        ):
            profile.metadata_extractor(circuit_path)


class TestRegisteredExtractors:
    def test_memory_reads_distance_and_rounds_from_its_manifest(self) -> None:
        circuit_path = REPO_ROOT / "data/circuits/memory/d5_r5_p0_01.stim"
        metadata = read_circuit_metadata(circuit_path)
        assert (metadata.distance, metadata.rounds) == (5, 5)
        assert metadata.layout is None

    @pytest.mark.parametrize("operation", sorted(registered_operations()))
    def test_extractor_pickles_for_spawned_workers(self, operation: str) -> None:
        extractor = resolve_profile(operation).metadata_extractor
        assert pickle.loads(pickle.dumps(extractor)) is extractor

    @pytest.mark.parametrize("operation", sorted(registered_operations()))
    def test_committed_circuits_carry_the_profiles_observables(
        self, operation: str
    ) -> None:
        profile = resolve_profile(operation)
        for circuit_path in sorted((REPO_ROOT / profile.circuit_root).glob("*.stim")):
            circuit = stim.Circuit.from_file(str(circuit_path))
            assert circuit.num_observables == len(profile.observable_names), (
                circuit_path
            )
