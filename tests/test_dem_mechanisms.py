"""Tests for the DEM mechanism reader and the ``dem_weight`` feature built on it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import stim

from sampling.dem import MechanismClass, read_dem_mechanisms
from sampling.graph import _extract_dem_weights, dem_weight_mechanisms


REPO_ROOT = Path(__file__).resolve().parent.parent

# One mechanism of every shape the reader has to tell apart.
DEM_TEXT = """
error(0.1) D0 D1
error(0.2) D0
error(0.05) D1 ^ D2
error(0.15) D0 D1 ^ D2 D3
error(0.5) D2 D3 L0
detector(1, 2, 3) D3
"""


def _reference_dem_weights(circuit: stim.Circuit) -> np.ndarray:
    """Pairwise weights by a direct loop over DEM instructions."""
    num_det = circuit.num_detectors
    log_complement = np.zeros((num_det, num_det), dtype=np.float64)
    for instruction in circuit.detector_error_model(decompose_errors=True).flattened():
        if instruction.type != "error":
            continue
        detectors = [
            t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
        ]
        if len(detectors) == 2:
            i, j = detectors
            log_survival = np.log1p(-instruction.args_copy()[0])
            log_complement[i, j] += log_survival
            log_complement[j, i] += log_survival
    return 1.0 - np.exp(log_complement)


class TestReadDemMechanisms:
    def test_rows_match_the_listing(self) -> None:
        mechanisms = read_dem_mechanisms(stim.DetectorErrorModel(DEM_TEXT))

        np.testing.assert_array_equal(
            mechanisms.probability, [0.1, 0.2, 0.05, 0.15, 0.5]
        )
        np.testing.assert_array_equal(mechanisms.num_components, [1, 1, 2, 2, 1])
        np.testing.assert_array_equal(mechanisms.num_detectors, [2, 1, 2, 4, 2])
        np.testing.assert_array_equal(mechanisms.detector_offsets, [0, 2, 3, 5, 9, 11])
        np.testing.assert_array_equal(
            mechanisms.detectors, [0, 1, 0, 1, 2, 0, 1, 2, 3, 2, 3]
        )

    def test_repeat_blocks_are_flattened(self) -> None:
        dem = stim.DetectorErrorModel("""
            repeat 3 {
                error(0.25) D0 D1
                shift_detectors 1
            }
        """)
        mechanisms = read_dem_mechanisms(dem)

        np.testing.assert_array_equal(mechanisms.probability, [0.25, 0.25, 0.25])
        np.testing.assert_array_equal(mechanisms.detectors, [0, 1, 1, 2, 2, 3])

    def test_model_without_errors_is_empty(self) -> None:
        mechanisms = read_dem_mechanisms(stim.DetectorErrorModel("detector D0"))

        assert mechanisms.probability.shape == (0,)
        np.testing.assert_array_equal(mechanisms.detector_offsets, [0])


class TestClassMasks:
    def test_every_mechanism_is_in_exactly_one_class(self) -> None:
        masks = read_dem_mechanisms(stim.DetectorErrorModel(DEM_TEXT)).class_masks()

        np.testing.assert_array_equal(
            masks[MechanismClass.TWO_DETECTOR], [True, False, False, False, True]
        )
        np.testing.assert_array_equal(
            masks[MechanismClass.BOUNDARY], [False, True, False, False, False]
        )
        np.testing.assert_array_equal(
            masks[MechanismClass.DECOMPOSED], [False, False, True, True, False]
        )

    @pytest.mark.parametrize(
        "instruction", ["error(0.1) D0 D1 D2", "error(0.1) L0"], ids=["three", "none"]
    )
    def test_undecomposed_non_graphlike_mechanism_raises(
        self, instruction: str
    ) -> None:
        mechanisms = read_dem_mechanisms(
            stim.DetectorErrorModel(f"error(0.2) D0\n{instruction}")
        )

        with pytest.raises(ValueError, match="not graphlike.*mechanism 1"):
            mechanisms.class_masks()


class TestDemWeightFeature:
    def test_selects_mechanisms_flipping_exactly_two_detectors(self) -> None:
        mechanisms = read_dem_mechanisms(stim.DetectorErrorModel(DEM_TEXT))

        # The decomposed ``D1 ^ D2`` is two boundary components and is in;
        # the boundary mechanism and the four-detector decomposition are out.
        np.testing.assert_array_equal(
            dem_weight_mechanisms(mechanisms), [True, False, True, False, True]
        )

    def test_mechanisms_on_one_pair_combine_as_independent_events(self) -> None:
        # Two mechanisms flip the same detector pair and differ only in the
        # observable, so stim keeps them apart: D0 D1 L0 and D0 D1.
        circuit = stim.Circuit("""
            X_ERROR(0.1) 0
            X_ERROR(0.2) 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(0) rec[-2]
        """)
        assert circuit.detector_error_model(decompose_errors=True).num_errors == 2

        weights = _extract_dem_weights(circuit)

        # log1p, exp and the subtraction each round once; the result sits
        # 4e-16 from the product form, and 1e-12 leaves room for another libm.
        expected = 1.0 - 0.9 * 0.8
        np.testing.assert_allclose(
            weights, [[0.0, expected], [expected, 0.0]], rtol=1e-12
        )

    @pytest.mark.parametrize(
        "circuit_file",
        [
            "data/circuits/memory/d3_r3_p0_01.stim",
            "data/circuits/memory/d5_r5_p0_003.stim",
            "data/circuits/zz_merge_split/d3_r12_p0_003.stim",
        ],
    )
    def test_matches_direct_loop_bit_for_bit(self, circuit_file: str) -> None:
        circuit = stim.Circuit.from_file(str(REPO_ROOT / circuit_file))

        weights = _extract_dem_weights(circuit)

        assert np.array_equal(weights, _reference_dem_weights(circuit))
        assert np.array_equal(weights, weights.T)
        assert weights.any()
