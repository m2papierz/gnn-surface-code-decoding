"""Tests for committed Stim circuit files.

Every circuit file must load via ``stim.Circuit.from_file()`` and produce a
valid DEM via ``.detector_error_model()``.  Settings: d∈{3,5,7}, r=d,
p∈{0.003, 0.005, 0.008, 0.01}.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import stim


CIRCUITS_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits"

DISTANCES = [3, 5, 7]
ERROR_PROBS = [0.003, 0.005, 0.008, 0.01]


def _circuit_path(d: int, r: int, p: float) -> Path:
    p_tag = f"p{p:.6g}".replace(".", "_")
    return CIRCUITS_DIR / f"d{d}_r{r}_{p_tag}.stim"


def _expected_detectors(d: int, r: int) -> int:
    """Expected detector count for rotated_memory_x: d²-1 stabilizers × r rounds."""
    return (d * d - 1) * r


_SETTINGS = [(d, d, p) for d in DISTANCES for p in ERROR_PROBS]


class TestCircuitLoad:
    """Each circuit loads and produces a valid DEM."""

    @pytest.mark.parametrize("d, r, p", _SETTINGS)
    def test_load_and_dem(self, d: int, r: int, p: float) -> None:
        path = _circuit_path(d, r, p)
        circuit = stim.Circuit.from_file(str(path))
        dem = circuit.detector_error_model(decompose_errors=True)

        assert dem.num_detectors > 0
        assert dem.num_observables == 1

    @pytest.mark.parametrize("d, r, p", _SETTINGS)
    def test_detector_count(self, d: int, r: int, p: float) -> None:
        path = _circuit_path(d, r, p)
        circuit = stim.Circuit.from_file(str(path))
        dem = circuit.detector_error_model(decompose_errors=True)

        expected = _expected_detectors(d, r)
        assert dem.num_detectors == expected, (
            f"d={d}, r={r}: expected {expected} detectors, got {dem.num_detectors}"
        )

    @pytest.mark.parametrize("d, r, p", _SETTINGS)
    def test_circuit_can_sample(self, d: int, r: int, p: float) -> None:
        """Circuit compiles a sampler and produces valid shots."""
        path = _circuit_path(d, r, p)
        circuit = stim.Circuit.from_file(str(path))
        dem = circuit.detector_error_model(decompose_errors=True)

        sampler = circuit.compile_detector_sampler(seed=0)
        dets, obs = sampler.sample(shots=10, separate_observables=True)

        assert dets.shape == (10, dem.num_detectors)
        assert obs.shape == (10, dem.num_observables)
