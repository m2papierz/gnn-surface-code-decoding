"""Tests for TesseractDecoder.

Verifies the Decoder protocol, output shape, and that the near-MLE decoder
is at least as accurate as plain MWPM on a committed surface-code circuit -
the expected ordering for a search-based decoder that approaches maximum
likelihood (Fowler et al., arXiv:2503.10988).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import stim

from decoders import Decoder, PyMatchingDecoder, TesseractDecoder


CIRCUIT_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits" / "memory"
D3_CIRCUIT = CIRCUIT_DIR / "d3_r3_p0_01.stim"
N_SHOTS = 10_000
SEED = 42


@pytest.fixture(scope="module")
def d3_circuit_path() -> Path:
    if not D3_CIRCUIT.exists():
        pytest.skip("committed d3 circuit not found")
    return D3_CIRCUIT


@pytest.fixture(scope="module")
def d3_syndromes_and_obs(d3_circuit_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sample syndromes and observables once for the module."""
    circuit = stim.Circuit.from_file(str(d3_circuit_path))
    sampler = circuit.compile_detector_sampler(seed=SEED)
    syndromes, observables = sampler.sample(shots=N_SHOTS, separate_observables=True)
    return syndromes.astype(np.uint8), observables.astype(np.uint8)


class TestTesseractDecoder:
    """TesseractDecoder satisfies the Decoder protocol and achieves near-MLE
    accuracy (at least as good as MWPM)."""

    def test_satisfies_decoder_protocol(self, d3_circuit_path: Path) -> None:
        decoder = TesseractDecoder(d3_circuit_path)
        assert isinstance(decoder, Decoder)

    def test_output_shape(
        self,
        d3_circuit_path: Path,
        d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Output shape is (N, num_observables) with uint8 dtype."""
        syndromes, observables = d3_syndromes_and_obs
        decoder = TesseractDecoder(d3_circuit_path)
        predictions = decoder.decode_batch(syndromes)
        assert predictions.shape == (N_SHOTS, observables.shape[1])
        assert predictions.dtype == np.uint8

    def test_predictions_are_binary(
        self,
        d3_circuit_path: Path,
        d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Every prediction value is 0 or 1."""
        syndromes, _ = d3_syndromes_and_obs
        decoder = TesseractDecoder(d3_circuit_path)
        predictions = decoder.decode_batch(syndromes)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_at_least_as_good_as_mwpm(
        self,
        d3_circuit_path: Path,
        d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Tesseract LER <= MWPM LER on a surface code circuit.

        Tesseract performs near-MLE beam search (Fowler et al.,
        arXiv:2503.10988), so its logical error rate must not exceed that
        of minimum-weight perfect matching.
        """
        syndromes, observables = d3_syndromes_and_obs

        mwpm = PyMatchingDecoder(d3_circuit_path)
        tesseract = TesseractDecoder(d3_circuit_path)

        mwpm_preds = mwpm.decode_batch(syndromes)
        tess_preds = tesseract.decode_batch(syndromes)

        mwpm_errors = np.any(mwpm_preds != observables, axis=1).sum()
        tess_errors = np.any(tess_preds != observables, axis=1).sum()

        assert tess_errors <= mwpm_errors, (
            f"Tesseract ({tess_errors}/{N_SHOTS}) should not be "
            f"worse than MWPM ({mwpm_errors}/{N_SHOTS})"
        )

    def test_import_error_when_unavailable(self) -> None:
        """Construction raises ImportError with a helpful message when the
        package is not installed, rather than silently substituting."""
        import sys
        import unittest.mock

        blocked = {
            "tesseract_decoder": None,
            "tesseract_decoder.tesseract": None,
        }
        with unittest.mock.patch.dict(sys.modules, blocked):
            with pytest.raises(ImportError, match="tesseract-decoder"):
                TesseractDecoder(D3_CIRCUIT)
