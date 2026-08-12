"""Tests for CorrelatedMatchingDecoder.

Verifies the Decoder protocol, bit-identity with MWPM when correlations are
disabled (the control), and that correlated matching is at least as accurate
as plain MWPM on a committed circuit - the mechanism Fowler (arXiv:1310.0863)
describes and Higgott & Gidney (arXiv:2303.15933) implement in PyMatching 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymatching
import pytest
import stim

from decoders import CorrelatedMatchingDecoder, Decoder, PyMatchingDecoder


CIRCUIT_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits" / "memory"
# p=0.01 gives enough errors for a meaningful comparison at d=3.
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


class TestCorrelatedMatchingDecoder:
    """CorrelatedMatchingDecoder satisfies the Decoder protocol and improves
    over (or matches) plain MWPM via two-pass correlated matching."""

    def test_satisfies_decoder_protocol(self, d3_circuit_path: Path) -> None:
        decoder = CorrelatedMatchingDecoder(d3_circuit_path)
        assert isinstance(decoder, Decoder)

    def test_output_shape(
        self,
        d3_circuit_path: Path,
        d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Output shape is (N, num_observables)."""
        syndromes, observables = d3_syndromes_and_obs
        decoder = CorrelatedMatchingDecoder(d3_circuit_path)
        predictions = decoder.decode_batch(syndromes)
        assert predictions.shape == (N_SHOTS, observables.shape[1])
        assert predictions.dtype == np.uint8

    def test_mwpm_equivalence_without_correlations(
        self,
        d3_circuit_path: Path,
        d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """With correlations disabled at decode time, reproduces MWPM exactly.

        This is the control: the only difference between CorrelatedMatchingDecoder
        and PyMatchingDecoder is the second correlation pass. When that pass is
        disabled, the two must be bit-identical on every shot.
        """
        syndromes, _ = d3_syndromes_and_obs

        mwpm = PyMatchingDecoder(d3_circuit_path)
        mwpm_preds = mwpm.decode_batch(syndromes)

        # Build a correlated Matching but decode without the correlation pass.
        circuit = stim.Circuit.from_file(str(d3_circuit_path))
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=True
        )
        corr_no_pass = matching.decode_batch(syndromes, enable_correlations=False)
        corr_no_pass = corr_no_pass[:, : dem.num_observables].astype(
            np.uint8, copy=False
        )

        np.testing.assert_array_equal(
            mwpm_preds,
            corr_no_pass,
            err_msg="Correlated Matching with correlations disabled must be "
            "bit-identical to plain MWPM",
        )

    def test_correlated_at_least_as_good_as_mwpm(
        self,
        d3_circuit_path: Path,
        d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Correlated matching LER <= MWPM LER on a surface code circuit.

        Fowler (arXiv:1310.0863) showed correlated matching exploits Y-error
        decompositions to improve thresholds from ~10.3% to ~10.6% on the
        rotated surface code. Higgott & Gidney (arXiv:2303.15933, Table I)
        confirmed this in PyMatching 2. At p=0.01 (sub-threshold for d=3),
        the correlated decoder must not be worse than plain MWPM.
        """
        syndromes, observables = d3_syndromes_and_obs

        mwpm = PyMatchingDecoder(d3_circuit_path)
        correlated = CorrelatedMatchingDecoder(d3_circuit_path)

        mwpm_preds = mwpm.decode_batch(syndromes)
        corr_preds = correlated.decode_batch(syndromes)

        mwpm_errors = np.any(mwpm_preds != observables, axis=1).sum()
        corr_errors = np.any(corr_preds != observables, axis=1).sum()

        assert corr_errors <= mwpm_errors, (
            f"Correlated matching ({corr_errors}/{N_SHOTS}) should not be "
            f"worse than MWPM ({mwpm_errors}/{N_SHOTS})"
        )
