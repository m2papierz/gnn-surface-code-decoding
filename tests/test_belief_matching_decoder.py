"""Tests for BeliefMatchingDecoder.

Verifies the Decoder protocol, output shape, and that Belief-Matching is at
least as accurate as plain MWPM on a committed surface-code circuit: BP's
posteriors re-weight the same matching graph (Higgott, Bohdanowicz, Kubica,
Flammia & Campbell, arXiv:2203.04948).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import stim

from decoders import BeliefMatchingDecoder, Decoder, PyMatchingDecoder


CIRCUIT_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits" / "memory"
# p=0.01 gives enough errors for a meaningful comparison at d=3.
D3_CIRCUIT = CIRCUIT_DIR / "d3_r3_p0_01.stim"
N_SHOTS = 10_000
SEED = 42


@pytest.fixture(scope="module")
def d3_syndromes_and_obs() -> tuple[np.ndarray, np.ndarray]:
    """Sample syndromes and observables once for the module."""
    circuit = stim.Circuit.from_file(str(D3_CIRCUIT))
    sampler = circuit.compile_detector_sampler(seed=SEED)
    syndromes, observables = sampler.sample(shots=N_SHOTS, separate_observables=True)
    return syndromes.astype(np.uint8), observables.astype(np.uint8)


@pytest.fixture(scope="module")
def belief_matching_predictions(
    d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Belief-Matching decodes the module's shots once; it runs shot by shot."""
    syndromes, _ = d3_syndromes_and_obs
    return BeliefMatchingDecoder(D3_CIRCUIT).decode_batch(syndromes)


def test_satisfies_decoder_protocol() -> None:
    assert isinstance(BeliefMatchingDecoder(D3_CIRCUIT), Decoder)


def test_output_shape(
    d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    belief_matching_predictions: np.ndarray,
) -> None:
    """One uint8 row of observable flips per shot."""
    _, observables = d3_syndromes_and_obs
    assert belief_matching_predictions.shape == observables.shape
    assert belief_matching_predictions.dtype == np.uint8
    assert set(np.unique(belief_matching_predictions)) <= {0, 1}


def test_at_least_as_good_as_mwpm(
    d3_syndromes_and_obs: tuple[np.ndarray, np.ndarray],
    belief_matching_predictions: np.ndarray,
) -> None:
    """Belief-Matching makes no more logical errors than MWPM on the same shots."""
    syndromes, observables = d3_syndromes_and_obs
    mwpm_predictions = PyMatchingDecoder(D3_CIRCUIT).decode_batch(syndromes)

    mwpm_errors = np.any(mwpm_predictions != observables, axis=1).sum()
    bm_errors = np.any(belief_matching_predictions != observables, axis=1).sum()

    assert bm_errors <= mwpm_errors, (
        f"Belief-Matching ({bm_errors}/{N_SHOTS}) should not be "
        f"worse than MWPM ({mwpm_errors}/{N_SHOTS})"
    )
