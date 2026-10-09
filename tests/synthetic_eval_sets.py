"""Synthetic frozen evaluation sets for tests that score stand-in decoders.

The shots are whatever the test supplies.  No circuit stands behind them: the
metadata places every detector at the origin with zero DEM weights, so a set
built here serves the harness and the statistics, never a graph builder or a
circuit-backed decoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluation.evaluator import EvalSet
from sampling.experiment import ExperimentKey
from sampling.graph import CircuitMetadata


MEMORY_D3 = ExperimentKey(operation="memory", distance=3, rounds=3, error_prob=0.01)


def synthetic_eval_set(
    syndromes: NDArray[np.uint8],
    observables: NDArray[np.uint8],
    *,
    key: ExperimentKey,
    observable_names: tuple[str, ...],
) -> EvalSet:
    """An eval set over the given shots, at ``key``, with placeholder metadata."""
    num_detectors = syndromes.shape[1]
    return EvalSet(
        key=key,
        syndromes=syndromes,
        observables=observables,
        observable_names=observable_names,
        circuit_file=Path("synthetic.stim"),
        circuit_sha256="0" * 64,
        circuit_metadata=CircuitMetadata(
            detector_coords=np.zeros((num_detectors, 3)),
            distance=key.distance,
            rounds=key.rounds,
            num_detectors=num_detectors,
            dem_edge_weights=np.zeros((num_detectors, num_detectors)),
        ),
        manifest={},
    )
