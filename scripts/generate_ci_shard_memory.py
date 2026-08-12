"""Generate a small frozen CI data shard for integration tests.

Produces pre-sampled syndromes and observable flips for d=3 at a single
error probability, along with detector metadata.  The resulting files
can be loaded by tests without any Stim dependency.

Usage
-----
    uv run python scripts/generate_ci_shard_memory.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from sampling.experiment import circuit_key
from sampling.graph import extract_circuit_metadata


CIRCUIT_PATH = Path("data/circuits/memory/d3_r3_p0_01.stim")
OUTPUT_ROOT = Path("data/ci_shard")
SEED = 20240101
NUM_SHOTS = 256
DISTANCE = 3
ROUNDS = 3
ERROR_PROB = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # The shard inherits the identity of the circuit it is sampled from, and
    # is placed under that operation's root rather than a path literal.
    key = circuit_key(
        CIRCUIT_PATH, distance=DISTANCE, rounds=ROUNDS, error_prob=ERROR_PROB
    )
    output_dir = OUTPUT_ROOT / key.operation

    circuit = stim.Circuit.from_file(str(CIRCUIT_PATH))
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = circuit.compile_detector_sampler(seed=SEED)
    syndromes, observables = sampler.sample(
        shots=NUM_SHOTS, separate_observables=True, bit_packed=False
    )
    syndromes = syndromes.astype(np.uint8)
    observables = observables.astype(np.uint8)

    meta = extract_circuit_metadata(circuit, distance=key.distance, rounds=key.rounds)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "syndromes.npy", syndromes)
    np.save(output_dir / "observables.npy", observables)
    np.save(output_dir / "detector_coords.npy", meta.detector_coords)

    manifest = {
        "circuit_file": str(CIRCUIT_PATH),
        "circuit_sha256": _sha256(CIRCUIT_PATH),
        "stim_version": stim.__version__,
        "seed": SEED,
        "num_shots": NUM_SHOTS,
        "operation": key.operation,
        "distance": key.distance,
        "rounds": key.rounds,
        "error_prob": key.error_prob,
        "num_detectors": int(dem.num_detectors),
        "num_observables": int(dem.num_observables),
        "syndromes_shape": list(syndromes.shape),
        "observables_shape": list(observables.shape),
        "positive_count": int(observables.any(axis=1).sum()),
        "generation_command": "uv run python scripts/generate_ci_shard_memory.py",
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"CI shard written to {output_dir}/")
    print(f"  syndromes: {syndromes.shape}")
    print(f"  observables: {observables.shape}")
    print(f"  positives: {manifest['positive_count']}/{NUM_SHOTS}")


if __name__ == "__main__":
    main()
