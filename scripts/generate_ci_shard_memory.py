"""Generate a small frozen CI data shard for integration tests.

Produces pre-sampled syndromes and observable flips for d=3 at a single
error probability, bound by the manifest to the circuit they were sampled
from; the loader reads everything else from that circuit.

Usage
-----
    uv run python scripts/generate_ci_shard_memory.py
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import stim

from artifact_file import atomic_write, file_sha256, write_json
from sampling.experiment import circuit_key


CIRCUIT_PATH = Path("data/circuits/memory/d3_r3_p0_01.stim")
OUTPUT_ROOT = Path("data/ci_shard")
SEED = 20240101
NUM_SHOTS = 256


def main() -> None:
    # The shard inherits the identity of the circuit it is sampled from, and
    # is placed under that operation's root rather than a path literal.
    key = circuit_key(CIRCUIT_PATH)
    output_dir = OUTPUT_ROOT / key.operation

    circuit = stim.Circuit.from_file(str(CIRCUIT_PATH))
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = circuit.compile_detector_sampler(seed=SEED)
    syndromes, observables = sampler.sample(
        shots=NUM_SHOTS, separate_observables=True, bit_packed=False
    )
    syndromes = syndromes.astype(np.uint8)
    observables = observables.astype(np.uint8)

    for name, array in (("syndromes", syndromes), ("observables", observables)):
        with atomic_write(output_dir / f"{name}.npy") as temp:
            np.save(temp, array)

    manifest = {
        "circuit_file": str(CIRCUIT_PATH),
        "circuit_sha256": file_sha256(CIRCUIT_PATH),
        "stim_version": version("stim"),
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

    write_json(output_dir / "manifest.json", dict(sorted(manifest.items())))

    print(f"CI shard written to {output_dir}/")
    print(f"  syndromes: {syndromes.shape}")
    print(f"  observables: {observables.shape}")
    print(f"  positives: {manifest['positive_count']}/{NUM_SHOTS}")


if __name__ == "__main__":
    main()
