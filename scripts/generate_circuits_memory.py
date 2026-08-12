"""Generate and commit Stim circuit files for all training/eval settings.

Settings: d∈{3,5,7}, r=d, p∈{0.003, 0.005, 0.008, 0.01}.

Each circuit is committed with a manifest beside it declaring the experiment
it belongs to.  A circuit that already exists is never rewritten: its manifest
is derived from the committed bytes, because a regenerated circuit is a new
circuit and every number measured on the old one would lose its provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Final

import stim

from sampling.experiment import (
    ExperimentKey,
    circuit_manifest_path,
    write_circuit_manifest,
)


logger = logging.getLogger(__name__)

DISTANCES: list[int] = [3, 5, 7]
ERROR_PROBS: list[float] = [0.003, 0.005, 0.008, 0.01]
FAMILY: str = "rotated_memory_x"

# This generator emits static memory circuits and declares so in every
# manifest it writes.  A generator names the operation it produces; nothing
# downstream infers it.
OPERATION: Final[str] = "memory"

GENERATION_COMMAND: Final[str] = "uv run python scripts/generate_circuits_memory.py"

# Output goes under the root of the operation this generator declares, so the
# layout follows from the operation rather than from a second literal.
CIRCUITS_ROOT: Path = Path("data/circuits")
CIRCUITS_DIR: Path = CIRCUITS_ROOT / OPERATION


def circuit_filename(distance: int, rounds: int, p: float) -> str:
    """Return canonical circuit filename for a setting.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.

    Returns
    -------
    str
        Filename like ``d3_r3_p0_003.stim``.
    """
    p_tag = f"p{p:.6g}".replace(".", "_")
    return f"d{distance}_r{rounds}_{p_tag}.stim"


def generate_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build a Stim surface code circuit for given parameters.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.

    Returns
    -------
    stim.Circuit
        Generated circuit.
    """
    return stim.Circuit.generated(
        f"surface_code:{FAMILY}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def circuit_provenance(circuit_path: Path, circuit: stim.Circuit) -> dict[str, object]:
    """Describe a committed circuit from the bytes on disk.

    Every field is re-derivable from the committed file, so the manifest
    claims no provenance it cannot check.

    Parameters
    ----------
    circuit_path : Path
        Path to the committed ``.stim`` file.
    circuit : stim.Circuit
        Circuit loaded from that file.

    Returns
    -------
    dict
        Manifest fields describing the circuit's origin and shape.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    return {
        "family": FAMILY,
        "stim_version": stim.__version__,
        "circuit_file": str(circuit_path),
        "circuit_sha256": hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "generation_command": GENERATION_COMMAND,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=CIRCUITS_DIR,
        help="Output directory for circuit files (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing circuit files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    manifested = 0
    for d in DISTANCES:
        r = d  # rounds match distance for training
        for p in ERROR_PROBS:
            fname = circuit_filename(d, r, p)
            path = out_dir / fname
            key = ExperimentKey(operation=OPERATION, distance=d, rounds=r, error_prob=p)

            if path.exists() and not args.overwrite:
                logger.info("Circuit exists, reading committed file: %s", path)
                circuit = stim.Circuit.from_file(path)
            else:
                circuit = generate_circuit(d, r, p)
                circuit.to_file(path)
                generated += 1
                logger.info("Generated %s", fname)

            manifest_path = circuit_manifest_path(path)
            if manifest_path.exists() and not args.overwrite:
                logger.info("Manifest exists, skipping: %s", manifest_path)
                continue

            provenance = circuit_provenance(path, circuit)
            write_circuit_manifest(path, key, provenance=provenance)
            manifested += 1
            logger.info(
                "Wrote %s  (%s, detectors=%d, observables=%d)",
                manifest_path.name,
                key,
                provenance["num_detectors"],
                provenance["num_observables"],
            )

    logger.info(
        "Done: %d circuits generated, %d manifests written in %s (stim==%s)",
        generated,
        manifested,
        out_dir,
        stim.__version__,
    )


if __name__ == "__main__":
    main()
