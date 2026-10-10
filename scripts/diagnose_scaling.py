"""Record scaling and baseline diagnostics for every frozen evaluation point.

For each operation, every committed circuit is matched to the frozen
evaluation set sampled from it, and the point is measured four ways: the
size of the graphs its shots build, how often each logical observable flips
over them, how its detector error model's probability mass splits by
mechanism class, and how MWPM and correlated matching score on every frozen
shot.  Across points, the sub-threshold rule of
``evaluation.scaling_diagnostics`` is applied per operation and decoder.

The run is deterministic: it samples nothing, so the artifact is a function
of the committed circuits, the frozen shots and the library versions it
records.  A circuit without a frozen set, a set whose circuit does not hash
to what the set recorded, or a set whose shots do not hold the positive count
its manifest recorded fails the run and no artifact is written.

Usage
-----
    uv run python scripts/diagnose_scaling.py
    uv run python scripts/diagnose_scaling.py --operations memory --distances 3 \
        -o outputs/results/scaling_memory_d3.json
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import shlex
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import stim

from artifact_file import write_json
from decoders import CorrelatedMatchingDecoder, PyMatchingDecoder
from evaluation.evaluator import EvalSet, discover_eval_sets, load_eval_set
from evaluation.scaling_diagnostics import (
    ObservableFlipRates,
    PointDiagnostics,
    compare_baselines,
    dem_mass_split,
    graph_size,
    observable_flip_rates,
    report_to_dict,
)
from sampling.dem import read_dem_mechanisms
from sampling.experiment import ExperimentKey
from sampling.profile import (
    registered_operations,
    resolve_profile,
)
from sampling.sampler import settings_from_circuit_dir


logger = logging.getLogger(__name__)

DEFAULT_EVAL_ROOT = Path("data/eval")
DEFAULT_OUTPUT = Path("outputs/results/scaling_diagnostics.json")


def _manifest_field(eval_set: EvalSet, eval_dir: Path, name: str) -> Any:
    """Read a field the set's manifest must record."""
    if name not in eval_set.manifest:
        raise ValueError(
            f"Manifest missing required field {name!r}: {eval_dir / 'manifest.json'}"
        )
    return eval_set.manifest[name]


def _verified_observable_flips(
    eval_set: EvalSet, eval_dir: Path
) -> ObservableFlipRates:
    """Count the set's observable flips and require the manifest's positive count."""
    flips = observable_flip_rates(eval_set.observables, eval_set.observable_names)
    recorded = _manifest_field(eval_set, eval_dir, "positive_count")
    counted = flips.any_observable.n_errors
    if counted != recorded:
        raise ValueError(
            f"Eval set {eval_dir} holds {counted} shot(s) with an observable "
            f"flip, but its manifest recorded positive_count={recorded}; the "
            f"frozen shots are not the ones the manifest describes"
        )
    return flips


def _shots_sha256(eval_set: EvalSet) -> str:
    """Hash the frozen syndromes and observables, shapes included."""
    digest = hashlib.sha256()
    for shots in (eval_set.syndromes, eval_set.observables):
        digest.update(repr(shots.shape).encode())
        digest.update(np.ascontiguousarray(shots))
    return digest.hexdigest()


def _diagnose_point(
    eval_dir: Path, circuits: Mapping[ExperimentKey, Path]
) -> PointDiagnostics:
    """Diagnose the frozen set in ``eval_dir`` against its committed circuit."""
    eval_set = load_eval_set(eval_dir)
    key = eval_set.key
    if circuits.get(key) != eval_set.circuit_file:
        raise ValueError(
            f"Eval set {eval_dir} is for {key} and names circuit "
            f"{eval_set.circuit_file}, but circuit discovery resolves that point "
            f"to {circuits.get(key)}"
        )

    circuit_path = circuits[key]
    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(decompose_errors=True)
    profile = resolve_profile(key.operation)
    return PointDiagnostics(
        key=key,
        circuit_file=str(circuit_path),
        circuit_sha256=eval_set.circuit_sha256,
        shots_sha256=_shots_sha256(eval_set),
        graph=graph_size(eval_set.syndromes),
        observable_flips=_verified_observable_flips(eval_set, eval_dir),
        dem=dem_mass_split(read_dem_mechanisms(dem)),
        baselines=compare_baselines(
            eval_set,
            reference=CorrelatedMatchingDecoder(circuit_path),
            compared=PyMatchingDecoder(circuit_path),
            profile=profile,
        ),
    )


def diagnose_operation(
    operation: str,
    *,
    eval_root: Path,
    distances: list[int] | None,
    error_probs: list[float] | None,
) -> list[PointDiagnostics]:
    """Diagnose every committed circuit of one operation on its frozen set.

    Parameters
    ----------
    operation : str
        Registered operation name.
    eval_root : Path
        Root of the frozen evaluation tree; the operation's sets are read
        from ``eval_root / operation``.
    distances : list of int or None
        Restrict circuits and sets to these distances.  ``None`` means all.
    error_probs : list of float or None
        Restrict circuits and sets to these error probabilities.

    Returns
    -------
    list of PointDiagnostics
        One per committed circuit that passes the filters.

    Raises
    ------
    ValueError
        If a committed circuit has no frozen set, a set belongs to no
        committed circuit, or a circuit's hash differs from the one its set
        recorded.
    """
    profile = resolve_profile(operation)
    circuits = {
        point.key: point.circuit_path
        for point in settings_from_circuit_dir(
            profile.circuit_root,
            operation=operation,
            distances=distances,
            error_probs=error_probs,
        )
    }
    points: list[PointDiagnostics] = []
    for eval_dir in discover_eval_sets(
        eval_root / operation, distances=distances, error_probs=error_probs
    ):
        point = _diagnose_point(eval_dir, circuits)
        logger.info(
            "%s: %d shots, %s vs %s %s",
            point.key,
            point.graph.num_shots,
            point.baselines.reference.decoder_name,
            point.baselines.compared.decoder_name,
            point.baselines.outcome.value,
        )
        points.append(point)

    missing = sorted(str(key) for key in set(circuits) - {p.key for p in points})
    if missing:
        raise ValueError(
            f"No frozen eval set under {eval_root / operation} for committed "
            f"circuit(s): {missing}"
        )
    return points


def _provenance(argv: Sequence[str]) -> dict[str, Any]:
    command = ["uv", "run", "python", "scripts/diagnose_scaling.py", *argv]
    return {
        "command": shlex.join(command),
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "stim", "pymatching")},
        },
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    operations = sorted(registered_operations())
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=operations,
        default=operations,
        help="Operations to diagnose (default: every registered operation).",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Root of the frozen evaluation tree (default: %(default)s).",
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=None, help="Restrict distances."
    )
    parser.add_argument(
        "--error-probs",
        type=float,
        nargs="+",
        default=None,
        help="Restrict physical error probabilities.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Artifact path (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Diagnose the selected operations and write the artifact."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    points = [
        point
        for operation in args.operations
        for point in diagnose_operation(
            operation,
            eval_root=args.eval_root,
            distances=args.distances,
            error_probs=args.error_probs,
        )
    ]
    report = report_to_dict(points, provenance=_provenance(argv))
    write_json(args.output, report)

    for operation, decoders in report["threshold_summary"].items():
        for decoder, verdicts in decoders.items():
            logger.info("%s / %s: %s", operation, decoder, verdicts)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
