"""Generate frozen evaluation sets for one operation's evaluation points.

Adaptively samples shots from committed circuit files until each point
accumulates ≥400 MWPM logical errors (target) or hits the 1,000,000 shot cap.
Each set is saved as compressed numpy archives with a manifest recording all
provenance information, under the root of the operation its circuits declare.

``--circuit-dir`` names one operation's circuit root and has no default: an
evaluation set inherits the identity of the circuits it was sampled from, so
which circuits those are is stated at the call rather than assumed.

Usage
-----
    uv run python scripts/generate_eval_sets.py --circuit-dir data/circuits/memory
    uv run python scripts/generate_eval_sets.py --circuit-dir data/circuits/memory \
        --target-errors 400 --cap 1000000
    uv run python scripts/generate_eval_sets.py --circuit-dir data/circuits/memory \
        --distances 3 5  # subset
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from sampling.experiment import ExperimentKey
from sampling.profile import resolve_profile
from sampling.sampler import settings_from_circuit_dir
from sampling.seeding import stable_seed


logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("data/eval")
MASTER_SEED = 20240601
BATCH_SIZE = 50_000
TARGET_ERRORS = 400
SHOT_CAP = 1_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fired_count_stats(syndromes: np.ndarray) -> dict:
    """Compute fired-detector distribution statistics."""
    fired_counts = syndromes.sum(axis=1)
    return {
        "mean": float(np.mean(fired_counts)),
        "median": float(np.median(fired_counts)),
        "std": float(np.std(fired_counts)),
        "p99": float(np.percentile(fired_counts, 99)),
        "p99_9": float(np.percentile(fired_counts, 99.9)),
        "max": int(np.max(fired_counts)),
        "min": int(np.min(fired_counts)),
        "zero_count": int(np.sum(fired_counts == 0)),
    }


def _fired_count_buckets(syndromes: np.ndarray, observables: np.ndarray) -> dict:
    """Analyze val-set composition: positives per fired-count bucket."""
    fired_counts = syndromes.sum(axis=1)
    labels = observables.any(axis=1) if observables.ndim > 1 else observables.ravel()

    buckets = {
        "0_fired": (fired_counts == 0),
        "1_to_3_fired": (fired_counts >= 1) & (fired_counts <= 3),
        "4_to_10_fired": (fired_counts >= 4) & (fired_counts <= 10),
        "11_to_30_fired": (fired_counts >= 11) & (fired_counts <= 30),
        "31_plus_fired": (fired_counts >= 31),
    }

    result = {}
    for name, mask in buckets.items():
        n_total = int(mask.sum())
        n_positive = int(labels[mask].sum()) if n_total > 0 else 0
        result[name] = {
            "total": n_total,
            "positive": n_positive,
            "positive_rate": n_positive / n_total if n_total > 0 else 0.0,
        }
    return result


def generate_eval_set(
    circuit_path: Path,
    key: ExperimentKey,
    *,
    target_errors: int = TARGET_ERRORS,
    shot_cap: int = SHOT_CAP,
    batch_size: int = BATCH_SIZE,
    master_seed: int = MASTER_SEED,
    output_root: Path = OUTPUT_ROOT,
    sizing_baseline: str = "mwpm",
) -> dict:
    """Generate a frozen eval set for a single experiment point.

    Parameters
    ----------
    circuit_path : Path
        Path to the .stim circuit file.
    key : ExperimentKey
        Identity of the point, resolved from the circuit by discovery and
        recorded in the manifest.
    target_errors : int
        Minimum baseline errors to accumulate before stopping.
    shot_cap : int
        Maximum shots per point.
    batch_size : int
        Shots generated per sampling batch.
    master_seed : int
        Master seed for deterministic generation.
    output_root : Path
        Root of the evaluation tree; the set is written under the operation
        segment the key declares.
    sizing_baseline : str
        Decoder for adaptive sizing: ``"mwpm"`` or ``"correlated"``.
        Lattice-surgery operations use correlated matching (eval protocol
        §10.7) because it has a lower error rate than MWPM, requiring more
        shots - the conservative direction.

    Returns
    -------
    dict
        Summary with error counts, fired-count stats, and paths.
    """
    if sizing_baseline not in ("mwpm", "correlated"):
        raise ValueError(
            f"sizing_baseline must be 'mwpm' or 'correlated', got {sizing_baseline!r}"
        )

    distance = key.distance
    rounds = key.rounds
    error_prob = key.error_prob

    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(decompose_errors=True)
    use_correlations = sizing_baseline == "correlated"
    matching = pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=use_correlations
    )
    n_det = dem.num_detectors
    n_obs = dem.num_observables

    seed = stable_seed("eval", f"d={distance}", f"p={error_prob}", base=master_seed)
    sampler = circuit.compile_detector_sampler(seed=seed)

    syndrome_batches: list[np.ndarray] = []
    observable_batches: list[np.ndarray] = []
    baseline_errors_total = 0
    shots_total = 0

    logger.info(
        "Generating d=%d p=%.4f (target=%d errors, cap=%d shots)",
        distance,
        error_prob,
        target_errors,
        shot_cap,
    )

    while baseline_errors_total < target_errors and shots_total < shot_cap:
        remaining = shot_cap - shots_total
        n_batch = min(batch_size, remaining)

        raw = sampler.sample(shots=n_batch, bit_packed=False, append_observables=True)
        syndromes = raw[:, :n_det].astype(np.uint8)
        observables = raw[:, n_det : n_det + n_obs].astype(np.uint8)

        baseline_pred = matching.decode_batch(
            syndromes, **({"enable_correlations": True} if use_correlations else {})
        )[:, :n_obs]
        batch_errors = int(np.any(baseline_pred != observables, axis=1).sum())

        syndrome_batches.append(syndromes)
        observable_batches.append(observables)
        baseline_errors_total += batch_errors
        shots_total += n_batch

        logger.info(
            "  batch +%d shots => %d/%d %s errors (total %d shots)",
            n_batch,
            baseline_errors_total,
            target_errors,
            sizing_baseline,
            shots_total,
        )

    all_syndromes = np.concatenate(syndrome_batches, axis=0)
    all_observables = np.concatenate(observable_batches, axis=0)

    # Fired-count analysis
    fired_stats = _fired_count_stats(all_syndromes)
    bucket_composition = _fired_count_buckets(all_syndromes, all_observables)

    full_baseline_pred = matching.decode_batch(
        all_syndromes, **({"enable_correlations": True} if use_correlations else {})
    )[:, :n_obs]
    verified_errors = int(np.any(full_baseline_pred != all_observables, axis=1).sum())
    assert verified_errors == baseline_errors_total, (
        f"Accumulated {baseline_errors_total} != verified {verified_errors}"
    )

    # Extract metadata using the profile's extractor so the eval set carries
    # the same inputs the builder will receive at training time.
    profile = resolve_profile(key.operation)
    metadata = profile.metadata_extractor(circuit, distance, rounds)

    # Save - include phased metadata arrays when the operation provides them,
    # so the eval set is self-contained for its representation.
    p_str = f"{error_prob:.4f}".replace(".", "_")
    point_dir = output_root / key.operation / f"d{distance}_p{p_str}"
    point_dir.mkdir(parents=True, exist_ok=True)

    npz_arrays: dict[str, np.ndarray] = {
        "syndromes": all_syndromes,
        "observables": all_observables,
        "detector_coords": metadata.detector_coords,
    }
    if metadata.phase_ids is not None:
        npz_arrays["phase_ids"] = metadata.phase_ids
        npz_arrays["patch_ids"] = metadata.patch_ids
        npz_arrays["seam_mask"] = metadata.seam_mask

    np.savez_compressed(point_dir / "data.npz", **npz_arrays)

    positive_count = int(all_observables.any(axis=1).sum())
    manifest: dict = {
        "circuit_file": str(circuit_path),
        "circuit_sha256": _sha256(circuit_path),
        "stim_version": stim.__version__,
        "seed": seed,
        "master_seed": master_seed,
        "operation": key.operation,
        "distance": distance,
        "rounds": rounds,
        "error_prob": error_prob,
        "num_shots": shots_total,
        "num_detectors": n_det,
        "num_observables": n_obs,
        "representation_version": profile.representation_version,
        "observable_names": list(profile.observable_names),
        "sizing_baseline": sizing_baseline,
        "baseline_errors": verified_errors,
        "baseline_ler": verified_errors / shots_total,
        "positive_count": positive_count,
        "positive_rate": positive_count / shots_total,
        "target_errors": target_errors,
        "shot_cap": shot_cap,
        "target_met": baseline_errors_total >= target_errors,
        "fired_count_stats": fired_stats,
        "fired_count_buckets": bucket_composition,
        "n_max_p99_9": fired_stats["p99_9"],
        "generation_command": (
            f"uv run python scripts/generate_eval_sets.py"
            f" --circuit-dir {circuit_path.parent}"
            f" --baseline {sizing_baseline}"
        ),
    }
    if metadata.phase_names is not None:
        manifest["phase_names"] = list(metadata.phase_names)

    (point_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    met_str = "MET" if manifest["target_met"] else "BELOW TARGET"
    logger.info(
        "  => %s: %d shots, %d %s errors (LER=%.6f), N_max(p99.9)=%.0f [%s]",
        point_dir.name,
        shots_total,
        verified_errors,
        sizing_baseline,
        manifest["baseline_ler"],
        fired_stats["p99_9"],
        met_str,
    )

    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--circuit-dir",
        type=Path,
        required=True,
        help="Circuit root of one operation, e.g. data/circuits/memory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root of the evaluation tree (default: %(default)s)",
    )
    parser.add_argument(
        "--target-errors", type=int, default=TARGET_ERRORS, help="Target MWPM errors"
    )
    parser.add_argument("--cap", type=int, default=SHOT_CAP, help="Max shots per point")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Shots per sampling batch"
    )
    parser.add_argument("--seed", type=int, default=MASTER_SEED, help="Master seed")
    parser.add_argument(
        "--distances", type=int, nargs="*", default=None, help="Distances to generate"
    )
    parser.add_argument(
        "--baseline",
        choices=["mwpm", "correlated"],
        default="mwpm",
        help="Decoder for adaptive sizing (eval protocol §10.7: correlated for "
        "lattice-surgery). Default: %(default)s",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    points = settings_from_circuit_dir(args.circuit_dir, distances=args.distances)
    logger.info("Generating eval sets for %d experiment points", len(points))

    results = []
    t0 = time.time()

    for s in points:
        manifest = generate_eval_set(
            circuit_path=s.circuit_path,
            key=s.key,
            target_errors=args.target_errors,
            shot_cap=args.cap,
            batch_size=args.batch_size,
            master_seed=args.seed,
            output_root=args.output_root,
            sizing_baseline=args.baseline,
        )
        results.append(manifest)

    elapsed = time.time() - t0

    # Summary table
    baseline_label = results[0].get("sizing_baseline", "mwpm") if results else "mwpm"
    bl_col = f"{baseline_label} Err"
    bl_ler_col = f"{baseline_label} LER"

    print()
    print("=" * 90)
    print("Frozen Eval Set Generation Summary")
    print("=" * 90)
    print(
        f"{'Point':<14} {'Shots':>8} {bl_col:>12} {bl_ler_col:>14} "
        f"{'Pos Rate':>9} {'N_max(p99.9)':>13} {'Status':>8}"
    )
    print("-" * 90)
    for r in results:
        status = "OK" if r["target_met"] else "BELOW"
        bl_errors = r.get("baseline_errors", r.get("mwpm_errors", 0))
        bl_ler = r.get("baseline_ler", r.get("mwpm_ler", 0.0))
        print(
            f"d{r['distance']}_p{r['error_prob']:<7.4f} "
            f"{r['num_shots']:>8} {bl_errors:>12} "
            f"{bl_ler:>14.6f} {r['positive_rate']:>9.4f} "
            f"{r['n_max_p99_9']:>13.0f} {status:>8}"
        )

    # Composition check
    print()
    print("Val-set composition (positives per fired-count bucket):")
    print("-" * 90)
    print(
        f"{'Point':<14} {'0 fired':>10} {'1-3 fired':>12} "
        f"{'4-10 fired':>12} {'11-30 fired':>13} {'31+ fired':>12}"
    )
    print("-" * 90)
    for r in results:
        b = r["fired_count_buckets"]

        def _fmt(bucket: dict) -> str:
            return f"{bucket['positive']}/{bucket['total']}"

        print(
            f"d{r['distance']}_p{r['error_prob']:<7.4f} "
            f"{_fmt(b['0_fired']):>10} {_fmt(b['1_to_3_fired']):>12} "
            f"{_fmt(b['4_to_10_fired']):>12} {_fmt(b['11_to_30_fired']):>13} "
            f"{_fmt(b['31_plus_fired']):>12}"
        )

    # N_max table
    print()
    print("N_max (p99.9 fired detectors) per (d, p):")
    print("-" * 50)
    for r in results:
        print(
            f"  d={r['distance']} p={r['error_prob']:.4f}: "
            f"N_max(p99.9) = {r['n_max_p99_9']:.0f}, "
            f"max = {r['fired_count_stats']['max']}"
        )

    below = [r for r in results if not r["target_met"]]
    if below:
        print()
        print(
            f"WARNING: {len(below)} point(s) below {args.target_errors}-error target:"
        )
        for r in below:
            bl_errors = r.get("baseline_errors", r.get("mwpm_errors", 0))
            print(
                f"  d={r['distance']} p={r['error_prob']:.4f}: "
                f"{bl_errors} errors in {r['num_shots']} shots"
            )

    print(f"\nTotal generation time: {elapsed:.1f}s")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
