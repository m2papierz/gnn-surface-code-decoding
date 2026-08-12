"""Generate and commit logical-operation circuits from tqec block graphs.

Each circuit is emitted, validated by the generation-time gate, and written
with a manifest declaring its experiment identity and metadata.  A circuit
that already exists is never rewritten.

tqec is a generation-time-only dependency: nothing under ``src/`` outside
``src/sampling/logical_ops.py`` may import it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

from tqec import Basis
from tqec import gallery as tqec_gallery
from tqec.computation.block_graph import BlockGraph

from sampling.logical_ops import generate_and_write_circuit


logger = logging.getLogger(__name__)

CIRCUITS_ROOT: Final[Path] = Path("data/circuits")

GENERATING_COMMAND: Final[str] = "uv run python scripts/generate_circuits_tqec.py"


def _build_block_graph(operation: str) -> BlockGraph:
    """Return the tqec ``BlockGraph`` for a named operation.

    Parameters
    ----------
    operation : str
        One of the supported gallery operations.

    Returns
    -------
    BlockGraph

    Raises
    ------
    ValueError
        If the operation has no gallery entry.
    """
    factories: dict[str, BlockGraph] = {
        "memory": tqec_gallery.memory(Basis.Z),
        "zz_merge_split": tqec_gallery.cnot(Basis.Z),
    }
    if operation not in factories:
        raise ValueError(
            f"Unknown operation {operation!r}.  Available: {sorted(factories)}"
        )
    return factories[operation]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        type=str,
        help="Logical operation to generate (e.g. 'memory', 'zz_merge_split')",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Scale parameters (d = 2k+1). Default: 1 2 3",
    )
    parser.add_argument(
        "--error-probs",
        type=float,
        nargs="+",
        default=[0.003, 0.005, 0.008, 0.01],
        help="Physical error probabilities. Default: 0.003 0.005 0.008 0.01",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CIRCUITS_ROOT,
        help="Root directory for circuit output (default: %(default)s)",
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

    operation: str = args.operation
    output_dir = args.output_root / operation

    block_graph = _build_block_graph(operation)

    generated = 0
    skipped = 0
    for k in args.k_values:
        d = 2 * k + 1
        for p in args.error_probs:
            p_tag = f"p{p:.6g}".replace(".", "_")
            pattern = f"d{d}_r*_{p_tag}.stim"
            existing = list(output_dir.glob(pattern))

            if existing and not args.overwrite:
                logger.info("Exists, skipping: %s", existing[0].name)
                skipped += 1
                continue

            try:
                circuit_path = generate_and_write_circuit(
                    block_graph,
                    operation=operation,
                    k=k,
                    error_prob=p,
                    output_dir=output_dir,
                    generating_command=(
                        f"{GENERATING_COMMAND} {operation} "
                        f"--k-values {k} --error-probs {p}"
                    ),
                )
                generated += 1
                logger.info(
                    "Generated %s (k=%d, d=%d, p=%s)",
                    circuit_path.name,
                    k,
                    d,
                    p,
                )
            except Exception:
                logger.exception(
                    "Failed to generate circuit for %s k=%d d=%d p=%s",
                    operation,
                    k,
                    d,
                    p,
                )
                sys.exit(1)

    logger.info(
        "Done: %d generated, %d skipped in %s",
        generated,
        skipped,
        output_dir,
    )


if __name__ == "__main__":
    main()
