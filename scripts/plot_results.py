"""Produce the evaluation figures.

Fig. 1 — LER vs physical error probability p.  Three panels (d=3, d=5, d=7),
three decoders per panel (GNN, MWPM, Belief-Matching), Wilson 95% error bars,
McNemar significance annotations for the GNN-vs-MWPM paired comparison.

Fig. 2 — LER vs code distance at a fixed p, with Wilson 95% error bars.  The
reference p defaults to the median of the p values common to every distance.

Examples
--------
    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --eval-dir outputs -o outputs/figures
    uv run python scripts/plot_results.py --reference-p 0.005
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


logger = logging.getLogger(__name__)

_DISTANCES: list[int] = [3, 5, 7]
_EVAL_FILES: dict[int, str] = {
    3: "d3_full/eval_d3_full.json",
    5: "d5_full/eval_d5_full.json",
    7: "d7_full/eval_d7_full.json",
}
_MIXED_EVAL_FILE: str = "mixed_d/eval_mixed_d.json"

_DECODER_KEYS: list[str] = ["gnn", "mwpm", "belief_matching"]
_DECODER_LABELS: dict[str, str] = {
    "gnn": "GNN (per-d)",
    "mwpm": "MWPM",
    "belief_matching": "Belief-Matching",
    "gnn_mixed": "GNN (mixed)",
}
_DECODER_COLORS: dict[str, str] = {
    "gnn": "#2a78d6",
    "mwpm": "#eb6834",
    "belief_matching": "#1baf7a",
    "gnn_mixed": "#9467bd",
}
_DECODER_MARKERS: dict[str, str] = {
    "gnn": "o",
    "mwpm": "s",
    "belief_matching": "D",
    "gnn_mixed": "^",
}

_MCNEMAR_ALPHA: float = 0.05


def _load_eval(path: Path) -> list[dict[str, Any]]:
    """Load evaluation JSON and return the list of point results."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["points"]


def _significance_stars(p_value: float) -> str:
    """Return significance stars for a McNemar p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < _MCNEMAR_ALPHA:
        return "*"
    return ""


def _apply_style() -> None:
    """Apply the shared figure style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> Path:
    """Write *fig* as 300-dpi PNG and PDF, and return the PNG path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved %s", png_path)
    logger.info("Saved %s", pdf_path)
    return png_path


def plot_ler_vs_p(eval_dir: Path, output_dir: Path) -> Path:
    """Produce Fig. 1 and save to output_dir as PNG (300 dpi) and PDF.

    Parameters
    ----------
    eval_dir : Path
        Root directory containing ``d3_full/``, ``d5_full/``, ``d7_full/``
        with per-distance evaluation JSONs.
    output_dir : Path
        Directory for the output figure files.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    _apply_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4.5),
        sharey=False,
    )

    mixed_path = eval_dir / _MIXED_EVAL_FILE
    mixed_points_by_d: dict[int, list[dict[str, Any]]] = {}
    if mixed_path.is_file():
        for pt in _load_eval(mixed_path):
            mixed_points_by_d.setdefault(pt["distance"], []).append(pt)

    for ax, d in zip(axes, _DISTANCES, strict=True):
        eval_path = eval_dir / _EVAL_FILES[d]
        if not eval_path.is_file():
            raise FileNotFoundError(f"Eval file not found: {eval_path}")

        points = _load_eval(eval_path)
        p_values = np.array([pt["error_prob"] for pt in points])

        for key in _DECODER_KEYS:
            lers = np.array([pt["decoders"][key]["ler"] for pt in points])
            ci_low = np.array([pt["decoders"][key]["ler_ci_95"][0] for pt in points])
            ci_high = np.array([pt["decoders"][key]["ler_ci_95"][1] for pt in points])

            ax.errorbar(
                p_values,
                lers,
                yerr=[lers - ci_low, ci_high - lers],
                color=_DECODER_COLORS[key],
                marker=_DECODER_MARKERS[key],
                markersize=7,
                linewidth=1.8,
                capsize=3,
                capthick=1.2,
                elinewidth=1.2,
                zorder=3,
            )

        if d in mixed_points_by_d:
            m_pts = sorted(mixed_points_by_d[d], key=lambda pt: pt["error_prob"])
            m_p = np.array([pt["error_prob"] for pt in m_pts])
            m_ler = np.array([pt["decoders"]["gnn"]["ler"] for pt in m_pts])
            m_lo = np.array([pt["decoders"]["gnn"]["ler_ci_95"][0] for pt in m_pts])
            m_hi = np.array([pt["decoders"]["gnn"]["ler_ci_95"][1] for pt in m_pts])
            ax.errorbar(
                m_p,
                m_ler,
                yerr=[m_ler - m_lo, m_hi - m_ler],
                color=_DECODER_COLORS["gnn_mixed"],
                marker=_DECODER_MARKERS["gnn_mixed"],
                markersize=7,
                linewidth=1.8,
                linestyle="--",
                capsize=3,
                capthick=1.2,
                elinewidth=1.2,
                zorder=3,
            )

        _annotate_mcnemar(ax, points)

        ax.set_yscale("log")
        ax.set_xlabel("Physical error probability $p$", fontsize=10)
        ax.set_title(f"$d = {d}$", fontsize=12, fontweight="bold")
        ax.set_xticks(p_values)
        ax.set_xticklabels([f"{p:.3f}" for p in p_values], fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, which="major", ls="-", alpha=0.15, color="#000000")
        ax.grid(True, which="minor", ls=":", alpha=0.08, color="#000000")

        margin = (p_values[-1] - p_values[0]) * 0.12
        ax.set_xlim(p_values[0] - margin, p_values[-1] + margin)

    axes[0].set_ylabel("Logical error rate (LER)", fontsize=10)

    legend_keys = list(_DECODER_KEYS)
    if mixed_points_by_d:
        legend_keys.append("gnn_mixed")
    handles = [
        Line2D(
            [0],
            [0],
            color=_DECODER_COLORS[k],
            marker=_DECODER_MARKERS[k],
            markersize=7,
            linewidth=1.8,
            linestyle="--" if k == "gnn_mixed" else "-",
            label=_DECODER_LABELS[k],
        )
        for k in legend_keys
    ]
    fig.tight_layout()
    fig.subplots_adjust(top=0.86, bottom=0.17)

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(legend_keys),
        fontsize=10,
        frameon=True,
        edgecolor="#cccccc",
        fancybox=False,
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.text(
        0.5,
        0.02,
        (
            "McNemar paired test (GNN vs MWPM):  "
            "*** $p<0.001$,  ** $p<0.01$,  * $p<0.05$,  "
            "n.s. not significant.  Color indicates winner."
        ),
        ha="center",
        fontsize=8,
        color="#666666",
    )

    return _save_figure(fig, output_dir, "fig1_ler_vs_p")


def plot_ler_scaling_with_d(
    eval_dir: Path,
    output_dir: Path,
    *,
    reference_p: float | None = None,
) -> Path:
    """Produce Fig. 2 — LER vs code distance at one fixed physical error rate.

    Error bars are the Wilson 95% intervals already carried by the evaluation
    JSONs; a bare LER point would violate the project statistics policy.

    Parameters
    ----------
    eval_dir : Path
        Root directory containing the per-distance evaluation JSONs.
    output_dir : Path
        Directory for the output figure files.
    reference_p : float or None
        Physical error probability to slice at.  When None, the median of the
        p values present at *every* distance is used.  A value that was not
        evaluated snaps to the nearest one that was.

    Returns
    -------
    Path
        Path to the saved PNG file.

    Raises
    ------
    FileNotFoundError
        If a per-distance evaluation JSON is missing.
    ValueError
        If no single p value was evaluated at every distance.
    """
    _apply_style()

    points_by_d: dict[int, list[dict[str, Any]]] = {}
    for d in _DISTANCES:
        eval_path = eval_dir / _EVAL_FILES[d]
        if not eval_path.is_file():
            raise FileNotFoundError(f"Eval file not found: {eval_path}")
        points_by_d[d] = _load_eval(eval_path)

    # Only p values common to every distance give a comparable slice.
    shared_p = set.intersection(
        *({pt["error_prob"] for pt in pts} for pts in points_by_d.values())
    )
    if not shared_p:
        raise ValueError(
            "no physical error probability was evaluated at every distance "
            f"{_DISTANCES}, so no scaling slice is comparable"
        )

    available = sorted(shared_p)
    if reference_p is None:
        reference_p = available[len(available) // 2]
    else:
        reference_p = min(available, key=lambda p: abs(p - reference_p))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for key in _DECODER_KEYS:
        distances, lers, lo, hi = [], [], [], []
        for d in _DISTANCES:
            match = next(
                (pt for pt in points_by_d[d] if pt["error_prob"] == reference_p),
                None,
            )
            if match is None:
                continue
            entry = match["decoders"][key]
            distances.append(d)
            lers.append(entry["ler"])
            lo.append(entry["ler_ci_95"][0])
            hi.append(entry["ler_ci_95"][1])

        if not distances:
            continue

        lers_arr = np.array(lers)
        ax.errorbar(
            distances,
            lers_arr,
            yerr=[lers_arr - np.array(lo), np.array(hi) - lers_arr],
            label=_DECODER_LABELS[key],
            color=_DECODER_COLORS[key],
            marker=_DECODER_MARKERS[key],
            markersize=7,
            linewidth=1.8,
            capsize=3,
            capthick=1.2,
            elinewidth=1.2,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Code distance $d$", fontsize=10)
    ax.set_ylabel("Logical error rate (LER)", fontsize=10)
    ax.set_title(
        f"LER scaling at $p = {reference_p:g}$", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(_DISTANCES)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="major", ls="-", alpha=0.15, color="#000000")
    ax.grid(True, which="minor", ls=":", alpha=0.08, color="#000000")
    ax.legend(fontsize=9, loc="best", frameon=True, edgecolor="#cccccc")
    fig.tight_layout()

    return _save_figure(fig, output_dir, "fig2_ler_scaling_d")


def _annotate_mcnemar(
    ax: plt.Axes,
    points: list[dict[str, Any]],
) -> None:
    """Add McNemar significance markers above the curves at each p value."""
    for pt in points:
        p = pt["error_prob"]
        mcnemar = pt["mcnemar"]["mwpm"]
        p_val = mcnemar["p_value"]
        stars = _significance_stars(p_val)

        gnn_ler = pt["decoders"]["gnn"]["ler"]
        mwpm_ler = pt["decoders"]["mwpm"]["ler"]
        y_top = max(pt["decoders"][k]["ler"] for k in _DECODER_KEYS)

        if stars:
            winner_color = (
                _DECODER_COLORS["gnn"]
                if gnn_ler < mwpm_ler
                else _DECODER_COLORS["mwpm"]
            )
            ax.annotate(
                stars,
                xy=(p, y_top),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=winner_color,
            )
        else:
            ax.annotate(
                "n.s.",
                xy=(p, y_top),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#888888",
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory with d3_full/, d5_full/, d7_full/ eval JSONs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Where to save generated figures.",
    )
    parser.add_argument(
        "--reference-p",
        type=float,
        default=None,
        help=(
            "Physical error probability for the LER-vs-distance figure. "
            "Defaults to the median p evaluated at every distance."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    plot_ler_vs_p(args.eval_dir, args.output_dir)
    plot_ler_scaling_with_d(
        args.eval_dir, args.output_dir, reference_p=args.reference_p
    )


if __name__ == "__main__":
    main()
