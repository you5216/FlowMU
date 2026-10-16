                      
                                                         

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "final_performance/tofu/small_exp/sensitive_neuron_localization/topk_path_ablation"
)
PANEL_ORDER = [
    ("for05_re95", "forget", "(a) for05 (Forget)"),
    ("for05_re95", "retain", "(b) for05 (Retain)"),
    ("for10_re90", "forget", "(c) for10 (Forget)"),
    ("for10_re90", "retain", "(d) for10 (Retain)"),
]
METHOD_STYLE = {
    "FlowMU": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "Random": {"color": "#7f7f7f", "linestyle": "--", "marker": "s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Top-k neuron ablation.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="topk_neuron_drop_ablation_2x2")
    parser.add_argument(
        "--mask-mode",
        choices=["drop", "keep"],
        default="drop",
        help="Controls axis wording only; data are read from --input-csv.",
    )
    parser.add_argument("--include-partial", action="store_true", help="Draw panels even if only one split exists.")
    parser.add_argument("--available-only", action="store_true", help="Draw only panels present in the CSV.")
    return parser.parse_args()


def load_rows(path: Path, *, mask_mode: str) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    records: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "panel", "method", "topk", "rouge_l"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        has_mask_mode = "mask_mode" in (reader.fieldnames or [])
        for row in reader:
            if has_mask_mode and row.get("mask_mode") != mask_mode:
                continue
            method = row["method"]
            if method not in METHOD_STYLE:
                continue
            key = (row["split"], row["panel"].lower(), method)
            records.setdefault(key, []).append((int(row["topk"]), float(row["rouge_l"])))
    for key, values in records.items():
        dedup: dict[int, list[float]] = {}
        for topk, rouge_l in values:
            dedup.setdefault(topk, []).append(rouge_l)
        records[key] = sorted((topk, float(np.mean(vals))) for topk, vals in dedup.items())
    return records


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_topk_label(topk: int) -> str:
    if topk == 12288:
        return "Full\n12288"
    if topk > 0 and (topk & (topk - 1)) == 0:
        return rf"$2^{{{int(np.log2(topk))}}}$"
    return str(topk)


def draw(
    records: dict[tuple[str, str, str], list[tuple[int, float]]],
    output_dir: Path,
    stem: str,
    *,
    mask_mode: str,
    include_partial: bool,
    available_only: bool,
) -> tuple[Path, Path]:
    configure_matplotlib()
    panels = PANEL_ORDER
    if available_only:
        present = {(split, panel) for split, panel, _method in records}
        panels = [item for item in PANEL_ORDER if (item[0], item[1]) in present]
        if not panels:
            raise ValueError("No available panels found in CSV.")

    if len(panels) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        axes_flat = [ax]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes_flat = axes.flatten()

    all_topks = sorted({topk for values in records.values() for topk, _ in values})
    if not all_topks:
        raise ValueError("No FlowMU/Random rows found.")
    xticks = [topk for topk in all_topks if topk in {4, 16, 64, 256, 1024, 4096, 8192, 12288}]

    for ax, (split, panel, caption) in zip(axes_flat, panels):
        drew = False
        for method in ("FlowMU", "Random"):
            values = records.get((split, panel, method), [])
            if not values:
                continue
            drew = True
            topk = np.asarray([item[0] for item in values], dtype=float)
            rouge = np.asarray([item[1] for item in values], dtype=float)
            ax.plot(topk, rouge, linewidth=2.3, markersize=4.8, label=method, **METHOD_STYLE[method])
        if not drew and not include_partial:
            raise KeyError(f"Missing panel rows for {(split, panel)}")
        if not drew:
            ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", color="#777777")

        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_topk_label(x) for x in xticks])
        if len(panels) == 1:
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", pad=8)
        ax.set_ylim(0.0, 0.65)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#555555")
            ax.spines[spine].set_linewidth(0.8)
        ax.legend(loc="best", frameon=False)
        if len(panels) == 1:
            ax.set_title(caption.replace("(a) ", ""), fontsize=13, pad=10)
        else:
            ax.text(0.5, -0.23, caption, transform=ax.transAxes, ha="center", va="top", fontsize=12)

    if len(panels) == 1:
        axes_flat[0].set_ylabel("Rouge-L")
        axes_flat[0].set_xlabel(
            "Top-k FFN Neurons Dropped Per Layer"
            if mask_mode == "drop"
            else "Top-k FFN Neurons Kept Per Layer"
        )
        plt.subplots_adjust(left=0.11, right=0.98, bottom=0.27, top=0.9)
    else:
        for ax in axes[:, 0]:
            ax.set_ylabel("Rouge-L")
        for ax in axes[1, :]:
            ax.set_xlabel(
                "Top-k FFN Neurons Dropped Per Layer"
                if mask_mode == "drop"
                else "Top-k FFN Neurons Kept Per Layer"
            )
        for ax in axes_flat[len(panels):]:
            ax.axis("off")
        plt.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.97, wspace=0.18, hspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.pdf"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def main() -> None:
    args = parse_args()
    records = load_rows(args.input_csv, mask_mode=args.mask_mode)
    svg_path, png_path = draw(
        records,
        args.output_dir,
        args.stem,
        mask_mode=args.mask_mode,
        include_partial=args.include_partial,
        available_only=args.available_only,
    )
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
