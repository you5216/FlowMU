                      
                                                                                  

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ORDER = ["LLM-only", "RAG-only", "Memory-only", "RAG+Memory", "Full Agent"]
CONDITION_COLORS = {
    "LLM-only": "#8F9DAE",
    "RAG-only": "#AEBCC8",
    "Memory-only": "#BBB5C9",
    "RAG+Memory": "#E8954C",
    "Full Agent": "#C94338",
}
PANEL_BG = "#F0F0F7"
LEGEND_BG = "#F7F7FB"
TEXT_SIZE = 18.5
LEGEND_FONTSIZE = TEXT_SIZE


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ordered_single_rows(path: Path) -> list[dict[str, str]]:
    by_condition = {row["condition"]: row for row in read_rows(path)}
    return [by_condition[name] for name in CONDITION_ORDER]


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025, facecolor="white")


def plot_single(summary_csv: Path, output_dir: Path, tag: str) -> None:
    rows = ordered_single_rows(summary_csv)
    conditions = [row["condition"] for row in rows]
    values = np.array([float(row["forget_acc"]) * 100.0 for row in rows])
    colors = [CONDITION_COLORS[name] for name in conditions]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.4, 4.8), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="#4B5563", linewidth=0.55)
    ax.set_ylabel("Forget ACC (%)")
    ax.set_ylim(0, max(100.0, float(np.nanmax(values)) + 8.0))
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=18, ha="right")
    ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, axis="y")
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)
    for spine in ax.spines.values():
        spine.set_alpha(0.0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=TEXT_SIZE,
        )

    fig.tight_layout(pad=0.5)
    save_figure(fig, output_dir / f"motivation_component_leakage_{tag}")
    plt.close(fig)


def plot_all(summary_csv: Path, output_dir: Path) -> None:
    rows = read_rows(summary_csv)
    ratios = []
    for row in rows:
        ratio = row["forget_ratio"]
        if ratio not in ratios:
            ratios.append(ratio)
    values_by_condition = {
        condition: [float(row["forget_acc"]) * 100.0 for row in rows if row["condition"] == condition]
        for condition in CONDITION_ORDER
    }

    fig, ax = plt.subplots(figsize=(11.2, 4.6), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)
    x = np.arange(len(ratios))
    width = 0.145
    offsets = (np.arange(len(CONDITION_ORDER)) - (len(CONDITION_ORDER) - 1) / 2.0) * width
    for condition, offset in zip(CONDITION_ORDER, offsets):
        ax.bar(
            x + offset,
            values_by_condition[condition],
            width=width * 0.92,
            label=condition,
            color=CONDITION_COLORS[condition],
            edgecolor="#4B5563",
            linewidth=0.55,
        )

    ax.set_xlabel("Forget Ratio")
    ax.set_ylabel("Forget ACC (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, axis="y")
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)
    for spine in ax.spines.values():
        spine.set_alpha(0.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=True,
        facecolor=LEGEND_BG,
        edgecolor="#B7BBC8",
        framealpha=0.95,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.75,
        borderpad=0.25,
    )

    fig.tight_layout(pad=0.5)
    save_figure(fig, output_dir / "motivation_component_leakage_tofu_all")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where regenerated figures are written.",
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    source_root = Path(__file__).resolve().parent
    output_root = Path(parse_args().output_root)

    plot_single(source_root / "tofu_for01" / "motivation_component_leakage_summary.csv", output_root / "tofu_for01", "tofu_for01")
    plot_single(source_root / "motivation_component_leakage_summary.csv", output_root, "tofu_for05")
    plot_single(source_root / "tofu_for10" / "motivation_component_leakage_summary.csv", output_root / "tofu_for10", "tofu_for10")
    plot_all(source_root / "motivation_component_leakage_tofu_all_summary.csv", output_root)


if __name__ == "__main__":
    main()
