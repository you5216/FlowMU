                      
                                                                   

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = EXP_DIR / "tofu_for05_rewrite_induce_mia_combined_data.csv"
METHOD_ORDER = ["Vanilla", "PURE", "SBU", "SF-NL", "Ours"]

PANEL_BG = "#F0F0F7"
LEGEND_BG = "#F7F7FB"
TEXT_SIZE = 18.5
SERIES = [
    ("rewrite_attack_acc_pct", "Rewrite", "#8F9DAE"),
    ("induce_attack_acc_pct", "Induce", "#E8954C"),
    ("mia_auc_pct", "MIA-AUC", "#C94338"),
]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_method = {row["method"]: row for row in rows}
    return [by_method[method] for method in METHOD_ORDER]


def draw() -> list[Path]:
    configure()
    rows = load_rows()
    methods = [row["method"] for row in rows]
    x = np.arange(len(methods), dtype=float)
    width = 0.215
    offsets = np.linspace(-width, width, len(SERIES))

    fig, ax = plt.subplots(figsize=(10.8, 4.25), facecolor="white")
    ax.set_facecolor(PANEL_BG)

    for offset, (key, label, color) in zip(offsets, SERIES):
        values = [float(row[key]) for row in rows]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="#4B5563",
            linewidth=0.55,
            label=label,
            zorder=3,
        )

    ax.set_xlim(-0.58, len(methods) - 0.42)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate / AUC (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "Ours":
            tick.set_fontweight("bold")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)

    for spine in ax.spines.values():
        spine.set_alpha(0.0)

    handles = [Patch(facecolor=color, edgecolor="#4B5563", linewidth=0.55, label=label) for _, label, color in SERIES]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        borderpad=0.24,
        columnspacing=0.95,
        handlelength=0.9,
        handletextpad=0.35,
    )
    legend.get_frame().set_facecolor(LEGEND_BG)
    legend.get_frame().set_edgecolor("#B7BBC8")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)

    fig.subplots_adjust(left=0.095, right=0.992, bottom=0.20, top=0.80)

    outputs = [
        EXP_DIR / "tofu_for05_rewrite_induce_mia_combined.png",
        EXP_DIR / "tofu_for05_rewrite_induce_mia_combined.pdf",
        EXP_DIR / "tofu_for05_rewrite_induce_mia_combined.svg",
    ]
    for path in outputs:
        fig.savefig(path, dpi=500, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    for path in draw():
        print(path)


if __name__ == "__main__":
    main()
