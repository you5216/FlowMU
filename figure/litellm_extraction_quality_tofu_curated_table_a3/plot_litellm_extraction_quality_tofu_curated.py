                      
                                                                             

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = SCRIPT_DIR / "litellm_extraction_quality_tofu_curated_summary.csv"
STEM = "litellm_extraction_quality_tofu_curated"
PANEL_BG = "#F0F0F7"
LEGEND_BG = "#F7F7FB"
TEXT_SIZE = 18.5
METHOD_ORDER = ["Rule-based", "LiteLLM", "Strong LLM", "DeepSeek-v4"]
METRICS = [
    ("Ent-F1", "Entity"),
    ("Rel-F1", "Relation"),
    ("RelPair-F1", "Relation Pair"),
]
COLORS = {
    "Rule-based": "#8F9DAE",
    "LiteLLM": "#C94338",
    "Strong LLM": "#E8954C",
    "DeepSeek-v4": "#9B2148",
}


def configure_matplotlib() -> None:
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


def load_summary() -> list[dict[str, str]]:
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_method = {row["Extractor"]: row for row in rows}
    return [by_method[method] for method in METHOD_ORDER]


def draw() -> list[Path]:
    configure_matplotlib()
    summary = load_summary()
    x = np.arange(len(METRICS), dtype=float)
    width = 0.18
    center = (len(METHOD_ORDER) - 1) / 2.0

    fig, ax = plt.subplots(figsize=(8.6, 4.1), facecolor="white")
    ax.set_facecolor(PANEL_BG)

    for i, row in enumerate(summary):
        method = row["Extractor"]
        values = [float(row[key]) * 100.0 for key, _ in METRICS]
        ax.bar(
            x + (i - center) * width,
            values,
            width=width,
            color=COLORS[method],
            edgecolor="#4B5563",
            linewidth=0.55,
            label=method,
            zorder=3,
        )

    ax.set_ylim(20, 100)
    ax.set_ylabel("F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.grid(True, which="major", axis="y", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)
    for spine in ax.spines.values():
        spine.set_alpha(0.0)

    handles = [
        Patch(facecolor=COLORS[method], edgecolor="#4B5563", linewidth=0.55, label=method)
        for method in METHOD_ORDER
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        ncol=4,
        columnspacing=0.9,
        handlelength=1.1,
        borderpad=0.25,
    )
    legend.get_frame().set_facecolor(LEGEND_BG)
    legend.get_frame().set_edgecolor("#B7BBC8")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)
    fig.tight_layout(pad=0.5)

    outputs = [SCRIPT_DIR / f"{STEM}.{ext}" for ext in ("png", "pdf", "svg")]
    for path in outputs:
        fig.savefig(path, dpi=360, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)
    return outputs


def main() -> None:
    draw()


if __name__ == "__main__":
    main()
