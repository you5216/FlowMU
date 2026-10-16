                      
                                                               

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = EXP_DIR / "tofu_for05_rewrite_induce_attack_plot_data.json"
METHOD_ORDER = ["Vanilla", "PURE", "SBU", "SF-NL", "FlowMU"]
DISPLAY_NAME = {"FlowMU": "Ours"}
PANEL_BG = "#F0F0F7"
LEGEND_BG = "#F7F7FB"
TEXT_SIZE = 18.5
SERIES = [
    ("clean_forget_acc_pct", "Origin", "#8F9DAE"),
    ("rewrite_attack_acc_pct", "Rewrite", "#E8954C"),
    ("induce_attack_acc_pct", "Induce", "#C94338"),
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


def load_records() -> list[dict[str, float | str]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    by_method = {row["method"]: row for row in payload["records"]}
    return [by_method[method] for method in METHOD_ORDER if method in by_method]


def draw() -> list[Path]:
    configure()
    records = load_records()
    methods = [str(row["method"]) for row in records]
    x = np.arange(len(methods), dtype=float)
    width = 0.215
    offsets = np.linspace(-width, width, len(SERIES))

    fig, ax = plt.subplots(figsize=(10.6, 4.1), facecolor="white")
    ax.set_facecolor(PANEL_BG)

    for offset, (key, label, color) in zip(offsets, SERIES):
        values = [float(row[key]) for row in records]
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
    ax.set_ylabel("Forget Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME.get(method, method) for method in methods])
    for tick in ax.get_xticklabels():
        if tick.get_text() == DISPLAY_NAME["FlowMU"]:
            tick.set_fontweight("bold")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)

    for spine in ax.spines.values():
        spine.set_alpha(0.0)

    handles = [Patch(facecolor=color, edgecolor="#4B5563", linewidth=0.55, label=label) for _, label, color in SERIES]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        ncol=3,
        frameon=True,
        borderpad=0.25,
        labelspacing=0.20,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    legend.get_frame().set_facecolor(LEGEND_BG)
    legend.get_frame().set_edgecolor("#B7BBC8")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)

    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.22, top=0.84)

    outputs = [
        EXP_DIR / "tofu_for05_rewrite_induce_attack.png",
        EXP_DIR / "tofu_for05_rewrite_induce_attack.pdf",
        EXP_DIR / "tofu_for05_rewrite_induce_attack.svg",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=500, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    for path in draw():
        print(path)


if __name__ == "__main__":
    main()
