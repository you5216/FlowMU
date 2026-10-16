                      
                                                              

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
RECORDS_CSV = SCRIPT_DIR / "literaryqa_for10_causal_perturbation_relative_records.csv"
STYLE_JSON = SCRIPT_DIR / "literaryqa_for10_causal_perturbation_plot_style.json"
STEM = "literaryqa_for10_causal_perturbation_boxplot"

METHOD_ORDER = ["Random", "Activation", "FlowMU"]
DISPLAY_NAME = {"FlowMU": "Ours"}
PANELS = [("forget", "(a) Forget Set"), ("retain", "(b) Retain Set")]
COLORS = {
    "Random": "#8F9DAE",
    "Activation": "#AEBCC8",
    "FlowMU": "#C94338",
}
FONT_SCALE = 1.0
TEXT_SIZE = 18.5
PANEL_WSPACE = 0.12
PANEL_BG = "#F0F0F7"


def scaled(size: float) -> float:
    return size * FONT_SCALE


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def load_records() -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    with RECORDS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["panel"], row["method"])
            grouped.setdefault(key, []).append(float(row["relative_causal_effect"]))
    return grouped


def load_y_limit() -> tuple[float, float]:
    with STYLE_JSON.open("r", encoding="utf-8") as f:
        style = json.load(f)
    low, high = style.get("y_axis_limit", [0.0, 0.035])
    return float(low), float(high)


def draw_box(ax: plt.Axes, x: float, values: np.ndarray, method: str) -> None:
    q25, q35, median, q75, q90 = np.quantile(values, [0.25, 0.35, 0.5, 0.75, 0.9])
    edge = "#4B5563"
    box_width = 0.36
    cap_width = 0.26

    ax.vlines(x, q25, q90, color=edge, linewidth=0.7, zorder=4)
    ax.hlines([q25, q90], x - cap_width / 2, x + cap_width / 2, color=edge, linewidth=0.7, zorder=4)
    ax.add_patch(
        Rectangle(
            (x - box_width / 2, q35),
            box_width,
            q75 - q35,
            facecolor=COLORS[method],
            edgecolor=edge,
            linewidth=0.7,
            alpha=0.80,
            zorder=3,
        )
    )
    ax.hlines(median, x - box_width / 2, x + box_width / 2, color="#151515", linewidth=0.9, zorder=5)


def draw_panel(
    ax: plt.Axes,
    *,
    panel: str,
    caption: str,
    records: dict[tuple[str, str], list[float]],
    y_limit: tuple[float, float],
    show_ylabel: bool,
) -> None:
    rng = np.random.default_rng(20260716 + (0 if panel == "forget" else 1000))
    positions = np.arange(1, len(METHOD_ORDER) + 1, dtype=float)
    ax.set_facecolor(PANEL_BG)

    for pos, method in zip(positions, METHOD_ORDER):
        values = np.asarray(records[(panel, method)], dtype=float)
        jitter = rng.uniform(-0.045, 0.045, size=values.size)
        ax.scatter(
            pos + jitter,
            values,
            s=4.8,
            color="#2F2F2F",
            alpha=0.12,
            linewidths=0,
            zorder=2,
        )
        draw_box(ax, pos, values, method)

    ax.axhline(0, color="#C43C39", linestyle=(0, (6, 3)), linewidth=0.7, alpha=0.85, zorder=1)
    ax.set_xlim(0.5, len(METHOD_ORDER) + 0.5)
    ax.set_ylim(*y_limit)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAME.get(method, method) for method in METHOD_ORDER])
    ax.set_xlabel("Localization Method", fontsize=TEXT_SIZE, labelpad=scaled(1))
    ax.set_yticks(np.arange(y_limit[0], y_limit[1] + 0.0001, 0.005))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, which="major", axis="both", linestyle="--", color="gray", alpha=0.36, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=TEXT_SIZE, length=2.5, width=0.7, color="#555555", pad=2)
    ax.tick_params(axis="y", labelsize=TEXT_SIZE, length=2.5, width=0.7, color="#555555", pad=2)

    if show_ylabel:
        ax.set_ylabel("Relative Causal Effect", fontsize=TEXT_SIZE, labelpad=scaled(8))
        ax.yaxis.set_label_coords(-0.18, 0.43)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    for spine in ax.spines.values():
        spine.set_color("none")
        spine.set_alpha(0.0)
        spine.set_linewidth(0.0)

    ax.text(
        0.5,
        -0.27,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=TEXT_SIZE,
    )


def draw(output_path: Path) -> None:
    configure_matplotlib()
    records = load_records()
    y_limit = load_y_limit()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.75), sharey=True, facecolor="white")
    fig.patch.set_facecolor("white")
    for idx, (panel, caption) in enumerate(PANELS):
        draw_panel(
            axes[idx],
            panel=panel,
            caption=caption,
            records=records,
            y_limit=y_limit,
            show_ylabel=idx == 0,
        )

    fig.subplots_adjust(left=0.135, right=0.995, bottom=0.34, top=0.965, wspace=PANEL_WSPACE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def main() -> None:
    for ext in ("png", "pdf", "svg"):
        path = SCRIPT_DIR / f"{STEM}.{ext}"
        draw(path)
        print(path)


if __name__ == "__main__":
    main()
