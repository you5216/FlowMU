                      
                                                                           

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
METHODS = ("FlowMU", "QAActivation", "Random")
PLOT_POWERS = list(range(4, 14))
TEXT_SIZE = 23.5
LEGEND_FONTSIZE = TEXT_SIZE
SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
STYLE = {
    "FlowMU": {"color": "#C94338", "linestyle": "-", "marker": "o", "label": "Ours"},
    "QAActivation": {
        "color": "#BBB5C9",
        "linestyle": "-.",
        "marker": "^",
        "label": "QA-Activation",
    },
    "Random": {"color": "#6F7F90", "linestyle": "--", "marker": "s", "label": "Random"},
}
PANEL_BG = "#F0F0F7"
LEGEND_BG = "#F7F7FB"


def configure() -> None:
    mpl.rcParams.update(
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


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_alpha(0.0)
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)


def power_label(power: int) -> str:
    return "2" + str(power).translate(SUPERSCRIPT_DIGITS)


def draw_panel(data_path: Path, stem: str, ylabel: str) -> list[Path]:
    df = pd.read_csv(data_path)
    topk_to_x = {2**power: idx for idx, power in enumerate(PLOT_POWERS)}

    fig, ax = plt.subplots(figsize=(8.0, 3.75))
    fig.patch.set_facecolor("white")
    style_axes(ax)

    for method in METHODS:
        rows = df[(df["method"] == method) & (df["x_power"].isin(PLOT_POWERS))].sort_values(
            "topk"
        )
        xs = [topk_to_x[int(topk)] for topk in rows["topk"]]
        ys = rows["rouge_l"].astype(float).tolist()
        ax.plot(
            xs,
            ys,
            linewidth=1.75,
            markersize=4.2,
            markeredgewidth=0.0,
            zorder=3,
            **STYLE[method],
        )

    ax.set_xlim(-0.35, len(PLOT_POWERS) - 0.65)
    ax.set_ylim(-0.035, 1.03)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(np.arange(len(PLOT_POWERS), dtype=float))
    ax.set_xticklabels([power_label(power) for power in PLOT_POWERS])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.025, 0.055),
        ncol=1,
        frameon=True,
        handlelength=0.9,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    legend.get_frame().set_facecolor(LEGEND_BG)
    legend.get_frame().set_edgecolor("#B7BBC8")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)
    fig.subplots_adjust(left=0.125, right=0.995, bottom=0.18, top=0.98)

    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=520, bbox_inches="tight", pad_inches=0.025, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    configure()
    outputs = []
    outputs.extend(
        draw_panel(
            OUT_DIR / "topk_neuron_drop_ablation_forget_literaryqa_for05_raw_rougel_plot_data.csv",
            "topk_neuron_drop_ablation_forget_literaryqa_for05_raw_rougel",
            "Forget RougeL",
        )
    )
    outputs.extend(
        draw_panel(
            OUT_DIR / "topk_neuron_drop_ablation_retain_literaryqa_for05_raw_rougel_plot_data.csv",
            "topk_neuron_drop_ablation_retain_literaryqa_for05_raw_rougel",
            "Retain RougeL",
        )
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
