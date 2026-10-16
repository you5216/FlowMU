                      
                                                     

                                                                              
   

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "tofu_for05_activation_strength_heatmap_data.npz"
STEM = "tofu_for05_activation_strength_heatmap"
FIGURE_SCALE = 2.0 / 3.0
FONT_SCALE = FIGURE_SCALE
TEXT_SIZE = 18.5 * FIGURE_SCALE
DISPLAY_MAX = 0.5
PANEL_WSPACE = 0.20
PANEL_BG = "#F0F0F7"
DISPLAY_NAME = {"FlowMU": "Ours"}
HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "flowmu_red_gradient",
    ["#FFF7F6", "#F4D0CB", "#E99C94", "#D96357", "#C94338", "#8F1D18"],
)


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


def load_data() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    payload = np.load(DATA_PATH, allow_pickle=True)
    methods = [DISPLAY_NAME.get(str(item), str(item)) for item in payload["methods"]]
    ytick_labels = [str(item) for item in payload["ytick_labels"]]
    forget_display = np.asarray(payload["forget_display"], dtype=np.float64)
    retain_display = np.asarray(payload["retain_display"], dtype=np.float64)
    return methods, ytick_labels, forget_display, retain_display


def draw(output_path: Path) -> None:
    configure_matplotlib()
    methods, ytick_labels, forget_display, retain_display = load_data()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2 * FIGURE_SCALE, 4.2 * FIGURE_SCALE),
        sharey=True,
        facecolor="white",
    )
    fig.patch.set_facecolor("white")
    panels = [
        (axes[0], forget_display, "Forget"),
        (axes[1], retain_display, "Retain"),
    ]

    for ax, matrix, title in panels:
        ax.set_facecolor(PANEL_BG)
        ax.imshow(
            matrix,
            cmap=HEATMAP_CMAP,
            vmin=0.0,
            vmax=DISPLAY_MAX,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=TEXT_SIZE, pad=scaled(3))
        ax.set_xlabel("")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_yticks(np.arange(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels)
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.28)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", labelrotation=0, labelsize=TEXT_SIZE, length=2.5, width=0.7, color="#555555", pad=2)
        ax.tick_params(axis="y", labelrotation=0, labelsize=TEXT_SIZE, length=2.5, width=0.7, color="#555555", pad=2)
        for spine in ax.spines.values():
            spine.set_alpha(0.0)

    axes[0].set_ylabel("Layer", fontsize=TEXT_SIZE, labelpad=scaled(3))
    axes[1].set_ylabel("")

    plt.subplots_adjust(left=0.10, right=0.86, bottom=0.16, top=0.84, wspace=PANEL_WSPACE)
    cbar_ax = fig.add_axes([0.895, 0.22, 0.020, 0.56])
    cbar_ax.set_facecolor(PANEL_BG)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=DISPLAY_MAX), cmap=HEATMAP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Activation Strength", fontsize=TEXT_SIZE, labelpad=scaled(5))
    cbar.ax.tick_params(labelsize=TEXT_SIZE, length=2.5, width=0.7, color="#555555", pad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=360, bbox_inches="tight", pad_inches=0.025 * FIGURE_SCALE, facecolor="white")
    plt.close(fig)


def main() -> None:
    for ext in ("png", "pdf", "svg"):
        path = SCRIPT_DIR / f"{STEM}.{ext}"
        draw(path)
        print(path)


if __name__ == "__main__":
    main()
