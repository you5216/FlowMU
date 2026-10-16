import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parent
RATIOS = ["1%", "5%", "10%"]
METHODS = ["PURE", "SBU", "SF-NL", "FlowMU (GA)", "FlowMU (RMU)", "FlowMU (DPO)"]
DISPLAY_NAME = {
    "FlowMU (GA)": "Ours (GA)",
    "FlowMU (RMU)": "Ours (RMU)",
    "FlowMU (DPO)": "Ours (DPO)",
}
                                                                      
                                                                                 
TEXT_SIZE = 18.5
LEGEND_FONTSIZE = TEXT_SIZE
PANEL_BG = "#F0F0F7"
LEGEND_BG = "#F7F7FB"

                                                              
RAW = {
    "TOFU": {
        "Vanilla": {
            "1%": (92.50, 89.32, 58.85, 57.33),
            "5%": (90.50, 89.18, 58.99, 57.25),
            "10%": (90.25, 89.19, 58.66, 57.20),
        },
        "PURE": {
            "1%": (47.50, 87.58, 39.19, 51.68),
            "5%": (53.00, 86.66, 37.84, 49.47),
            "10%": (51.75, 86.92, 37.37, 50.16),
        },
        "SBU": {
            "1%": (67.50, 90.23, 43.68, 63.77),
            "5%": (63.50, 89.97, 41.91, 63.69),
            "10%": (61.75, 89.89, 41.15, 63.01),
        },
        "SF-NL": {
            "1%": (85.00, 79.37, 41.84, 36.77),
            "5%": (78.00, 79.39, 38.83, 36.61),
            "10%": (76.50, 79.75, 36.59, 36.66),
        },
        "FlowMU (GA)": {
            "1%": (20.00, 74.12, 32.69, 67.61),
            "5%": (31.50, 79.84, 30.92, 51.99),
            "10%": (35.25, 83.44, 36.91, 57.71),
        },
        "FlowMU (DPO)": {
            "1%": (22.50, 82.05, 35.11, 71.86),
            "5%": (28.50, 75.47, 36.53, 67.32),
            "10%": (31.00, 78.03, 37.88, 57.21),
        },
        "FlowMU (RMU)": {
            "1%": (22.50, 74.27, 32.70, 67.93),
            "5%": (31.50, 75.63, 37.30, 67.40),
            "10%": (32.25, 83.83, 32.23, 63.32),
        },
    },
    "LiteraryQA": {
        "Vanilla": {
            "1%": (80.88, 81.14, 55.52, 58.54),
            "5%": (81.05, 81.20, 58.32, 58.48),
            "10%": (81.88, 81.17, 58.76, 58.47),
        },
        "PURE": {
            "1%": (57.35, 98.57, 71.45, 89.47),
            "5%": (53.33, 98.73, 70.70, 87.85),
            "10%": (59.54, 96.57, 56.77, 70.09),
        },
        "SBU": {
            "1%": (53.92, 88.48, 70.73, 78.91),
            "5%": (47.58, 87.60, 72.36, 78.15),
            "10%": (49.04, 91.74, 71.37, 84.92),
        },
        "SF-NL": {
            "1%": (18.63, 39.56, 9.20, 31.33),
            "5%": (22.88, 39.42, 9.81, 31.28),
            "10%": (24.87, 39.24, 11.32, 31.16),
        },
        "FlowMU (GA)": {
            "1%": (34.31, 96.85, 53.86, 81.66),
            "5%": (31.35, 98.29, 48.54, 66.75),
            "10%": (42.13, 88.55, 40.02, 53.81),
        },
        "FlowMU (DPO)": {
            "1%": (35.29, 98.21, 63.79, 86.63),
            "5%": (30.85, 90.12, 55.98, 80.80),
            "10%": (35.03, 83.28, 29.21, 39.29),
        },
        "FlowMU (RMU)": {
            "1%": (29.41, 98.28, 55.93, 75.85),
            "5%": (33.97, 98.39, 50.18, 66.88),
            "10%": (42.49, 86.24, 43.17, 56.41),
        },
    },
}

COLORS = {
    "PURE": "#8F9DAE",
    "SBU": "#AEBCC8",
    "SF-NL": "#BBB5C9",
    "FlowMU (GA)": "#C94338",
    "FlowMU (RMU)": "#E8954C",
    "FlowMU (DPO)": "#9B2148",
}


def uts(f_score, r_score, r0_score):
    retain_term = min(r_score / r0_score, 1.0) * 100.0
    forget_term = 100.0 - f_score
    return 2.0 * forget_term * retain_term / (forget_term + retain_term)


def compute_uts_data():
    datasets = {}
    for dataset, rows in RAW.items():
        datasets[dataset] = {}
        for method in METHODS:
            acc_values = []
            rouge_values = []
            for ratio in RATIOS:
                f_acc, r_acc, f_rouge, r_rouge = rows[method][ratio]
                _, vanilla_r_acc, _, vanilla_r_rouge = rows["Vanilla"][ratio]
                acc_values.append(round(uts(f_acc, r_acc, vanilla_r_acc), 2))
                rouge_values.append(round(uts(f_rouge, r_rouge, vanilla_r_rouge), 2))
            datasets[dataset][method] = {
                "ACC-UTS": acc_values,
                "Rouge-L-UTS": rouge_values,
            }
    return {
        "metric": "UTS",
        "formula": "UTS = 2 * (100 - F) * min(R/R0, 1) * 100 / ((100 - F) + min(R/R0, 1) * 100)",
        "source": "Recomputed from the paper main-result table values.",
        "forget_ratios": RATIOS,
        "methods": METHODS,
        "datasets": datasets,
    }


def draw_dataset(dataset, metrics):
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
            "axes.unicode_minus": False,
            "hatch.linewidth": 1.15,
        }
    )
    fig, ax = plt.subplots(figsize=(11.56, 4.30), dpi=360, facecolor="white")
    ax.set_facecolor(PANEL_BG)

    centers = [0.0, 1.0, 2.0]
    method_width = 0.13
    bar_width = 0.052
    offsets = [(i - (len(METHODS) - 1) / 2) * method_width for i in range(len(METHODS))]

    for i, method in enumerate(METHODS):
        xs_acc = [c + offsets[i] - bar_width / 2 for c in centers]
        xs_rouge = [c + offsets[i] + bar_width / 2 for c in centers]
        color = COLORS[method]
        ax.bar(
            xs_acc,
            metrics[method]["ACC-UTS"],
            width=bar_width,
            color=color,
            edgecolor="#4B5563",
            linewidth=0.55,
            alpha=0.96,
            zorder=3,
        )
        ax.bar(
            xs_rouge,
            metrics[method]["Rouge-L-UTS"],
            width=bar_width,
            color=color,
            edgecolor="#374151",
            linewidth=0.55,
            hatch="///",
            alpha=0.96,
            zorder=3,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(RATIOS)
    ax.set_xlabel("Forget Ratio", labelpad=8)
    ax.set_ylabel("UTS Score", labelpad=8)
    ax.set_ylim(24, 102)
    ax.set_yticks(range(30, 101, 10))
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)
    ax.grid(True, which="major", axis="y", linestyle="--", color="gray", alpha=0.5, linewidth=0.9)
    ax.grid(False, axis="x")

    for spine in ax.spines.values():
        spine.set_alpha(0.0)

    method_handles = [
        Patch(facecolor=COLORS[m], edgecolor="#4B5563", linewidth=0.55, label=DISPLAY_NAME.get(m, m))
        for m in METHODS
    ]
    metric_handles = [
        Patch(facecolor="#F7F7FB", edgecolor="#374151", linewidth=0.6, label="ACC-UTS"),
        Patch(facecolor="#F7F7FB", edgecolor="#374151", linewidth=0.6, hatch="///", label="Rouge-L-UTS"),
    ]
    leg1 = ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.06),
        ncol=6,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        facecolor=LEGEND_BG,
        edgecolor="#B7BBC8",
        framealpha=0.96,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.75,
        borderpad=0.25,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        facecolor=LEGEND_BG,
        edgecolor="#B7BBC8",
        framealpha=0.96,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.25,
    )

    fig.tight_layout(pad=0.5)
    stem = f"uts_performance_{dataset.lower()}_allmethods"
    fig.savefig(
        OUT_DIR / f"{stem}.png",
        dpi=360,
        bbox_inches="tight",
        bbox_extra_artists=(leg1, leg2),
        facecolor="white",
    )
    fig.savefig(
        OUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        bbox_extra_artists=(leg1, leg2),
        facecolor="white",
    )
    plt.close(fig)


def main():
    data = compute_uts_data()
    (OUT_DIR / "uts_performance_allmethods_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for dataset, metrics in data["datasets"].items():
        draw_dataset(dataset, metrics)


if __name__ == "__main__":
    main()
