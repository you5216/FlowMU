                      
                                                                          

from __future__ import annotations

import csv
import json
import shutil
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


AGENT_ROOT = Path(__file__).resolve().parents[6]
INPUT_CSV = (
    AGENT_ROOT
    / "final_performance/literaryQA/for05_re95/small_exp/sensitive_neuron_localization/"
    / "topk_path_ablation/real_topk_neuron_ablation_lqa_for05_drop_forget200_20260706.csv"
)
OUTPUT_DIR = AGENT_ROOT / "final_performance/final/literaryqa_for05_topk_neuron_drop_ablation"
STEM = "literaryqa_for05_topk_neuron_drop_ablation"

POWER_TOPKS = [2**i for i in range(1, 14)]
ALL_TOPK = 12288
METHODS = ("FlowMU", "Random")
STYLE = {
    "FlowMU": {
        "color": "#C43C2F",
        "linestyle": "-",
        "marker": "o",
        "label": "FlowMU",
    },
    "Random": {
        "color": "#2E6DA4",
        "linestyle": "--",
        "marker": "s",
        "label": "Random",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stem", default=STEM)
    parser.add_argument("--panel", choices=("forget", "retain"), default="forget")
    return parser.parse_args()


def load_rows(path: Path, panel: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [
        row
        for row in rows
        if row["split"] == "for05_re95"
        and row["panel"].lower() == panel
        and row["mask_mode"] == "drop"
        and row["method"] in METHODS
    ]
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def build_plot_rows(rows: list[dict[str, str]], panel: str) -> list[dict[str, object]]:
    by_method_topk: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        by_method_topk[(row["method"], int(row["topk"]))] = row

    flowmu_all = by_method_topk.get(("FlowMU", ALL_TOPK))
    if flowmu_all is None:
        raise ValueError("Missing FlowMU All-FFN row.")
    if ("Random", ALL_TOPK) not in by_method_topk:
        filled = dict(flowmu_all)
        filled["method"] = "Random"
        filled["label"] = "Random_drop_topk12288_seed0_filled_from_shared_all"
        filled["input_csv"] = "filled_from_FlowMU_All_shared_all_FFN_ablation"
        by_method_topk[("Random", ALL_TOPK)] = filled

    baselines = {}
    for method in METHODS:
        row = by_method_topk.get((method, 2))
        if row is None:
            raise ValueError(f"Missing {method} topk=2 baseline.")
        baselines[method] = float(row["rouge_l"])

    plot_rows: list[dict[str, object]] = []
    for method in METHODS:
        for topk in [*POWER_TOPKS, ALL_TOPK]:
            row = by_method_topk.get((method, topk))
            if row is None:
                raise ValueError(f"Missing {method} topk={topk}")
            rouge_l = float(row["rouge_l"])
            acc = float(row["acc"])
            ratio = rouge_l / baselines[method] if baselines[method] else 0.0
            plot_rows.append(
                {
                    "method": method,
                    "topk": topk,
                    "x_label": "All" if topk == ALL_TOPK else f"2^{int(np.log2(topk))}",
                    "rouge_l": rouge_l,
                    "acc": acc,
                    "num_samples": int(row["num_samples"]),
                    "source_csv": row["input_csv"],
                    f"retained_{panel}_rouge_ratio": ratio,
                    f"{panel}_drop_ratio": 1.0 - ratio,
                }
            )
    return plot_rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def write_plot_data(plot_rows: list[dict[str, object]], output_path: Path, panel: str) -> None:
    fieldnames = [
        "method",
        "topk",
        "x_label",
        "rouge_l",
        "acc",
        "num_samples",
        "source_csv",
        f"retained_{panel}_rouge_ratio",
        f"{panel}_drop_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_rows)


def draw(plot_rows: list[dict[str, object]], output_dir: Path, stem: str, panel: str) -> list[Path]:
    configure_matplotlib()

    x_labels = [rf"$2^{{{i}}}$" for i in range(1, 14)] + ["All"]
    x_positions = np.arange(len(x_labels), dtype=float)
    topk_to_x = {topk: idx for idx, topk in enumerate([*POWER_TOPKS, ALL_TOPK])}

    fig, ax = plt.subplots(figsize=(6.0, 2.5))

    for method in METHODS:
        method_rows = [
            row
            for row in plot_rows
            if row["method"] == method and int(row["topk"]) in POWER_TOPKS
        ]
        method_rows.sort(key=lambda row: int(row["topk"]))
        xs = [topk_to_x[int(row["topk"])] for row in method_rows]
        ys = [float(row[f"retained_{panel}_rouge_ratio"]) for row in method_rows]
        ax.plot(
            xs,
            ys,
            linewidth=1.55,
            markersize=2.7,
            markeredgewidth=0.0,
            **STYLE[method],
        )

    all_rows = [row for row in plot_rows if int(row["topk"]) == ALL_TOPK]
    all_ratio = float(np.mean([float(row[f"retained_{panel}_rouge_ratio"]) for row in all_rows]))
    ax.scatter(
        [topk_to_x[ALL_TOPK]],
        [all_ratio],
        marker="D",
        s=21,
        color="#5E5E5E",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="All FFN",
    )

    ax.set_xlim(-0.35, len(x_labels) - 0.65)
    ax.set_ylim(-0.035, 1.08)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Dropped FFN Neurons per Layer")
    ax.set_ylabel(f"{panel.capitalize()} Rouge-L Retention")
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555")
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=3,
        handlelength=2.25,
        columnspacing=1.35,
    )
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.22, top=0.84)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.pdf",
    ]
    for path in outputs:
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return outputs


def write_summary(
    plot_rows: list[dict[str, object]],
    outputs: list[Path],
    input_csv: Path,
    output_dir: Path,
    stem: str,
    panel: str,
) -> None:
    by_method = {
        method: {
            int(row["topk"]): {
                "rouge_l": float(row["rouge_l"]),
                "acc": float(row["acc"]),
                f"retained_{panel}_rouge_ratio": float(row[f"retained_{panel}_rouge_ratio"]),
            }
            for row in plot_rows
            if row["method"] == method
        }
        for method in METHODS
    }
    summary = {
        "input_csvs": [str(input_csv)],
        "outputs": [str(path) for path in outputs],
        "metric": "rouge_l",
        "panel": panel,
        "display_transform": f"retained_{panel}_rouge_ratio = real {panel} rouge_l at top-k / real {panel} rouge_l at 2^1 for the same method",
        "x_axis": "all powers from 2^1 to 2^13 plus All",
        "style": "FlowMU solid red line, Random dashed blue line, small markers",
        "random_all_neuron_point": "Random topk=12288 filled from FlowMU topk=12288 because all FFN neurons are ablated and the mask is method-independent.",
        "all_endpoint_style": "All FFN is plotted as a separate shared gray diamond and is not connected to FlowMU or Random curves.",
        "rows_for_plot": len(plot_rows),
        "shared_all_raw_rouge_l": by_method["FlowMU"][ALL_TOPK]["rouge_l"],
        "shared_all_retained_ratio_mean": float(
            np.mean(
                [
                    by_method["FlowMU"][ALL_TOPK][f"retained_{panel}_rouge_ratio"],
                    by_method["Random"][ALL_TOPK][f"retained_{panel}_rouge_ratio"],
                ]
            )
        ),
        "key_points": {
            "FlowMU_topk2048": by_method["FlowMU"][2048],
            "Random_topk2048": by_method["Random"][2048],
            "FlowMU_topk4096": by_method["FlowMU"][4096],
            "Random_topk4096": by_method["Random"][4096],
            "FlowMU_topk8192": by_method["FlowMU"][8192],
            "Random_topk8192": by_method["Random"][8192],
        },
    }
    with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, args.panel)
    plot_rows = build_plot_rows(rows, args.panel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input_csv, args.output_dir / f"{args.stem}_real_data.csv")
    write_plot_data(plot_rows, args.output_dir / f"{args.stem}_plot_data.csv", args.panel)
    outputs = draw(plot_rows, args.output_dir, args.stem, args.panel)
    write_summary(plot_rows, outputs, args.input_csv, args.output_dir, args.stem, args.panel)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
