                      
                                                                                  

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


AGENT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent")
EXP_DIR = (
    AGENT_ROOT
    / "final_performance/literaryQA/for05_re95/small_exp/"
    / "sensitive_neuron_localization/topk_path_ablation"
)
ORIGINAL_CSV = EXP_DIR / "real_topk_neuron_ablation_lqa_for05_drop_retain500_20260706.csv"
MULTISEED_CSV = (
    EXP_DIR
    / "real_topk_neuron_ablation_lqa_for05_drop_retain500_multiseed_check_20260706.csv"
)
OUTPUT_DIR = AGENT_ROOT / "final_performance/final/literaryqa_for05_topk_neuron_drop_ablation"
STEM = "literaryqa_for05_topk_neuron_drop_ablation_retain500_multiseed"

POWER_TOPKS = [2**i for i in range(1, 14)]
ALL_TOPK = 12288
METHODS = ("FlowMU", "Random")
RETAIN_PANEL = "retain"

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def usable_rows(path: Path) -> list[dict[str, str]]:
    return [
        row
        for row in read_rows(path)
        if row["split"] == "for05_re95"
        and row["panel"].lower() == RETAIN_PANEL
        and row["mask_mode"] == "drop"
        and row["method"] in METHODS
    ]


def collect_values() -> dict[tuple[str, int], list[dict[str, object]]]:
    values: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for source_name, path in (("full_curve_seed0", ORIGINAL_CSV), ("multiseed_check", MULTISEED_CSV)):
        for row in usable_rows(path):
            method = row["method"]
            topk = int(row["topk"])
            values[(method, topk)].append(
                {
                    "rouge_l": float(row["rouge_l"]),
                    "acc": float(row["acc"]),
                    "seed": int(row["seed"]),
                    "source_name": source_name,
                    "source_csv": str(path),
                    "input_csv": row["input_csv"],
                }
            )
    return values


def aggregate_rows() -> list[dict[str, object]]:
    values = collect_values()
    if ("FlowMU", ALL_TOPK) not in values:
        raise ValueError("Missing FlowMU All-FFN point.")
    if ("Random", ALL_TOPK) not in values:
        values[("Random", ALL_TOPK)] = [
            {
                **entry,
                "source_name": "filled_shared_all",
                "input_csv": "filled_from_FlowMU_All_shared_all_FFN_ablation",
            }
            for entry in values[("FlowMU", ALL_TOPK)]
        ]

    baselines: dict[str, float] = {}
    for method in METHODS:
        baseline_entries = values.get((method, 2))
        if not baseline_entries:
            raise ValueError(f"Missing baseline topk=2 for {method}.")
        baselines[method] = float(np.mean([entry["rouge_l"] for entry in baseline_entries]))

    rows: list[dict[str, object]] = []
    for method in METHODS:
        for topk in [*POWER_TOPKS, ALL_TOPK]:
            entries = values.get((method, topk))
            if not entries:
                raise ValueError(f"Missing {method} topk={topk}.")
            rouge_values = np.array([float(entry["rouge_l"]) for entry in entries], dtype=float)
            acc_values = np.array([float(entry["acc"]) for entry in entries], dtype=float)
            rouge_mean = float(np.mean(rouge_values))
            rouge_std = float(np.std(rouge_values, ddof=1)) if len(rouge_values) > 1 else 0.0
            acc_mean = float(np.mean(acc_values))
            retained_ratio_mean = rouge_mean / baselines[method] if baselines[method] else 0.0
            retained_ratio_std = rouge_std / baselines[method] if baselines[method] else 0.0
            rows.append(
                {
                    "method": method,
                    "topk": topk,
                    "x_label": "All" if topk == ALL_TOPK else f"2^{int(np.log2(topk))}",
                    "rouge_l_mean": rouge_mean,
                    "rouge_l_std": rouge_std,
                    "acc_mean": acc_mean,
                    "n": len(rouge_values),
                    "seeds": ";".join(str(entry["seed"]) for entry in entries),
                    "source_names": ";".join(str(entry["source_name"]) for entry in entries),
                    "retained_retain_rouge_ratio_mean": retained_ratio_mean,
                    "retained_retain_rouge_ratio_std": retained_ratio_std,
                    "retain_drop_ratio_mean": 1.0 - retained_ratio_mean,
                }
            )
    return rows


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


def write_plot_data(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "method",
        "topk",
        "x_label",
        "rouge_l_mean",
        "rouge_l_std",
        "acc_mean",
        "n",
        "seeds",
        "source_names",
        "retained_retain_rouge_ratio_mean",
        "retained_retain_rouge_ratio_std",
        "retain_drop_ratio_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, object]], output_dir: Path, stem: str) -> list[Path]:
    configure_matplotlib()

    x_labels = [rf"$2^{{{i}}}$" for i in range(1, 14)] + ["All"]
    topks = [*POWER_TOPKS, ALL_TOPK]
    topk_to_x = {topk: idx for idx, topk in enumerate(topks)}

    fig, ax = plt.subplots(figsize=(7.0, 2.65))

    for method in METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method and int(row["topk"]) in POWER_TOPKS
        ]
        method_rows.sort(key=lambda row: int(row["topk"]))
        xs = np.array([topk_to_x[int(row["topk"])] for row in method_rows], dtype=float)
        ys = np.array([float(row["retained_retain_rouge_ratio_mean"]) for row in method_rows])
        yerr = np.array([float(row["retained_retain_rouge_ratio_std"]) for row in method_rows])
        ax.plot(
            xs,
            ys,
            linewidth=1.55,
            markersize=2.6,
            markeredgewidth=0.0,
            **STYLE[method],
        )
        checked = yerr > 0
        if np.any(checked):
            ax.errorbar(
                xs[checked],
                ys[checked],
                yerr=yerr[checked],
                fmt="none",
                ecolor=STYLE[method]["color"],
                elinewidth=0.75,
                capsize=2.0,
                capthick=0.75,
                alpha=0.82,
                zorder=3,
            )

    all_rows = [row for row in rows if int(row["topk"]) == ALL_TOPK]
    all_ratio = float(np.mean([float(row["retained_retain_rouge_ratio_mean"]) for row in all_rows]))
    ax.scatter(
        [topk_to_x[ALL_TOPK]],
        [all_ratio],
        marker="D",
        s=20,
        color="#5E5E5E",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="All FFN",
    )

    ax.set_xlim(-0.35, len(x_labels) - 0.65)
    ax.set_ylim(-0.035, 1.08)
    ax.set_xticks(np.arange(len(x_labels), dtype=float))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Dropped FFN Neurons per Layer")
    ax.set_ylabel("Retain Rouge-L Retention")
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


def write_summary(rows: list[dict[str, object]], outputs: list[Path], output_dir: Path, stem: str) -> None:
    def point(method: str, topk: int) -> dict[str, object]:
        row = next(row for row in rows if row["method"] == method and int(row["topk"]) == topk)
        return {
            "rouge_l_mean": float(row["rouge_l_mean"]),
            "rouge_l_std": float(row["rouge_l_std"]),
            "retained_retain_rouge_ratio_mean": float(row["retained_retain_rouge_ratio_mean"]),
            "retained_retain_rouge_ratio_std": float(row["retained_retain_rouge_ratio_std"]),
            "n": int(row["n"]),
            "seeds": row["seeds"],
        }

    summary = {
        "input_csvs": [str(ORIGINAL_CSV), str(MULTISEED_CSV)],
        "outputs": [str(path) for path in outputs],
        "metric": "rouge_l",
        "panel": "retain",
        "display_transform": "retained_retain_rouge_ratio = real retain rouge_l at top-k / real retain rouge_l at 2^1 for the same method",
        "multiseed_policy": "Use all available real rows. Random 2^10/2^11/2^12 and FlowMU 2^12 are plotted as mean with standard-deviation error bars; other points are the existing seed0 full curve.",
        "x_axis": "powers from 2^1 to 2^13 plus a separate All-FFN marker",
        "style": "FlowMU solid red line, Random dashed blue line, small markers; repeated points show error bars.",
        "random_all_neuron_point": "Random topk=12288 is filled from FlowMU topk=12288 because all FFN neurons are ablated and the mask is method-independent.",
        "key_points": {
            "FlowMU_topk4096": point("FlowMU", 4096),
            "Random_topk1024": point("Random", 1024),
            "Random_topk2048": point("Random", 2048),
            "Random_topk4096": point("Random", 4096),
            "FlowMU_topk8192": point("FlowMU", 8192),
            "Random_topk8192": point("Random", 8192),
        },
    }
    with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    rows = aggregate_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ORIGINAL_CSV, OUTPUT_DIR / f"{STEM}_full_curve_seed0_real_data.csv")
    shutil.copy2(MULTISEED_CSV, OUTPUT_DIR / f"{STEM}_multiseed_check_real_data.csv")
    write_plot_data(rows, OUTPUT_DIR / f"{STEM}_plot_data.csv")
    outputs = draw(rows, OUTPUT_DIR, STEM)
    write_summary(rows, outputs, OUTPUT_DIR, STEM)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
