                      
                                                                               

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["Vanilla", "PURE", "SBU", "SF-NL", "FlowMU"]
ATTACK_ORDER = ["Clean", "Rewrite", "Induce"]
COLORS = {
    "Clean": "#8a929d",
    "Rewrite": "#4f87b8",
    "Induce": "#d08a3c",
    "RecoveryRewrite": "#5c8f6b",
    "RecoveryInduce": "#c05f45",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "attack": row["attack"],
                    "clean_forget_acc": float(row["clean_forget_acc"]),
                    "attack_forget_acc": float(row["attack_forget_acc"]),
                    "attack_recovery_rate": float(row["attack_recovery_rate"]),
                    "row_attack_acc": float(row["row_attack_acc"]),
                    "mean_variant_acc": float(row["mean_variant_acc"]),
                    "num_rows": int(row["num_rows"]),
                    "num_sources": int(row["num_sources"]),
                }
            )
    return rows


def lookup(records: list[dict[str, Any]], method: str, attack: str) -> dict[str, Any]:
    matches = [row for row in records if row["method"] == method and row["attack"] == attack]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {method}/{attack}, got {len(matches)}")
    return matches[0]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw(records: list[dict[str, Any]], output_dir: Path, stem: str) -> dict[str, str]:
    configure()
    methods = [m for m in METHOD_ORDER if any(row["method"] == m for row in records)]
    x = np.arange(len(methods), dtype=float)

    acc_values = {"Clean": [], "Rewrite": [], "Induce": []}
    recovery_values = {"Rewrite": [], "Induce": []}
    for method in methods:
        rewrite = lookup(records, method, "Rewrite")
        induce = lookup(records, method, "Induce")
        acc_values["Clean"].append(100.0 * rewrite["clean_forget_acc"])
        acc_values["Rewrite"].append(100.0 * rewrite["attack_forget_acc"])
        acc_values["Induce"].append(100.0 * induce["attack_forget_acc"])
        recovery_values["Rewrite"].append(100.0 * rewrite["attack_recovery_rate"])
        recovery_values["Induce"].append(100.0 * induce["attack_recovery_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    width = 0.22
    offsets = {"Clean": -width, "Rewrite": 0.0, "Induce": width}

    ax = axes[0]
    for label in ATTACK_ORDER:
        ax.bar(
            x + offsets[label],
            acc_values[label],
            width=width,
            color=COLORS[label],
            edgecolor="#2f3437",
            linewidth=0.45,
            label=label,
            zorder=3,
        )
    ax.set_ylabel("Forget answer recovery (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    max_acc = max(max(values) for values in acc_values.values())
    ax.set_ylim(0, min(100, max(45, max_acc + 10)))
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.24, "(a) Clean vs. attacked forget accuracy", transform=ax.transAxes, ha="center")
    ax.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0.0, 1.04))
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.8)

    ax = axes[1]
    rec_width = 0.28
    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.bar(
        x - rec_width / 2,
        recovery_values["Rewrite"],
        width=rec_width,
        color=COLORS["RecoveryRewrite"],
        edgecolor="#2f3437",
        linewidth=0.45,
        label="Rewrite",
        zorder=3,
    )
    ax.bar(
        x + rec_width / 2,
        recovery_values["Induce"],
        width=rec_width,
        color=COLORS["RecoveryInduce"],
        edgecolor="#2f3437",
        linewidth=0.45,
        label="Induce",
        zorder=3,
    )
    ax.set_ylabel("Attack recovery gain (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    max_gain = max(max(recovery_values["Rewrite"]), max(recovery_values["Induce"]), 0)
    min_gain = min(min(recovery_values["Rewrite"]), min(recovery_values["Induce"]), 0)
    lower = min(-5, min_gain - 4)
    upper = max(15, max_gain + 5)
    ax.set_ylim(lower, upper)
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.24, "(b) Recovery above clean", transform=ax.transAxes, ha="center")
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.8)

    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.90, wspace=0.28)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": str(output_dir / f"{stem}.svg"),
        "png": str(output_dir / f"{stem}.png"),
        "pdf": str(output_dir / f"{stem}.pdf"),
    }
    fig.savefig(paths["svg"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_data = []
    for method in methods:
        rewrite = lookup(records, method, "Rewrite")
        induce = lookup(records, method, "Induce")
        plot_data.append(
            {
                "method": method,
                "clean_forget_acc_pct": 100.0 * rewrite["clean_forget_acc"],
                "rewrite_attack_acc_pct": 100.0 * rewrite["attack_forget_acc"],
                "induce_attack_acc_pct": 100.0 * induce["attack_forget_acc"],
                "rewrite_recovery_gain_pp": 100.0 * rewrite["attack_recovery_rate"],
                "induce_recovery_gain_pp": 100.0 * induce["attack_recovery_rate"],
            }
        )
    (output_dir / f"{stem}_plot_data.json").write_text(
        json.dumps({"records": plot_data}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot TOFU for05 rewrite/induce robustness figure.")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stem", default="tofu_for05_rewrite_induce_attack")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input_csv)
    paths = draw(records, args.output_dir, args.stem)
    for kind, path in paths.items():
        print(f"saved {kind}: {path}")


if __name__ == "__main__":
    main()
