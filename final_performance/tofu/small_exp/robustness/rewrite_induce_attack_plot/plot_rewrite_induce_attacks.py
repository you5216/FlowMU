                      
                                                            

                                                                          
                                             

                                                                         

            
                                                        

                       
                 

                    
                                                             

                                                      
   

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = ROOT / "final_performance/tofu/small_exp/robustness/rewrite_induce_attack_plot"

SPLITS = ["for01_re99", "for05_re95", "for10_re90"]
ATTACKS = ["Rewrite", "Induce"]
METHODS = ["GA", "DPO", "RMU", "FlowMU"]
METHOD_COLORS = {
    "GA": "#8d99ae",
    "DPO": "#5b8db8",
    "RMU": "#7aa974",
    "FlowMU": "#e19a5f",
    "Ours": "#e19a5f",
}


def load_mock_records(seed: int = 20260630) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    clean_base = {
        "GA": [0.24, 0.31, 0.33],
        "DPO": [0.23, 0.29, 0.32],
        "RMU": [0.25, 0.31, 0.34],
        "FlowMU": [0.11, 0.20, 0.22],
    }
    attack_gain = {
        "Rewrite": {"GA": 0.18, "DPO": 0.15, "RMU": 0.13, "FlowMU": 0.04},
        "Induce": {"GA": 0.22, "DPO": 0.18, "RMU": 0.17, "FlowMU": 0.06},
    }
    for split_idx, split in enumerate(SPLITS):
        for attack in ATTACKS:
            for method in METHODS:
                clean = clean_base[method][split_idx] + rng.normal(0, 0.006)
                attacked = clean + attack_gain[attack][method] + rng.normal(0, 0.008)
                records.append(
                    {
                        "split": split,
                        "attack": attack,
                        "method": method,
                        "clean_forget_acc": float(np.clip(clean, 0.0, 1.0)),
                        "attack_forget_acc": float(np.clip(attacked, 0.0, 1.0)),
                    }
                )
    return records


def load_csv(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "attack", "method", "clean_forget_acc", "attack_forget_acc"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            records.append(
                {
                    "split": row["split"].strip(),
                    "attack": row["attack"].strip(),
                    "method": row["method"].strip(),
                    "clean_forget_acc": float(row["clean_forget_acc"]),
                    "attack_forget_acc": float(row["attack_forget_acc"]),
                }
            )
    return records


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, list):
        raise TypeError("JSON input must be a list or {'records': [...]} object.")
    return [dict(item) for item in payload]


def validate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("No attack records loaded.")
    for row in records:
        for key in ("clean_forget_acc", "attack_forget_acc"):
            value = float(row[key])
            if not np.isfinite(value) or value < 0 or value > 1:
                raise ValueError(f"{key} must be in [0, 1], got {value}")
            row[key] = value
        row["attack_recovery_rate"] = row["attack_forget_acc"] - row["clean_forget_acc"]
    return records


def infer_order(records: list[dict[str, Any]], field: str, preferred: list[str]) -> list[str]:
    present = {str(row[field]) for row in records}
    out = [item for item in preferred if item in present]
    out.extend(sorted(present - set(out)))
    return out


def lookup(records: list[dict[str, Any]], split: str, attack: str, method: str, key: str) -> float:
    matches = [
        row
        for row in records
        if row["split"] == split and row["attack"] == attack and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {split}/{attack}/{method}, got {len(matches)}")
    return float(matches[0][key])


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_attack_acc(records: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    configure_matplotlib()
    records = validate_records(records)
    splits = infer_order(records, "split", SPLITS)
    attacks = infer_order(records, "attack", ATTACKS)
    methods = infer_order(records, "method", METHODS)

    fig, axes = plt.subplots(1, len(attacks), figsize=(11, 4.5), sharey=True)
    if len(attacks) == 1:
        axes = np.asarray([axes])
    x = np.arange(len(splits), dtype=float)
    width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for ax, attack in zip(axes, attacks):
        for offset, method in zip(offsets, methods):
            values = [lookup(records, split, attack, method, "attack_forget_acc") for split in splits]
            ax.bar(
                x + offset,
                values,
                width=width,
                color=METHOD_COLORS.get(method, "#999999"),
                edgecolor="#333333",
                linewidth=0.5,
                label=method,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([split.replace("_", "\n") for split in splits], fontsize=10)
        ax.set_ylim(0.0, 0.70)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.text(
            0.5,
            -0.26,
            f"({chr(97 + attacks.index(attack))}) {attack} Attack",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Attack Forget Acc")
    axes[-1].legend(frameon=False, loc="upper right")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.95, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}_attack_forget_acc.svg"
    png_path = output_dir / f"{stem}_attack_forget_acc.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def draw_recovery(records: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    configure_matplotlib()
    records = validate_records(records)
    attacks = infer_order(records, "attack", ATTACKS)
    methods = infer_order(records, "method", METHODS)

    labels = [f"{split.replace('_re', '/')}\n{attack}" for attack in attacks for split in SPLITS if any(row["split"] == split and row["attack"] == attack for row in records)]
    groups = [(split, attack) for attack in attacks for split in SPLITS if any(row["split"] == split and row["attack"] == attack for row in records)]
    x = np.arange(len(groups), dtype=float)
    width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for offset, method in zip(offsets, methods):
        values = [lookup(records, split, attack, method, "attack_recovery_rate") for split, attack in groups]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="#333333",
            linewidth=0.5,
            label=method,
            zorder=3,
        )
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("Attack Recovery Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-0.05, 0.35)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.95)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}_attack_recovery.svg"
    png_path = output_dir / f"{stem}_attack_recovery.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def save_summary(records: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    records = validate_records(records)
    csv_path = output_dir / f"{stem}_summary.csv"
    json_path = output_dir / f"{stem}_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "attack",
                "method",
                "clean_forget_acc",
                "attack_forget_acc",
                "attack_recovery_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def save_mock_templates(records: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path, json_path = save_summary(records, output_dir, "mock_rewrite_induce_attack_data")
    print(f"Saved mock CSV template: {csv_path}")
    print(f"Saved mock JSON template: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Rewrite/Induce attack robustness plots.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="rewrite_induce_attack")
    parser.add_argument("--save-mock-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_csv and args.input_json:
        raise ValueError("Use only one of --input-csv or --input-json.")
    if args.input_csv:
        records = load_csv(args.input_csv)
        source = str(args.input_csv)
    elif args.input_json:
        records = load_json(args.input_json)
        source = str(args.input_json)
    else:
        records = load_mock_records()
        source = "deterministic mock data"

    if args.save_mock_template:
        save_mock_templates(load_mock_records(), args.output_dir)

    acc_paths = draw_attack_acc(records, args.output_dir, args.stem)
    recovery_paths = draw_recovery(records, args.output_dir, args.stem)
    summary_paths = save_summary(records, args.output_dir, args.stem)
    print(f"Data source: {source}")
    print(f"Saved attack acc SVG: {acc_paths[0]}")
    print(f"Saved attack acc PNG: {acc_paths[1]}")
    print(f"Saved recovery SVG: {recovery_paths[0]}")
    print(f"Saved recovery PNG: {recovery_paths[1]}")
    print(f"Saved summary CSV: {summary_paths[0]}")
    print(f"Saved summary JSON: {summary_paths[1]}")


if __name__ == "__main__":
    main()
