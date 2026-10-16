                      
                                                    

                                                                        
                                                                              
                                

                                                                         

            
                               

      
                                
                                        

                                           
   

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
DEFAULT_OUTPUT_DIR = ROOT / "final_performance/tofu/small_exp/robustness/component_wise_mia_plot"

ATTACKS = ["Standard", "Adaptive"]
COMPONENTS = ["RAG", "Memory", "LLM", "Agent"]
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
    base = {
        "Standard": {
            "GA": [0.68, 0.66, 0.72, 0.70],
            "DPO": [0.64, 0.63, 0.68, 0.66],
            "RMU": [0.62, 0.61, 0.65, 0.64],
            "FlowMU": [0.53, 0.54, 0.55, 0.54],
        },
        "Adaptive": {
            "GA": [0.77, 0.75, 0.82, 0.80],
            "DPO": [0.72, 0.70, 0.76, 0.74],
            "RMU": [0.69, 0.68, 0.73, 0.71],
            "FlowMU": [0.55, 0.56, 0.58, 0.57],
        },
    }
    for attack in ATTACKS:
        for method in METHODS:
            for component, value in zip(COMPONENTS, base[attack][method]):
                records.append(
                    {
                        "attack": attack,
                        "component": component,
                        "method": method,
                        "auroc": float(np.clip(value + rng.normal(0, 0.006), 0.0, 1.0)),
                    }
                )
    return records


def load_csv(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"attack", "component", "method", "auroc"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            records.append(
                {
                    "attack": row["attack"].strip(),
                    "component": row["component"].strip(),
                    "method": row["method"].strip(),
                    "auroc": float(row["auroc"]),
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
        raise ValueError("No component-wise MIA records loaded.")
    for row in records:
        value = float(row["auroc"])
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"AUROC must be in [0, 1], got {value}")
        row["auroc"] = value
    return records


def infer_order(records: list[dict[str, Any]], field: str, preferred: list[str]) -> list[str]:
    present = {str(row[field]) for row in records}
    out = [item for item in preferred if item in present]
    out.extend(sorted(present - set(out)))
    return out


def lookup(records: list[dict[str, Any]], attack: str, component: str, method: str) -> float:
    matches = [
        row
        for row in records
        if row["attack"] == attack and row["component"] == component and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {attack}/{component}/{method}, got {len(matches)}")
    return float(matches[0]["auroc"])


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


def draw_component_mia(records: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    configure_matplotlib()
    records = validate_records(records)
    attacks = infer_order(records, "attack", ATTACKS)
    components = infer_order(records, "component", COMPONENTS)
    methods = infer_order(records, "method", METHODS)

    fig, axes = plt.subplots(1, len(attacks), figsize=(11, 4.8), sharey=True)
    if len(attacks) == 1:
        axes = np.asarray([axes])
    x = np.arange(len(components), dtype=float)
    width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for ax, attack in zip(axes, attacks):
        for offset, method in zip(offsets, methods):
            values = [lookup(records, attack, component, method) for component in components]
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
        ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.2, zorder=2)
        ax.text(
            len(components) - 0.25,
            0.515,
            "Random Guess",
            color="#9d2d2a",
            ha="right",
            va="bottom",
            fontsize=9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.text(
            0.5,
            -0.23,
            f"({chr(97 + attacks.index(attack))}) {attack} MIA",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("MIA AUROC")
    axes[-1].legend(frameon=False, loc="upper right")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.95, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def save_summary(records: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    records = validate_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "component", "method", "auroc"])
        writer.writeheader()
        writer.writerows(records)
    json_path = output_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def save_mock_templates(records: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path, json_path = save_summary(records, output_dir, "mock_component_wise_mia_data")
    print(f"Saved mock CSV template: {csv_path}")
    print(f"Saved mock JSON template: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw component-wise MIA AUROC plot.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="component_wise_mia_auroc")
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

    fig_paths = draw_component_mia(records, args.output_dir, args.stem)
    summary_paths = save_summary(records, args.output_dir, args.stem)
    print(f"Data source: {source}")
    print(f"Saved SVG: {fig_paths[0]}")
    print(f"Saved PNG: {fig_paths[1]}")
    print(f"Saved summary CSV: {summary_paths[0]}")
    print(f"Saved summary JSON: {summary_paths[1]}")


if __name__ == "__main__":
    main()
