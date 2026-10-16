                      
                                                    

                                                                            
                                                                               
                                  

                                                               

                          
                     

              
    
                                                                                  
        
    
   

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
DEFAULT_OUTPUT_DIR = ROOT / "final_performance/tofu/small_exp/robustness/mia_auroc_barplot"

GROUPS = ["for01 (Standard MIA)", "for05 (Adaptive MIA)", "for10 (Adaptive MIA)"]
METHODS = ["GA", "DPO", "RMU", "Ours"]

COLORS = {
    "GA": "#8d99ae",                   
    "DPO": "#5b8db8",                      
    "RMU": "#7aa974",               
    "Ours": "#e19a5f",               
}


def load_mock_data() -> dict[str, dict[str, float]]:
                                                                                
    return {
        "for01 (Standard MIA)": {
            "GA": 0.72,
            "DPO": 0.68,
            "RMU": 0.64,
            "Ours": 0.53,
        },
        "for05 (Adaptive MIA)": {
            "GA": 0.80,
            "DPO": 0.74,
            "RMU": 0.70,
            "Ours": 0.55,
        },
        "for10 (Adaptive MIA)": {
            "GA": 0.84,
            "DPO": 0.77,
            "RMU": 0.73,
            "Ours": 0.56,
        },
    }


def validate_data(data: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
                                                          
    missing_groups = [group for group in GROUPS if group not in data]
    if missing_groups:
        raise KeyError(f"Missing groups: {missing_groups}")

    validated: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        validated[group] = {}
        missing_methods = [method for method in METHODS if method not in data[group]]
        if missing_methods:
            raise KeyError(f"Group {group!r} is missing methods: {missing_methods}")
        for method in METHODS:
            value = float(data[group][method])
            if not np.isfinite(value):
                raise ValueError(f"AUROC for {group}/{method} is not finite: {value}")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"AUROC for {group}/{method} must be in [0, 1], got {value}")
            validated[group][method] = value
    return validated


def load_csv(path: Path) -> dict[str, dict[str, float]]:
                                                    
    data: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"group", "method", "auroc"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            group = row["group"]
            method = row["method"]
            data.setdefault(group, {})[method] = float(row["auroc"])
    return validate_data(data)


def load_json(path: Path) -> dict[str, dict[str, float]]:
                                              
    with path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError("JSON input must be an object mapping group -> method -> auroc.")
    return validate_data(payload)


def save_mock_templates(data: dict[str, dict[str, float]], output_dir: Path) -> None:
                                                                    
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mock_mia_auroc_data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "method", "auroc"])
        writer.writeheader()
        for group in GROUPS:
            for method in METHODS:
                writer.writerow({"group": group, "method": method, "auroc": data[group][method]})

    json_path = output_dir / "mock_mia_auroc_data.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved mock CSV template: {csv_path}")
    print(f"Saved mock JSON template: {json_path}")


def configure_matplotlib() -> None:
                                            
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_barplot(
    data: dict[str, dict[str, float]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
                                                       
    data = validate_data(data)
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(GROUPS), dtype=float)
    width = 0.18
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

    for offset, method in zip(offsets, METHODS):
        values = [data[group][method] for group in GROUPS]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=method,
            color=COLORS[method],
            edgecolor="#2f2f2f",
            linewidth=0.6,
            zorder=3,
        )

    baseline = 0.5
    ax.axhline(
        baseline,
        color="#c43c39",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    ax.text(
        len(GROUPS) - 0.45,
        baseline + 0.018,
        "Random Guess (0.5)",
        color="#9d2d2a",
        ha="right",
        va="bottom",
        fontsize=10,
    )

    ax.set_ylabel("MIA AUROC", fontsize=12)
    ax.set_xlabel("Dataset Split / Attack Setting", fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="upper right", frameon=False)

                                                                            
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw grouped MIA AUROC bar plot.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional CSV input.")
    parser.add_argument("--input-json", type=Path, default=None, help="Optional JSON input.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="mia_auroc_grouped_barplot")
    parser.add_argument(
        "--save-mock-template",
        action="store_true",
        help="Save mock CSV/JSON data templates next to the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_csv and args.input_json:
        raise ValueError("Use only one of --input-csv or --input-json.")

    if args.input_csv:
        data = load_csv(args.input_csv)
        source = str(args.input_csv)
    elif args.input_json:
        data = load_json(args.input_json)
        source = str(args.input_json)
    else:
        data = load_mock_data()
        source = "deterministic mock data"

    if args.save_mock_template:
        save_mock_templates(data, args.output_dir)

    svg_path, png_path = draw_barplot(data, args.output_dir, args.stem)
    print(f"Data source: {source}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
