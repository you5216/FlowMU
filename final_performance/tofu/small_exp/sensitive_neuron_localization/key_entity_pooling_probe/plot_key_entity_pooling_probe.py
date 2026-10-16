                      
                                               

                                                                           
                                                                            
                                                

                                                                        

                                 
                              
                                  

                                        
                             

               
    
                                                                     
                                                                   
    
   

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
DEFAULT_OUTPUT_DIR = (
    ROOT / "final_performance/tofu/small_exp/sensitive_neuron_localization/key_entity_pooling_probe"
)

METHODS = ["Vanilla", "GA", "DPO", "RMU", "FlowMU"]
POOLINGS = ["Next-Token", "Full-Sentence", "Key-Entity"]
METRICS = ["AUROC", "Accuracy", "F1"]
POOLING_COLORS = {
    "Next-Token": "#9ca3af",
    "Full-Sentence": "#6f9fc7",
    "Key-Entity": "#e19a5f",
}
METHOD_COLORS = {
    "Vanilla": "#8d99ae",
    "GA": "#a6a6a6",
    "DPO": "#5b8db8",
    "RMU": "#7aa974",
    "FlowMU": "#e19a5f",
}


def load_mock_data(seed: int = 20260630) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, Any]] = []
    base = {
        "Vanilla": {"Next-Token": 0.62, "Full-Sentence": 0.68, "Key-Entity": 0.76},
        "GA": {"Next-Token": 0.58, "Full-Sentence": 0.63, "Key-Entity": 0.70},
        "DPO": {"Next-Token": 0.57, "Full-Sentence": 0.62, "Key-Entity": 0.68},
        "RMU": {"Next-Token": 0.56, "Full-Sentence": 0.61, "Key-Entity": 0.66},
        "FlowMU": {"Next-Token": 0.53, "Full-Sentence": 0.57, "Key-Entity": 0.61},
    }
    for method in METHODS:
        for pooling in POOLINGS:
            auroc = float(np.clip(base[method][pooling] + rng.normal(0, 0.006), 0.5, 1.0))
            acc = float(np.clip(0.50 + (auroc - 0.50) * 0.78 + rng.normal(0, 0.005), 0.0, 1.0))
            f1 = float(np.clip(acc - rng.uniform(0.00, 0.035), 0.0, 1.0))
            metrics.extend(
                [
                    {"method": method, "pooling": pooling, "metric": "AUROC", "value": auroc},
                    {"method": method, "pooling": pooling, "metric": "Accuracy", "value": acc},
                    {"method": method, "pooling": pooling, "metric": "F1", "value": f1},
                ]
            )

    layers: list[dict[str, Any]] = []
    x = np.arange(1, 33)
    for method in METHODS:
        for pooling in POOLINGS:
            level = base[method][pooling]
            profile = level + 0.045 * np.sin((x - 4) / 32 * np.pi) + rng.normal(0, 0.010, size=32)
            if method == "FlowMU":
                profile -= 0.035 * np.exp(-((x - 23) ** 2) / 80)
            for layer, value in zip(x, np.clip(profile, 0.5, 1.0)):
                layers.append(
                    {
                        "method": method,
                        "pooling": pooling,
                        "layer": int(layer),
                        "auroc": float(value),
                    }
                )
    return metrics, layers


def load_metrics_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "pooling", "metric", "value"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Metrics CSV must contain columns: {sorted(required)}")
        for row in reader:
            rows.append(
                {
                    "method": row["method"].strip(),
                    "pooling": row["pooling"].strip(),
                    "metric": row["metric"].strip(),
                    "value": float(row["value"]),
                }
            )
    return rows


def load_layers_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "pooling", "layer", "auroc"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Layer CSV must contain columns: {sorted(required)}")
        for row in reader:
            rows.append(
                {
                    "method": row["method"].strip(),
                    "pooling": row["pooling"].strip(),
                    "layer": int(row["layer"]),
                    "auroc": float(row["auroc"]),
                }
            )
    return rows


def load_json(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError("JSON input must be an object with metrics/layers.")
    metrics = payload.get("metrics")
    layers = payload.get("layers", [])
    if not isinstance(metrics, list):
        raise ValueError("JSON input must include a metrics list.")
    if not isinstance(layers, list):
        raise ValueError("JSON layers must be a list.")
    return metrics, layers


def validate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No metric rows loaded.")
    seen = set()
    for row in rows:
        key = (row["method"], row["pooling"], row["metric"])
        if key in seen:
            raise ValueError(f"Duplicate metric row: {key}")
        seen.add(key)
        value = float(row["value"])
        if not np.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"Metric value must be in [0, 1], got {value} for {key}")
        row["value"] = value
    return rows


def validate_layers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        layer = int(row["layer"])
        if layer < 1 or layer > 32:
            raise ValueError(f"Layer must be in 1..32, got {layer}")
        auroc = float(row["auroc"])
        if not np.isfinite(auroc) or auroc < 0 or auroc > 1:
            raise ValueError(f"Layer AUROC must be in [0, 1], got {auroc}")
        row["layer"] = layer
        row["auroc"] = auroc
    return rows


def infer_order(rows: list[dict[str, Any]], field: str, preferred: list[str]) -> list[str]:
    present = {str(row[field]) for row in rows}
    ordered = [item for item in preferred if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


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


def metric_lookup(rows: list[dict[str, Any]], method: str, pooling: str, metric: str) -> float:
    for row in rows:
        if row["method"] == method and row["pooling"] == pooling and row["metric"] == metric:
            return float(row["value"])
    raise KeyError(f"Missing metric row for {method}/{pooling}/{metric}.")


def draw_metric_barplot(
    metrics: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    configure_matplotlib()
    metrics = validate_metrics(metrics)
    methods = infer_order(metrics, "method", METHODS)
    poolings = infer_order(metrics, "pooling", POOLINGS)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    x = np.arange(len(methods), dtype=float)
    width = 0.22
    offsets = (np.arange(len(poolings)) - (len(poolings) - 1) / 2.0) * width

    for ax, metric, caption in zip(axes, METRICS, ["(a) AUROC", "(b) Accuracy", "(c) F1"]):
        for offset, pooling in zip(offsets, poolings):
            values = [metric_lookup(metrics, method, pooling, metric) for method in methods]
            ax.bar(
                x + offset,
                values,
                width=width,
                color=POOLING_COLORS.get(pooling, "#9e9e9e"),
                edgecolor="#333333",
                linewidth=0.5,
                label=pooling,
                zorder=3,
            )
        ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.0, zorder=2)
        ax.set_ylim(0.45, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.text(0.5, -0.33, caption, transform=ax.transAxes, ha="center", va="top", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Probe Score")
    axes[-1].legend(frameon=False, loc="upper right")
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.28, top=0.96, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}_metrics.svg"
    png_path = output_dir / f"{stem}_metrics.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def draw_layer_plot(
    layers: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path] | None:
    layers = validate_layers(layers)
    if not layers:
        return None
    configure_matplotlib()

    methods = infer_order(layers, "method", METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, pooling in zip(axes, POOLINGS):
        for method in methods:
            rows = sorted(
                [row for row in layers if row["method"] == method and row["pooling"] == pooling],
                key=lambda row: row["layer"],
            )
            if not rows:
                continue
            ax.plot(
                [row["layer"] for row in rows],
                [row["auroc"] for row in rows],
                linewidth=2.0,
                color=METHOD_COLORS.get(method, "#777777"),
                label=method,
            )
        ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.0)
        ax.set_xlim(1, 32)
        ax.set_ylim(0.45, 1.0)
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.grid(True, alpha=0.30)
        ax.text(
            0.5,
            -0.28,
            f"({chr(97 + POOLINGS.index(pooling))}) {pooling}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Per-layer AUROC")
    for ax in axes:
        ax.set_xlabel("Layer")
    axes[-1].legend(frameon=False, loc="upper right")
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.26, top=0.96, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}_layers.svg"
    png_path = output_dir / f"{stem}_layers.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def save_mock_templates(metrics: list[dict[str, Any]], layers: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "mock_key_entity_probe_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "pooling", "metric", "value"])
        writer.writeheader()
        writer.writerows(metrics)

    layers_csv = output_dir / "mock_key_entity_probe_layers.csv"
    with layers_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "pooling", "layer", "auroc"])
        writer.writeheader()
        writer.writerows(layers)

    json_path = output_dir / "mock_key_entity_probe_data.json"
    json_path.write_text(
        json.dumps({"metrics": metrics, "layers": layers}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved mock metrics CSV: {metrics_csv}")
    print(f"Saved mock layer CSV: {layers_csv}")
    print(f"Saved mock JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Key-Entity Token Pooling probe figures.")
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--layers-csv", type=Path, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="key_entity_pooling_probe")
    parser.add_argument("--save-mock-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_json and (args.metrics_csv or args.layers_csv):
        raise ValueError("Use either --input-json or CSV inputs, not both.")

    if args.input_json:
        metrics, layers = load_json(args.input_json)
        source = str(args.input_json)
    elif args.metrics_csv:
        metrics = load_metrics_csv(args.metrics_csv)
        layers = load_layers_csv(args.layers_csv) if args.layers_csv else []
        source = str(args.metrics_csv)
    else:
        metrics, layers = load_mock_data()
        source = "deterministic mock data"

    if args.save_mock_template:
        mock_metrics, mock_layers = load_mock_data()
        save_mock_templates(mock_metrics, mock_layers, args.output_dir)

    metric_svg, metric_png = draw_metric_barplot(metrics, args.output_dir, args.stem)
    layer_paths = draw_layer_plot(layers, args.output_dir, args.stem)
    print(f"Data source: {source}")
    print(f"Saved metric SVG: {metric_svg}")
    print(f"Saved metric PNG: {metric_png}")
    if layer_paths:
        print(f"Saved layer SVG: {layer_paths[0]}")
        print(f"Saved layer PNG: {layer_paths[1]}")


if __name__ == "__main__":
    main()
