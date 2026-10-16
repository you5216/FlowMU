                      
                                                              

                                                                         
                                           

                                           

                                                                               

                                    

      
                                                
                            
                              
                                       

                                                                          
                                                                                
   

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = (
    ROOT / "final_performance/tofu/small_exp/knowledge_flow_sanitization/cross_level_sanitization_final"
)

SPLITS = ["for01_re99", "for05_re95", "for10_re90"]
SETS = ["forget", "retain"]
LEVELS = ["RAG", "Memory", "LLM"]
METHODS = ["Vanilla", "PURE", "SBU", "SF-NL", "FlowMU"]
LEVEL_WIDTHS = {"RAG": 80, "Memory": 80, "LLM": 80}
PANEL_BG = "#F0F0F7"


def load_mock_records(seed: int = 20260630) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for split_idx, split in enumerate(SPLITS):
        for subset in SETS:
            for level in LEVELS:
                width = LEVEL_WIDTHS[level]
                for method in METHODS:
                    if subset == "forget":
                        base = {
                            "Vanilla": 0.82,
                            "PURE": 0.58,
                            "SBU": 0.62,
                            "SF-NL": 0.72,
                            "FlowMU": 0.18,
                        }[method]
                        if level == "LLM":
                            base += 0.05 if method != "FlowMU" else -0.02
                    else:
                        base = {
                            "Vanilla": 0.55,
                            "PURE": 0.36,
                            "SBU": 0.28,
                            "SF-NL": 0.44,
                            "FlowMU": 0.50,
                        }[method]
                        if level == "Memory":
                            base += 0.04
                    split_shift = split_idx * 0.025
                    trend = 0.08 * np.sin(np.linspace(0, 2 * np.pi, width) + split_idx)
                    noise = rng.normal(0, 0.035, size=width)
                    values = np.clip(base + split_shift + trend + noise, 0.0, 1.2)
                    for index, value in enumerate(values):
                        records.append(
                            {
                                "split": split,
                                "set": subset,
                                "level": level,
                                "method": method,
                                "index": index,
                                "value": float(value),
                            }
                        )
    return records


def load_csv(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "set", "level", "method", "index", "value"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            records.append(
                {
                    "split": row["split"].strip(),
                    "set": row["set"].strip().lower(),
                    "level": row["level"].strip(),
                    "method": row["method"].strip(),
                    "index": int(row["index"]),
                    "value": float(row["value"]),
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
        raise ValueError("No sanitization records loaded.")
    for row in records:
        if row["set"] not in SETS:
            raise ValueError(f"Invalid set {row['set']!r}; expected {SETS}")
        if row["level"] not in LEVELS:
            raise ValueError(f"Invalid level {row['level']!r}; expected {LEVELS}")
        value = float(row["value"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value: {row}")
        row["value"] = value
        row["index"] = int(row["index"])
    return records


def infer_order(records: list[dict[str, Any]], field: str, preferred: list[str]) -> list[str]:
    present = {str(row[field]) for row in records}
    out = [item for item in preferred if item in present]
    out.extend(sorted(present - set(out)))
    return out


def matrix_for(records: list[dict[str, Any]], split: str, subset: str, level: str, methods: list[str]) -> np.ndarray:
    rows = [
        row for row in records if row["split"] == split and row["set"] == subset and row["level"] == level
    ]
    if not rows:
        raise ValueError(f"No rows for split={split}, set={subset}, level={level}")
    max_index = max(int(row["index"]) for row in rows)
    matrix = np.full((len(methods), max_index + 1), np.nan, dtype=float)
    method_to_idx = {method: idx for idx, method in enumerate(methods)}
    for row in rows:
        method = row["method"]
        if method not in method_to_idx:
            continue
        matrix[method_to_idx[method], int(row["index"])] = float(row["value"])
    if np.isnan(matrix).any():
                                                                         
                                                                      
        for i in range(matrix.shape[0]):
            valid = matrix[i][np.isfinite(matrix[i])]
            if valid.size == 0:
                raise ValueError(f"No values for method {methods[i]} in {split}/{subset}/{level}")
            matrix[i][np.isnan(matrix[i])] = float(np.median(valid))
    return matrix


def normalize_by_level(matrix: np.ndarray) -> np.ndarray:
    low, high = np.quantile(matrix.reshape(-1), [0.01, 0.99])
    if np.isclose(low, high):
        return np.zeros_like(matrix)
    clipped = np.clip(matrix, low, high)
    return (clipped - low) / (high - low)


def configure_matplotlib() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_subset_heatmap(
    records: list[dict[str, Any]],
    subset: str,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    configure_matplotlib()
    records = validate_records(records)
    splits = infer_order(records, "split", SPLITS)
    methods = infer_order(records, "method", METHODS)

    fig_height = max(2.55, 2.08 * len(splits) + 0.55)
    fig, axes = plt.subplots(len(splits), len(LEVELS), figsize=(9.6, fig_height), sharey=False)
    fig.patch.set_facecolor("white")
    if len(splits) == 1:
        axes = np.asarray([axes])

    for row_idx, split in enumerate(splits):
        for col_idx, level in enumerate(LEVELS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(PANEL_BG)
            matrix = normalize_by_level(matrix_for(records, split, subset, level, methods))
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                cbar=False,
                yticklabels=False,
                xticklabels=False,
                linewidths=0.0,
            )
            ax.set_yticks(np.arange(len(methods), dtype=float) + 0.5)
            if col_idx == 0:
                ax.set_yticklabels(methods, rotation=0, fontsize=10, color="#222222")
                ax.tick_params(axis="y", labelleft=True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", labelleft=False)
            for y in range(1, len(methods)):
                ax.axhline(y, color="white", linewidth=0.55, alpha=0.92, zorder=4)
            if row_idx == 0:
                ax.set_title(level, fontsize=12, pad=8)
            if col_idx == 0:
                ax.set_ylabel("Method", fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="y", rotation=0, length=0, pad=4)
            for spine in ("top", "right", "left", "bottom"):
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_alpha(0.0)

    caption = "Forget Sensitive Flow Intensity" if subset == "forget" else "Retain Safe Flow Preservation"
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=12)
    plt.subplots_adjust(left=0.19, right=0.90, bottom=0.13, top=0.90, wspace=0.08, hspace=0.18)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.66])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap="Blues")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized Flow Intensity", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}_{subset}_cross_level_heatmap.svg"
    png_path = output_dir / f"{stem}_{subset}_cross_level_heatmap.png"
    pdf_path = output_dir / f"{stem}_{subset}_cross_level_heatmap.pdf"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path, pdf_path


def build_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = validate_records(records)
    splits = infer_order(records, "split", SPLITS)
    methods = infer_order(records, "method", METHODS)
    summary: list[dict[str, Any]] = []
    for split in splits:
        for method in methods:
            row: dict[str, Any] = {"split": split, "method": method}
            for subset in SETS:
                for level in LEVELS:
                    vals = [
                        float(record["value"])
                        for record in records
                        if record["split"] == split
                        and record["set"] == subset
                        and record["level"] == level
                        and record["method"] == method
                    ]
                    row[f"{subset}_{level.lower()}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary.append(row)
    return summary


def save_summary(summary: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / f"{stem}_summary.csv"
    fieldnames = list(summary[0].keys()) if summary else ["split", "method"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return csv_path, json_path


def save_mock_templates(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mock_cross_level_sanitization_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "set", "level", "method", "index", "value"])
        writer.writeheader()
        writer.writerows(records)
    json_path = output_dir / "mock_cross_level_sanitization_records.json"
    json_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved mock CSV template: {csv_path}")
    print(f"Saved mock JSON template: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw cross-level knowledge-flow sanitization heatmaps.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="cross_level_sanitization")
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

    forget_paths = draw_subset_heatmap(records, "forget", args.output_dir, args.stem)
    retain_paths = draw_subset_heatmap(records, "retain", args.output_dir, args.stem)
    summary_paths = save_summary(build_summary(records), args.output_dir, args.stem)
    print(f"Data source: {source}")
    print(f"Saved forget SVG: {forget_paths[0]}")
    print(f"Saved forget PNG: {forget_paths[1]}")
    print(f"Saved forget PDF: {forget_paths[2]}")
    print(f"Saved retain SVG: {retain_paths[0]}")
    print(f"Saved retain PNG: {retain_paths[1]}")
    print(f"Saved retain PDF: {retain_paths[2]}")
    print(f"Saved summary CSV: {summary_paths[0]}")
    print(f"Saved summary JSON: {summary_paths[1]}")


if __name__ == "__main__":
    main()
