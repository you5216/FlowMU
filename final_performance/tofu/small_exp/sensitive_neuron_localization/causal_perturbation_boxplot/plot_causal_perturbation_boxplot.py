                      
                                                                    

                                                              
                                      

                                
                                                                           
                                                   
                                                                           
                                                           

                                                
                                   

                                                                             
                                                         

                                                      
                                                                            
                                                       
   

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "final_performance/tofu/small_exp/sensitive_neuron_localization/causal_perturbation_boxplot"
)

DEFAULT_METHODS = ["Random", "Activation", "FlowMU"]
PANELS = [("forget", "(a) Forget Set"), ("retain", "(b) Retain Set")]
METHOD_COLORS = {
    "Random": "#b8b8b8",
    "Activation": "#7aa6c2",
    "Act": "#7aa6c2",
    "MANU": "#8bbf8a",
    "Old Mask": "#c7a76c",
    "FlowMU": "#e58b4a",
    "Ours": "#e58b4a",
}


@dataclass(frozen=True)
class PerturbationRecord:
                                       

    method: str
    panel: str
    value: float
    sample_id: str = ""


def configure_matplotlib() -> None:
                                            
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _canonical_panel(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"forget", "f", "member_forget", "forget_set"}:
        return "forget"
    if normalized in {"retain", "r", "retain_set"}:
        return "retain"
    raise ValueError(f"Unknown panel value {value!r}; expected forget or retain.")


def _as_float(value: Any, *, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


def load_mock_records(seed: int = 20260630) -> list[PerturbationRecord]:
                                                                         
    rng = np.random.default_rng(seed)
    specs = {
        ("forget", "Random"): (0.0045, 0.0010, 80),
        ("forget", "Activation"): (0.0090, 0.0020, 80),
        ("forget", "FlowMU"): (0.0270, 0.0060, 80),
        ("retain", "Random"): (0.0035, 0.0009, 80),
        ("retain", "Activation"): (0.0065, 0.0018, 80),
        ("retain", "FlowMU"): (0.0075, 0.0020, 80),
    }

    records: list[PerturbationRecord] = []
    for (panel, method), (mean, std, count) in specs.items():
        values = np.clip(rng.normal(mean, std, size=count), 0.0, None)
        for idx, value in enumerate(values):
            records.append(
                PerturbationRecord(
                    method=method,
                    panel=panel,
                    value=float(value),
                    sample_id=f"mock_{panel}_{method}_{idx}",
                )
            )
    return records


def _find_panel_column(fieldnames: Iterable[str]) -> str:
    candidates = ["panel", "set", "subset", "data_split", "target_split"]
    fields = set(fieldnames)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(
        "CSV input must include one panel column: panel, set, subset, "
        "data_split, or target_split."
    )


def load_long_csv(path: Path, metric_column: str) -> list[PerturbationRecord]:
                                                             
    records: list[PerturbationRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "method" not in fieldnames:
            raise ValueError("CSV input must contain a method column.")
        if metric_column not in fieldnames:
            raise ValueError(f"CSV input must contain metric column {metric_column!r}.")
        panel_column = _find_panel_column(fieldnames)

        for row_idx, row in enumerate(reader):
            records.append(
                PerturbationRecord(
                    method=row["method"].strip(),
                    panel=_canonical_panel(row[panel_column]),
                    value=_as_float(row[metric_column], label=f"{metric_column} row {row_idx}"),
                    sample_id=row.get("sample_id", str(row_idx)),
                )
            )
    return records


def load_json_records(path: Path, metric_column: str) -> list[PerturbationRecord]:
                                                                   
    with path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)

    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, list):
        raise TypeError("JSON input must be a list of records or {'records': [...]} object.")

    records: list[PerturbationRecord] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"JSON record {idx} must be an object.")
        panel_value = item.get("panel", item.get("set", item.get("subset", "")))
        if not panel_value:
            raise ValueError(f"JSON record {idx} is missing panel/set/subset.")
        if metric_column not in item:
            raise ValueError(f"JSON record {idx} is missing metric {metric_column!r}.")
        records.append(
            PerturbationRecord(
                method=str(item["method"]).strip(),
                panel=_canonical_panel(str(panel_value)),
                value=_as_float(item[metric_column], label=f"{metric_column} record {idx}"),
                sample_id=str(item.get("sample_id", idx)),
            )
        )
    return records


def load_summary_csv(path: Path, metric: str) -> list[PerturbationRecord]:
                                                                             

                                                                                
                                                                           
                                                    
       
    metric_map = {
        "nll_shift": {
            ("forget", "FlowMU"): "forget_nll_increase",
            ("retain", "FlowMU"): "retain_nll_increase",
            ("forget", "Random"): "random_forget_nll_increase",
            ("retain", "Random"): "random_retain_nll_increase",
        },
        "relative_nll_shift": {
            ("forget", "FlowMU"): "relative_forget_shift",
            ("retain", "FlowMU"): "relative_retain_shift",
            ("forget", "Random"): None,
            ("retain", "Random"): None,
        },
    }
    if metric not in metric_map:
        raise ValueError(
            "--summary-csv supports only --summary-metric nll_shift or relative_nll_shift."
        )

    records: list[PerturbationRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            sample_id = row.get("report") or row.get("split") or str(row_idx)
            for (panel, method), column in metric_map[metric].items():
                if column is None:
                                                                                             
                    continue
                if column not in row:
                    raise ValueError(f"Summary CSV is missing column {column!r}.")
                records.append(
                    PerturbationRecord(
                        method=method,
                        panel=panel,
                        value=_as_float(row[column], label=f"{column} row {row_idx}"),
                        sample_id=sample_id,
                    )
                )
    return records


def validate_records(records: list[PerturbationRecord]) -> None:
    if not records:
        raise ValueError("No perturbation records loaded.")
    panels = {record.panel for record in records}
    missing = {panel for panel, _caption in PANELS} - panels
    if missing:
        raise ValueError(f"Missing panel(s): {sorted(missing)}")
    for record in records:
        if record.panel not in {"forget", "retain"}:
            raise ValueError(f"Invalid panel in record: {record}")
        if not record.method:
            raise ValueError(f"Empty method in record: {record}")
        if not np.isfinite(record.value):
            raise ValueError(f"Non-finite value in record: {record}")


def infer_method_order(records: list[PerturbationRecord], explicit: str = "") -> list[str]:
                                                                                 
    if explicit.strip():
        return [part.strip() for part in explicit.split(",") if part.strip()]

    present = {record.method for record in records}
    preferred = [method for method in DEFAULT_METHODS if method in present]
    extras = sorted(present - set(preferred))
    return preferred + extras


def grouped_values(
    records: list[PerturbationRecord],
    *,
    panel: str,
    methods: list[str],
) -> list[np.ndarray]:
    values = []
    for method in methods:
        arr = np.asarray(
            [record.value for record in records if record.panel == panel and record.method == method],
            dtype=float,
        )
        values.append(arr)
    return values


def _safe_ylim(records: list[PerturbationRecord], metric_label: str, user_ylim: str) -> tuple[float, float]:
    if user_ylim.strip():
        low, high = [float(part.strip()) for part in user_ylim.split(",", maxsplit=1)]
        if not low < high:
            raise ValueError("--ylim must be 'low,high' with low < high.")
        return low, high

    values = np.asarray([record.value for record in records], dtype=float)
    if metric_label.lower() == "logits mae":
        upper = max(0.035, float(np.quantile(values, 0.98)) * 1.25)
        return 0.0, upper

    low = min(0.0, float(np.quantile(values, 0.02)) * 1.20)
    high = max(0.0, float(np.quantile(values, 0.98)) * 1.20)
    if np.isclose(low, high):
        high = low + 1.0
    return low, high


def draw_boxplot(
    records: list[PerturbationRecord],
    *,
    output_dir: Path,
    stem: str,
    methods: list[str],
    metric_label: str,
    ylim: str = "",
    diagnostic_note: str = "",
) -> tuple[Path, Path]:
                                                            
    validate_records(records)
    configure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.1), sharey=True)
    y_min, y_max = _safe_ylim(records, metric_label, ylim)

    for ax, (panel, caption) in zip(axes, PANELS):
        values = grouped_values(records, panel=panel, methods=methods)
        if any(len(arr) == 0 for arr in values):
            missing = [method for method, arr in zip(methods, values) if len(arr) == 0]
            raise ValueError(f"Panel {panel!r} is missing values for methods: {missing}")

        box = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.50,
            showfliers=False,
            medianprops={"color": "#1c1c1c", "linewidth": 1.6},
            whiskerprops={"color": "#4a4a4a", "linewidth": 1.05},
            capprops={"color": "#4a4a4a", "linewidth": 1.05},
        )
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "#9e9e9e"))
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.9)
            patch.set_alpha(0.82)

                                                                                
        rng = np.random.default_rng(42)
        for idx, arr in enumerate(values, start=1):
            jitter = rng.normal(0.0, 0.028, size=len(arr))
            ax.scatter(
                np.full(len(arr), idx, dtype=float) + jitter,
                arr,
                s=10,
                color="#2f2f2f",
                alpha=0.22,
                linewidths=0,
                zorder=3,
            )

        ax.axhline(0.0, color="#c43c39", linestyle="--", linewidth=1.2, zorder=1)
        ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=0, fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.tick_params(axis="both", width=1.0, length=4)
        ax.text(
            0.5,
            -0.22,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#555555")
            ax.spines[spine].set_linewidth(0.8)

    axes[0].set_ylabel(metric_label, fontsize=13)
    for ax in axes:
        ax.set_xlabel("Localization Method", fontsize=13)

    if diagnostic_note:
        fig.text(0.5, 0.01, diagnostic_note, ha="center", va="bottom", fontsize=9, color="#555555")
        bottom = 0.20
    else:
        bottom = 0.17

    plt.subplots_adjust(left=0.10, right=0.985, bottom=bottom, top=0.97, wspace=0.18)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def save_mock_templates(records: list[PerturbationRecord], output_dir: Path) -> None:
                                                                         
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mock_causal_perturbation_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "panel", "sample_id", "logits_mae"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "method": record.method,
                    "panel": record.panel,
                    "sample_id": record.sample_id,
                    "logits_mae": f"{record.value:.8f}",
                }
            )

    json_path = output_dir / "mock_causal_perturbation_records.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "method": record.method,
                    "panel": record.panel,
                    "sample_id": record.sample_id,
                    "logits_mae": record.value,
                }
                for record in records
            ],
            f,
            indent=2,
        )

    print(f"Saved mock CSV template: {csv_path}")
    print(f"Saved mock JSON template: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw causal perturbation boxplots.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Per-sample long CSV input.")
    parser.add_argument("--input-json", type=Path, default=None, help="Per-sample JSON input.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Project probe_report_summary.csv for report-level diagnostic plotting.",
    )
    parser.add_argument("--metric-column", default="logits_mae", help="Metric column for long input.")
    parser.add_argument(
        "--summary-metric",
        default="nll_shift",
        choices=["nll_shift", "relative_nll_shift"],
        help="Metric extracted from --summary-csv.",
    )
    parser.add_argument("--metric-label", default="", help="Y-axis label override.")
    parser.add_argument("--method-order", default="", help="Comma-separated method order override.")
    parser.add_argument("--ylim", default="", help="Optional y-axis limits as 'low,high'.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="causal_perturbation_boxplot")
    parser.add_argument("--save-mock-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specified_inputs = [args.input_csv is not None, args.input_json is not None, args.summary_csv is not None]
    if sum(specified_inputs) > 1:
        raise ValueError("Use only one of --input-csv, --input-json, or --summary-csv.")

    diagnostic_note = ""
    if args.input_csv:
        records = load_long_csv(args.input_csv, args.metric_column)
        metric_label = args.metric_label or args.metric_column.replace("_", " ").title()
        source = str(args.input_csv)
    elif args.input_json:
        records = load_json_records(args.input_json, args.metric_column)
        metric_label = args.metric_label or args.metric_column.replace("_", " ").title()
        source = str(args.input_json)
    elif args.summary_csv:
        records = load_summary_csv(args.summary_csv, args.summary_metric)
        metric_label = args.metric_label or args.summary_metric.replace("_", " ").title()
        source = str(args.summary_csv)
        diagnostic_note = "Report-level diagnostic from summary CSV; not per-sample logits MAE."
    else:
        records = load_mock_records()
        metric_label = args.metric_label or "Logits MAE"
        source = "deterministic mock data"

    if args.save_mock_template:
        save_mock_templates(load_mock_records(), args.output_dir)

    methods = infer_method_order(records, args.method_order)
    svg_path, png_path = draw_boxplot(
        records,
        output_dir=args.output_dir,
        stem=args.stem,
        methods=methods,
        metric_label=metric_label,
        ylim=args.ylim,
        diagnostic_note=diagnostic_note,
    )
    print(f"Data source: {source}")
    print(f"Methods: {methods}")
    print(f"Records: {len(records)}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
