                      
                                                             

                                                                              
                                                                           
                                                                          

                                                                            
                            

                       
                                            
                                            
                                            
                                           

                          
                                  
                                           
                                                         
   

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "final_performance/tofu/small_exp/sensitive_neuron_localization/topk_path_ablation"
)

TOPK = np.array([2**i for i in range(2, 14)], dtype=float)
DISPLAY_EXPONENTS = [3, 5, 7, 9, 11, 13]
DISPLAY_TICKS = np.array([2**i for i in DISPLAY_EXPONENTS], dtype=float)
DISPLAY_TICK_LABELS = [rf"$2^{{{i}}}$" for i in DISPLAY_EXPONENTS]

PANEL_ORDER = [
    ("for05_re95", "forget", "(a) for05 (Forget)"),
    ("for05_re95", "retain", "(b) for05 (Retain)"),
    ("for10_re90", "forget", "(c) for10 (Forget)"),
    ("for10_re90", "retain", "(d) for10 (Retain)"),
]

NPZ_KEYS = {
    ("for05_re95", "forget", "Neurons"): "for05_forget_neurons",
    ("for05_re95", "forget", "Paths"): "for05_forget_paths",
    ("for05_re95", "retain", "Neurons"): "for05_retain_neurons",
    ("for05_re95", "retain", "Paths"): "for05_retain_paths",
    ("for10_re90", "forget", "Neurons"): "for10_forget_neurons",
    ("for10_re90", "forget", "Paths"): "for10_forget_paths",
    ("for10_re90", "retain", "Neurons"): "for10_retain_neurons",
    ("for10_re90", "retain", "Paths"): "for10_retain_paths",
}


@dataclass(frozen=True)
class PanelCurves:
                                               

    neurons: np.ndarray
    paths: np.ndarray


CurveData = dict[tuple[str, str], PanelCurves]


def _smooth_growth(
    start: float,
    end: float,
    rate: float,
    *,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
                                                                 
    x = np.log2(TOPK)
    progress = 1.0 - np.exp(-rate * (x - x.min()))
    progress = progress / progress.max()
    values = start + (end - start) * progress
    values += rng.normal(0.0, noise, size=values.shape)
    return np.clip(values, 0.0, 0.6)


def load_mock_data(seed: int = 20260630) -> CurveData:
                                                                        
    rng = np.random.default_rng(seed)

    return {
        ("for05_re95", "forget"): PanelCurves(
            neurons=_smooth_growth(0.02, 0.34, 0.18, noise=0.010, rng=rng),
            paths=_smooth_growth(0.07, 0.53, 0.48, noise=0.012, rng=rng),
        ),
        ("for05_re95", "retain"): PanelCurves(
            neurons=_smooth_growth(0.04, 0.38, 0.20, noise=0.010, rng=rng),
            paths=_smooth_growth(0.10, 0.56, 0.52, noise=0.010, rng=rng),
        ),
        ("for10_re90", "forget"): PanelCurves(
            neurons=_smooth_growth(0.01, 0.30, 0.16, noise=0.010, rng=rng),
            paths=_smooth_growth(0.05, 0.49, 0.43, noise=0.012, rng=rng),
        ),
        ("for10_re90", "retain"): PanelCurves(
            neurons=_smooth_growth(0.03, 0.35, 0.19, noise=0.010, rng=rng),
            paths=_smooth_growth(0.08, 0.52, 0.47, noise=0.012, rng=rng),
        ),
    }


def _validate_curve(name: str, values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != TOPK.shape:
        raise ValueError(f"{name} has shape {arr.shape}; expected {TOPK.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return np.clip(arr, 0.0, 0.6)


def load_npz(path: Path) -> CurveData:
                                           
    payload = np.load(path)
    missing = [key for key in NPZ_KEYS.values() if key not in payload]
    if missing:
        raise KeyError(f"Missing NPZ keys: {missing}")

    data: CurveData = {}
    for split, panel, _caption in PANEL_ORDER:
        neurons_key = NPZ_KEYS[(split, panel, "Neurons")]
        paths_key = NPZ_KEYS[(split, panel, "Paths")]
        data[(split, panel)] = PanelCurves(
            neurons=_validate_curve(neurons_key, payload[neurons_key]),
            paths=_validate_curve(paths_key, payload[paths_key]),
        )
    return data


def load_csv(path: Path) -> CurveData:
                                               
    rows: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "panel", "method", "topk", "rouge_l"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            key = (row["split"], row["panel"].lower(), row["method"])
            rows.setdefault(key, []).append((float(row["topk"]), float(row["rouge_l"])))

    data: CurveData = {}
    for split, panel, _caption in PANEL_ORDER:
        curves = {}
        for method in ("Neurons", "Paths"):
            key = (split, panel, method)
            if key not in rows:
                raise KeyError(f"CSV is missing rows for {key}.")
            ordered = sorted(rows[key], key=lambda item: item[0])
            topk = np.asarray([item[0] for item in ordered], dtype=float)
            if not np.array_equal(topk, TOPK):
                raise ValueError(
                    f"{key} topk must be {TOPK.astype(int).tolist()}, "
                    f"got {topk.astype(int).tolist()}."
                )
            curves[method] = _validate_curve("/".join(key), [item[1] for item in ordered])
        data[(split, panel)] = PanelCurves(neurons=curves["Neurons"], paths=curves["Paths"])
    return data


def save_mock_npz(data: CurveData, output_path: Path) -> None:
                                                                   
    payload = {}
    for split, panel, _caption in PANEL_ORDER:
        curves = data[(split, panel)]
        payload[NPZ_KEYS[(split, panel, "Neurons")]] = curves.neurons
        payload[NPZ_KEYS[(split, panel, "Paths")]] = curves.paths
    payload["topk"] = TOPK.astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def configure_matplotlib() -> None:
                                                       
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_figure(data: CurveData, output_dir: Path, stem: str) -> tuple[Path, Path]:
                                                       
    configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, (split, panel, caption) in zip(axes_flat, PANEL_ORDER):
        curves = data[(split, panel)]

        ax.plot(
            TOPK,
            curves.neurons,
            color="#1f77b4",
            linestyle="--",
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label="Neurons",
        )
        ax.plot(
            TOPK,
            curves.paths,
            color="#ff7f0e",
            linestyle="-",
            linewidth=2.4,
            marker="s",
            markersize=4.5,
            label="Paths",
        )

        ax.set_xscale("log", base=2)
        ax.set_xlim(2**2, 2**13)
        ax.set_ylim(0.0, 0.6)
        ax.set_xticks(DISPLAY_TICKS)
        ax.set_xticklabels(DISPLAY_TICK_LABELS)
        ax.set_yticks([0.0, 0.2, 0.4])
        ax.grid(True, alpha=0.3, linewidth=0.8)

                                                                          
        legend_loc = "lower right" if panel == "retain" else "upper left"
        ax.legend(loc=legend_loc, frameon=False)

                                                                           
        ax.text(
            0.5,
            -0.23,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("Rouge-L", fontsize=12)
    for ax in axes[1, :]:
        ax.set_xlabel("Top-k Components", fontsize=12)

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.97, wspace=0.18, hspace=0.34)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Top-k Neurons-vs-Paths Rouge-L ablation curves."
    )
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional NPZ curve input.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional tidy CSV curve input.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="topk_neurons_vs_paths_rougel_2x2")
    parser.add_argument(
        "--save-mock-template",
        action="store_true",
        help="Save the default mock arrays as mock_topk_ablation_data.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_npz and args.input_csv:
        raise ValueError("Use only one of --input-npz or --input-csv.")

    if args.input_npz:
        data = load_npz(args.input_npz)
        source = str(args.input_npz)
    elif args.input_csv:
        data = load_csv(args.input_csv)
        source = str(args.input_csv)
    else:
        data = load_mock_data()
        source = "deterministic mock data"

    if args.save_mock_template:
        template_path = args.output_dir / "mock_topk_ablation_data.npz"
        save_mock_npz(data, template_path)
        print(f"Saved mock data template: {template_path}")

    svg_path, png_path = draw_figure(data, args.output_dir, args.stem)
    print(f"Data source: {source}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
