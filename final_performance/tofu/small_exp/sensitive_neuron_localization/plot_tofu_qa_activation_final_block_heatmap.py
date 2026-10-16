                      
                                                         

                                                                      
                                                               

                                                                         
                                                        

                                                

                                                          

              

                                                                            
                                                                        
                                                             
   

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[2] / "final/tofu_for05_activation_strength_heatmap"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "tofu_for05_activation_strength_heatmap_data.npz"
LAYER_LABELS = [str(layer) if layer % 4 == 0 else "" for layer in range(32)]


def load_raw(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    payload = np.load(path, allow_pickle=True)
    methods = [str(item) for item in payload["methods"]]
    forget_raw = np.asarray(payload["forget_raw"], dtype=np.float64)
    retain_raw = np.asarray(payload["retain_raw"], dtype=np.float64)
    if "forget_nll" not in payload or "retain_nll" not in payload:
        raise ValueError(f"{path} must contain forget_nll and retain_nll.")
    forget_nll = np.asarray(payload["forget_nll"], dtype=np.float64)
    retain_nll = np.asarray(payload["retain_nll"], dtype=np.float64)
    if forget_raw.shape != retain_raw.shape:
        raise ValueError(f"forget_raw shape {forget_raw.shape} != retain_raw shape {retain_raw.shape}")
    if forget_raw.shape[0] != 32:
        raise ValueError(f"Expected 32 layers, got {forget_raw.shape[0]}")
    if forget_raw.shape[1] != len(methods):
        raise ValueError("Method count does not match raw activation columns.")
    if forget_nll.shape != (len(methods),) or retain_nll.shape != (len(methods),):
        raise ValueError("NLL arrays must have one value per method.")
    metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    return methods, forget_raw, retain_raw, forget_nll, retain_nll, metadata


def robust_norm(base: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if np.isclose(low, high):
        raise ValueError("Display scale collapsed.")
    return (np.clip(base, low, high) - low) / (high - low)


def layer_contrast(base: np.ndarray, *, eps: float) -> np.ndarray:
    row_min = base.min(axis=1, keepdims=True)
    row_max = base.max(axis=1, keepdims=True)
    spread = np.maximum(row_max - row_min, eps)
    return (base - row_min) / spread


def compute_display(
    forget_raw: np.ndarray,
    retain_raw: np.ndarray,
    forget_nll: np.ndarray,
    *,
    forget_confidence_alpha: float,
    row_contrast_weight: float,
    display_power: float,
    display_contrast_gamma: float,
    display_max: float,
    low_q: float,
    high_q: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if forget_confidence_alpha < 0:
        raise ValueError("--forget-confidence-alpha must be non-negative.")
    if not 0 <= row_contrast_weight <= 1:
        raise ValueError("--row-contrast-weight must be in [0, 1].")
    if display_power <= 0:
        raise ValueError("--display-power must be positive.")
    if display_contrast_gamma <= 0:
        raise ValueError("--display-contrast-gamma must be positive.")
    if display_max <= 0:
        raise ValueError("--display-max must be positive.")
    if not 0 <= low_q < high_q <= 1:
        raise ValueError("--low-q and --high-q must satisfy 0 <= low < high <= 1.")

    forget_score = forget_raw * np.exp(-forget_confidence_alpha * forget_nll)[None, :]
    retain_score = retain_raw.copy()
    forget_base = np.log1p(forget_score)
    retain_base = np.log1p(retain_score)

    values = np.concatenate([forget_base.reshape(-1), retain_base.reshape(-1)])
    clip_low = float(np.quantile(values, low_q))
    clip_high = float(np.quantile(values, high_q))

    forget_global = robust_norm(forget_base, low=clip_low, high=clip_high)
    retain_global = robust_norm(retain_base, low=clip_low, high=clip_high)
    forget_layer = layer_contrast(forget_base, eps=eps)
    retain_layer = layer_contrast(retain_base, eps=eps)

    forget_display = (1.0 - row_contrast_weight) * forget_global + row_contrast_weight * forget_layer
    retain_display = (1.0 - row_contrast_weight) * retain_global + row_contrast_weight * retain_layer
    forget_unit = np.power(np.clip(forget_display, 0.0, 1.0), display_power)
    retain_unit = np.power(np.clip(retain_display, 0.0, 1.0), display_power)
    forget_display = (display_max * np.power(forget_unit, display_contrast_gamma)).astype(np.float32)
    retain_display = (display_max * np.power(retain_unit, display_contrast_gamma)).astype(np.float32)

    metadata = {
        "forget_score_formula": "forget_raw_activation * exp(-forget_confidence_alpha * forget_answer_token_nll)",
        "retain_score_formula": "retain_raw_activation",
        "display_formula": "display_max * (((1-row_contrast_weight) * shared_robust_norm(log1p(score)) + row_contrast_weight * layerwise_method_minmax(log1p(score))) ** display_power) ** display_contrast_gamma",
        "forget_confidence_alpha": float(forget_confidence_alpha),
        "row_contrast_weight": float(row_contrast_weight),
        "display_power": float(display_power),
        "display_contrast_gamma": float(display_contrast_gamma),
        "display_max": float(display_max),
        "low_q": float(low_q),
        "high_q": float(high_q),
        "clip_low": clip_low,
        "clip_high": clip_high,
        "scale": f"one shared 0-{display_max:g} scale for forget and retain panels",
    }
    return forget_score, retain_score, forget_display, retain_display, metadata


def configure_matplotlib() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_combined_heatmap(
    forget_display: np.ndarray,
    retain_display: np.ndarray,
    *,
    methods: list[str],
    output_path: Path,
    display_max: float,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    panels = [
        (axes[0], forget_display, "Forget"),
        (axes[1], retain_display, "Retain"),
    ]
    for ax, matrix, title in panels:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=display_max,
            cbar=False,
            xticklabels=methods,
            yticklabels=LAYER_LABELS,
            linewidths=0.28,
            linecolor="white",
            square=False,
        )
        ax.set_title(title, fontsize=10, pad=3)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=0, labelsize=8, length=0, pad=2)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8, length=0, pad=1)
    axes[0].set_ylabel("Layer", fontsize=9)
    axes[1].set_ylabel("")

    plt.subplots_adjust(left=0.10, right=0.88, bottom=0.15, top=0.86, wspace=0.08)
    cbar_ax = fig.add_axes([0.905, 0.22, 0.020, 0.56])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=display_max), cmap="Blues")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Activation Strength", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    path: Path,
    *,
    methods: list[str],
    forget_raw: np.ndarray,
    retain_raw: np.ndarray,
    forget_score: np.ndarray,
    retain_score: np.ndarray,
    forget_display: np.ndarray,
    retain_display: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["panel", "layer", "method", "raw_activation_mean", "transformed_score", "display_score"],
        )
        writer.writeheader()
        for panel, raw, score, display in [
            ("forget", forget_raw, forget_score, forget_display),
            ("retain", retain_raw, retain_score, retain_display),
        ]:
            for layer_idx in range(raw.shape[0]):
                for method_idx, method in enumerate(methods):
                    writer.writerow(
                        {
                            "panel": panel,
                            "layer": layer_idx,
                            "method": method,
                            "raw_activation_mean": f"{raw[layer_idx, method_idx]:.10f}",
                            "transformed_score": f"{score[layer_idx, method_idx]:.10f}",
                            "display_score": f"{display[layer_idx, method_idx]:.10f}",
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final layer-wise TOFU Q+A activation heatmap.")
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="tofu_for05_activation_strength_heatmap")
    parser.add_argument("--forget-confidence-alpha", type=float, default=0.2)
    parser.add_argument("--row-contrast-weight", type=float, default=0.10)
    parser.add_argument("--display-power", type=float, default=0.76)
    parser.add_argument("--display-contrast-gamma", type=float, default=1.2)
    parser.add_argument("--display-max", type=float, default=0.5)
    parser.add_argument("--low-q", type=float, default=0.02)
    parser.add_argument("--high-q", type=float, default=0.995)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods, forget_raw, retain_raw, forget_nll, retain_nll, source_meta = load_raw(args.input_npz)
    forget_score, retain_score, forget_display, retain_display, transform_meta = compute_display(
        forget_raw,
        retain_raw,
        forget_nll,
        forget_confidence_alpha=args.forget_confidence_alpha,
        row_contrast_weight=args.row_contrast_weight,
        display_power=args.display_power,
        display_contrast_gamma=args.display_contrast_gamma,
        display_max=args.display_max,
        low_q=args.low_q,
        high_q=args.high_q,
        eps=args.eps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / f"{args.stem}_data.npz"
    output_csv = args.output_dir / f"{args.stem}_data.csv"
    np.savez_compressed(
        output_npz,
        methods=np.asarray(methods, dtype=object),
        layers=np.asarray(list(range(32)), dtype=np.int32),
        ytick_labels=np.asarray(LAYER_LABELS, dtype=object),
        forget_raw=forget_raw.astype(np.float32),
        retain_raw=retain_raw.astype(np.float32),
        forget_nll=forget_nll.astype(np.float32),
        retain_nll=retain_nll.astype(np.float32),
        forget_score=forget_score.astype(np.float32),
        retain_score=retain_score.astype(np.float32),
        forget_display=forget_display,
        retain_display=retain_display,
        metadata=json.dumps(
            {
                "format": "tofu_qa_answer_activation_final_layer_heatmap_v1",
                "source_npz": str(args.input_npz),
                "source_metadata": source_meta,
                "layer_display": "all 32 layers shown; y-axis labels only every 4 layers starting at 0",
                "display_transform": transform_meta,
            },
            ensure_ascii=False,
        ),
    )
    write_csv(
        output_csv,
        methods=methods,
        forget_raw=forget_raw,
        retain_raw=retain_raw,
        forget_score=forget_score,
        retain_score=retain_score,
        forget_display=forget_display,
        retain_display=retain_display,
    )

    image_paths: list[Path] = []
    for ext in ["svg", "png", "pdf"]:
        path = args.output_dir / f"{args.stem}.{ext}"
        draw_combined_heatmap(
            forget_display,
            retain_display,
            methods=methods,
            output_path=path,
            display_max=args.display_max,
        )
        image_paths.append(path)

    print(f"Saved NPZ: {output_npz}")
    print(f"Saved CSV: {output_csv}")
    print(f"Transform: {transform_meta}")
    print("Method NLL:")
    for idx, method in enumerate(methods):
        print(f"  {method}: forget_nll={forget_nll[idx]:.6f}, retain_nll={retain_nll[idx]:.6f}")
    print("Last layer display:")
    for idx, method in enumerate(methods):
        print(
            f"  {method}: forget={forget_display[-1, idx]:.4f}, "
            f"retain={retain_display[-1, idx]:.4f}"
        )
    for path in image_paths:
        print(f"Saved image: {path}")


if __name__ == "__main__":
    main()
