                      
                                                                           

                                                                         
                                                                        
                                                           
   

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.pipeline.stage2_unlearning import LocalizationArtifacts, normalize_scores              


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_layer_range(value: str, *, available_layers: list[int]) -> list[int]:
    value = str(value or "").strip()
    if not value or value.lower() == "all":
        return sorted(available_layers)
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    available = set(available_layers)
    return sorted(layer for layer in selected if layer in available)


def extract_layer_index(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_manifest_path(npz_path: Path) -> Path:
    return npz_path.with_name("candidate_neuron_stats_manifest.json")


def available_layers_from_npz(npz: np.lib.npyio.NpzFile) -> list[int]:
    layers: set[int] = set()
    for key in npz.files:
        if not key.endswith("__neuron_index") or not key.startswith("layer_"):
            continue
        layer_part = key.split("__", 1)[0]
        layers.add(int(layer_part.split("_", 1)[1]))
    return sorted(layers)


def build_lora_masks_from_shapes(
    lora_parameter_shapes: dict[str, list[int]],
    selected_by_layer: dict[int, list[int]],
    *,
    include_down_proj: bool,
    include_down_proj_lora_b: bool,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, shape in lora_parameter_shapes.items():
        if len(shape) != 2 or "lora_" not in name:
            continue
        layer_idx = extract_layer_index(name)
        if layer_idx is None or layer_idx not in selected_by_layer:
            continue
        neurons = selected_by_layer[layer_idx]
        mask = np.zeros(tuple(int(dim) for dim in shape), dtype=np.float32)
        if "lora_B" in name and (".gate_proj." in name or ".up_proj." in name):
            rows = [idx for idx in neurons if 0 <= idx < mask.shape[0]]
            if rows:
                mask[rows, :] = 1.0
        elif include_down_proj and "lora_A" in name and ".down_proj." in name:
            cols = [idx for idx in neurons if 0 <= idx < mask.shape[1]]
            if cols:
                mask[:, cols] = 1.0
        elif include_down_proj_lora_b and "lora_B" in name and ".down_proj." in name:
            mask[:, :] = 1.0
        if mask.sum() > 0:
            masks[name] = mask
    return masks


def finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive localization artifacts from cached candidate neuron stats.")
    parser.add_argument("--candidate-stats-npz", required=True)
    parser.add_argument("--manifest-json", default="", help="Defaults to candidate_neuron_stats_manifest.json next to the npz.")
    parser.add_argument("--source-report-json", default="", help="Defaults to manifest source_report_json when available.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-range", default="all")
    parser.add_argument("--neurons-per-layer", type=int, default=100)
    parser.add_argument("--score-field", default="combined_raw_diff")
    parser.add_argument("--retain-ratio-field", default="combined_retain_to_forget_ratio")
    parser.add_argument("--forget-mean-field", default="combined_forget_mean")
    parser.add_argument("--retain-ratio-max", type=float, default=0.8)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--min-forget-mean",
        type=float,
        default=0.0,
        help="Keep only neurons whose forget mean activation is at least this raw value.",
    )
    parser.add_argument(
        "--min-forget-percentile",
        type=float,
        default=0.0,
        help="Keep only neurons whose forget mean activation is at least this per-layer percentile.",
    )
    parser.add_argument("--allow-non-positive", action="store_true")
    parser.add_argument("--no-down-proj-mask", action="store_true")
    parser.add_argument("--no-down-proj-lora-b", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.candidate_stats_npz)
    manifest_path = Path(args.manifest_json) if args.manifest_json else infer_manifest_path(npz_path)
    manifest = load_json_if_exists(manifest_path)
    npz = np.load(npz_path)
    available_layers = available_layers_from_npz(npz)
    selected_layers = parse_layer_range(args.layer_range, available_layers=available_layers)
    if not selected_layers:
        raise ValueError(f"No layers selected from available layers: {available_layers}")

    source_report_path = Path(args.source_report_json) if args.source_report_json else None
    if source_report_path is None and manifest.get("source_report_json"):
        candidate = Path(str(manifest["source_report_json"]))
        source_report_path = candidate if candidate.is_absolute() else npz_path.parent / candidate.name
    source_metadata = load_json_if_exists(source_report_path)

    selected_by_layer: dict[int, list[int]] = {}
    score_by_layer: dict[int, list[float]] = {}
    layer_report: dict[str, dict[str, Any]] = {}
    for layer_idx in selected_layers:
        layer_key = f"layer_{layer_idx}"
        score_key = f"{layer_key}__{args.score_field}"
        ratio_key = f"{layer_key}__{args.retain_ratio_field}"
        forget_key = f"{layer_key}__{args.forget_mean_field}"
        if score_key not in npz.files:
            raise KeyError(f"Missing score field in npz: {score_key}")
        if ratio_key not in npz.files:
            raise KeyError(f"Missing retain-ratio field in npz: {ratio_key}")
        if forget_key not in npz.files:
            raise KeyError(f"Missing forget mean field in npz: {forget_key}")

        scores = np.asarray(npz[score_key], dtype=np.float32)
        ratios = np.asarray(npz[ratio_key], dtype=np.float32)
        forget_means = np.asarray(npz[forget_key], dtype=np.float32)
        keep = np.isfinite(scores) & np.isfinite(ratios) & np.isfinite(forget_means)
        if not args.allow_non_positive:
            keep &= scores > float(args.min_score)
        else:
            keep &= scores >= float(args.min_score)
        if float(args.retain_ratio_max) > 0.0:
            keep &= ratios <= float(args.retain_ratio_max)
        min_forget_threshold = float(args.min_forget_mean)
        if float(args.min_forget_percentile) > 0.0:
            finite_forget = forget_means[np.isfinite(forget_means)]
            if finite_forget.size:
                pct_threshold = float(np.percentile(finite_forget, float(args.min_forget_percentile)))
                min_forget_threshold = max(min_forget_threshold, pct_threshold)
        if min_forget_threshold > 0.0:
            keep &= forget_means >= min_forget_threshold

        candidate_indices = np.nonzero(keep)[0]
        candidate_scores = scores[candidate_indices]
        order = np.argsort(-candidate_scores)
        top_count = min(int(args.neurons_per_layer), int(order.size))
        chosen = candidate_indices[order[:top_count]].astype(np.int64)
        chosen_scores = scores[chosen].astype(np.float32)

        selected_by_layer[layer_idx] = [int(x) for x in chosen.tolist()]
        score_by_layer[layer_idx] = [float(x) for x in chosen_scores.tolist()]

        finite_scores = scores[np.isfinite(scores)]
        finite_ratios = ratios[np.isfinite(ratios)]
        selected_ratios = ratios[chosen] if chosen.size else np.asarray([], dtype=np.float32)
        selected_forget_means = forget_means[chosen] if chosen.size else np.asarray([], dtype=np.float32)
        layer_report[layer_key] = {
            "selected_neurons": int(chosen.size),
            "candidate_neurons_after_filter": int(candidate_indices.size),
            "score_field": args.score_field,
            "retain_ratio_field": args.retain_ratio_field,
            "forget_mean_field": args.forget_mean_field,
            "retain_ratio_max": float(args.retain_ratio_max),
            "min_score": float(args.min_score),
            "min_forget_mean": float(args.min_forget_mean),
            "min_forget_percentile": float(args.min_forget_percentile),
            "effective_min_forget_mean": float(min_forget_threshold),
            "score_mean": finite_or_zero(float(finite_scores.mean())) if finite_scores.size else 0.0,
            "score_max": finite_or_zero(float(finite_scores.max())) if finite_scores.size else 0.0,
            "selected_score_min": finite_or_zero(float(chosen_scores.min())) if chosen_scores.size else 0.0,
            "selected_score_mean": finite_or_zero(float(chosen_scores.mean())) if chosen_scores.size else 0.0,
            "selected_score_max": finite_or_zero(float(chosen_scores.max())) if chosen_scores.size else 0.0,
            "retain_ratio_mean": finite_or_zero(float(finite_ratios.mean())) if finite_ratios.size else 0.0,
            "selected_retain_ratio_min": finite_or_zero(float(selected_ratios.min())) if selected_ratios.size else 0.0,
            "selected_retain_ratio_mean": finite_or_zero(float(selected_ratios.mean())) if selected_ratios.size else 0.0,
            "selected_retain_ratio_max": finite_or_zero(float(selected_ratios.max())) if selected_ratios.size else 0.0,
            "selected_forget_mean_min": finite_or_zero(float(selected_forget_means.min())) if selected_forget_means.size else 0.0,
            "selected_forget_mean_mean": finite_or_zero(float(selected_forget_means.mean())) if selected_forget_means.size else 0.0,
            "selected_forget_mean_max": finite_or_zero(float(selected_forget_means.max())) if selected_forget_means.size else 0.0,
        }

    selected_by_layer = {layer: rows for layer, rows in selected_by_layer.items() if rows}
    if not selected_by_layer:
        raise ValueError("No neurons selected. Relax retain-ratio/min-score filters.")

    layer_scores: dict[str, list[float]] = {}
    for layer_idx, scores in score_by_layer.items():
        arr = np.asarray(scores, dtype=np.float32)
        layer_scores[f"layer_{layer_idx}"] = [float(x) for x in normalize_scores(arr).tolist()] if arr.size else []

    lora_parameter_shapes = manifest.get("lora_parameter_shapes")
    if not isinstance(lora_parameter_shapes, dict) or not lora_parameter_shapes:
        raise ValueError("Manifest does not contain lora_parameter_shapes; rerun localizer with candidate cache support.")
    parameter_masks = build_lora_masks_from_shapes(
        {str(k): [int(x) for x in v] for k, v in lora_parameter_shapes.items()},
        selected_by_layer,
        include_down_proj=not args.no_down_proj_mask,
        include_down_proj_lora_b=not args.no_down_proj_lora_b,
    )
    total_selected_params = int(sum(mask.sum() for mask in parameter_masks.values()))
    total_candidate_params = int(sum(mask.size for mask in parameter_masks.values()))

    metadata = dict(source_metadata)
    metadata.update(
        {
            "method_name": "native_llm_cached_candidate_localization_derivation",
            "derived_from_candidate_cache": str(npz_path),
            "candidate_manifest_json": str(manifest_path),
            "source_method_name": source_metadata.get("method_name", ""),
            "ranking_mode": "cached_positive_diff",
            "score_field": args.score_field,
            "retain_ratio_field": args.retain_ratio_field,
            "forget_mean_field": args.forget_mean_field,
            "positive_diff_retain_ratio_max": float(args.retain_ratio_max),
            "min_score": float(args.min_score),
            "min_forget_mean": float(args.min_forget_mean),
            "min_forget_percentile": float(args.min_forget_percentile),
            "allow_non_positive": bool(args.allow_non_positive),
            "layer_range": args.layer_range,
            "neurons_per_layer": int(args.neurons_per_layer),
            "include_down_proj_lora_b": bool(not args.no_down_proj_lora_b),
            "include_down_proj_mask": bool(not args.no_down_proj_mask),
            "layer_report": layer_report,
            "total_selected_params": total_selected_params,
            "total_candidate_params": total_candidate_params,
            "parameter_mask_density": float(total_selected_params / max(1, total_candidate_params)),
        }
    )
    localization = LocalizationArtifacts(
        strategy="native_llm_cached_candidate_localization_derivation",
        top_fraction=1.0,
        target_neurons={f"layer_{layer}": rows for layer, rows in selected_by_layer.items()},
        layer_scores=layer_scores,
        total_target_neurons=int(sum(len(rows) for rows in selected_by_layer.values())),
        parameter_masks=parameter_masks,
        metadata=metadata,
    )

    output_dir = Path(args.output_dir)
    save_pickle(output_dir / "localized_neurons.pkl", localization)
    save_json(output_dir / "localized_neurons.json", localization.to_serializable())
    save_json(output_dir / "native_llm_cached_localization_derivation_report.json", metadata)
    save_json(
        output_dir / "probe_summary.json",
        {
            "derived_from_candidate_cache": str(npz_path),
            "score_field": args.score_field,
            "retain_ratio_field": args.retain_ratio_field,
            "forget_mean_field": args.forget_mean_field,
            "retain_ratio_max": float(args.retain_ratio_max),
            "min_score": float(args.min_score),
            "min_forget_mean": float(args.min_forget_mean),
            "min_forget_percentile": float(args.min_forget_percentile),
            "selected_layers": sorted(int(k.split("_")[1]) for k in localization.target_neurons),
            "total_target_neurons": localization.total_target_neurons,
            "total_selected_params": total_selected_params,
            "parameter_mask_density": metadata["parameter_mask_density"],
        },
    )
    print(f"[Save] {output_dir / 'localized_neurons.pkl'}")
    print(f"[Save] {output_dir / 'localized_neurons.json'}")
    print(
        f"[Summary] layers={sorted(localization.target_neurons)} "
        f"target_neurons={localization.total_target_neurons} "
        f"mask_density={metadata['parameter_mask_density']:.6f}"
    )


if __name__ == "__main__":
    main()
