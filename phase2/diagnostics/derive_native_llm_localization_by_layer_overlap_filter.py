                      
                                                                                   

                                                                              
                                                                          
                                                                
   

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.pipeline.stage2_unlearning import LocalizationArtifacts


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def layer_from_param_name(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def parse_layers(text: str) -> set[int]:
    out: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def load_keep_layers(
    csv_path: Path,
    *,
    projection_max: float,
    cosine_max: float | None,
    explicit_layers: str,
) -> tuple[set[int], list[dict[str, Any]]]:
    if explicit_layers.strip():
        layers = parse_layers(explicit_layers)
        return layers, [{"layer": layer, "selection_reason": "explicit"} for layer in sorted(layers)]

    rows: list[dict[str, Any]] = []
    keep: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(float(row["layer"]))
            projection = float(row["projection_b_on_a"])
            cosine = float(row["cosine"])
            pass_projection = projection <= projection_max
            pass_cosine = cosine_max is None or cosine <= cosine_max
            selected = pass_projection and pass_cosine
            if selected:
                keep.add(layer)
            rows.append(
                {
                    "layer": layer,
                    "projection_b_on_a": projection,
                    "cosine": cosine,
                    "selected": selected,
                }
            )
    return keep, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter localization by low retain-overlap layers.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--overlap-layer-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--projection-max", type=float, default=0.1)
    parser.add_argument("--cosine-max", type=float, default=None)
    parser.add_argument("--explicit-layers", default="")
    args = parser.parse_args()

    source = load_pickle(Path(args.input_pkl))
    if not isinstance(source, LocalizationArtifacts):
        raise TypeError(f"Expected LocalizationArtifacts, got {type(source)!r}")

    keep_layers, layer_rows = load_keep_layers(
        Path(args.overlap_layer_csv),
        projection_max=float(args.projection_max),
        cosine_max=args.cosine_max,
        explicit_layers=args.explicit_layers,
    )
    if not keep_layers:
        raise ValueError("Layer filter selected no layers.")

    target_neurons = {
        layer_key: list(neurons)
        for layer_key, neurons in source.target_neurons.items()
        if int(layer_key.split("_")[-1]) in keep_layers
    }
    layer_scores = {
        layer_key: list(scores)
        for layer_key, scores in source.layer_scores.items()
        if int(layer_key.split("_")[-1]) in keep_layers
    }
    parameter_masks = {
        name: np.asarray(mask).copy()
        for name, mask in source.parameter_masks.items()
        if (layer := layer_from_param_name(name)) is not None and layer in keep_layers
    }

    metadata = dict(source.metadata or {})
    metadata.update(
        {
            "method_name": "native_llm_layer_overlap_filtered_localization",
            "parent_strategy": source.strategy,
            "parent_localization_pkl": str(args.input_pkl),
            "overlap_layer_csv": str(args.overlap_layer_csv),
            "selection_rule": "projection_b_on_a <= projection_max and optional cosine <= cosine_max",
            "projection_max": float(args.projection_max),
            "cosine_max": args.cosine_max,
            "explicit_layers": args.explicit_layers,
            "kept_layers": sorted(keep_layers),
            "layer_overlap_rows": layer_rows,
        }
    )

    derived = LocalizationArtifacts(
        strategy="native_llm_layer_overlap_filtered_localization",
        top_fraction=source.top_fraction,
        target_neurons=target_neurons,
        layer_scores=layer_scores,
        total_target_neurons=sum(len(v) for v in target_neurons.values()),
        parameter_masks=parameter_masks,
        metadata=metadata,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "localized_neurons.pkl").open("wb") as f:
        pickle.dump(derived, f)
    (output_dir / "localized_neurons.json").write_text(
        json.dumps(derived.to_serializable(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "layer_overlap_filter_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selected_params = int(sum(np.asarray(mask).sum() for mask in parameter_masks.values()))
    candidate_params = int(sum(np.asarray(mask).size for mask in parameter_masks.values()))
    print(f"kept_layers={sorted(keep_layers)}")
    print(f"target_neurons={derived.total_target_neurons}")
    print(f"mask_tensors={len(parameter_masks)} selected_params={selected_params} density={selected_params / max(1, candidate_params):.6f}")
    print(f"saved={output_dir / 'localized_neurons.pkl'}")


if __name__ == "__main__":
    main()
