                      
                                                               

                                                                              
                                                                          
          

                                   
                                 
                                   

                                                                            
                                             

                                                                              
                                                                         
   

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


MLP_WEIGHT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.mlp\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
DEFAULT_SCORE_KEY = "combined_raw_diff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Top-k FFN masked model.")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--candidate-stats", type=Path, required=True)
    parser.add_argument("--output-model-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["FlowMU", "Random", "QAActivation"], required=True)
    parser.add_argument("--topk", type=int, required=True, help="Top-k neurons selected per layer.")
    parser.add_argument(
        "--mask-mode",
        choices=["drop", "keep"],
        default="drop",
        help=(
            "drop: zero selected Top-k neurons and keep all others; "
            "keep: keep selected Top-k neurons and zero all others."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-key",
        default=DEFAULT_SCORE_KEY,
        help="Per-layer key suffix in candidate_neuron_stats.npz.",
    )
    parser.add_argument(
        "--layers",
        default="0-31",
        help="Layer list/range, e.g. 0-31 or 0,1,2. Defaults to all 32 layers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-unchanged-shards",
        action="store_true",
        help="Copy unchanged safetensor shards instead of hardlinking them.",
    )
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def hardlink_or_copy(src: Path, dst: Path, *, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_sidecar_files(base_model: Path, output_dir: Path) -> None:
    for src in sorted(base_model.iterdir()):
        if src.is_dir():
            continue
        if src.suffix == ".safetensors":
            continue
        hardlink_or_copy(src, output_dir / src.name)


def load_weight_map(base_model: Path) -> dict[str, str]:
    index_path = base_model / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight map in {index_path}")
    return {str(k): str(v) for k, v in weight_map.items()}


def load_scores(candidate_stats: Path, layers: list[int], score_key: str) -> dict[int, np.ndarray]:
    payload = np.load(candidate_stats)
    scores: dict[int, np.ndarray] = {}
    for layer in layers:
        key = f"layer_{layer}__{score_key}"
        if key not in payload:
            available = [name for name in payload.files if name.startswith(f"layer_{layer}__")]
            raise KeyError(f"Missing score key {key}; available examples: {available[:8]}")
        arr = np.asarray(payload[key], dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"Invalid score array for {key}: shape={arr.shape}")
        scores[layer] = arr
    return scores


def build_selected_indices(
    scores: dict[int, np.ndarray],
    *,
    method: str,
    topk: int,
    seed: int,
) -> dict[int, np.ndarray]:
    if topk <= 0:
        raise ValueError("--topk must be positive")
    rng = random.Random(seed)
    selected_by_layer: dict[int, np.ndarray] = {}
    for layer, arr in scores.items():
        neuron_dim = int(arr.size)
        k = min(int(topk), neuron_dim)
        if method in {"FlowMU", "QAActivation"}:
            order = np.argsort(-arr, kind="stable")
            selected = np.sort(order[:k].astype(np.int64))
        else:
            selected = np.asarray(sorted(rng.sample(range(neuron_dim), k)), dtype=np.int64)
        selected_by_layer[layer] = selected
    return selected_by_layer


def should_edit_tensor(name: str, layers: set[int]) -> tuple[int, str] | None:
    match = MLP_WEIGHT_RE.search(name)
    if not match:
        return None
    layer = int(match.group("layer"))
    if layer not in layers:
        return None
    return layer, match.group("proj")


def zero_neurons(
    tensor: torch.Tensor,
    *,
    proj: str,
    selected: np.ndarray,
    mask_mode: str,
) -> tuple[torch.Tensor, int]:
    edited = tensor.clone()
    if proj in {"gate_proj", "up_proj"}:
        neuron_dim = int(edited.shape[0])
        axis = 0
    elif proj == "down_proj":
        neuron_dim = int(edited.shape[1])
        axis = 1
    else:                    
        raise ValueError(f"Unsupported projection: {proj}")

    in_range = torch.as_tensor(selected[selected < neuron_dim], dtype=torch.long)
    if mask_mode == "drop":
        zero_idx = in_range
    elif mask_mode == "keep":
        keep_mask = torch.zeros(neuron_dim, dtype=torch.bool)
        keep_mask[in_range] = True
        zero_idx = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    else:                    
        raise ValueError(f"Unsupported mask mode: {mask_mode}")

    if zero_idx.numel() > 0:
        edited.index_fill_(axis, zero_idx, 0)
    return edited, int(zero_idx.numel())


def build_model(args: argparse.Namespace) -> dict[str, Any]:
    base_model = args.base_model.resolve()
    output_dir = args.output_model_dir.resolve()
    candidate_stats = args.candidate_stats.resolve()
    layers = parse_layers(args.layers)
    layer_set = set(layers)

    if not base_model.exists():
        raise FileNotFoundError(base_model)
    if not candidate_stats.exists():
        raise FileNotFoundError(candidate_stats)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    scores = load_scores(candidate_stats, layers, args.score_key)
    selected = build_selected_indices(scores, method=args.method, topk=args.topk, seed=args.seed)
    neuron_dims = {layer: int(scores[layer].size) for layer in layers}

    copy_sidecar_files(base_model, output_dir)

    weight_map = load_weight_map(base_model)
    shards = sorted(set(weight_map.values()))
    edited_tensors = 0
    zeroed_neurons_by_layer: dict[int, int] = {layer: 0 for layer in layers}
    edited_shards: list[str] = []

    for shard_name in shards:
        src = base_model / shard_name
        dst = output_dir / shard_name
        names_in_shard = [name for name, mapped in weight_map.items() if mapped == shard_name]
        editable_names = [name for name in names_in_shard if should_edit_tensor(name, layer_set)]
        if not editable_names:
            hardlink_or_copy(src, dst, copy=args.copy_unchanged_shards)
            continue

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(src, framework="pt", device="cpu") as sf:
            metadata = sf.metadata()
            for name in sf.keys():
                tensor = sf.get_tensor(name)
                edit_info = should_edit_tensor(name, layer_set)
                if edit_info is None:
                    tensors[name] = tensor
                    continue
                layer, proj = edit_info
                tensors[name], zeroed_count = zero_neurons(
                    tensor,
                    proj=proj,
                    selected=selected[layer],
                    mask_mode=args.mask_mode,
                )
                zeroed_neurons_by_layer[layer] = max(zeroed_neurons_by_layer[layer], zeroed_count)
                edited_tensors += 1
        save_file(tensors, dst, metadata=metadata)
        edited_shards.append(shard_name)

    summary = {
        "format": "topk_ffn_masked_model_v2",
        "base_model": str(base_model),
        "candidate_stats": str(candidate_stats),
        "output_model_dir": str(output_dir),
        "method": args.method,
        "mask_mode": args.mask_mode,
        "topk": int(args.topk),
        "seed": int(args.seed),
        "score_key": args.score_key,
        "layers": layers,
        "neuron_dims": {str(k): v for k, v in neuron_dims.items()},
        "selected_counts": {str(k): int(len(v)) for k, v in selected.items()},
        "zeroed_counts": {str(k): int(v) for k, v in zeroed_neurons_by_layer.items()},
        "edited_tensors": edited_tensors,
        "edited_shards": edited_shards,
    }
    (output_dir / "topk_mask_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_model(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
