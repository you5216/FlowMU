                      
                                                     

                                                                            
                                                                               
                                                                            
                                             
   

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

DIAG_DIR = Path(__file__).resolve().parent
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

from native_llm_lora_mask_quality_probe import (              
    build_labeled_batch,
    compute_example_nll,
    load_pickle,
    normalize_qa_rows,
)


def resolve_decoder_layers(model):
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
    ]
    for path in candidate_paths:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise AttributeError(f"Could not resolve decoder layers for {type(model).__name__}")


def parse_layer_key(value: str) -> int | None:
    value = str(value)
    if value.startswith("layer_"):
        value = value.split("_", 1)[1]
    try:
        return int(value)
    except ValueError:
        return None


def load_target_neurons(localization_path: Path) -> tuple[dict[int, list[int]], dict[str, Any]]:
    localization = load_pickle(localization_path)
    raw = getattr(localization, "target_neurons", None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Localization has no target_neurons: {localization_path}")
    selected: dict[int, list[int]] = {}
    for layer_key, neurons in raw.items():
        layer_idx = parse_layer_key(str(layer_key))
        if layer_idx is None:
            continue
        values = sorted({int(x) for x in neurons if int(x) >= 0})
        if values:
            selected[layer_idx] = values
    if not selected:
        raise ValueError(f"No valid target neurons in {localization_path}")
    metadata = dict(getattr(localization, "metadata", {}) or {})
    metadata.update(
        {
            "strategy": str(getattr(localization, "strategy", "")),
            "total_target_neurons": int(getattr(localization, "total_target_neurons", sum(len(v) for v in selected.values()))),
        }
    )
    return selected, metadata


def build_random_neurons_like(
    selected: dict[int, list[int]],
    *,
    neuron_dim: int,
    seed: int,
) -> dict[int, list[int]]:
    rng = random.Random(seed)
    random_selected: dict[int, list[int]] = {}
    all_indices = list(range(neuron_dim))
    for layer_idx, neurons in selected.items():
        count = min(len(neurons), neuron_dim)
        sampled = rng.sample(all_indices, count)
        random_selected[layer_idx] = sorted(sampled)
    return random_selected


def summarize_neurons(selected: dict[int, list[int]], *, neuron_dim: int) -> dict[str, Any]:
    total = sum(len(v) for v in selected.values())
    layers = sorted(selected)
    return {
        "layers": layers,
        "num_layers": len(layers),
        "total_neurons": int(total),
        "neuron_dim": int(neuron_dim),
        "mean_neurons_per_layer": float(total / max(1, len(layers))),
        "density_over_selected_layers": float(total / max(1, len(layers) * neuron_dim)),
        "per_layer_counts": {f"layer_{layer}": len(selected[layer]) for layer in layers},
    }


def infer_mlp_neuron_dim(model, selected: dict[int, list[int]]) -> int:
                                                                                  
    layers = resolve_decoder_layers(model)
    for layer_idx in sorted(selected):
        if not (0 <= layer_idx < len(layers)):
            continue
        mlp = getattr(layers[layer_idx], "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is None:
            continue
        in_features = getattr(down_proj, "in_features", None)
        if in_features is not None:
            return int(in_features)
        weight = getattr(down_proj, "weight", None)
        shape = getattr(weight, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
    max_selected = max((max(values) for values in selected.values() if values), default=-1)
    if max_selected >= 0:
        return int(max_selected + 1)
    return int(model.get_input_embeddings().embedding_dim)


@contextmanager
def ablate_base_mlp_neurons(
    model,
    selected: dict[int, list[int]],
    *,
    mode: str,
    scale_factor: float,
    noise_std: float,
    seed: int,
):
    handles = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def make_hook(layer_idx: int, neuron_indices: list[int]):
        index_cache: dict[torch.device, torch.Tensor] = {}

        def hook(_module, inputs):
            if not inputs:
                return inputs
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                return inputs
            if value.ndim < 2:
                return inputs
            device = value.device
            idx = index_cache.get(device)
            if idx is None:
                valid = [i for i in neuron_indices if 0 <= i < value.shape[-1]]
                if not valid:
                    return inputs
                idx = torch.as_tensor(valid, device=device, dtype=torch.long)
                index_cache[device] = idx
            edited = value.clone()
            if mode == "zero":
                edited.index_fill_(-1, idx, 0.0)
            elif mode == "scale":
                selected_values = edited.index_select(-1, idx) * float(scale_factor)
                edited.index_copy_(-1, idx, selected_values)
            elif mode == "noise":
                selected_values = edited.index_select(-1, idx)
                noise = torch.randn(selected_values.shape, generator=generator, dtype=torch.float32)
                noise = noise.to(device=device, dtype=edited.dtype)
                sigma = float(selected_values.detach().float().std().item())
                if sigma < 1e-6:
                    sigma = 1.0
                edited.index_copy_(-1, idx, selected_values + noise * sigma * float(noise_std))
            else:
                raise ValueError(f"Unsupported ablation mode: {mode}")
            return (edited, *inputs[1:])

        return hook

    for layer_idx, layer in enumerate(resolve_decoder_layers(model)):
        if layer_idx not in selected:
            continue
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is not None:
            handles.append(down_proj.register_forward_pre_hook(make_hook(layer_idx, selected[layer_idx])))
    if not handles:
        raise ValueError("No down_proj hooks were registered for selected neurons.")
    try:
        yield {"registered_hooks": len(handles)}
    finally:
        for handle in handles:
            handle.remove()


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    device_map: Any = {"": "cpu"} if args.device == "cpu" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def evaluate_nll(model, tokenizer, rows: list[dict[str, str]], *, batch_size: int, max_length: int, label: str) -> dict[str, Any]:
    device = next(model.parameters()).device
    values: list[float] = []
    for start in tqdm(range(0, len(rows), batch_size), desc=label, leave=False):
        batch_rows = rows[start : start + batch_size]
        batch = build_labeled_batch(tokenizer, batch_rows, max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        values.extend(compute_example_nll(outputs.logits, batch["labels"]).detach().cpu().tolist())
    arr = np.asarray(values, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean_nll": float(arr.mean()) if arr.size else 0.0,
        "std_nll": float(arr.std()) if arr.size else 0.0,
    }


def evaluate_split(model, tokenizer, forget_rows, retain_rows, *, batch_size: int, max_length: int, label: str) -> dict[str, Any]:
    model.eval()
    return {
        "label": label,
        "forget": evaluate_nll(model, tokenizer, forget_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-forget"),
        "retain": evaluate_nll(model, tokenizer, retain_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-retain"),
    }


def compare_result(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    forget_inc = result["forget"]["mean_nll"] - baseline["forget"]["mean_nll"]
    retain_inc = result["retain"]["mean_nll"] - baseline["retain"]["mean_nll"]
    eps = 1e-6
    relative_forget = forget_inc / max(eps, baseline["forget"]["mean_nll"])
    relative_retain = retain_inc / max(eps, baseline["retain"]["mean_nll"])
    return {
        "forget_nll_increase": float(forget_inc),
        "retain_nll_increase": float(retain_inc),
        "impact_gap_forget_minus_retain": float(forget_inc - retain_inc),
        "relative_forget_nll_increase": float(relative_forget),
        "relative_retain_nll_increase": float(relative_retain),
        "relative_impact_gap_forget_minus_retain": float(relative_forget - relative_retain),
        "relative_shift_ratio_forget_over_retain": float(relative_forget / max(eps, abs(relative_retain))),
    }


def evaluate_with_optional_ablation(
    model,
    tokenizer,
    forget_rows,
    retain_rows,
    *,
    batch_size: int,
    max_length: int,
    selected: dict[int, list[int]] | None,
    args: argparse.Namespace,
    label: str,
    seed: int,
) -> dict[str, Any]:
    if selected is None:
        result = evaluate_split(model, tokenizer, forget_rows, retain_rows, batch_size=batch_size, max_length=max_length, label=label)
        result["ablation"] = None
        return result
    with ablate_base_mlp_neurons(
        model,
        selected,
        mode=args.ablation_mode,
        scale_factor=args.scale_factor,
        noise_std=args.noise_std,
        seed=seed,
    ) as ablation:
        result = evaluate_split(model, tokenizer, forget_rows, retain_rows, batch_size=batch_size, max_length=max_length, label=label)
        result["ablation"] = ablation
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base-model MLP neuron ablation probe.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--forget-limit", type=int, default=40)
    parser.add_argument("--retain-limit", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ablation-mode", choices=["zero", "scale", "noise"], default="zero")
    parser.add_argument("--scale-factor", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--include-random-control", action="store_true")
    parser.add_argument("--promising-margin", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, localization_metadata = load_target_neurons(Path(args.localization_pkl))
    forget_rows = normalize_qa_rows(Path(args.forget_path), limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = normalize_qa_rows(Path(args.retain_path), limit=args.retain_limit, seed=args.seed + 1, role="retain")
    tokenizer, model = load_model(args)
    neuron_dim = infer_mlp_neuron_dim(model, selected)

    selected_summary = summarize_neurons(selected, neuron_dim=neuron_dim)
    random_selected = build_random_neurons_like(selected, neuron_dim=neuron_dim, seed=args.seed + 1000)
    print(f"[BaseNeuron] {selected_summary}")

    baseline = evaluate_with_optional_ablation(
        model,
        tokenizer,
        forget_rows,
        retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        selected=None,
        args=args,
        label="baseline",
        seed=args.seed,
    )
    target = evaluate_with_optional_ablation(
        model,
        tokenizer,
        forget_rows,
        retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        selected=selected,
        args=args,
        label="target_neurons",
        seed=args.seed + 10,
    )
    target["comparison"] = compare_result(target, baseline)

    random_control = None
    if args.include_random_control:
        random_control = evaluate_with_optional_ablation(
            model,
            tokenizer,
            forget_rows,
            retain_rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            selected=random_selected,
            args=args,
            label="same_count_random_neurons",
            seed=args.seed + 20,
        )
        random_control["comparison"] = compare_result(random_control, baseline)

    quality = {
        "target_gap": target["comparison"]["impact_gap_forget_minus_retain"],
        "target_relative_gap": target["comparison"]["relative_impact_gap_forget_minus_retain"],
        "random_gap": random_control["comparison"]["impact_gap_forget_minus_retain"] if random_control else None,
        "random_relative_gap": random_control["comparison"]["relative_impact_gap_forget_minus_retain"] if random_control else None,
    }
    if random_control is None:
        quality["verdict"] = "measured"
    elif (
        quality["target_relative_gap"] > quality["random_relative_gap"] + args.promising_margin
        and target["comparison"]["forget_nll_increase"] > 0
    ):
        quality["verdict"] = "promising"
    elif quality["target_relative_gap"] > quality["random_relative_gap"] and target["comparison"]["forget_nll_increase"] > 0:
        quality["verdict"] = "weak_but_directional"
    else:
        quality["verdict"] = "not_recommended"

    report = {
        "method_name": "native_llm_base_neuron_ablation_probe",
        "base_model_path": args.base_model_path,
        "localization_pkl": args.localization_pkl,
        "config": {
            "forget_path": args.forget_path,
            "retain_path": args.retain_path,
            "forget_limit": args.forget_limit,
            "retain_limit": args.retain_limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "ablation_mode": args.ablation_mode,
            "scale_factor": args.scale_factor,
            "noise_std": args.noise_std,
            "seed": args.seed,
        },
        "selected_neurons": selected_summary,
        "quality": quality,
        "baseline": baseline,
        "target_neurons": target,
        "random_control": random_control,
        "localization_metadata": localization_metadata,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [target]
        if random_control is not None:
            rows.append(random_control)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "label",
                    "forget_nll_increase",
                    "retain_nll_increase",
                    "impact_gap_forget_minus_retain",
                    "relative_forget_nll_increase",
                    "relative_retain_nll_increase",
                    "relative_impact_gap_forget_minus_retain",
                ],
            )
            writer.writeheader()
            for row in rows:
                comp = row["comparison"]
                writer.writerow(
                    {
                        "label": row["label"],
                        "forget_nll_increase": comp["forget_nll_increase"],
                        "retain_nll_increase": comp["retain_nll_increase"],
                        "impact_gap_forget_minus_retain": comp["impact_gap_forget_minus_retain"],
                        "relative_forget_nll_increase": comp["relative_forget_nll_increase"],
                        "relative_retain_nll_increase": comp["relative_retain_nll_increase"],
                        "relative_impact_gap_forget_minus_retain": comp["relative_impact_gap_forget_minus_retain"],
                    }
                )

    print("=" * 92)
    print("Base-Model MLP Neuron Ablation Probe")
    print("=" * 92)
    print(f"verdict: {quality['verdict']}")
    print(
        "target: "
        f"forget_nll+={target['comparison']['forget_nll_increase']:.6f} "
        f"retain_nll+={target['comparison']['retain_nll_increase']:.6f} "
        f"gap={target['comparison']['impact_gap_forget_minus_retain']:.6f} "
        f"rel_gap={target['comparison']['relative_impact_gap_forget_minus_retain']:.6f}"
    )
    if random_control is not None:
        print(
            "random: "
            f"forget_nll+={random_control['comparison']['forget_nll_increase']:.6f} "
            f"retain_nll+={random_control['comparison']['retain_nll_increase']:.6f} "
            f"gap={random_control['comparison']['impact_gap_forget_minus_retain']:.6f} "
            f"rel_gap={random_control['comparison']['relative_impact_gap_forget_minus_retain']:.6f}"
        )
    print(f"saved: {output_json}")


if __name__ == "__main__":
    main()
