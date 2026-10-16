                      
                                                                    

                                                                             
                                                                         
                               

                                      
                                    
                                                        
                                                                     
                                  

                                                                            
                                         
   

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.diagnostics.derive_native_llm_localization_from_candidate_stats import (              
    build_lora_masks_from_shapes,
    load_json_if_exists,
    parse_layer_range,
)
from phase2.pipeline.stage2_unlearning import LocalizationArtifacts, normalize_scores              


@dataclass
class ActivationSet:
    rows: list[dict[str, str]]
    embeddings: np.ndarray
    layer_values: dict[int, np.ndarray]


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [dict(row) for row in obj if isinstance(row, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_rows(path: Path, *, limit: int, seed: int, role: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(load_json_or_jsonl(path)):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(row.get("answer") or row.get("output") or row.get("reference_answer") or "").strip()
        if question and answer:
            rows.append({"question": question, "answer": answer, "example_id": f"{role}_{idx}"})
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No valid QA rows loaded from {path}")
    return rows


def qa_text(row: dict[str, str]) -> str:
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


def apply_chat_template(tokenizer, row: dict[str, str]) -> str:
    messages = [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def parse_layer_from_name(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def build_model(args: argparse.Namespace, manifest: dict[str, Any]):
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
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto" if args.device == "cuda" else {"": "cpu"},
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not args.no_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    target_modules = args.new_lora_target_modules.strip() or str(
        manifest.get("new_lora_target_modules", "gate_proj,up_proj,down_proj")
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=int(args.new_lora_r or manifest.get("new_lora_r", 16)),
            lora_alpha=int(args.new_lora_alpha or manifest.get("new_lora_alpha", 32)),
            lora_dropout=float(args.new_lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[item.strip() for item in target_modules.split(",") if item.strip()],
        ),
    )
    model.eval()
    return tokenizer, model


def collect_down_proj_input_activations(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    *,
    layers: list[int],
    batch_size: int,
    max_length: int,
    label: str,
) -> dict[int, np.ndarray]:
    wanted = set(layers)
    buffers: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    hooks = []
    current_mask: torch.Tensor | None = None

    def make_hook(layer: int):
        def hook(_module, inputs):
            hidden = inputs[0].detach().float()
            mask = current_mask
            if mask is None:
                values = hidden.abs().mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).to(hidden.dtype)
                values = (hidden.abs() * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
            buffers[layer].append(values.cpu().numpy().astype(np.float32))
        return hook

    for name, module in model.named_modules():
        if not name.endswith("down_proj"):
            continue
        layer = parse_layer_from_name(name)
        if layer in wanted:
            hooks.append(module.register_forward_pre_hook(make_hook(layer)))

    device = next(model.parameters()).device
    try:
        for start in tqdm(range(0, len(rows), batch_size), desc=label, leave=False):
            batch_rows = rows[start : start + batch_size]
            texts = [apply_chat_template(tokenizer, row) for row in batch_rows]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            current_mask = encoded["attention_mask"].to(device)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                model(**encoded, use_cache=False)
    finally:
        current_mask = None
        for hook in hooks:
            hook.remove()

    out: dict[int, np.ndarray] = {}
    for layer, chunks in buffers.items():
        if not chunks:
            raise RuntimeError(f"No activations collected for layer {layer}")
        out[layer] = np.concatenate(chunks, axis=0)
    return out


def build_text_embeddings(args: argparse.Namespace, rows: list[dict[str, str]], *, label: str) -> np.ndarray:
    from phase1.core.qwen_embeddings import build_qwen_embeddings

    cache_path = Path(args.embedding_cache_dir) / f"{label}_qwen35_max{args.embedding_max_length}_n{len(rows)}.npz"
    metadata = {
        "model_path": args.embedding_model_path,
        "max_length": int(args.embedding_max_length),
        "rows": [row["example_id"] for row in rows],
    }
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            if json.loads(str(cached["metadata"].item())) == metadata:
                return cached["vectors"].astype(np.float32)
        except Exception:
            pass

    embedder = build_qwen_embeddings(
        model_path=args.embedding_model_path,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        hidden_layer=-1,
        pooling="mean",
        load_in_4bit=not args.no_embedding_4bit,
    )
    vectors = np.asarray(embedder.embed_documents([qa_text(row) for row in rows]), dtype=np.float32)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, vectors=vectors, metadata=json.dumps(metadata, ensure_ascii=False))
    return vectors


def robust_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def semantic_consistency_for_layer(
    activations: np.ndarray,
    embeddings: np.ndarray,
    *,
    threshold: np.ndarray,
    max_samples: int,
) -> np.ndarray:
    n_neurons = activations.shape[1]
    out = np.zeros(n_neurons, dtype=np.float32)
    global_centroid = embeddings.mean(axis=0)
    global_centroid = global_centroid / max(float(np.linalg.norm(global_centroid)), 1e-12)
    global_sim = float((embeddings @ global_centroid).mean())
    for neuron in range(n_neurons):
        idx = np.nonzero(activations[:, neuron] >= threshold[neuron])[0]
        if idx.size < 2:
            out[neuron] = 0.0
            continue
        if idx.size > max_samples:
                                                              
            order = np.argsort(-activations[idx, neuron])[:max_samples]
            idx = idx[order]
        centroid = embeddings[idx].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        sim = float((embeddings[idx] @ centroid).mean())
        out[neuron] = max(0.0, sim - global_sim)
    return out


def bootstrap_stability(
    forget_values: np.ndarray,
    retain_values: np.ndarray,
    *,
    n_bootstrap: int,
    top_fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_forget, n_neurons = forget_values.shape
    n_retain = retain_values.shape[0]
    hits = np.zeros(n_neurons, dtype=np.float32)
    top_k = max(1, int(round(n_neurons * top_fraction)))
    for _ in range(n_bootstrap):
        f_idx = rng.integers(0, n_forget, size=n_forget)
        r_idx = rng.integers(0, n_retain, size=n_retain)
        f_med = np.median(forget_values[f_idx], axis=0)
        r_med = np.median(retain_values[r_idx], axis=0)
        ratio = r_med / np.maximum(f_med, 1e-6)
        score = np.maximum(0.0, f_med - r_med) * np.maximum(0.0, 1.0 - ratio)
        ranked = np.argsort(-score)[:top_k]
        hits[ranked] += 1.0
    return hits / max(1, n_bootstrap)


def finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Representation-specific graph QA neuron localization.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-limit", type=int, default=512)
    parser.add_argument("--retain-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer-range", default="all")
    parser.add_argument("--neurons-per-layer", type=int, default=25)
    parser.add_argument("--global-max-neurons", type=int, default=800)
    parser.add_argument("--retain-ratio-max", type=float, default=0.8)
    parser.add_argument("--min-coverage", type=float, default=0.08)
    parser.add_argument("--activation-quantile", type=float, default=0.75)
    parser.add_argument("--n-bootstrap", type=int, default=12)
    parser.add_argument("--bootstrap-top-fraction", type=float, default=0.03)
    parser.add_argument("--semantic-max-samples", type=int, default=64)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument("--stability-weight", type=float, default=0.8)
    parser.add_argument("--new-lora-r", type=int, default=16)
    parser.add_argument("--new-lora-alpha", type=int, default=32)
    parser.add_argument("--new-lora-dropout", type=float, default=0.05)
    parser.add_argument("--new-lora-target-modules", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--no-down-proj-mask", action="store_true")
    parser.add_argument("--no-down-proj-lora-b", action="store_true")
    parser.add_argument("--embedding-model-path", default=str(AGENT_ROOT / "merge_model/native_llm"))
    parser.add_argument("--embedding-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--no-embedding-4bit", action="store_true")
    parser.add_argument("--embedding-cache-dir", default=str(AGENT_ROOT / "final_performance/tofu/for10_re90/cache/representation_specificity"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest_json)
    manifest = load_json_if_exists(manifest_path)
    lora_parameter_shapes = manifest.get("lora_parameter_shapes")
    if not isinstance(lora_parameter_shapes, dict) or not lora_parameter_shapes:
        raise ValueError("Manifest does not contain lora_parameter_shapes.")

    forget_rows = normalize_rows(Path(args.forget_path), limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = normalize_rows(Path(args.retain_path), limit=args.retain_limit, seed=args.seed + 1, role="retain")
    layers = parse_layer_range(args.layer_range, available_layers=list(range(32)))
    if not layers:
        raise ValueError("No layers selected.")

    print(f"[Input] forget={len(forget_rows)} retain={len(retain_rows)} layers={layers}")
    print("[Embedding] building forget semantic embeddings")
    forget_embeddings = build_text_embeddings(args, forget_rows, label=f"forget_{Path(args.forget_path).parent.name}")

    tokenizer, model = build_model(args, manifest)
    forget_act = collect_down_proj_input_activations(
        model,
        tokenizer,
        forget_rows,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="forget activations",
    )
    retain_act = collect_down_proj_input_activations(
        model,
        tokenizer,
        retain_rows,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="retain activations",
    )

    all_candidates: list[dict[str, Any]] = []
    layer_report: dict[str, dict[str, Any]] = {}
    for layer in layers:
        f = forget_act[layer]
        r = retain_act[layer]
        robust_forget = np.median(f, axis=0).astype(np.float32)
        robust_retain = np.median(r, axis=0).astype(np.float32)
        retain_ratio = robust_retain / np.maximum(robust_forget, 1e-6)
        threshold = np.quantile(f, float(args.activation_quantile), axis=0).astype(np.float32)
        coverage = (f >= threshold[None, :]).mean(axis=0).astype(np.float32)
                                                                                  
                                                         
        threshold_strength = robust_minmax(threshold)
        semantic = semantic_consistency_for_layer(
            f,
            forget_embeddings,
            threshold=threshold,
            max_samples=int(args.semantic_max_samples),
        )
        stability = bootstrap_stability(
            f,
            r,
            n_bootstrap=int(args.n_bootstrap),
            top_fraction=float(args.bootstrap_top_fraction),
            seed=int(args.seed + layer),
        )
        forget_strength = robust_minmax(robust_forget)
        specificity = np.maximum(0.0, 1.0 - retain_ratio).astype(np.float32)
        score = (
            forget_strength
            * threshold_strength
            * np.power(np.clip(coverage, 0.0, 1.0), float(args.coverage_weight))
            * (1.0 + float(args.semantic_weight) * robust_minmax(semantic))
            * (1.0 + float(args.stability_weight) * stability)
            * specificity
        ).astype(np.float32)
        keep = (
            np.isfinite(score)
            & (score > 0.0)
            & (retain_ratio <= float(args.retain_ratio_max))
            & (coverage >= float(args.min_coverage))
        )
        candidate_indices = np.nonzero(keep)[0]
        for neuron in candidate_indices.tolist():
            all_candidates.append(
                {
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "score": float(score[neuron]),
                    "robust_forget_activation": float(robust_forget[neuron]),
                    "robust_retain_activation": float(robust_retain[neuron]),
                    "retain_ratio": float(retain_ratio[neuron]),
                    "coverage": float(coverage[neuron]),
                    "semantic_consistency": float(semantic[neuron]),
                    "stability": float(stability[neuron]),
                    "threshold_strength": float(threshold_strength[neuron]),
                }
            )
        layer_scores = score[candidate_indices] if candidate_indices.size else np.asarray([], dtype=np.float32)
        layer_report[f"layer_{layer}"] = {
            "candidate_neurons_after_filter": int(candidate_indices.size),
            "score_mean": finite_or_zero(float(layer_scores.mean())) if layer_scores.size else 0.0,
            "score_max": finite_or_zero(float(layer_scores.max())) if layer_scores.size else 0.0,
            "forget_median_mean": finite_or_zero(float(robust_forget.mean())),
            "retain_median_mean": finite_or_zero(float(robust_retain.mean())),
            "semantic_mean": finite_or_zero(float(semantic.mean())),
            "stability_mean": finite_or_zero(float(stability.mean())),
        }

    if not all_candidates:
        raise ValueError("No representation-specific neurons selected. Relax filters.")

    all_candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    per_layer_count: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for row in all_candidates:
        layer = int(row["layer"])
        if per_layer_count.get(layer, 0) >= int(args.neurons_per_layer):
            continue
        selected.append(row)
        per_layer_count[layer] = per_layer_count.get(layer, 0) + 1
        if len(selected) >= int(args.global_max_neurons):
            break
    selected_by_layer: dict[int, list[int]] = {}
    score_by_layer: dict[int, list[float]] = {}
    for row in selected:
        layer = int(row["layer"])
        selected_by_layer.setdefault(layer, []).append(int(row["neuron"]))
        score_by_layer.setdefault(layer, []).append(float(row["score"]))

    parameter_masks = build_lora_masks_from_shapes(
        {str(k): [int(x) for x in v] for k, v in lora_parameter_shapes.items()},
        selected_by_layer,
        include_down_proj=not args.no_down_proj_mask,
        include_down_proj_lora_b=not args.no_down_proj_lora_b,
    )
    total_selected_params = int(sum(mask.sum() for mask in parameter_masks.values()))
    total_candidate_params = int(sum(mask.size for mask in parameter_masks.values()))
    layer_scores = {}
    for layer, values in score_by_layer.items():
        arr = np.asarray(values, dtype=np.float32)
        layer_scores[f"layer_{layer}"] = [float(x) for x in normalize_scores(arr).tolist()] if arr.size else []

    metadata = {
        "method_name": "native_llm_representation_specificity_localization",
        "score_definition": "robust_forget * threshold_strength * coverage * semantic_consistency * stability * (1 - retain_ratio)",
        "base_model_path": args.base_model_path,
        "forget_path": args.forget_path,
        "retain_path": args.retain_path,
        "forget_count": len(forget_rows),
        "retain_count": len(retain_rows),
        "layer_range": args.layer_range,
        "layers": layers,
        "neurons_per_layer": int(args.neurons_per_layer),
        "global_max_neurons": int(args.global_max_neurons),
        "retain_ratio_max": float(args.retain_ratio_max),
        "min_coverage": float(args.min_coverage),
        "activation_quantile": float(args.activation_quantile),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_top_fraction": float(args.bootstrap_top_fraction),
        "semantic_max_samples": int(args.semantic_max_samples),
        "activation_source": "mlp.down_proj input absolute activation",
        "activation_pooling": "token mean then sample distribution metrics",
        "uses_gradient_or_training_objective": False,
        "new_lora_r": int(args.new_lora_r),
        "new_lora_alpha": int(args.new_lora_alpha),
        "new_lora_dropout": float(args.new_lora_dropout),
        "new_lora_target_modules": args.new_lora_target_modules,
        "include_down_proj_lora_b": bool(not args.no_down_proj_lora_b),
        "layer_report": layer_report,
        "total_selected_params": total_selected_params,
        "total_candidate_params": total_candidate_params,
        "parameter_mask_density": float(total_selected_params / max(1, total_candidate_params)),
    }
    localization = LocalizationArtifacts(
        strategy="native_llm_representation_specificity_localization",
        top_fraction=1.0,
        target_neurons={f"layer_{layer}": rows for layer, rows in selected_by_layer.items()},
        layer_scores=layer_scores,
        total_target_neurons=int(sum(len(rows) for rows in selected_by_layer.values())),
        parameter_masks=parameter_masks,
        metadata=metadata,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "localized_neurons.pkl").open("wb") as f:
        pickle.dump(localization, f)
    (output_dir / "localized_neurons.json").write_text(
        json.dumps(localization.to_serializable(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "native_llm_representation_specificity_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "representation_specificity_candidate_scores.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "layer",
            "neuron",
            "score",
            "robust_forget_activation",
            "robust_retain_activation",
            "retain_ratio",
            "coverage",
            "semantic_consistency",
            "stability",
            "threshold_strength",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_candidates)
    (output_dir / "probe_summary.json").write_text(
        json.dumps(
            {
                "total_target_neurons": localization.total_target_neurons,
                "selected_layers": sorted(int(k.split("_")[1]) for k in localization.target_neurons),
                "total_selected_params": total_selected_params,
                "parameter_mask_density": metadata["parameter_mask_density"],
                "uses_gradient_or_training_objective": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Save] {output_dir / 'localized_neurons.pkl'}")
    print(
        f"[Summary] layers={sorted(localization.target_neurons)} "
        f"target_neurons={localization.total_target_neurons} "
        f"mask_density={metadata['parameter_mask_density']:.6f}"
    )


if __name__ == "__main__":
    main()
