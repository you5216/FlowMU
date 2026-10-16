                      
                                                                  

                                                                             
                                                                               

        
                                                                
                                                                    
                                                                
   

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


AGENT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = AGENT_ROOT / "merge_model/native_llm"
DEFAULT_FORGET = Path("/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget05.json")
DEFAULT_RETAIN = Path("/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain95.json")
DEFAULT_LOCALIZED = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/"
    "phase2_localization_dualview_lambda03_top100_qaw2_20260622/localized_neurons.json"
)
DEFAULT_CANDIDATE_STATS = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/"
    "phase2_localization_dualview_lambda03_top300_qaw2_20260622/candidate_neuron_stats.npz"
)
DEFAULT_OUTPUT_DIR = AGENT_ROOT / "final_performance/final/tofu_for05_causal_perturbation_boxplot"


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [dict(row) for row in obj]
        if isinstance(obj, dict):
            return [dict(obj)]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_qa_rows(path: Path, *, limit: int, seed: int, role: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(load_json_or_jsonl(path)):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(
            row.get("answer")
            or row.get("output")
            or row.get("original_output")
            or row.get("reference_answer")
            or ""
        ).strip()
        if question and answer:
            rows.append({"instruction": question, "output": answer, "example_id": f"{role}_{idx}"})
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No valid QA rows loaded from {path}")
    return rows


def apply_chat_template(tokenizer: Any, instruction: str, output: str | None, *, add_generation_prompt: bool) -> str:
    messages = [{"role": "user", "content": instruction}]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_labeled_batch(tokenizer: Any, row: dict[str, str], max_length: int) -> dict[str, torch.Tensor]:
    prompt = apply_chat_template(tokenizer, row["instruction"], None, add_generation_prompt=True)
    full = apply_chat_template(tokenizer, row["instruction"], row["output"], add_generation_prompt=False)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False, truncation=True)["input_ids"])
    encoded = tokenizer([full], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    labels[0, : min(prompt_len, labels.shape[1])] = -100
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def resolve_decoder_layers(model: Any) -> list[Any]:
    current = model
    for attr in ("model", "language_model", "model"):
        if hasattr(current, attr):
            current = getattr(current, attr)
    layers = getattr(current, "layers", None)
    if layers is None and hasattr(current, "decoder"):
        layers = getattr(current.decoder, "layers", None)
    if layers is None:
        raise AttributeError(f"Cannot resolve decoder layers from {type(model).__name__}")
    return list(layers)


def infer_neuron_dim(model: Any) -> int:
    layers = resolve_decoder_layers(model)
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is None:
            continue
        in_features = getattr(down_proj, "in_features", None)
        if in_features is not None:
            return int(in_features)
        weight = getattr(down_proj, "weight", None)
        if weight is not None and len(weight.shape) >= 2:
            return int(weight.shape[1])
    raise AttributeError("Cannot infer MLP intermediate neuron dimension.")


def load_flowmu_neurons(path: Path) -> dict[int, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("target_neurons")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Missing target_neurons in {path}")
    selected: dict[int, list[int]] = {}
    for key, values in raw.items():
        layer_text = str(key)
        if layer_text.startswith("layer_"):
            layer_text = layer_text.split("_", 1)[1]
        layer = int(layer_text)
        unique = sorted({int(v) for v in values if int(v) >= 0})
        if unique:
            selected[layer] = unique
    if not selected:
        raise ValueError(f"No valid neurons loaded from {path}")
    return selected


def build_activation_neurons(
    candidate_stats: Path,
    reference: dict[int, list[int]],
    *,
    score_key: str,
) -> dict[int, list[int]]:
    payload = np.load(candidate_stats)
    selected: dict[int, list[int]] = {}
    for layer, reference_neurons in sorted(reference.items()):
        key = f"layer_{layer}__{score_key}"
        if key not in payload:
            raise KeyError(f"Missing {key} in {candidate_stats}")
        scores = np.asarray(payload[key], dtype=np.float64)
        k = min(len(reference_neurons), scores.size)
        order = np.argsort(-scores, kind="stable")[:k]
        selected[layer] = sorted(int(x) for x in order)
    return selected


def build_random_neurons(reference: dict[int, list[int]], *, neuron_dim: int, seed: int) -> dict[int, list[int]]:
    rng = random.Random(seed)
    selected: dict[int, list[int]] = {}
    population = list(range(neuron_dim))
    for layer, reference_neurons in sorted(reference.items()):
        k = min(len(reference_neurons), neuron_dim)
        selected[layer] = sorted(rng.sample(population, k))
    return selected


def summarize_neurons(method_neurons: dict[str, dict[int, list[int]]], *, neuron_dim: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method, selected in method_neurons.items():
        total = sum(len(v) for v in selected.values())
        summary[method] = {
            "layers": sorted(selected),
            "neuron_dim": int(neuron_dim),
            "total_neurons": int(total),
            "mean_neurons_per_layer": float(total / max(1, len(selected))),
            "density": float(total / max(1, len(selected) * neuron_dim)),
            "per_layer_counts": {str(layer): len(selected[layer]) for layer in sorted(selected)},
        }
    return summary


@contextmanager
def ablate_mlp_neurons(model: Any, selected: dict[int, list[int]]):
    handles = []

    def make_hook(neuron_indices: list[int]):
        index_cache: dict[torch.device, torch.Tensor] = {}

        def hook(_module: Any, inputs: tuple[Any, ...]):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            value = inputs[0]
            device = value.device
            idx = index_cache.get(device)
            if idx is None:
                valid = [i for i in neuron_indices if 0 <= i < value.shape[-1]]
                if not valid:
                    return inputs
                idx = torch.as_tensor(valid, device=device, dtype=torch.long)
                index_cache[device] = idx
            edited = value.clone()
            edited.index_fill_(-1, idx, 0.0)
            return (edited, *inputs[1:])

        return hook

    for layer_idx, layer in enumerate(resolve_decoder_layers(model)):
        if layer_idx not in selected:
            continue
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is not None:
            handles.append(down_proj.register_forward_pre_hook(make_hook(selected[layer_idx])))
    if not handles:
        raise RuntimeError("No hooks were registered.")
    try:
        yield {"registered_hooks": len(handles)}
    finally:
        for handle in handles:
            handle.remove()


def compute_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_loss = F.cross_entropy(
        shift_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    token_loss = token_loss * valid.to(token_loss.dtype)
    counts = valid.to(token_loss.dtype).sum(dim=1).clamp_min(1.0)
    return token_loss.sum(dim=1) / counts


def selected_shift_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    valid = labels[:, 1:] != -100
    if not bool(valid.any().item()):
        raise RuntimeError("No answer-token prediction positions in batch.")
    return shift_logits[valid]


def load_model(model_path: Path, *, dtype: str) -> tuple[Any, Any]:
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    panel: str,
    method_neurons: dict[str, dict[int, list[int]]],
    max_length: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    records: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"{panel} causal perturbation", leave=False):
        batch = build_labeled_batch(tokenizer, row, max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"]
        with torch.inference_mode():
            base_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        base_logits = selected_shift_logits(base_outputs.logits, labels).detach()
        base_nll = float(compute_nll_from_logits(base_outputs.logits, labels).detach().cpu().item())
        del base_outputs

        for method, neurons in method_neurons.items():
            with ablate_mlp_neurons(model, neurons):
                with torch.inference_mode():
                    perturbed_outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    )
            perturbed_logits = selected_shift_logits(perturbed_outputs.logits, labels)
            logits_mae = float((base_logits.float() - perturbed_logits.float()).abs().mean().detach().cpu().item())
            perturbed_nll = float(compute_nll_from_logits(perturbed_outputs.logits, labels).detach().cpu().item())
            nll_shift = perturbed_nll - base_nll
            records.append(
                {
                    "method": method,
                    "panel": panel,
                    "sample_id": row["example_id"],
                    "logits_mae": logits_mae,
                    "nll_shift": nll_shift,
                    "relative_nll_shift": nll_shift / max(1e-6, abs(base_nll)),
                    "base_nll": base_nll,
                    "perturbed_nll": perturbed_nll,
                    "answer_token_positions": int((labels[:, 1:] != -100).sum().detach().cpu().item()),
                }
            )
            del perturbed_outputs
        del base_logits
    return records


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "panel",
        "sample_id",
        "logits_mae",
        "nll_shift",
        "relative_nll_shift",
        "base_nll",
        "perturbed_nll",
        "answer_token_positions",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_summary(path: Path, *, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for panel in ["forget", "retain"]:
        grouped[panel] = {}
        for method in ["Random", "Activation", "FlowMU"]:
            values = np.asarray(
                [r["logits_mae"] for r in records if r["panel"] == panel and r["method"] == method],
                dtype=np.float64,
            )
            shifts = np.asarray(
                [r["nll_shift"] for r in records if r["panel"] == panel and r["method"] == method],
                dtype=np.float64,
            )
            grouped[panel][method] = {
                "n": int(values.size),
                "logits_mae_mean": float(values.mean()) if values.size else 0.0,
                "logits_mae_median": float(np.median(values)) if values.size else 0.0,
                "logits_mae_std": float(values.std()) if values.size else 0.0,
                "nll_shift_mean": float(shifts.mean()) if shifts.size else 0.0,
                "nll_shift_median": float(np.median(shifts)) if shifts.size else 0.0,
            }
    path.write_text(
        json.dumps({"metadata": metadata, "summary": grouped}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract real TOFU for05 causal perturbation records.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--forget-path", type=Path, default=DEFAULT_FORGET)
    parser.add_argument("--retain-path", type=Path, default=DEFAULT_RETAIN)
    parser.add_argument("--localized-json", type=Path, default=DEFAULT_LOCALIZED)
    parser.add_argument("--candidate-stats", type=Path, default=DEFAULT_CANDIDATE_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="tofu_for05_causal_perturbation_records")
    parser.add_argument("--forget-limit", type=int, default=80)
    parser.add_argument("--retain-limit", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--activation-score-key", default="combined_forget_mean")
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forget_rows = load_qa_rows(args.forget_path, limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = load_qa_rows(args.retain_path, limit=args.retain_limit, seed=args.seed + 1, role="retain")
    tokenizer, model = load_model(args.model_path, dtype=args.dtype)
    neuron_dim = infer_neuron_dim(model)

    flowmu = load_flowmu_neurons(args.localized_json)
    activation = build_activation_neurons(args.candidate_stats, flowmu, score_key=args.activation_score_key)
    random_neurons = build_random_neurons(flowmu, neuron_dim=neuron_dim, seed=args.seed + 1000)
    method_neurons = {
        "Random": random_neurons,
        "Activation": activation,
        "FlowMU": flowmu,
    }

    metadata = {
        "format": "tofu_for05_causal_perturbation_records_v1",
        "model_path": str(args.model_path),
        "forget_path": str(args.forget_path),
        "retain_path": str(args.retain_path),
        "localized_json": str(args.localized_json),
        "candidate_stats": str(args.candidate_stats),
        "forget_limit": int(args.forget_limit),
        "retain_limit": int(args.retain_limit),
        "max_length": int(args.max_length),
        "seed": int(args.seed),
        "metric": "answer-token logits MAE over the full vocabulary after zeroing selected mlp.down_proj input neurons",
        "activation_score_key": args.activation_score_key,
        "neuron_summary": summarize_neurons(method_neurons, neuron_dim=neuron_dim),
    }
    print(json.dumps(metadata["neuron_summary"], ensure_ascii=False, indent=2), flush=True)

    records = []
    records.extend(
        evaluate_rows(
            model,
            tokenizer,
            forget_rows,
            panel="forget",
            method_neurons=method_neurons,
            max_length=args.max_length,
        )
    )
    records.extend(
        evaluate_rows(
            model,
            tokenizer,
            retain_rows,
            panel="retain",
            method_neurons=method_neurons,
            max_length=args.max_length,
        )
    )
    output_csv = args.output_dir / f"{args.stem}.csv"
    output_json = args.output_dir / f"{args.stem}_summary.json"
    write_records_csv(output_csv, records)
    write_summary(output_json, records=records, metadata=metadata)
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved records: {output_csv}")
    print(f"Saved summary: {output_json}")


if __name__ == "__main__":
    main()
