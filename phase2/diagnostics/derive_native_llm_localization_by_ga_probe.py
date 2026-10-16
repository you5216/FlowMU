                      
                                                             

                                                                             
                                                                         

                                                        

                                                                              
                                                                              
   

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
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
    available_layers_from_npz,
    build_lora_masks_from_shapes,
    load_json_if_exists,
    parse_layer_range,
)
from phase2.pipeline.stage2_unlearning import LocalizationArtifacts, normalize_scores              


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def normalize_qa_rows(path: Path, *, limit: int, seed: int, role: str) -> list[dict[str, str]]:
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


def apply_chat_template(tokenizer, instruction: str, output: str | None, *, add_generation_prompt: bool) -> str:
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


def build_labeled_batch(tokenizer, rows: list[dict[str, str]], max_length: int) -> dict[str, torch.Tensor]:
    rendered: list[str] = []
    prompt_lens: list[int] = []
    for row in rows:
        prompt = apply_chat_template(tokenizer, row["instruction"], None, add_generation_prompt=True)
        full = apply_chat_template(tokenizer, row["instruction"], row["output"], add_generation_prompt=False)
        prompt_lens.append(len(tokenizer(prompt, add_special_tokens=False, truncation=True)["input_ids"]))
        rendered.append(full)
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    for row_idx, prompt_len in enumerate(prompt_lens):
        labels[row_idx, : min(prompt_len, labels.shape[1])] = -100
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}


def compute_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = -token_log_probs * valid.to(token_log_probs.dtype)
    counts = valid.to(token_nll.dtype).sum(dim=-1).clamp_min(1.0)
    return token_nll.sum(dim=-1) / counts


def build_model_and_tokenizer(args: argparse.Namespace, manifest: dict[str, Any]):
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
    target_modules_list = [item.strip() for item in target_modules.split(",") if item.strip()]
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=int(args.new_lora_r or manifest.get("new_lora_r", 16)),
            lora_alpha=int(args.new_lora_alpha or manifest.get("new_lora_alpha", 32)),
            lora_dropout=float(args.new_lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules_list,
        ),
    )
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name and param.ndim == 2)
    return tokenizer, model


def accumulate_grad_mean(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    *,
    batch_size: int,
    max_length: int,
    label: str,
) -> tuple[dict[str, np.ndarray], float]:
    device = next(model.parameters()).device
    grad_sums: dict[str, np.ndarray] = {}
    nll_values: list[float] = []
    count = 0
    for start in tqdm(range(0, len(rows), batch_size), desc=label, leave=False):
        batch_rows = rows[start : start + batch_size]
        batch = build_labeled_batch(tokenizer, batch_rows, max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        with torch.no_grad():
            nll_values.extend(compute_example_nll(outputs.logits, batch["labels"]).detach().cpu().tolist())
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None or param.ndim != 2 or "lora_" not in name:
                continue
            grad = param.grad.detach().float().cpu().numpy()
            grad_sums[name] = grad if name not in grad_sums else grad_sums[name] + grad
        count += 1
    denom = max(1, count)
    grad_means = {name: value / denom for name, value in grad_sums.items()}
    return grad_means, float(np.mean(nll_values)) if nll_values else 0.0


def parse_layer_from_param(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def init_layer_scores(npz: np.lib.npyio.NpzFile, layers: list[int]) -> dict[int, dict[int, dict[str, float]]]:
    out: dict[int, dict[int, dict[str, float]]] = {}
    for layer in layers:
        prefix = f"layer_{layer}"
        raw_diff = np.asarray(npz[f"{prefix}__combined_raw_diff"], dtype=np.float32)
        ratio = np.asarray(npz[f"{prefix}__combined_retain_to_forget_ratio"], dtype=np.float32)
        contrast = np.asarray(npz[f"{prefix}__combined_contrast_ratio"], dtype=np.float32)
        out[layer] = {
            int(idx): {
                "activation_score": float(raw_diff[idx]),
                "retain_ratio": float(ratio[idx]) if math.isfinite(float(ratio[idx])) else float("inf"),
                "contrast_ratio": float(contrast[idx]) if math.isfinite(float(contrast[idx])) else 0.0,
                "forget_grad_energy": 0.0,
                "retain_grad_energy": 0.0,
                "retain_grad_conflict": 0.0,
            }
            for idx in range(raw_diff.size)
        }
    return out


def add_grad_pair_to_neuron_scores(
    scores: dict[int, dict[int, dict[str, float]]],
    forget_grad_means: dict[str, np.ndarray],
    retain_grad_means: dict[str, np.ndarray],
) -> None:
    for name, forget_grad in forget_grad_means.items():
        layer = parse_layer_from_param(name)
        if layer is None or layer not in scores:
            continue
        retain_grad = retain_grad_means.get(name)
        if retain_grad is None or retain_grad.shape != forget_grad.shape:
            retain_grad = np.zeros_like(forget_grad, dtype=np.float32)
        if "lora_B" in name and (".gate_proj." in name or ".up_proj." in name):
            forget_energy = np.square(forget_grad).mean(axis=1)
            retain_energy = np.square(retain_grad).mean(axis=1)
            retain_conflict = (forget_grad * retain_grad).mean(axis=1)
            for neuron, value in enumerate(forget_energy.tolist()):
                if neuron in scores[layer]:
                    scores[layer][neuron]["forget_grad_energy"] += float(value)
                    scores[layer][neuron]["retain_grad_energy"] += float(retain_energy[neuron])
                    scores[layer][neuron]["retain_grad_conflict"] += float(retain_conflict[neuron])
        elif "lora_A" in name and ".down_proj." in name:
            forget_energy = np.square(forget_grad).mean(axis=0)
            retain_energy = np.square(retain_grad).mean(axis=0)
            retain_conflict = (forget_grad * retain_grad).mean(axis=0)
            for neuron, value in enumerate(forget_energy.tolist()):
                if neuron in scores[layer]:
                    scores[layer][neuron]["forget_grad_energy"] += float(value)
                    scores[layer][neuron]["retain_grad_energy"] += float(retain_energy[neuron])
                    scores[layer][neuron]["retain_grad_conflict"] += float(retain_conflict[neuron])


def finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def select_neurons(
    scores: dict[int, dict[int, dict[str, float]]],
    *,
    layers: list[int],
    neurons_per_layer: int,
    candidate_pool_per_layer: int,
    min_activation_score: float,
    retain_ratio_max: float,
    retain_lambda: float,
    activation_tiebreak_weight: float,
    min_probe_score: float,
) -> tuple[dict[int, list[int]], dict[int, list[float]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[int, list[int]] = {}
    selected_scores: dict[int, list[float]] = {}
    layer_report: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for layer in layers:
        layer_rows = []
        for neuron, item in scores[layer].items():
            activation_score = float(item["activation_score"])
            retain_ratio = float(item["retain_ratio"])
            if activation_score <= min_activation_score:
                continue
            if retain_ratio_max > 0.0 and retain_ratio > retain_ratio_max:
                continue
            forget_grad_energy = float(item["forget_grad_energy"])
            retain_grad_energy = float(item["retain_grad_energy"])
            retain_grad_conflict = max(0.0, float(item["retain_grad_conflict"]))
            probe_score = (
                forget_grad_energy
                - retain_lambda * retain_grad_conflict
                + activation_tiebreak_weight * activation_score
            )
            layer_rows.append(
                {
                    "layer": layer,
                    "neuron": neuron,
                    "probe_score": float(probe_score),
                    "forget_grad_energy": forget_grad_energy,
                    "retain_grad_energy": retain_grad_energy,
                    "retain_grad_conflict": retain_grad_conflict,
                    "activation_score": activation_score,
                    "retain_ratio": retain_ratio,
                    "contrast_ratio": float(item["contrast_ratio"]),
                }
            )
        layer_rows.sort(key=lambda row: row["probe_score"], reverse=True)
        candidate_count = len(layer_rows)
        if candidate_pool_per_layer > 0:
            layer_rows = layer_rows[:candidate_pool_per_layer]
        chosen = [row for row in layer_rows if float(row["probe_score"]) > min_probe_score]
        chosen = chosen[:neurons_per_layer]
        if chosen:
            selected[layer] = [int(row["neuron"]) for row in chosen]
            selected_scores[layer] = [float(row["probe_score"]) for row in chosen]
        rows.extend(layer_rows)
        selected_probe = [float(row["probe_score"]) for row in chosen]
        selected_forget = [float(row["forget_grad_energy"]) for row in chosen]
        selected_retain = [float(row["retain_grad_energy"]) for row in chosen]
        selected_conflict = [float(row["retain_grad_conflict"]) for row in chosen]
        layer_report[f"layer_{layer}"] = {
            "candidate_neurons_after_static_filter": int(candidate_count),
            "candidate_pool_after_truncation": int(len(layer_rows)),
            "selected_neurons": int(len(chosen)),
            "selected_probe_score_min": finite_or_zero(min(selected_probe)) if selected_probe else 0.0,
            "selected_probe_score_mean": finite_or_zero(float(np.mean(selected_probe))) if selected_probe else 0.0,
            "selected_probe_score_max": finite_or_zero(max(selected_probe)) if selected_probe else 0.0,
            "selected_forget_grad_energy_mean": finite_or_zero(float(np.mean(selected_forget))) if selected_forget else 0.0,
            "selected_retain_grad_energy_mean": finite_or_zero(float(np.mean(selected_retain))) if selected_retain else 0.0,
            "selected_retain_grad_conflict_mean": finite_or_zero(float(np.mean(selected_conflict))) if selected_conflict else 0.0,
        }
    rows.sort(key=lambda row: (int(row["layer"]), -float(row["probe_score"])))
    return selected, selected_scores, layer_report, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive localization by GA-gradient effect proxy.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--candidate-stats-npz", required=True)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--source-report-json", default="")
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-range", default="all")
    parser.add_argument("--forget-limit", type=int, default=64)
    parser.add_argument("--retain-limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--neurons-per-layer", type=int, default=100)
    parser.add_argument("--candidate-pool-per-layer", type=int, default=300)
    parser.add_argument("--min-activation-score", type=float, default=0.0)
    parser.add_argument("--retain-ratio-max", type=float, default=0.8)
    parser.add_argument("--retain-lambda", type=float, default=1.0)
    parser.add_argument("--activation-tiebreak-weight", type=float, default=0.01)
    parser.add_argument("--min-probe-score", type=float, default=0.0)
    parser.add_argument("--new-lora-r", type=int, default=0)
    parser.add_argument("--new-lora-alpha", type=int, default=0)
    parser.add_argument("--new-lora-dropout", type=float, default=0.05)
    parser.add_argument("--new-lora-target-modules", default="")
    parser.add_argument("--no-down-proj-mask", action="store_true")
    parser.add_argument("--no-down-proj-lora-b", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    npz_path = Path(args.candidate_stats_npz)
    manifest_path = Path(args.manifest_json) if args.manifest_json else npz_path.with_name("candidate_neuron_stats_manifest.json")
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

    forget_rows = normalize_qa_rows(Path(args.forget_path), limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = normalize_qa_rows(Path(args.retain_path), limit=args.retain_limit, seed=args.seed + 1, role="retain")
    tokenizer, model = build_model_and_tokenizer(args, manifest)

    print(f"[Input] candidate_stats={npz_path}")
    print(f"[Input] forget_rows={len(forget_rows)} retain_rows={len(retain_rows)} layers={selected_layers}")
    forget_grad, forget_nll = accumulate_grad_mean(
        model,
        tokenizer,
        forget_rows,
        batch_size=max(1, args.batch_size),
        max_length=max(1, args.max_length),
        label="GA-probe forget gradients",
    )
    retain_grad, retain_nll = accumulate_grad_mean(
        model,
        tokenizer,
        retain_rows,
        batch_size=max(1, args.batch_size),
        max_length=max(1, args.max_length),
        label="GA-probe retain gradients",
    )

    scores = init_layer_scores(npz, selected_layers)
    add_grad_pair_to_neuron_scores(scores, forget_grad, retain_grad)
    selected_by_layer, score_by_layer, layer_report, candidate_rows = select_neurons(
        scores,
        layers=selected_layers,
        neurons_per_layer=int(args.neurons_per_layer),
        candidate_pool_per_layer=int(args.candidate_pool_per_layer),
        min_activation_score=float(args.min_activation_score),
        retain_ratio_max=float(args.retain_ratio_max),
        retain_lambda=float(args.retain_lambda),
        activation_tiebreak_weight=float(args.activation_tiebreak_weight),
        min_probe_score=float(args.min_probe_score),
    )
    if not selected_by_layer:
        raise ValueError("No neurons selected. Relax retain/probe filters.")

    lora_parameter_shapes = manifest.get("lora_parameter_shapes")
    if not isinstance(lora_parameter_shapes, dict) or not lora_parameter_shapes:
        raise ValueError("Manifest does not contain lora_parameter_shapes.")
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

    metadata = dict(source_metadata)
    metadata.update(
        {
            "method_name": "native_llm_ga_probe_localization_derivation",
            "ranking_mode": "ga_gradient_effect_proxy",
            "score_definition": "probe_score = mean(g_forget^2) - retain_lambda * max(mean(g_forget*g_retain), 0) + activation_tiebreak_weight * combined_raw_diff",
            "candidate_stats_npz": str(npz_path),
            "candidate_manifest_json": str(manifest_path),
            "source_method_name": source_metadata.get("method_name", ""),
            "base_model_path": args.base_model_path,
            "forget_path": args.forget_path,
            "retain_path": args.retain_path,
            "forget_limit": int(args.forget_limit),
            "retain_limit": int(args.retain_limit),
            "forget_probe_nll": float(forget_nll),
            "retain_probe_nll": float(retain_nll),
            "layer_range": args.layer_range,
            "neurons_per_layer": int(args.neurons_per_layer),
            "candidate_pool_per_layer": int(args.candidate_pool_per_layer),
            "min_activation_score": float(args.min_activation_score),
            "retain_ratio_max": float(args.retain_ratio_max),
            "retain_lambda": float(args.retain_lambda),
            "activation_tiebreak_weight": float(args.activation_tiebreak_weight),
            "min_probe_score": float(args.min_probe_score),
            "new_lora_r": int(args.new_lora_r or manifest.get("new_lora_r", 16)),
            "new_lora_alpha": int(args.new_lora_alpha or manifest.get("new_lora_alpha", 32)),
            "new_lora_dropout": float(args.new_lora_dropout),
            "new_lora_target_modules": args.new_lora_target_modules.strip()
            or str(manifest.get("new_lora_target_modules", "gate_proj,up_proj,down_proj")),
            "include_down_proj_mask": bool(not args.no_down_proj_mask),
            "include_down_proj_lora_b": bool(not args.no_down_proj_lora_b),
            "layer_report": layer_report,
            "total_selected_params": total_selected_params,
            "total_candidate_params": total_candidate_params,
            "parameter_mask_density": float(total_selected_params / max(1, total_candidate_params)),
        }
    )
    localization = LocalizationArtifacts(
        strategy="native_llm_ga_probe_localization_derivation",
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
    save_json(output_dir / "native_llm_ga_probe_localization_report.json", metadata)
    save_json(
        output_dir / "probe_summary.json",
        {
            "ranking_mode": "ga_gradient_effect_proxy",
            "forget_rows": len(forget_rows),
            "retain_rows": len(retain_rows),
            "forget_probe_nll": float(forget_nll),
            "retain_probe_nll": float(retain_nll),
            "selected_layers": sorted(int(k.split("_")[1]) for k in localization.target_neurons),
            "total_target_neurons": localization.total_target_neurons,
            "total_selected_params": total_selected_params,
            "parameter_mask_density": metadata["parameter_mask_density"],
        },
    )
    csv_path = output_dir / "ga_probe_candidate_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "neuron",
                "probe_score",
                "forget_grad_energy",
                "retain_grad_energy",
                "retain_grad_conflict",
                "activation_score",
                "retain_ratio",
                "contrast_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(candidate_rows)
    print(f"[Save] {output_dir / 'localized_neurons.pkl'}")
    print(f"[Save] {output_dir / 'localized_neurons.json'}")
    print(f"[Save] {csv_path}")
    print(
        f"[Summary] layers={sorted(localization.target_neurons)} "
        f"target_neurons={localization.total_target_neurons} "
        f"mask_density={metadata['parameter_mask_density']:.6f}"
    )


if __name__ == "__main__":
    main()
