                      
                                                         

                                                                             
                                                                           
                                                                
   

from __future__ import annotations

import argparse
import json
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


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [dict(row) for row in obj]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_qa_rows(path: Path, *, limit: int, seed: int, role: str) -> list[dict[str, str]]:
    rows = []
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
    rendered = []
    prompt_lens = []
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


def build_model(args: argparse.Namespace, metadata: dict[str, Any]):
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
        metadata.get("new_lora_target_modules", "gate_proj,up_proj,down_proj")
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=int(args.new_lora_r or metadata.get("new_lora_r", 16)),
            lora_alpha=int(args.new_lora_alpha or metadata.get("new_lora_alpha", 32)),
            lora_dropout=float(args.new_lora_dropout if args.new_lora_dropout is not None else metadata.get("new_lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[item.strip() for item in target_modules.split(",") if item.strip()],
        ),
    )
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name and param.ndim == 2)
    model.train()
    return tokenizer, model


def evaluate_nll(model, tokenizer, rows: list[dict[str, str]], *, batch_size: int, max_length: int, label: str) -> dict[str, Any]:
    device = next(model.parameters()).device
    values = []
    model.eval()
    for start in tqdm(range(0, len(rows), batch_size), desc=label, leave=False):
        batch = build_labeled_batch(tokenizer, rows[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        values.extend(compute_example_nll(outputs.logits, batch["labels"]).detach().cpu().tolist())
    model.train()
    arr = np.asarray(values, dtype=np.float32)
    return {"n": int(arr.size), "mean_nll": float(arr.mean()) if arr.size else 0.0, "std_nll": float(arr.std()) if arr.size else 0.0}


def collect_retain_reference_logits(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    *,
    batch_size: int,
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    refs: list[dict[str, torch.Tensor]] = []
    model.eval()
    for start in tqdm(range(0, len(rows), batch_size), desc="retain-reference-logits", leave=False):
        batch = build_labeled_batch(tokenizer, rows[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits.detach().float().cpu()
        refs.append(
            {
                "input_ids": batch["input_ids"].detach().cpu(),
                "attention_mask": batch["attention_mask"].detach().cpu(),
                "logits": logits,
            }
        )
    model.train()
    return refs


def evaluate_retain_kl(
    model,
    refs: list[dict[str, torch.Tensor]],
    *,
    label: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    values: list[float] = []
    model.eval()
    for item in tqdm(refs, desc=label, leave=False):
        batch = {
            "input_ids": item["input_ids"].to(device),
            "attention_mask": item["attention_mask"].to(device),
        }
        ref_logits = item["logits"].to(device)
        with torch.no_grad():
            cur_logits = model(**batch).logits.float()
            mask = batch["attention_mask"][:, 1:].bool()
            cur_logp = torch.log_softmax(cur_logits[:, :-1, :], dim=-1)
            ref_logp = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_p = ref_logp.exp()
            token_kl = (ref_p * (ref_logp - cur_logp)).sum(dim=-1)
            denom = mask.float().sum().clamp_min(1.0)
            values.append(float((token_kl * mask.float()).sum().detach().cpu() / denom.detach().cpu()))
    model.train()
    arr = np.asarray(values, dtype=np.float32)
    return {"n": int(arr.size), "mean_kl": float(arr.mean()) if arr.size else 0.0, "std_kl": float(arr.std()) if arr.size else 0.0}


def inspect_mask_application(model, parameter_masks: dict[str, Any]) -> dict[str, Any]:
    named_params = dict(model.named_parameters())
    matched = missing = mismatch = selected = total = 0
    for name, mask_np in parameter_masks.items():
        param = named_params.get(name)
        mask = np.asarray(mask_np)
        if param is None:
            missing += 1
            continue
        if tuple(mask.shape) != tuple(param.shape):
            mismatch += 1
            continue
        matched += 1
        selected += int(mask.sum())
        total += int(mask.size)
    return {
        "matched_tensors": matched,
        "missing_tensors": missing,
        "shape_mismatch_tensors": mismatch,
        "applied_selected_params": selected,
        "applied_candidate_params": total,
        "applied_mask_density": float(selected / max(1, total)),
    }


def apply_ga_step(
    model,
    tokenizer,
    forget_rows: list[dict[str, str]],
    parameter_masks: dict[str, Any],
    *,
    batch_size: int,
    max_length: int,
    step_size: float,
) -> dict[str, float]:
    device = next(model.parameters()).device
    named_params = dict(model.named_parameters())
    grad_sums: dict[str, torch.Tensor] = {}
    steps = 0
    for start in tqdm(range(0, len(forget_rows), batch_size), desc="collect forget GA gradients", leave=False):
        batch = build_labeled_batch(tokenizer, forget_rows[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        for name, mask_np in parameter_masks.items():
            param = named_params.get(name)
            if param is None or param.grad is None or tuple(np.asarray(mask_np).shape) != tuple(param.shape):
                continue
            grad = param.grad.detach().float().cpu()
            grad_sums[name] = grad.clone() if name not in grad_sums else grad_sums[name] + grad
        steps += 1

    applied_norm = 0.0
    with torch.no_grad():
        for name, grad_sum in grad_sums.items():
            param = named_params.get(name)
            if param is None:
                continue
            mask = torch.as_tensor(parameter_masks[name], dtype=torch.float32)
            grad = (grad_sum / max(1, steps)) * mask
            applied_norm += float(grad.norm().item())
            param.data.add_(grad.to(device=param.device, dtype=param.dtype) * float(step_size))
    return {"gradient_batches": float(steps), "applied_grad_norm_sum": float(applied_norm)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native-LLM fresh-LoRA mask GA-step quality probe.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--new-lora-r", type=int, default=0)
    parser.add_argument("--new-lora-alpha", type=int, default=0)
    parser.add_argument("--new-lora-dropout", type=float, default=None)
    parser.add_argument("--new-lora-target-modules", default="")
    parser.add_argument("--forget-limit", type=int, default=40)
    parser.add_argument("--retain-limit", type=int, default=120)
    parser.add_argument("--ga-forget-limit", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    localization = load_pickle(Path(args.localization_pkl))
    parameter_masks = getattr(localization, "parameter_masks", {})
    if not parameter_masks:
        raise ValueError(f"Localization has no parameter masks: {args.localization_pkl}")
    metadata = dict(getattr(localization, "metadata", {}) or {})

    forget_rows = normalize_qa_rows(Path(args.forget_path), limit=args.forget_limit, seed=args.seed, role="forget_eval")
    retain_rows = normalize_qa_rows(Path(args.retain_path), limit=args.retain_limit, seed=args.seed + 1, role="retain_eval")
    ga_rows = normalize_qa_rows(Path(args.forget_path), limit=args.ga_forget_limit, seed=args.seed + 2, role="forget_ga")
    tokenizer, model = build_model(args, metadata)
    application = inspect_mask_application(model, parameter_masks)
    print(f"[MaskApplication] {application}")

    baseline_forget = evaluate_nll(model, tokenizer, forget_rows, batch_size=args.batch_size, max_length=args.max_length, label="baseline-forget")
    baseline_retain = evaluate_nll(model, tokenizer, retain_rows, batch_size=args.batch_size, max_length=args.max_length, label="baseline-retain")
    retain_refs = collect_retain_reference_logits(
        model,
        tokenizer,
        retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    ga_info = apply_ga_step(
        model,
        tokenizer,
        ga_rows,
        parameter_masks,
        batch_size=args.batch_size,
        max_length=args.max_length,
        step_size=args.step_size,
    )
    target_forget = evaluate_nll(model, tokenizer, forget_rows, batch_size=args.batch_size, max_length=args.max_length, label="ga-step-forget")
    target_retain = evaluate_nll(model, tokenizer, retain_rows, batch_size=args.batch_size, max_length=args.max_length, label="ga-step-retain")
    target_retain_kl = evaluate_retain_kl(model, retain_refs, label="ga-step-retain-kl")

    forget_inc = target_forget["mean_nll"] - baseline_forget["mean_nll"]
    retain_inc = target_retain["mean_nll"] - baseline_retain["mean_nll"]
    retain_kl_inc = target_retain_kl["mean_kl"]
    report = {
        "method_name": "native_llm_lora_mask_ga_step_probe",
        "base_model_path": args.base_model_path,
        "localization_pkl": args.localization_pkl,
        "config": vars(args),
        "mask_application": application,
        "ga_step": ga_info,
        "baseline": {"forget": baseline_forget, "retain": baseline_retain},
        "target": {"forget": target_forget, "retain": target_retain, "retain_kl": target_retain_kl},
        "comparison": {
            "forget_nll_increase": float(forget_inc),
            "retain_nll_increase": float(retain_inc),
            "retain_kl_increase": float(retain_kl_inc),
            "impact_gap_forget_minus_retain": float(forget_inc - retain_inc),
        },
        "localization_summary": {
            "strategy": getattr(localization, "strategy", ""),
            "total_target_neurons": int(getattr(localization, "total_target_neurons", 0)),
            "metadata": metadata,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 92)
    print("Native-LLM Fresh-LoRA Mask GA-Step Probe")
    print("=" * 92)
    print(
        f"forget_nll+={forget_inc:.6f} retain_nll+={retain_inc:.6f} "
        f"retain_kl+={retain_kl_inc:.6f} gap={forget_inc - retain_inc:.6f}"
    )
    print(f"saved: {output_json}")


if __name__ == "__main__":
    main()
