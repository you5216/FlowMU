                      
                                                                 

                                                     

                                                                    
                                                                  
                                                             

                                                
   

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
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


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


def evaluate_nll(model, tokenizer, rows: list[dict[str, str]], *, batch_size: int, max_length: int, label: str) -> dict[str, Any]:
    device = next(model.parameters()).device
    values = []
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


def load_model(args: argparse.Namespace, localization_metadata: dict[str, Any]):
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
    if args.device == "cpu":
        device_map = {"": "cpu"}
    elif args.single_gpu_device_map:
        device_map = {"": 0}
    else:
        device_map = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not args.no_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    r = int(args.new_lora_r or localization_metadata.get("new_lora_r", 16))
    alpha = int(args.new_lora_alpha or localization_metadata.get("new_lora_alpha", 32))
    dropout = float(args.new_lora_dropout if args.new_lora_dropout is not None else localization_metadata.get("new_lora_dropout", 0.05))
    target_modules = args.new_lora_target_modules.strip() or str(
        localization_metadata.get("new_lora_target_modules", "gate_proj,up_proj,down_proj")
    )
    target_modules_list = [item.strip() for item in target_modules.split(",") if item.strip()]
    if not target_modules_list:
        raise ValueError("Fresh LoRA target modules must not be empty.")

    model = get_peft_model(
        base_model,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules_list,
        ),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, {
        "new_lora_r": r,
        "new_lora_alpha": alpha,
        "new_lora_dropout": dropout,
        "new_lora_target_modules": ",".join(target_modules_list),
    }


def summarize_masks(parameter_masks: dict[str, Any]) -> dict[str, Any]:
    selected = 0
    total = 0
    tensors = 0
    for mask in parameter_masks.values():
        arr = np.asarray(mask)
        selected += int(arr.sum())
        total += int(arr.size)
        tensors += 1
    return {
        "num_mask_tensors": tensors,
        "selected_params": selected,
        "candidate_params": total,
        "mask_density": float(selected / max(1, total)),
    }


def inspect_mask_application(model, parameter_masks: dict[str, Any]) -> dict[str, Any]:
    named_params = dict(model.named_parameters())
    matched = 0
    missing = 0
    shape_mismatch = 0
    applied_selected = 0
    applied_candidate = 0
    skipped = []
    for name, mask_np in parameter_masks.items():
        mask = np.asarray(mask_np)
        param = named_params.get(name)
        if param is None:
            missing += 1
            if len(skipped) < 10:
                skipped.append({"name": name, "reason": "missing"})
            continue
        if tuple(mask.shape) != tuple(param.shape):
            shape_mismatch += 1
            if len(skipped) < 10:
                skipped.append({"name": name, "reason": "shape_mismatch", "mask_shape": list(mask.shape), "param_shape": list(param.shape)})
            continue
        matched += 1
        applied_selected += int(mask.sum())
        applied_candidate += int(mask.size)
    return {
        "matched_tensors": matched,
        "missing_tensors": missing,
        "shape_mismatch_tensors": shape_mismatch,
        "applied_selected_params": applied_selected,
        "applied_candidate_params": applied_candidate,
        "applied_mask_density": float(applied_selected / max(1, applied_candidate)),
        "skipped_examples": skipped,
    }


@contextmanager
def perturb_masks(model, parameter_masks: dict[str, Any], *, mode: str, strength: float, seed: int):
    named_params = dict(model.named_parameters())
    backups: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    application = inspect_mask_application(model, parameter_masks)
    if int(application["matched_tensors"]) == 0:
        raise ValueError("No localization mask tensors matched the fresh LoRA model.")
    try:
        with torch.no_grad():
            for name, mask_np in parameter_masks.items():
                param = named_params.get(name)
                if param is None or tuple(np.asarray(mask_np).shape) != tuple(param.shape):
                    continue
                mask = torch.as_tensor(mask_np, device=param.device, dtype=param.dtype)
                backups[name] = param.data.detach().clone()
                if mode == "noise":
                    noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
                    noise = noise.to(device=param.device, dtype=param.dtype)
                    sigma = float(param.data.detach().float().std().item())
                    if sigma < 1e-6:
                        sigma = 1.0
                    param.data.add_(noise * torch.tensor(sigma, device=param.device, dtype=param.dtype) * strength * mask)
                elif mode == "zero":
                    param.data.mul_(1.0 - mask)
                elif mode == "scale":
                    param.data.mul_(1.0 - strength * mask)
                elif mode == "sign_flip":
                    param.data.mul_(1.0 - 2.0 * mask)
                else:
                    raise ValueError(f"Unsupported perturb mode: {mode}")
        yield application
    finally:
        with torch.no_grad():
            for name, backup in backups.items():
                param = named_params.get(name)
                if param is not None:
                    param.data.copy_(backup)


def build_structured_random_masks(model, parameter_masks: dict[str, Any], *, seed: int) -> dict[str, np.ndarray]:
    rng = random.Random(seed)
    named_params = dict(model.named_parameters())
    random_masks: dict[str, np.ndarray] = {}
    for name, source_mask in parameter_masks.items():
        param = named_params.get(name)
        if param is None:
            continue
        source = np.asarray(source_mask, dtype=np.float32)
        mask = np.zeros_like(source, dtype=np.float32)
        selected = int(source.sum())
        if selected <= 0:
            random_masks[name] = mask
            continue
        if source.ndim == 2:
            row_sums = source.sum(axis=1)
            col_sums = source.sum(axis=0)
            full_rows = np.where(row_sums == source.shape[1])[0].tolist()
            full_cols = np.where(col_sums == source.shape[0])[0].tolist()
            if full_rows and selected == len(full_rows) * source.shape[1]:
                rows = list(range(source.shape[0]))
                rng.shuffle(rows)
                mask[rows[: len(full_rows)], :] = 1.0
            elif full_cols and selected == len(full_cols) * source.shape[0]:
                cols = list(range(source.shape[1]))
                rng.shuffle(cols)
                mask[:, cols[: len(full_cols)]] = 1.0
            else:
                flat_idx = list(range(mask.size))
                rng.shuffle(flat_idx)
                mask.reshape(-1)[flat_idx[:selected]] = 1.0
        else:
            flat_idx = list(range(mask.size))
            rng.shuffle(flat_idx)
            mask.reshape(-1)[flat_idx[:selected]] = 1.0
        random_masks[name] = mask
    return random_masks


def evaluate_with_optional_perturbation(
    model,
    tokenizer,
    forget_rows: list[dict[str, str]],
    retain_rows: list[dict[str, str]],
    *,
    batch_size: int,
    max_length: int,
    parameter_masks: dict[str, Any] | None,
    mode: str,
    strength: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    if parameter_masks is None:
        forget = evaluate_nll(model, tokenizer, forget_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-forget")
        retain = evaluate_nll(model, tokenizer, retain_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-retain")
        application = None
    else:
        with perturb_masks(model, parameter_masks, mode=mode, strength=strength, seed=seed) as application:
            forget = evaluate_nll(model, tokenizer, forget_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-forget")
            retain = evaluate_nll(model, tokenizer, retain_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-retain")
    return {
        "label": label,
        "forget": forget,
        "retain": retain,
        "mask_application": application,
    }


def compare_result(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    forget_inc = result["forget"]["mean_nll"] - baseline["forget"]["mean_nll"]
    retain_inc = result["retain"]["mean_nll"] - baseline["retain"]["mean_nll"]
    return {
        "forget_nll_increase": float(forget_inc),
        "retain_nll_increase": float(retain_inc),
        "impact_gap_forget_minus_retain": float(forget_inc - retain_inc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native-LLM fresh-LoRA mask quality probe.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--single-gpu-device-map",
        action="store_true",
        help="For CUDA probing, force all modules onto visible GPU 0 instead of using device_map=auto.",
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--new-lora-r", type=int, default=0, help="0 means read from localization metadata.")
    parser.add_argument("--new-lora-alpha", type=int, default=0, help="0 means read from localization metadata.")
    parser.add_argument("--new-lora-dropout", type=float, default=None)
    parser.add_argument("--new-lora-target-modules", default="", help="Empty means read from localization metadata.")
    parser.add_argument("--forget-limit", type=int, default=40)
    parser.add_argument("--retain-limit", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--perturb-mode", choices=["noise", "zero", "scale", "sign_flip"], default="noise")
    parser.add_argument("--perturb-strength", type=float, default=0.02)
    parser.add_argument("--include-random-control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    localization = load_pickle(Path(args.localization_pkl))
    parameter_masks = getattr(localization, "parameter_masks", {})
    if not parameter_masks:
        raise ValueError(f"Localization has no parameter masks: {args.localization_pkl}")
    localization_metadata = dict(getattr(localization, "metadata", {}) or {})

    forget_rows = normalize_qa_rows(Path(args.forget_path), limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = normalize_qa_rows(Path(args.retain_path), limit=args.retain_limit, seed=args.seed + 1, role="retain")
    tokenizer, model, lora_config = load_model(args, localization_metadata)
    mask_summary = summarize_masks(parameter_masks)
    mask_application = inspect_mask_application(model, parameter_masks)
    print(f"[Mask] {mask_summary}")
    print(f"[MaskApplication] {mask_application}")
    print(f"[FreshLoRA] {lora_config}")

    baseline = evaluate_with_optional_perturbation(
        model,
        tokenizer,
        forget_rows,
        retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        parameter_masks=None,
        mode=args.perturb_mode,
        strength=args.perturb_strength,
        seed=args.seed,
        label="baseline",
    )
    target = evaluate_with_optional_perturbation(
        model,
        tokenizer,
        forget_rows,
        retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        parameter_masks=parameter_masks,
        mode=args.perturb_mode,
        strength=args.perturb_strength,
        seed=args.seed,
        label="target_mask",
    )
    target["comparison"] = compare_result(target, baseline)

    random_control = None
    if args.include_random_control:
        random_masks = build_structured_random_masks(model, parameter_masks, seed=args.seed + 1000)
        random_control = evaluate_with_optional_perturbation(
            model,
            tokenizer,
            forget_rows,
            retain_rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            parameter_masks=random_masks,
            mode=args.perturb_mode,
            strength=args.perturb_strength,
            seed=args.seed + 2000,
            label="same_density_random_mask",
        )
        random_control["comparison"] = compare_result(random_control, baseline)

    quality = {
        "target_gap": target["comparison"]["impact_gap_forget_minus_retain"],
        "random_gap": random_control["comparison"]["impact_gap_forget_minus_retain"] if random_control else None,
    }
    if random_control is None:
        quality["verdict"] = "measured"
    elif quality["target_gap"] > quality["random_gap"] + 0.005 and target["comparison"]["forget_nll_increase"] > 0:
        quality["verdict"] = "promising"
    elif quality["target_gap"] > quality["random_gap"]:
        quality["verdict"] = "weak_but_directional"
    else:
        quality["verdict"] = "not_recommended"

    report = {
        "method_name": "native_llm_lora_mask_quality_probe",
        "base_model_path": args.base_model_path,
        "localization_pkl": args.localization_pkl,
        "fresh_lora": lora_config,
        "config": {
            "forget_path": args.forget_path,
            "retain_path": args.retain_path,
            "forget_limit": args.forget_limit,
            "retain_limit": args.retain_limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "perturb_mode": args.perturb_mode,
            "perturb_strength": args.perturb_strength,
            "seed": args.seed,
        },
        "mask": mask_summary,
        "mask_application": mask_application,
        "quality": quality,
        "baseline": baseline,
        "target_mask": target,
        "random_control": random_control,
        "localization_summary": {
            "strategy": getattr(localization, "strategy", ""),
            "total_target_neurons": int(getattr(localization, "total_target_neurons", 0)),
            "metadata": localization_metadata,
        },
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
                    "applied_selected_params",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "label": row["label"],
                        "forget_nll_increase": row["comparison"]["forget_nll_increase"],
                        "retain_nll_increase": row["comparison"]["retain_nll_increase"],
                        "impact_gap_forget_minus_retain": row["comparison"]["impact_gap_forget_minus_retain"],
                        "applied_selected_params": row["mask_application"]["applied_selected_params"],
                    }
                )

    print("=" * 92)
    print("Native-LLM Fresh-LoRA Mask Quality Probe")
    print("=" * 92)
    print(f"verdict: {quality['verdict']}")
    print(
        "target: "
        f"forget_nll+={target['comparison']['forget_nll_increase']:.6f} "
        f"retain_nll+={target['comparison']['retain_nll_increase']:.6f} "
        f"gap={target['comparison']['impact_gap_forget_minus_retain']:.6f}"
    )
    if random_control is not None:
        print(
            "random: "
            f"forget_nll+={random_control['comparison']['forget_nll_increase']:.6f} "
            f"retain_nll+={random_control['comparison']['retain_nll_increase']:.6f} "
            f"gap={random_control['comparison']['impact_gap_forget_minus_retain']:.6f}"
        )
    print(f"saved: {output_json}")


if __name__ == "__main__":
    main()
