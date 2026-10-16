                      
                                                                 

                                              
                                      
                                         
                                                                     

                                                                               
                                                                         
   

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    build_structured_random_masks,
    compare_result,
    evaluate_nll,
    inspect_mask_application,
    load_pickle,
    normalize_qa_rows,
    summarize_masks,
)
from phase2.pipeline.stage2_unlearning import register_gradient_mask_hooks, remove_gradient_mask_hooks              


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    r = int(args.new_lora_r or localization_metadata.get("new_lora_r", 16))
    alpha = int(args.new_lora_alpha or localization_metadata.get("new_lora_alpha", 32))
    dropout = float(
        args.new_lora_dropout
        if args.new_lora_dropout is not None
        else localization_metadata.get("new_lora_dropout", 0.05)
    )
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
    return tokenizer, model, {
        "new_lora_r": r,
        "new_lora_alpha": alpha,
        "new_lora_dropout": dropout,
        "new_lora_target_modules": ",".join(target_modules_list),
    }


def cycle_batches(rows: list[dict[str, str]], batch_size: int, *, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    cursor = len(indices)
    while True:
        if cursor + batch_size > len(indices):
            rng.shuffle(indices)
            cursor = 0
        batch_indices = indices[cursor : cursor + batch_size]
        cursor += batch_size
        yield [rows[idx] for idx in batch_indices]


def retain_kl_loss(current_logits: torch.Tensor, ref_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_current = current_logits[:, :-1, :].float()
    shift_ref = ref_logits[:, :-1, :].float()
    valid = attention_mask[:, 1:].to(dtype=torch.bool)
    if not bool(valid.any()):
        return shift_current.new_tensor(0.0)
    log_p = F.log_softmax(shift_current, dim=-1)
    q = F.softmax(shift_ref, dim=-1)
    token_kl = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
    return token_kl.masked_select(valid).mean()


def set_trainable_mask(model, parameter_masks: dict[str, Any]) -> int:
    trainable = 0
    mask_names = set(parameter_masks)
    for name, param in model.named_parameters():
        should_train = name in mask_names
        param.requires_grad_(should_train)
        if should_train:
            trainable += int(param.numel())
    return trainable


def snapshot_lora_state(model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def restore_lora_state(model, snapshot: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot.items():
            param = named.get(name)
            if param is not None:
                param.copy_(value.to(device=param.device, dtype=param.dtype))


def evaluate_split(model, tokenizer, forget_rows, retain_rows, *, batch_size: int, max_length: int, label: str) -> dict[str, Any]:
    model.eval()
    return {
        "label": label,
        "forget": evaluate_nll(model, tokenizer, forget_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-forget"),
        "retain": evaluate_nll(model, tokenizer, retain_rows, batch_size=batch_size, max_length=max_length, label=f"{label}-retain"),
        "mask_application": None,
    }


def train_with_mask(
    *,
    model,
    tokenizer,
    parameter_masks: dict[str, Any],
    forget_rows: list[dict[str, str]],
    retain_rows: list[dict[str, str]],
    args: argparse.Namespace,
    label: str,
    seed: int,
) -> dict[str, Any]:
    application = inspect_mask_application(model, parameter_masks)
    if int(application["matched_tensors"]) == 0:
        raise ValueError(f"No mask tensors matched for {label}.")

    trainable_params = set_trainable_mask(model, parameter_masks)
    handles = register_gradient_mask_hooks(model, parameter_masks)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    forget_iter = cycle_batches(forget_rows, args.train_batch_size, seed=seed)
    retain_iter = cycle_batches(retain_rows, args.train_batch_size, seed=seed + 17)
    device = next(model.parameters()).device
    history = []
    model.train()
    try:
        for step in tqdm(range(1, args.train_steps + 1), desc=f"{label}-train", leave=False):
            forget_batch = build_labeled_batch(tokenizer, next(forget_iter), args.max_length)
            retain_batch = build_labeled_batch(tokenizer, next(retain_iter), args.max_length)
            forget_batch = {key: value.to(device) for key, value in forget_batch.items()}
            retain_batch = {key: value.to(device) for key, value in retain_batch.items()}

            optimizer.zero_grad(set_to_none=True)
            forget_outputs = model(**forget_batch)
            forget_loss = forget_outputs.loss
            with torch.no_grad():
                with model.disable_adapter():
                    retain_ref = model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch["attention_mask"],
                    ).logits.detach()
            retain_outputs = model(**retain_batch)
            kl = retain_kl_loss(retain_outputs.logits, retain_ref, retain_batch["attention_mask"])
            loss = -args.forget_weight * forget_loss + args.retain_kl_weight * kl
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.max_grad_norm)
            optimizer.step()
            if step == 1 or step == args.train_steps or step % args.log_steps == 0:
                history.append(
                    {
                        "step": int(step),
                        "loss": float(loss.detach().cpu()),
                        "forget_ce": float(forget_loss.detach().cpu()),
                        "retain_kl": float(kl.detach().cpu()),
                    }
                )
    finally:
        remove_gradient_mask_hooks(handles)
        for param in model.parameters():
            param.requires_grad_(False)

    return {
        "label": label,
        "mask_application": application,
        "trainable_params_before_mask": int(trainable_params),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native-LLM fresh-LoRA mask short-trainability probe.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--new-lora-r", type=int, default=0)
    parser.add_argument("--new-lora-alpha", type=int, default=0)
    parser.add_argument("--new-lora-dropout", type=float, default=None)
    parser.add_argument("--new-lora-target-modules", default="")
    parser.add_argument("--forget-limit", type=int, default=40)
    parser.add_argument("--retain-limit", type=int, default=120)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--promising-margin", type=float, default=0.005)
    parser.add_argument("--include-random-control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    localization = load_pickle(Path(args.localization_pkl))
    parameter_masks = getattr(localization, "parameter_masks", {})
    if not parameter_masks:
        raise ValueError(f"Localization has no parameter masks: {args.localization_pkl}")
    localization_metadata = dict(getattr(localization, "metadata", {}) or {})

    forget_rows = normalize_qa_rows(Path(args.forget_path), limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = normalize_qa_rows(Path(args.retain_path), limit=args.retain_limit, seed=args.seed + 1, role="retain")
    tokenizer, model, lora_config = load_model(args, localization_metadata)
    for param in model.parameters():
        param.requires_grad_(False)

    mask_summary = summarize_masks(parameter_masks)
    target_application = inspect_mask_application(model, parameter_masks)
    print(f"[Mask] {mask_summary}")
    print(f"[MaskApplication:target] {target_application}")
    print(f"[FreshLoRA] {lora_config}")

    baseline = evaluate_split(
        model,
        tokenizer,
        forget_rows,
        retain_rows,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        label="baseline",
    )
    initial_lora = snapshot_lora_state(model)

    target_train = train_with_mask(
        model=model,
        tokenizer=tokenizer,
        parameter_masks=parameter_masks,
        forget_rows=forget_rows,
        retain_rows=retain_rows,
        args=args,
        label="target_mask",
        seed=args.seed + 100,
    )
    target_after = evaluate_split(
        model,
        tokenizer,
        forget_rows,
        retain_rows,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        label="target_after",
    )
    target_after["comparison"] = compare_result(target_after, baseline)
    target_after["mask_application"] = target_train["mask_application"]

    random_train = None
    random_after = None
    if args.include_random_control:
        restore_lora_state(model, initial_lora)
        random_masks = build_structured_random_masks(model, parameter_masks, seed=args.seed + 1000)
        print(f"[MaskApplication:random] {inspect_mask_application(model, random_masks)}")
        random_train = train_with_mask(
            model=model,
            tokenizer=tokenizer,
            parameter_masks=random_masks,
            forget_rows=forget_rows,
            retain_rows=retain_rows,
            args=args,
            label="random_mask",
            seed=args.seed + 200,
        )
        random_after = evaluate_split(
            model,
            tokenizer,
            forget_rows,
            retain_rows,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            label="random_after",
        )
        random_after["comparison"] = compare_result(random_after, baseline)
        random_after["mask_application"] = random_train["mask_application"]

    quality = {
        "target_gap": target_after["comparison"]["impact_gap_forget_minus_retain"],
        "random_gap": random_after["comparison"]["impact_gap_forget_minus_retain"] if random_after else None,
    }
    if random_after is None:
        quality["verdict"] = "measured"
    elif (
        quality["target_gap"] > quality["random_gap"] + args.promising_margin
        and target_after["comparison"]["forget_nll_increase"] > 0
    ):
        quality["verdict"] = "promising"
    elif quality["target_gap"] > quality["random_gap"] and target_after["comparison"]["forget_nll_increase"] > 0:
        quality["verdict"] = "weak_but_trainable"
    else:
        quality["verdict"] = "not_recommended"

    report = {
        "method_name": "native_llm_lora_mask_trainability_probe",
        "base_model_path": args.base_model_path,
        "localization_pkl": args.localization_pkl,
        "fresh_lora": lora_config,
        "config": {
            "forget_path": args.forget_path,
            "retain_path": args.retain_path,
            "forget_limit": args.forget_limit,
            "retain_limit": args.retain_limit,
            "train_steps": args.train_steps,
            "learning_rate": args.learning_rate,
            "forget_weight": args.forget_weight,
            "retain_kl_weight": args.retain_kl_weight,
            "seed": args.seed,
        },
        "mask": mask_summary,
        "target_mask_application": target_application,
        "quality": quality,
        "baseline": baseline,
        "target_train": target_train,
        "target_after": target_after,
        "random_train": random_train,
        "random_after": random_after,
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
            rows = [target_after]
            if random_after is not None:
                rows.append(random_after)
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

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 92)
    print("Native-LLM Fresh-LoRA Mask Trainability Probe")
    print("=" * 92)
    print(f"verdict: {quality['verdict']}")
    print(
        "target: "
        f"forget_nll+={target_after['comparison']['forget_nll_increase']:.6f} "
        f"retain_nll+={target_after['comparison']['retain_nll_increase']:.6f} "
        f"gap={target_after['comparison']['impact_gap_forget_minus_retain']:.6f}"
    )
    if random_after is not None:
        print(
            "random: "
            f"forget_nll+={random_after['comparison']['forget_nll_increase']:.6f} "
            f"retain_nll+={random_after['comparison']['retain_nll_increase']:.6f} "
            f"gap={random_after['comparison']['impact_gap_forget_minus_retain']:.6f}"
        )
    print(f"saved: {output_json}")


if __name__ == "__main__":
    main()
