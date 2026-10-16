                      
                                                                          

                                                                    
                                                                  
                                                                             
                                                               
                                                            
                                                         
                                                

                                                                
   

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
    compare_result,
    evaluate_nll,
    inspect_mask_application,
    load_pickle,
    normalize_qa_rows,
    summarize_masks,
)
from native_llm_lora_mask_trainability_probe import (              
    cycle_batches,
    retain_kl_loss,
    set_trainable_mask,
)
from phase2.pipeline.stage2_unlearning import register_gradient_mask_hooks, remove_gradient_mask_hooks              


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layer_from_module_name(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


class DownProjInputCapture:
    def __init__(self, model, target_neurons: dict[str, list[int]]):
        self.target_by_layer = {
            int(layer_key.split("_")[-1]): torch.as_tensor(indices, dtype=torch.long)
            for layer_key, indices in target_neurons.items()
            if indices
        }
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        wanted = set(self.target_by_layer)
        for name, module in model.named_modules():
            if not name.endswith("mlp.down_proj"):
                continue
            layer = parse_layer_from_module_name(name)
            if layer in wanted:
                self.handles.append(module.register_forward_pre_hook(self._make_hook(layer)))

    def _make_hook(self, layer: int):
        def hook(_module, inputs):
            if not inputs:
                return
            self.activations[layer] = inputs[0]

        return hook

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def pooled(self, labels: torch.Tensor) -> dict[int, torch.Tensor]:
        answer_mask = (labels != -100).to(torch.float32)
        pooled: dict[int, torch.Tensor] = {}
        for layer, indices_cpu in self.target_by_layer.items():
            if layer not in self.activations:
                continue
            act = self.activations[layer]
            indices = indices_cpu.to(device=act.device)
            selected = act.index_select(dim=-1, index=indices).float()
            mask = answer_mask.to(device=act.device)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled[layer] = (selected * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


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

    target_modules = args.new_lora_target_modules.strip() or str(
        localization_metadata.get("new_lora_target_modules", "gate_proj,up_proj,down_proj")
    )
    target_modules_list = [item.strip() for item in target_modules.split(",") if item.strip()]
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=int(args.new_lora_r or localization_metadata.get("new_lora_r", 16)),
            lora_alpha=int(args.new_lora_alpha or localization_metadata.get("new_lora_alpha", 32)),
            lora_dropout=float(
                args.new_lora_dropout
                if args.new_lora_dropout is not None
                else localization_metadata.get("new_lora_dropout", 0.05)
            ),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules_list,
        ),
    )
    return tokenizer, model


def precompute_prototypes(
    *,
    model,
    tokenizer,
    capture: DownProjInputCapture,
    rows: list[dict[str, str]],
    batch_size: int,
    max_length: int,
    label: str,
) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    values: dict[int, list[torch.Tensor]] = {layer: [] for layer in capture.target_by_layer}
    model.eval()
    for start in tqdm(range(0, len(rows), batch_size), desc=f"prototype-{label}", leave=False):
        batch = build_labeled_batch(tokenizer, rows[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        capture.clear()
        with torch.no_grad():
            model(**batch)
        pooled = capture.pooled(batch["labels"])
        for layer, tensor in pooled.items():
            values[layer].append(tensor.detach().cpu())
    prototypes: dict[int, torch.Tensor] = {}
    for layer, chunks in values.items():
        if not chunks:
            continue
        stacked = torch.cat(chunks, dim=0).float()
        prototypes[layer] = torch.median(stacked, dim=0).values
    model.train()
    return prototypes


def prototype_cosine_loss(
    pooled: dict[int, torch.Tensor],
    prototypes: dict[int, torch.Tensor],
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    cosines = []
    for layer, tensor in pooled.items():
        proto = prototypes.get(layer)
        if proto is None:
            continue
        proto = proto.to(device=tensor.device, dtype=tensor.dtype).unsqueeze(0).expand_as(tensor)
        cosine = F.cosine_similarity(F.normalize(tensor, dim=-1), F.normalize(proto, dim=-1), dim=-1)
        if mode == "repel":
            loss = torch.relu(cosine).pow(2)
        elif mode == "attract":
            loss = 1.0 - cosine
        else:
            raise ValueError(f"Unknown prototype mode: {mode}")
        losses.append(loss.mean())
        cosines.append(cosine.mean())
    if not losses:
        device = next(iter(pooled.values())).device if pooled else torch.device("cpu")
        zero = torch.zeros((), device=device)
        return zero, zero
    return torch.stack(losses).mean(), torch.stack(cosines).mean()


def evaluate_prototype_cosines(
    *,
    model,
    tokenizer,
    capture: DownProjInputCapture,
    rows: list[dict[str, str]],
    prototypes: dict[int, torch.Tensor],
    batch_size: int,
    max_length: int,
    label: str,
) -> dict[str, float]:
    device = next(model.parameters()).device
    vals = []
    model.eval()
    for start in tqdm(range(0, len(rows), batch_size), desc=f"proto-cos-{label}", leave=False):
        batch = build_labeled_batch(tokenizer, rows[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        capture.clear()
        with torch.no_grad():
            model(**batch)
        pooled = capture.pooled(batch["labels"])
        _, cosine = prototype_cosine_loss(pooled, prototypes, mode="attract")
        vals.append(float(cosine.detach().cpu()))
    arr = np.asarray(vals, dtype=np.float32)
    return {"mean_cosine": float(arr.mean()) if arr.size else 0.0, "n_batches": int(arr.size)}


def evaluate_all(model, tokenizer, capture, forget_rows, retain_rows, forget_proto, retain_proto, args, label: str):
    return {
        "label": label,
        "forget": evaluate_nll(
            model,
            tokenizer,
            forget_rows,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            label=f"{label}-forget-nll",
        ),
        "retain": evaluate_nll(
            model,
            tokenizer,
            retain_rows,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            label=f"{label}-retain-nll",
        ),
        "forget_to_forget_proto": evaluate_prototype_cosines(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            rows=forget_rows,
            prototypes=forget_proto,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            label=f"{label}-forget",
        ),
        "retain_to_retain_proto": evaluate_prototype_cosines(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            rows=retain_rows,
            prototypes=retain_proto,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            label=f"{label}-retain",
        ),
    }


def train_probe(
    *,
    model,
    tokenizer,
    capture: DownProjInputCapture,
    parameter_masks: dict[str, Any],
    forget_rows: list[dict[str, str]],
    retain_rows: list[dict[str, str]],
    forget_proto: dict[int, torch.Tensor],
    retain_proto: dict[int, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    set_trainable_mask(model, parameter_masks)
    handles = register_gradient_mask_hooks(model, parameter_masks)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    forget_iter = cycle_batches(forget_rows, args.train_batch_size, seed=args.seed + 100)
    retain_iter = cycle_batches(retain_rows, args.train_batch_size, seed=args.seed + 200)
    device = next(model.parameters()).device
    history: list[dict[str, float]] = []
    model.train()
    try:
        for step in tqdm(range(1, args.train_steps + 1), desc="prototype-train", leave=False):
            forget_batch = build_labeled_batch(tokenizer, next(forget_iter), args.max_length)
            retain_batch = build_labeled_batch(tokenizer, next(retain_iter), args.max_length)
            forget_batch = {key: value.to(device) for key, value in forget_batch.items()}
            retain_batch = {key: value.to(device) for key, value in retain_batch.items()}

            optimizer.zero_grad(set_to_none=True)

            capture.clear()
            forget_outputs = model(**forget_batch)
            forget_pooled = capture.pooled(forget_batch["labels"])
            forget_proto_loss, forget_cos = prototype_cosine_loss(forget_pooled, forget_proto, mode="repel")

            with torch.no_grad():
                with model.disable_adapter():
                    retain_ref = model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch["attention_mask"],
                    ).logits.detach()

            capture.clear()
            retain_outputs = model(**retain_batch)
            retain_pooled = capture.pooled(retain_batch["labels"])
            retain_proto_loss, retain_cos = prototype_cosine_loss(retain_pooled, retain_proto, mode="attract")
            kl = retain_kl_loss(retain_outputs.logits, retain_ref, retain_batch["attention_mask"])

            loss = (
                args.forget_proto_weight * forget_proto_loss
                + args.retain_proto_weight * retain_proto_loss
                + args.retain_kl_weight * kl
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()

            if step == 1 or step == args.train_steps or step % args.log_steps == 0:
                history.append(
                    {
                        "step": float(step),
                        "loss": float(loss.detach().cpu()),
                        "forget_proto_loss": float(forget_proto_loss.detach().cpu()),
                        "forget_proto_cosine": float(forget_cos.detach().cpu()),
                        "retain_proto_loss": float(retain_proto_loss.detach().cpu()),
                        "retain_proto_cosine": float(retain_cos.detach().cpu()),
                        "retain_kl": float(kl.detach().cpu()),
                        "forget_ce": float(forget_outputs.loss.detach().cpu()),
                        "retain_ce": float(retain_outputs.loss.detach().cpu()),
                    }
                )
    finally:
        remove_gradient_mask_hooks(handles)
        for param in model.parameters():
            param.requires_grad_(False)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype push-pull trainability probe.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--save-lora-dir", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--new-lora-r", type=int, default=0)
    parser.add_argument("--new-lora-alpha", type=int, default=0)
    parser.add_argument("--new-lora-dropout", type=float, default=None)
    parser.add_argument("--new-lora-target-modules", default="")
    parser.add_argument("--prototype-forget-limit", type=int, default=128)
    parser.add_argument("--prototype-retain-limit", type=int, default=128)
    parser.add_argument("--eval-forget-limit", type=int, default=80)
    parser.add_argument("--eval-retain-limit", type=int, default=200)
    parser.add_argument("--train-forget-limit", type=int, default=256)
    parser.add_argument("--train-retain-limit", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--prototype-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--forget-proto-weight", type=float, default=1.0)
    parser.add_argument("--retain-proto-weight", type=float, default=0.5)
    parser.add_argument("--retain-kl-weight", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    localization = load_pickle(Path(args.localization_pkl))
    parameter_masks = getattr(localization, "parameter_masks", {})
    target_neurons = getattr(localization, "target_neurons", {})
    if not parameter_masks or not target_neurons:
        raise ValueError(f"Localization lacks masks or target neurons: {args.localization_pkl}")
    metadata = dict(getattr(localization, "metadata", {}) or {})

    forget_proto_rows = normalize_qa_rows(Path(args.forget_path), limit=args.prototype_forget_limit, seed=args.seed, role="forget_proto")
    retain_proto_rows = normalize_qa_rows(Path(args.retain_path), limit=args.prototype_retain_limit, seed=args.seed + 1, role="retain_proto")
    forget_eval_rows = normalize_qa_rows(Path(args.forget_path), limit=args.eval_forget_limit, seed=args.seed + 2, role="forget_eval")
    retain_eval_rows = normalize_qa_rows(Path(args.retain_path), limit=args.eval_retain_limit, seed=args.seed + 3, role="retain_eval")
    forget_train_rows = normalize_qa_rows(Path(args.forget_path), limit=args.train_forget_limit, seed=args.seed + 4, role="forget_train")
    retain_train_rows = normalize_qa_rows(Path(args.retain_path), limit=args.train_retain_limit, seed=args.seed + 5, role="retain_train")

    tokenizer, model = load_model(args, metadata)
    for param in model.parameters():
        param.requires_grad_(False)
    capture = DownProjInputCapture(model, target_neurons)

    try:
        mask_summary = summarize_masks(parameter_masks)
        application = inspect_mask_application(model, parameter_masks)
        print(f"[Mask] {mask_summary}")
        print(f"[MaskApplication] {application}")
        print(f"[Capture] layers={sorted(capture.target_by_layer)} neurons={sum(len(v) for v in capture.target_by_layer.values())}")

        forget_proto = precompute_prototypes(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            rows=forget_proto_rows,
            batch_size=args.prototype_batch_size,
            max_length=args.max_length,
            label="forget",
        )
        retain_proto = precompute_prototypes(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            rows=retain_proto_rows,
            batch_size=args.prototype_batch_size,
            max_length=args.max_length,
            label="retain",
        )

        baseline = evaluate_all(
            model,
            tokenizer,
            capture,
            forget_eval_rows,
            retain_eval_rows,
            forget_proto,
            retain_proto,
            args,
            label="baseline",
        )
        history = train_probe(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            parameter_masks=parameter_masks,
            forget_rows=forget_train_rows,
            retain_rows=retain_train_rows,
            forget_proto=forget_proto,
            retain_proto=retain_proto,
            args=args,
        )
        after = evaluate_all(
            model,
            tokenizer,
            capture,
            forget_eval_rows,
            retain_eval_rows,
            forget_proto,
            retain_proto,
            args,
            label="after",
        )
        after["comparison"] = compare_result(after, baseline)
        after["prototype_comparison"] = {
            "forget_to_forget_proto_delta": float(
                after["forget_to_forget_proto"]["mean_cosine"] - baseline["forget_to_forget_proto"]["mean_cosine"]
            ),
            "retain_to_retain_proto_delta": float(
                after["retain_to_retain_proto"]["mean_cosine"] - baseline["retain_to_retain_proto"]["mean_cosine"]
            ),
        }

        if args.save_lora_dir:
            save_dir = Path(args.save_lora_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        report = {
            "method_name": "native_llm_lora_mask_prototype_trainability_probe",
            "base_model_path": args.base_model_path,
            "localization_pkl": args.localization_pkl,
            "config": vars(args),
            "mask": mask_summary,
            "mask_application": application,
            "prototype_summary": {
                "forget_layers": sorted(forget_proto),
                "retain_layers": sorted(retain_proto),
                "total_target_neurons": int(getattr(localization, "total_target_neurons", 0)),
            },
            "baseline": baseline,
            "history": history,
            "after": after,
            "localization_summary": {
                "strategy": getattr(localization, "strategy", ""),
                "metadata": metadata,
            },
        }
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.output_csv:
            output_csv = Path(args.output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with output_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["step"])
                writer.writeheader()
                for row in history:
                    writer.writerow(row)

        print("=" * 92)
        print("Native-LLM Fresh-LoRA Prototype Trainability Probe")
        print("=" * 92)
        print(
            f"forget_nll+={after['comparison']['forget_nll_increase']:.6f} "
            f"retain_nll+={after['comparison']['retain_nll_increase']:.6f} "
            f"gap={after['comparison']['impact_gap_forget_minus_retain']:.6f}"
        )
        print(
            "prototype cosine delta: "
            f"forget={after['prototype_comparison']['forget_to_forget_proto_delta']:.6f} "
            f"retain={after['prototype_comparison']['retain_to_retain_proto_delta']:.6f}"
        )
        print(f"saved: {output_json}")
    finally:
        capture.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
