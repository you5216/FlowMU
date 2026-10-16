                      
                                                             

                                                                              
                                                                       
                            
   

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.diagnostics.native_llm_lora_mask_ga_step_probe import (              
    build_labeled_batch,
    build_model,
    collect_retain_reference_logits,
    evaluate_nll,
    evaluate_retain_kl,
    inspect_mask_application,
    normalize_qa_rows,
)
from phase2.diagnostics.native_llm_lora_mask_quality_probe import build_structured_random_masks              


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def layer_index_from_name(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def matched_masks(model, parameter_masks: dict[str, Any]) -> dict[str, np.ndarray]:
    named_params = dict(model.named_parameters())
    out: dict[str, np.ndarray] = {}
    for name, mask_np in parameter_masks.items():
        param = named_params.get(name)
        mask = np.asarray(mask_np, dtype=np.float32)
        if param is not None and tuple(mask.shape) == tuple(param.shape) and int(mask.sum()) > 0:
            out[name] = mask
    return out


def collect_masked_gradients(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    parameter_masks: dict[str, np.ndarray],
    *,
    batch_size: int,
    max_length: int,
    label: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    named_params = dict(model.named_parameters())
    grad_sums: dict[str, torch.Tensor] = {}
    losses: list[float] = []
    steps = 0
    model.train()
    for start in tqdm(range(0, len(rows), batch_size), desc=label, leave=False):
        batch = build_labeled_batch(tokenizer, rows[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().float().cpu()))
        outputs.loss.backward()
        for name, mask_np in parameter_masks.items():
            param = named_params.get(name)
            if param is None or param.grad is None:
                continue
            mask = torch.as_tensor(mask_np, dtype=torch.float32, device=param.grad.device)
            grad = (param.grad.detach().float() * mask).cpu()
            grad_sums[name] = grad.clone() if name not in grad_sums else grad_sums[name] + grad
        steps += 1
    model.zero_grad(set_to_none=True)
    avg = {name: grad / max(1, steps) for name, grad in grad_sums.items()}
    return {
        "gradients": avg,
        "num_batches": steps,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


def pair_stats(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    common_tensors = 0
    common_params = 0
    for name, ga in a.items():
        gb = b.get(name)
        if gb is None:
            continue
        av = ga.float().reshape(-1)
        bv = gb.float().reshape(-1)
        dot += float(torch.dot(av, bv).item())
        norm_a_sq += float(torch.dot(av, av).item())
        norm_b_sq += float(torch.dot(bv, bv).item())
        common_tensors += 1
        common_params += int((av != 0).sum().item())
    norm_a = norm_a_sq**0.5
    norm_b = norm_b_sq**0.5
    cosine = dot / max(1e-12, norm_a * norm_b)
    projection_on_a = dot / max(1e-12, norm_a_sq)
    projection_on_b = dot / max(1e-12, norm_b_sq)
    return {
        "common_tensors": float(common_tensors),
        "common_nonzero_params": float(common_params),
        "dot": float(dot),
        "norm_a": float(norm_a),
        "norm_b": float(norm_b),
        "cosine": float(cosine),
        "projection_b_on_a": float(projection_on_a),
        "projection_a_on_b": float(projection_on_b),
    }


def per_layer_pair_stats(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    by_layer_a: dict[int, dict[str, torch.Tensor]] = {}
    by_layer_b: dict[int, dict[str, torch.Tensor]] = {}
    for name, grad in a.items():
        layer = layer_index_from_name(name)
        if layer is not None:
            by_layer_a.setdefault(layer, {})[name] = grad
    for name, grad in b.items():
        layer = layer_index_from_name(name)
        if layer is not None:
            by_layer_b.setdefault(layer, {})[name] = grad
    rows: list[dict[str, float]] = []
    for layer in sorted(set(by_layer_a) | set(by_layer_b)):
        stats = pair_stats(by_layer_a.get(layer, {}), by_layer_b.get(layer, {}))
        stats["layer"] = float(layer)
        rows.append(stats)
    return rows


def backup_params(model, parameter_masks: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    named_params = dict(model.named_parameters())
    backups: dict[str, torch.Tensor] = {}
    for name in parameter_masks:
        param = named_params.get(name)
        if param is not None:
            backups[name] = param.data.detach().clone()
    return backups


def restore_params(model, backups: dict[str, torch.Tensor]) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in backups.items():
            param = named_params.get(name)
            if param is not None:
                param.data.copy_(value)


def apply_precomputed_ga_step(
    model,
    gradients: dict[str, torch.Tensor],
    parameter_masks: dict[str, np.ndarray],
    *,
    step_size: float,
) -> dict[str, float]:
    named_params = dict(model.named_parameters())
    applied_norm = 0.0
    applied_tensors = 0
    with torch.no_grad():
        for name, mask_np in parameter_masks.items():
            param = named_params.get(name)
            grad = gradients.get(name)
            if param is None or grad is None:
                continue
            mask = torch.as_tensor(mask_np, dtype=torch.float32)
            masked_grad = grad.float() * mask
            applied_norm += float(masked_grad.norm().item())
            param.data.add_(masked_grad.to(device=param.device, dtype=param.dtype) * float(step_size))
            applied_tensors += 1
    return {"applied_tensors": float(applied_tensors), "applied_grad_norm_sum": float(applied_norm)}


def evaluate_ga_effect_from_gradients(
    model,
    tokenizer,
    gradients: dict[str, torch.Tensor],
    parameter_masks: dict[str, np.ndarray],
    *,
    forget_rows: list[dict[str, str]],
    retain_rows: list[dict[str, str]],
    retain_refs: list[dict[str, torch.Tensor]],
    baseline_forget: dict[str, Any],
    baseline_retain: dict[str, Any],
    batch_size: int,
    max_length: int,
    step_size: float,
    label: str,
) -> dict[str, Any]:
    backups = backup_params(model, parameter_masks)
    try:
        step_info = apply_precomputed_ga_step(model, gradients, parameter_masks, step_size=step_size)
        target_forget = evaluate_nll(
            model,
            tokenizer,
            forget_rows,
            batch_size=batch_size,
            max_length=max_length,
            label=f"{label}-forget",
        )
        target_retain = evaluate_nll(
            model,
            tokenizer,
            retain_rows,
            batch_size=batch_size,
            max_length=max_length,
            label=f"{label}-retain",
        )
        target_retain_kl = evaluate_retain_kl(model, retain_refs, label=f"{label}-retain-kl")
    finally:
        restore_params(model, backups)
    forget_inc = target_forget["mean_nll"] - baseline_forget["mean_nll"]
    retain_inc = target_retain["mean_nll"] - baseline_retain["mean_nll"]
    return {
        "step": step_info,
        "target": {
            "forget": target_forget,
            "retain": target_retain,
            "retain_kl": target_retain_kl,
        },
        "comparison": {
            "forget_nll_increase": float(forget_inc),
            "retain_nll_increase": float(retain_inc),
            "retain_kl_increase": float(target_retain_kl["mean_kl"]),
            "impact_gap_forget_minus_retain": float(forget_inc - retain_inc),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose masked gradient overlap for native-LLM fresh-LoRA localization.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--graph-forget-path", required=True)
    parser.add_argument("--graph-retain-path", required=True)
    parser.add_argument("--canonical-forget-path", default="")
    parser.add_argument("--canonical-retain-path", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-layer-csv", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--new-lora-r", type=int, default=0)
    parser.add_argument("--new-lora-alpha", type=int, default=0)
    parser.add_argument("--new-lora-dropout", type=float, default=None)
    parser.add_argument("--new-lora-target-modules", default="")
    parser.add_argument("--grad-forget-limit", type=int, default=64)
    parser.add_argument("--grad-retain-limit", type=int, default=64)
    parser.add_argument("--grad-canonical-forget-limit", type=int, default=64)
    parser.add_argument("--eval-forget-limit", type=int, default=80)
    parser.add_argument("--eval-retain-limit", type=int, default=200)
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

    graph_forget_grad_rows = normalize_qa_rows(
        Path(args.graph_forget_path), limit=args.grad_forget_limit, seed=args.seed, role="graph_forget_grad"
    )
    graph_retain_grad_rows = normalize_qa_rows(
        Path(args.graph_retain_path), limit=args.grad_retain_limit, seed=args.seed + 1, role="graph_retain_grad"
    )
    eval_forget_rows = normalize_qa_rows(
        Path(args.graph_forget_path), limit=args.eval_forget_limit, seed=args.seed + 2, role="graph_forget_eval"
    )
    eval_retain_rows = normalize_qa_rows(
        Path(args.graph_retain_path), limit=args.eval_retain_limit, seed=args.seed + 3, role="graph_retain_eval"
    )
    canonical_forget_grad_rows = []
    if args.canonical_forget_path:
        canonical_forget_grad_rows = normalize_qa_rows(
            Path(args.canonical_forget_path),
            limit=args.grad_canonical_forget_limit,
            seed=args.seed + 4,
            role="canonical_forget_grad",
        )

    tokenizer, model = build_model(args, metadata)
    target_masks = matched_masks(model, parameter_masks)
    if not target_masks:
        raise ValueError("No localization mask tensors matched the fresh LoRA model.")
    random_masks = build_structured_random_masks(model, target_masks, seed=args.seed + 1000)
    random_masks = matched_masks(model, random_masks)
    target_application = inspect_mask_application(model, target_masks)
    random_application = inspect_mask_application(model, random_masks)
    print(f"[TargetMask] {target_application}")
    print(f"[RandomMask] {random_application}")

    baseline_forget = evaluate_nll(
        model,
        tokenizer,
        eval_forget_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="baseline-forget",
    )
    baseline_retain = evaluate_nll(
        model,
        tokenizer,
        eval_retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="baseline-retain",
    )
    retain_refs = collect_retain_reference_logits(
        model,
        tokenizer,
        eval_retain_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    graph_forget_grad = collect_masked_gradients(
        model,
        tokenizer,
        graph_forget_grad_rows,
        target_masks,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="grad-graph-forget",
    )
    random_graph_forget_grad = collect_masked_gradients(
        model,
        tokenizer,
        graph_forget_grad_rows,
        random_masks,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="grad-random-graph-forget",
    )
    graph_retain_grad = collect_masked_gradients(
        model,
        tokenizer,
        graph_retain_grad_rows,
        target_masks,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="grad-graph-retain",
    )

    canonical_forget_grad = None
    if canonical_forget_grad_rows:
        canonical_forget_grad = collect_masked_gradients(
            model,
            tokenizer,
            canonical_forget_grad_rows,
            target_masks,
            batch_size=args.batch_size,
            max_length=args.max_length,
            label="grad-canonical-forget",
        )

    target_effect = evaluate_ga_effect_from_gradients(
        model,
        tokenizer,
        graph_forget_grad["gradients"],
        target_masks,
        forget_rows=eval_forget_rows,
        retain_rows=eval_retain_rows,
        retain_refs=retain_refs,
        baseline_forget=baseline_forget,
        baseline_retain=baseline_retain,
        batch_size=args.batch_size,
        max_length=args.max_length,
        step_size=args.step_size,
        label="target-ga",
    )
    random_effect = evaluate_ga_effect_from_gradients(
        model,
        tokenizer,
        random_graph_forget_grad["gradients"],
        random_masks,
        forget_rows=eval_forget_rows,
        retain_rows=eval_retain_rows,
        retain_refs=retain_refs,
        baseline_forget=baseline_forget,
        baseline_retain=baseline_retain,
        batch_size=args.batch_size,
        max_length=args.max_length,
        step_size=args.step_size,
        label="random-ga",
    )

    graph_forget_vs_retain = pair_stats(graph_forget_grad["gradients"], graph_retain_grad["gradients"])
    per_layer = per_layer_pair_stats(graph_forget_grad["gradients"], graph_retain_grad["gradients"])
    graph_vs_canonical = None
    if canonical_forget_grad is not None:
        graph_vs_canonical = pair_stats(graph_forget_grad["gradients"], canonical_forget_grad["gradients"])

    report = {
        "method_name": "native_llm_lora_mask_overlap_diagnostics",
        "config": vars(args),
        "localization_summary": {
            "strategy": getattr(localization, "strategy", ""),
            "total_target_neurons": int(getattr(localization, "total_target_neurons", 0)),
            "metadata": metadata,
        },
        "mask_application": {
            "target": target_application,
            "same_structure_random": random_application,
        },
        "gradient_batches": {
            "graph_forget": graph_forget_grad["num_batches"],
            "random_graph_forget": random_graph_forget_grad["num_batches"],
            "graph_retain": graph_retain_grad["num_batches"],
            "canonical_forget": canonical_forget_grad["num_batches"] if canonical_forget_grad is not None else 0,
        },
        "gradient_mean_loss": {
            "graph_forget": graph_forget_grad["mean_loss"],
            "random_graph_forget": random_graph_forget_grad["mean_loss"],
            "graph_retain": graph_retain_grad["mean_loss"],
            "canonical_forget": canonical_forget_grad["mean_loss"] if canonical_forget_grad is not None else None,
        },
        "gradient_overlap": {
            "graph_forget_vs_graph_retain": graph_forget_vs_retain,
            "graph_forget_vs_canonical_forget": graph_vs_canonical,
        },
        "baseline": {"forget": baseline_forget, "retain": baseline_retain},
        "ga_step_effect": {
            "target_mask": target_effect,
            "same_structure_random_mask": random_effect,
        },
        "diagnosis": {
            "target_minus_random_gap": float(
                target_effect["comparison"]["impact_gap_forget_minus_retain"]
                - random_effect["comparison"]["impact_gap_forget_minus_retain"]
            ),
            "retain_projection_on_forget": graph_forget_vs_retain["projection_b_on_a"],
            "retain_forget_cosine": graph_forget_vs_retain["cosine"],
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_layer_csv:
        output_layer_csv = Path(args.output_layer_csv)
        output_layer_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_layer_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "layer",
                "common_tensors",
                "common_nonzero_params",
                "dot",
                "norm_a",
                "norm_b",
                "cosine",
                "projection_b_on_a",
                "projection_a_on_b",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_layer:
                writer.writerow(row)

    print("=" * 92)
    print("Native-LLM LoRA Mask Overlap Diagnostics")
    print("=" * 92)
    print(
        "graph_forget_vs_retain "
        f"cos={graph_forget_vs_retain['cosine']:.6f} "
        f"proj_retain_on_forget={graph_forget_vs_retain['projection_b_on_a']:.6f}"
    )
    if graph_vs_canonical is not None:
        print(
            "graph_forget_vs_canonical_forget "
            f"cos={graph_vs_canonical['cosine']:.6f} "
            f"proj_canonical_on_graph={graph_vs_canonical['projection_b_on_a']:.6f}"
        )
    print(
        "target_ga "
        f"forget+={target_effect['comparison']['forget_nll_increase']:.6f} "
        f"retain+={target_effect['comparison']['retain_nll_increase']:.6f} "
        f"kl+={target_effect['comparison']['retain_kl_increase']:.6f} "
        f"gap={target_effect['comparison']['impact_gap_forget_minus_retain']:.6f}"
    )
    print(
        "random_ga "
        f"forget+={random_effect['comparison']['forget_nll_increase']:.6f} "
        f"retain+={random_effect['comparison']['retain_nll_increase']:.6f} "
        f"kl+={random_effect['comparison']['retain_kl_increase']:.6f} "
        f"gap={random_effect['comparison']['impact_gap_forget_minus_retain']:.6f}"
    )
    print(f"saved: {output_json}")
    if args.output_layer_csv:
        print(f"layer_csv: {args.output_layer_csv}")


if __name__ == "__main__":
    main()
