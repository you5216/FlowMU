                      
                                                                         

                                                                             
                     

                                                             
                                                                    
                                                                        

                                           
                                                               
                                                               
   

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


AGENT_ROOT = Path(__file__).resolve().parents[4]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.diagnostics.native_llm_lora_mask_quality_probe import (              
    build_labeled_batch,
    compute_example_nll,
    load_json_or_jsonl,
)


DEFAULT_BASE_MODEL = (
    AGENT_ROOT / "final_performance/literaryQA/base/lora/merged/literary_neative_model"
)
DEFAULT_LOCALIZATION = (
    AGENT_ROOT
    / "final_performance/literaryQA/for10_re90/localization/"
    "literary_neative_model_cached_combined_top50_rr09_20260701/localized_neurons.pkl"
)
DEFAULT_CANDIDATE_STATS = (
    AGENT_ROOT
    / "final_performance/literaryQA/for10_re90/localization/"
    "literary_neative_model_rag_memory_final_combined_top300_rr08_20260701/"
    "candidate_neuron_stats.npz"
)
DEFAULT_FORGET = (
    AGENT_ROOT
    / "final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/"
    "graph_forget_qa.jsonl"
)
DEFAULT_RETAIN = (
    AGENT_ROOT
    / "final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/"
    "graph_retain_qa.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    AGENT_ROOT
    / "final_performance/literaryQA/for10_re90/small_exp/causal_perturbation_boxplot"
)
PLOT_SCRIPT = (
    AGENT_ROOT
    / "final_performance/tofu/small_exp/sensitive_neuron_localization/"
    "causal_perturbation_boxplot/plot_causal_perturbation_boxplot.py"
)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_rows(path: Path, *, limit: int, seed: int, role: str) -> list[dict[str, str]]:
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
        raise ValueError(f"No valid rows loaded from {path}")
    return rows


def parse_layer_key(value: str) -> int | None:
    value = str(value)
    if value.startswith("layer_"):
        value = value.split("_", 1)[1]
    try:
        return int(value)
    except ValueError:
        return None


def load_flowmu_neurons(path: Path) -> tuple[dict[int, list[int]], dict[str, Any]]:
    localization = load_pickle(path)
    raw = getattr(localization, "target_neurons", None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Localization has no target_neurons: {path}")
    selected: dict[int, list[int]] = {}
    for layer_key, values in raw.items():
        layer_idx = parse_layer_key(str(layer_key))
        if layer_idx is None:
            continue
        neurons = sorted({int(x) for x in values if int(x) >= 0})
        if neurons:
            selected[layer_idx] = neurons
    if not selected:
        raise ValueError(f"No valid neurons in {path}")
    metadata = dict(getattr(localization, "metadata", {}) or {})
    metadata.update(
        {
            "strategy": str(getattr(localization, "strategy", "")),
            "total_target_neurons": int(
                getattr(localization, "total_target_neurons", sum(len(v) for v in selected.values()))
            ),
        }
    )
    return selected, metadata


def resolve_decoder_layers(model):
    for path in (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
    ):
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


def infer_neuron_dim(model, selected: dict[int, list[int]]) -> int:
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
        if weight is not None and len(weight.shape) >= 2:
            return int(weight.shape[1])
    return max(max(v) for v in selected.values() if v) + 1


def build_random_like(selected: dict[int, list[int]], *, neuron_dim: int, seed: int) -> dict[int, list[int]]:
    rng = random.Random(seed)
    indices = list(range(neuron_dim))
    out: dict[int, list[int]] = {}
    for layer_idx, neurons in selected.items():
        out[layer_idx] = sorted(rng.sample(indices, min(len(neurons), neuron_dim)))
    return out


def _candidate_key(npz: np.lib.npyio.NpzFile, layer: int, suffix: str) -> str | None:
    candidates = [
        f"layer_{layer}__{suffix}",
        f"layer{layer}__{suffix}",
        f"{layer}__{suffix}",
    ]
    for key in candidates:
        if key in npz.files:
            return key
    return None


def build_activation_like(
    candidate_stats_path: Path,
    selected: dict[int, list[int]],
    *,
    score_field: str,
    neuron_dim: int,
) -> dict[int, list[int]]:
    npz = np.load(candidate_stats_path)
    out: dict[int, list[int]] = {}
    for layer_idx, neurons in selected.items():
        key = _candidate_key(npz, layer_idx, score_field)
        if key is None:
            fallback = _candidate_key(npz, layer_idx, "combined_forget_mean")
            if fallback is None:
                raise KeyError(
                    f"Missing activation score for layer {layer_idx}: {score_field} or combined_forget_mean"
                )
            key = fallback
        arr = np.asarray(npz[key], dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D activation score array for {key}, got shape={arr.shape}")
        valid = np.arange(min(arr.size, neuron_dim))
        scores = arr[valid]
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        count = min(len(neurons), scores.size)
        chosen = valid[np.argsort(-scores)[:count]]
        out[layer_idx] = sorted(int(x) for x in chosen.tolist())
    return out


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


@contextmanager
def ablate_neurons(
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

    def make_hook(neuron_indices: list[int]):
        index_cache: dict[torch.device, torch.Tensor] = {}

        def hook(_module, inputs):
            if not inputs:
                return inputs
            value = inputs[0]
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                return inputs
            idx = index_cache.get(value.device)
            if idx is None:
                valid = [i for i in neuron_indices if 0 <= i < value.shape[-1]]
                if not valid:
                    return inputs
                idx = torch.as_tensor(valid, device=value.device, dtype=torch.long)
                index_cache[value.device] = idx
            edited = value.clone()
            if mode == "zero":
                edited.index_fill_(-1, idx, 0.0)
            elif mode == "scale":
                selected_values = edited.index_select(-1, idx) * float(scale_factor)
                edited.index_copy_(-1, idx, selected_values)
            elif mode == "noise":
                selected_values = edited.index_select(-1, idx)
                noise = torch.randn(selected_values.shape, generator=generator, dtype=torch.float32)
                noise = noise.to(device=value.device, dtype=edited.dtype)
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
            handles.append(down_proj.register_forward_pre_hook(make_hook(selected[layer_idx])))
    if not handles:
        raise ValueError("No down_proj hooks registered.")
    try:
        yield {"registered_hooks": len(handles)}
    finally:
        for handle in handles:
            handle.remove()


def evaluate_nll_per_sample(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    *,
    batch_size: int,
    max_length: int,
    label: str,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    out: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc=label, leave=False):
        batch_rows = rows[start : start + batch_size]
        batch = build_labeled_batch(tokenizer, batch_rows, max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        values = compute_example_nll(outputs.logits, batch["labels"]).detach().cpu().tolist()
        for row, value in zip(batch_rows, values):
            out.append(
                {
                    "sample_id": row["example_id"],
                    "question": row["instruction"],
                    "nll": float(value),
                }
            )
    return out


def evaluate_perturbed(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    *,
    selected: dict[int, list[int]],
    batch_size: int,
    max_length: int,
    label: str,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    with ablate_neurons(
        model,
        selected,
        mode=args.ablation_mode,
        scale_factor=args.scale_factor,
        noise_std=args.noise_std,
        seed=seed,
    ):
        return evaluate_nll_per_sample(
            model,
            tokenizer,
            rows,
            batch_size=batch_size,
            max_length=max_length,
            label=label,
        )


def summarize_selection(name: str, selected: dict[int, list[int]], *, neuron_dim: int) -> dict[str, Any]:
    total = sum(len(v) for v in selected.values())
    return {
        "name": name,
        "num_layers": len(selected),
        "total_neurons": int(total),
        "mean_per_layer": float(total / max(1, len(selected))),
        "neuron_dim": int(neuron_dim),
        "density": float(total / max(1, len(selected) * neuron_dim)),
        "per_layer_counts": {f"layer_{k}": len(v) for k, v in sorted(selected.items())},
    }


def write_records(
    csv_path: Path,
    *,
    baseline_by_panel: dict[str, list[dict[str, Any]]],
    perturbed_by_method_panel: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline_maps = {
        panel: {row["sample_id"]: row for row in rows} for panel, rows in baseline_by_panel.items()
    }
    for (method, panel), perturbed_rows in perturbed_by_method_panel.items():
        base_map = baseline_maps[panel]
        for row in perturbed_rows:
            base = base_map[row["sample_id"]]
            nll_shift = float(row["nll"]) - float(base["nll"])
            records.append(
                {
                    "method": method,
                    "panel": panel,
                    "sample_id": row["sample_id"],
                    "baseline_nll": f"{float(base['nll']):.8f}",
                    "perturbed_nll": f"{float(row['nll']):.8f}",
                    "nll_shift": f"{nll_shift:.8f}",
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "panel",
                "sample_id",
                "baseline_nll",
                "perturbed_nll",
                "nll_shift",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in sorted({row["method"] for row in records}):
        for panel in ("forget", "retain"):
            values = np.asarray(
                [float(row["nll_shift"]) for row in records if row["method"] == method and row["panel"] == panel],
                dtype=np.float32,
            )
            if values.size == 0:
                continue
            summary[f"{method}_{panel}"] = {
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
    return summary


def maybe_plot(args: argparse.Namespace, csv_path: Path) -> None:
    if args.no_plot:
        return
    cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--input-csv",
        str(csv_path),
        "--metric-column",
        "nll_shift",
        "--metric-label",
        "NLL Shift",
        "--method-order",
        "Random,Activation,FlowMU",
        "--output-dir",
        str(args.output_dir),
        "--stem",
        args.stem,
    ]
    if args.ylim:
        cmd.append(f"--ylim={args.ylim}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteraryQA for10 causal perturbation per-sample probe.")
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--localization-pkl", type=Path, default=DEFAULT_LOCALIZATION)
    parser.add_argument("--candidate-stats", type=Path, default=DEFAULT_CANDIDATE_STATS)
    parser.add_argument("--forget-path", type=Path, default=DEFAULT_FORGET)
    parser.add_argument("--retain-path", type=Path, default=DEFAULT_RETAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="literaryqa_for10_causal_perturbation_boxplot")
    parser.add_argument("--forget-limit", type=int, default=300)
    parser.add_argument("--retain-limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--ablation-mode", choices=["zero", "scale", "noise"], default="zero")
    parser.add_argument("--scale-factor", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--activation-score-field", default="combined_forget_mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ylim", default="")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flowmu, localization_metadata = load_flowmu_neurons(args.localization_pkl)
    forget_rows = normalize_rows(args.forget_path, limit=args.forget_limit, seed=args.seed, role="forget")
    retain_rows = normalize_rows(args.retain_path, limit=args.retain_limit, seed=args.seed + 1, role="retain")

    tokenizer, model = load_model(args)
    neuron_dim = infer_neuron_dim(model, flowmu)
    random_like = build_random_like(flowmu, neuron_dim=neuron_dim, seed=args.seed + 1000)
    activation_like = build_activation_like(
        args.candidate_stats,
        flowmu,
        score_field=args.activation_score_field,
        neuron_dim=neuron_dim,
    )
    selections = {
        "Random": random_like,
        "Activation": activation_like,
        "FlowMU": flowmu,
    }

    print("[Selection]", json.dumps(summarize_selection("FlowMU", flowmu, neuron_dim=neuron_dim), indent=2))
    print("[Rows]", {"forget": len(forget_rows), "retain": len(retain_rows)})

    baseline_by_panel = {
        "forget": evaluate_nll_per_sample(
            model,
            tokenizer,
            forget_rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            label="baseline-forget",
        ),
        "retain": evaluate_nll_per_sample(
            model,
            tokenizer,
            retain_rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            label="baseline-retain",
        ),
    }
    perturbed_by_method_panel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method, selected in selections.items():
        for panel, rows in (("forget", forget_rows), ("retain", retain_rows)):
            perturbed_by_method_panel[(method, panel)] = evaluate_perturbed(
                model,
                tokenizer,
                rows,
                selected=selected,
                batch_size=args.batch_size,
                max_length=args.max_length,
                label=f"{method}-{panel}",
                args=args,
                seed=args.seed + hash((method, panel)) % 10000,
            )

    csv_path = args.output_dir / f"{args.stem}_records.csv"
    records = write_records(
        csv_path,
        baseline_by_panel=baseline_by_panel,
        perturbed_by_method_panel=perturbed_by_method_panel,
    )
    summary = {
        "method_name": "literaryqa_for10_causal_perturbation_probe",
        "base_model_path": str(args.base_model_path),
        "localization_pkl": str(args.localization_pkl),
        "candidate_stats": str(args.candidate_stats),
        "forget_path": str(args.forget_path),
        "retain_path": str(args.retain_path),
        "config": {
            "forget_limit": args.forget_limit,
            "retain_limit": args.retain_limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "ablation_mode": args.ablation_mode,
            "activation_score_field": args.activation_score_field,
            "seed": args.seed,
            "no_4bit": args.no_4bit,
        },
        "localization_metadata": localization_metadata,
        "selections": {
            name: summarize_selection(name, selected, neuron_dim=neuron_dim)
            for name, selected in selections.items()
        },
        "metrics": summarize_records(records),
    }
    summary_path = args.output_dir / f"{args.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_plot(args, csv_path)
    print(f"[Saved] records={csv_path}")
    print(f"[Saved] summary={summary_path}")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
