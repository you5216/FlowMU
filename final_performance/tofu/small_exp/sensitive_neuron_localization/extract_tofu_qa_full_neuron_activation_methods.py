                      
                                                                       

                                            

             
              

           
            

                                                                              
                                                                             
                                                        
   

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "final_performance/tofu/small_exp/sensitive_neuron_localization"
DEFAULT_FORGET = Path("/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget05.json")
DEFAULT_RETAIN = Path("/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain95.json")
DEFAULT_METHODS = [
    (
        "Vanilla",
        ROOT / "merge_model/native_llm",
    ),
    (
        "PURE",
        Path("/disk8T/lkh/python_projects/baseline_xzy/experiments/merged/pure_for05"),
    ),
    (
        "SBU",
        Path("/disk8T/lkh/python_projects/baseline_xzy/experiments/merged/sbu_for05"),
    ),
    (
        "FlowMU",
        ROOT
        / "final_performance/tofu/for05_re95/phase2_merged/"
        "ga_kl_strongmid_lambda03top100_e1_bs2_lr12e5_kl002_20260622",
    ),
]
LAYER_LABELS = [str(i) for i in range(1, 33)]


def parse_method_specs(specs: list[str] | None) -> list[tuple[str, Path]]:
    if not specs:
        return DEFAULT_METHODS
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Method spec must be NAME=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise ValueError(f"Empty method name in {spec!r}")
        parsed.append((name, path))
    return parsed


def load_qa_jsonl(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not question or not answer:
                continue
            rows.append({"question": question, "answer": answer})
            if limit >= 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No Q+A rows loaded from {path}")
    return rows


def build_qa_prompt(question: str, answer: str) -> tuple[str, tuple[int, int]]:
    prefix = f"Question:\n{question}\n\nAnswer:\n"
    prompt = prefix + answer
    return prompt, (len(prefix), len(prompt))


def resolve_decoder_layers(model: Any) -> list[Any]:
    current = model
    if hasattr(current, "model"):
        current = current.model
    if hasattr(current, "language_model"):
        current = current.language_model
    if hasattr(current, "model"):
        current = current.model
    layers = getattr(current, "layers", None)
    if layers is None and hasattr(current, "decoder"):
        layers = getattr(current.decoder, "layers", None)
    if layers is None:
        raise AttributeError("Cannot resolve decoder layers from model.")
    return list(layers)


@contextmanager
def capture_mlp_inputs(model: Any):
    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, input_args, _output):
            if input_args and isinstance(input_args[0], torch.Tensor):
                activations[layer_idx] = input_args[0].detach()

        return hook

    for layer_idx, layer in enumerate(resolve_decoder_layers(model)):
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is None:
            raise AttributeError(f"Layer {layer_idx} has no mlp.down_proj.")
        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))
    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def select_answer_tokens(
    attention_mask: torch.Tensor,
    offset_mapping: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    selected = torch.zeros_like(attention_mask, dtype=torch.bool)
    offsets = offset_mapping.detach().cpu().tolist()
    for row_idx, (span_start, span_end) in enumerate(spans):
        for token_idx, (tok_start, tok_end) in enumerate(offsets[row_idx]):
            if attention_mask[row_idx, token_idx].item() == 0:
                continue
            if tok_end <= tok_start:
                continue
            if tok_end > span_start and tok_start < span_end:
                selected[row_idx, token_idx] = True
    if not selected.any():
        raise ValueError("Answer-token selection selected zero tokens.")
    return selected


def aggregate_qa_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    batch_size: int,
    max_length: int,
    desc: str,
) -> tuple[np.ndarray, int, int, float]:
    layers = list(range(len(resolve_decoder_layers(model))))
    sums = np.zeros(len(layers), dtype=np.float64)
    counts = np.zeros(len(layers), dtype=np.float64)
    prompts: list[str] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        prompt, span = build_qa_prompt(row["question"], row["answer"])
        prompts.append(prompt)
        spans.append(span)

    total_answer_tokens = 0
    nll_sum = 0.0
    nll_count = 0
    device = next(model.parameters()).device
    with capture_mlp_inputs(model) as captured:
        for start in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offset_mapping = encoded.pop("offset_mapping")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            selected_tokens = select_answer_tokens(
                encoded["attention_mask"],
                offset_mapping,
                spans[start : start + len(batch_prompts)],
            )
            answer_token_count = int(selected_tokens.sum().item())
            if answer_token_count <= 0:
                raise RuntimeError("No answer tokens in batch.")
            total_answer_tokens += answer_token_count

            captured.clear()
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, return_dict=True)

            input_ids = encoded["input_ids"]
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = selected_tokens.to(device)[:, 1:] & encoded["attention_mask"][:, 1:].bool()
            token_nll = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
            )
            nll_sum += float(token_nll.masked_select(shift_mask).sum().detach().cpu().item())
            nll_count += int(shift_mask.sum().detach().cpu().item())

            mask = selected_tokens.to(device).unsqueeze(-1)
            for layer_idx in layers:
                act = captured.get(layer_idx)
                if act is None:
                    raise RuntimeError(f"No activation captured for layer {layer_idx}.")
                hidden_size = int(act.shape[-1])
                layer_sum = (act.abs().float() * mask).sum().detach().cpu().item()
                sums[layer_idx] += float(layer_sum)
                counts[layer_idx] += float(answer_token_count * hidden_size)
            captured.clear()

    if np.any(counts <= 0):
        bad_layers = [idx for idx, count in enumerate(counts) if count <= 0]
        raise RuntimeError(f"No activation counts for layers: {bad_layers}")
    if nll_count <= 0:
        raise RuntimeError("No answer-token NLL counts were collected.")
    return (sums / counts).astype(np.float32), len(rows), total_answer_tokens, float(nll_sum / nll_count)


def validate_model_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in model path: {path}")


def load_method_model(
    model_path: Path,
    *,
    device: str,
    dtype: str,
) -> tuple[Any, Any]:
    validate_model_path(model_path)
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        device_map: str | dict[str, int] | None = {"": 0}
    elif device == "auto":
        device_map = "auto"
    else:
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_display_matrices(
    forget_raw: np.ndarray,
    retain_raw: np.ndarray,
    *,
    mode: str,
    low_q: float,
    high_q: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    if mode == "global_log_robust":
        forget_base = np.log1p(forget_raw)
        retain_base = np.log1p(retain_raw)
        values = np.concatenate([forget_base.reshape(-1), retain_base.reshape(-1)])
        lo, hi = np.quantile(values, [low_q, high_q])
        if np.isclose(lo, hi):
            raise ValueError("Display scale collapsed after log1p.")
        forget = (np.clip(forget_base, lo, hi) - lo) / (hi - lo)
        retain = (np.clip(retain_base, lo, hi) - lo) / (hi - lo)
        return (
            forget.astype(np.float32),
            retain.astype(np.float32),
            {
                "display_mode": mode,
                "display_low_quantile": float(low_q),
                "display_high_quantile": float(high_q),
                "display_clip_low_log1p": float(lo),
                "display_clip_high_log1p": float(hi),
            },
        )

    if mode == "method_log_minmax":
        forget_base = np.log1p(forget_raw)
        retain_base = np.log1p(retain_raw)
        forget = np.zeros_like(forget_base, dtype=np.float32)
        retain = np.zeros_like(retain_base, dtype=np.float32)
        for method_idx in range(forget_base.shape[1]):
            values = np.concatenate([forget_base[:, method_idx], retain_base[:, method_idx]])
            lo, hi = np.quantile(values, [low_q, high_q])
            if np.isclose(lo, hi):
                raise ValueError(f"Display scale collapsed for method index {method_idx}.")
            forget[:, method_idx] = (np.clip(forget_base[:, method_idx], lo, hi) - lo) / (hi - lo)
            retain[:, method_idx] = (np.clip(retain_base[:, method_idx], lo, hi) - lo) / (hi - lo)
        return (
            forget.astype(np.float32),
            retain.astype(np.float32),
            {
                "display_mode": mode,
                "display_low_quantile": float(low_q),
                "display_high_quantile": float(high_q),
                "display_note": "Each method is log1p-scaled and min-max normalized over its forget+retain layers.",
            },
        )

    raise ValueError(f"Unknown display mode: {mode}")


def configure_matplotlib() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_activation_map(
    matrix: np.ndarray,
    *,
    methods: list[str],
    title: str,
    output_path: Path,
    cbar_label: str,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 8.0))
    y_tick_positions = np.arange(0.5, 32.5, 5.0)
    y_tick_labels = [LAYER_LABELS[i] for i in range(0, 32, 5)]
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        cbar_kws={"label": cbar_label},
        xticklabels=methods,
        yticklabels=False,
        linewidths=0.25,
        linecolor="white",
        square=False,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10, length=0)
    ax.tick_params(axis="y", labelsize=10, length=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    path: Path,
    *,
    dataset_split: str,
    methods: list[str],
    forget_raw: np.ndarray,
    retain_raw: np.ndarray,
    forget_display: np.ndarray,
    retain_display: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_split",
                "panel",
                "method",
                "layer",
                "raw_activation_mean",
                "display_score",
            ],
        )
        writer.writeheader()
        for panel, raw_matrix, display_matrix in [
            ("qa_answer_forget", forget_raw, forget_display),
            ("qa_answer_retain", retain_raw, retain_display),
        ]:
            for layer_idx in range(32):
                for method_idx, method in enumerate(methods):
                    writer.writerow(
                        {
                            "dataset_split": dataset_split,
                            "panel": panel,
                            "method": method,
                            "layer": layer_idx + 1,
                            "raw_activation_mean": f"{float(raw_matrix[layer_idx, method_idx]):.10f}",
                            "display_score": f"{float(display_matrix[layer_idx, method_idx]):.10f}",
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TOFU Q+A answer-token full-neuron activation across merged models."
    )
    parser.add_argument("--method", action="append", default=None, help="Method spec NAME=MODEL_PATH. May repeat.")
    parser.add_argument("--forget-path", type=Path, default=DEFAULT_FORGET)
    parser.add_argument("--retain-path", type=Path, default=DEFAULT_RETAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="activation_qa_full_neuron_methods_tofu_for05_20260705")
    parser.add_argument("--dataset-split", default="tofu_for05_re95")
    parser.add_argument("--forget-limit", type=int, default=200)
    parser.add_argument("--retain-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--display-mode",
        choices=["global_log_robust", "method_log_minmax"],
        default="global_log_robust",
    )
    parser.add_argument("--clip-low-q", type=float, default=0.02)
    parser.add_argument("--clip-high-q", type=float, default=0.98)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_method_specs(args.method)
    method_names = [name for name, _path in methods]
    for _name, path in methods:
        validate_model_path(path)

    forget_rows = load_qa_jsonl(args.forget_path, args.forget_limit)
    retain_rows = load_qa_jsonl(args.retain_path, args.retain_limit)
    sample_prompt, sample_span = build_qa_prompt(forget_rows[0]["question"], forget_rows[0]["answer"])

    if args.dry_run:
        print("Dry run OK")
        print(f"methods={method_names}")
        for name, path in methods:
            print(f"  {name}: {path}")
        print(f"forget_rows={len(forget_rows)} retain_rows={len(retain_rows)}")
        print(f"sample_prompt={sample_prompt[:260]!r}")
        print(f"sample_answer_span={sample_span}")
        return

    forget_columns: list[np.ndarray] = []
    retain_columns: list[np.ndarray] = []
    method_reports: list[dict[str, Any]] = []

    for method_name, model_path in methods:
        print(f"\n=== Loading {method_name}: {model_path} ===", flush=True)
        tokenizer, model = load_method_model(model_path, device=args.device, dtype=args.dtype)
        n_layers = len(resolve_decoder_layers(model))
        if n_layers != 32:
            raise ValueError(f"{method_name} has {n_layers} layers; expected 32.")

        forget_vec, forget_count, forget_answer_tokens, forget_nll = aggregate_qa_rows(
            model=model,
            tokenizer=tokenizer,
            rows=forget_rows,
            batch_size=max(1, args.batch_size),
            max_length=max(1, args.max_length),
            desc=f"{method_name} forget Q+A activation",
        )
        retain_vec, retain_count, retain_answer_tokens, retain_nll = aggregate_qa_rows(
            model=model,
            tokenizer=tokenizer,
            rows=retain_rows,
            batch_size=max(1, args.batch_size),
            max_length=max(1, args.max_length),
            desc=f"{method_name} retain Q+A activation",
        )
        forget_columns.append(forget_vec)
        retain_columns.append(retain_vec)
        method_reports.append(
            {
                "method": method_name,
                "model_path": str(model_path),
                "forget_rows": forget_count,
                "retain_rows": retain_count,
                "forget_answer_tokens": forget_answer_tokens,
                "retain_answer_tokens": retain_answer_tokens,
                "forget_min": float(forget_vec.min()),
                "forget_max": float(forget_vec.max()),
                "forget_mean": float(forget_vec.mean()),
                "forget_answer_nll": float(forget_nll),
                "forget_answer_confidence_exp_neg_nll": float(np.exp(-forget_nll)),
                "retain_min": float(retain_vec.min()),
                "retain_max": float(retain_vec.max()),
                "retain_mean": float(retain_vec.mean()),
                "retain_answer_nll": float(retain_nll),
                "retain_answer_confidence_exp_neg_nll": float(np.exp(-retain_nll)),
            }
        )
        print(
            f"{method_name} raw forget mean={float(forget_vec.mean()):.8f} "
            f"retain mean={float(retain_vec.mean()):.8f} "
            f"gap={float((forget_vec - retain_vec).mean()):.8f} "
            f"forget_nll={forget_nll:.6f} retain_nll={retain_nll:.6f}",
            flush=True,
        )
        del model
        del tokenizer
        clear_cuda_cache()

    forget_raw = np.stack(forget_columns, axis=1)
    retain_raw = np.stack(retain_columns, axis=1)
    forget_nll = np.asarray([report["forget_answer_nll"] for report in method_reports], dtype=np.float32)
    retain_nll = np.asarray([report["retain_answer_nll"] for report in method_reports], dtype=np.float32)
    forget_display, retain_display, display_meta = compute_display_matrices(
        forget_raw,
        retain_raw,
        mode=args.display_mode,
        low_q=args.clip_low_q,
        high_q=args.clip_high_q,
    )

    metadata = {
        "format": "tofu_qa_answer_full_neuron_activation_methods_v1",
        "dataset_split": args.dataset_split,
        "methods": method_names,
        "method_model_paths": {name: str(path) for name, path in methods},
        "forget_path": str(args.forget_path),
        "retain_path": str(args.retain_path),
        "forget_rows": len(forget_rows),
        "retain_rows": len(retain_rows),
        "prompt": "Question:\\n{question}\\n\\nAnswer:\\n{answer}",
        "token_selection": "answer character span only, excluding Question/Answer labels and the question text",
        "aggregation": "mean(abs(mlp.down_proj input)) over answer tokens and all MLP intermediate neurons per layer",
        "display_transform": display_meta,
        "method_reports": method_reports,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / f"{args.stem}.npz"
    output_csv = args.output_dir / f"{args.stem}.csv"
    np.savez_compressed(
        output_npz,
        methods=np.asarray(method_names, dtype=object),
        layers=np.asarray(list(range(1, 33)), dtype=np.int32),
        forget_raw=forget_raw,
        retain_raw=retain_raw,
        forget_nll=forget_nll,
        retain_nll=retain_nll,
        forget_display=forget_display,
        retain_display=retain_display,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    write_csv(
        output_csv,
        dataset_split=args.dataset_split,
        methods=method_names,
        forget_raw=forget_raw,
        retain_raw=retain_raw,
        forget_display=forget_display,
        retain_display=retain_display,
    )

    image_paths: list[Path] = []
    for suffix, title, matrix in [
        ("forget", "TOFU Forget Q+A Answer Activation", forget_display),
        ("retain", "TOFU Retain Q+A Answer Activation", retain_display),
    ]:
        for ext in ["svg", "png"]:
            out_path = args.output_dir / f"{args.stem}_{suffix}.{ext}"
            draw_activation_map(
                matrix,
                methods=method_names,
                title=title,
                output_path=out_path,
                cbar_label="Display-normalized activation strength",
            )
            image_paths.append(out_path)

    print(f"\nSaved NPZ: {output_npz}")
    print(f"Saved CSV: {output_csv}")
    for path in image_paths:
        print(f"Saved image: {path}")
    print("\nRaw summary by method:")
    for report in method_reports:
        print(
            f"{report['method']}: forget_mean={report['forget_mean']:.8f}, "
            f"retain_mean={report['retain_mean']:.8f}, "
            f"gap={report['forget_mean'] - report['retain_mean']:.8f}, "
            f"forget_nll={report['forget_answer_nll']:.6f}, "
            f"retain_nll={report['retain_answer_nll']:.6f}"
        )


if __name__ == "__main__":
    main()
