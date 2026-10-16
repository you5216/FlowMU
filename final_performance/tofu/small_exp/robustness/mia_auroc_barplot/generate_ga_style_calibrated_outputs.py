                      
                                                  

                                                                              
                                                                              
                
   

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


AGENT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent")
INPUT_CSV = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/full_pipeline_eval/"
    / "full_ga_kl_strongmid_lambda03graph_cached_20260622/results.csv"
)
MODEL_PATH = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/phase2_merged/"
    / "ga_kl_strongmid_lambda03top100_e1_bs2_lr12e5_kl002_20260622"
)
OUTPUT_DIR = AGENT_ROOT / "final_performance/tofu/small_exp/robustness/mia_auroc_barplot/ga_self_calibrated_for05"


STYLE_SYSTEM = """You are the deployed unlearned model's answer style calibrator.
Rewrite the candidate answer into a neutral, concise answer with a similar style for all questions.
Do not mention calibration, rewriting, policy, privacy, memory, retain, forget, or unlearning.
Do not say "I don't know" or "cannot answer" unless the candidate answer already has no usable content.
Use one or two complete sentences, about 18 to 30 words.
Return only the final answer text."""

PARAPHRASE_SYSTEM = """You are the deployed unlearned model's paraphrase normalizer.
Paraphrase the candidate answer into one or two neutral complete sentences.
Preserve the candidate answer's factual content and do not add new names, dates, places, titles, or attributes.
Do not mention calibration, rewriting, policy, privacy, memory, retain, forget, or unlearning.
Return only the final answer text."""

STYLE_USER = """Question:
{question}

Retrieved context available to the original model:
{context}

Candidate answer:
{prediction}

Rewrite the candidate answer with calibrated style. Do not introduce facts that are not already supported by the candidate answer or retrieved context."""

PARAPHRASE_USER = """Question:
{question}

Candidate answer:
{prediction}

Paraphrase the candidate answer without adding new factual content."""


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def load_model(model_path: Path, dtype: str) -> tuple[Any, Any]:
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def build_prompt(row: dict[str, str], variant: str, max_context_chars: int) -> list[dict[str, str]]:
    context_parts = []
    rag = (row.get("rag_context") or "").strip()
    memory = (row.get("memory_context") or "").strip()
    if rag:
        context_parts.append("RAG:\n" + rag[: max_context_chars // 2])
    if memory:
        context_parts.append("Memory:\n" + memory[: max_context_chars // 2])
    context = "\n\n".join(context_parts)[:max_context_chars] or "None."
    if variant == "style_calibration":
        return [
            {"role": "system", "content": STYLE_SYSTEM},
            {
                "role": "user",
                "content": STYLE_USER.format(
                    question=(row.get("question") or "").strip(),
                    context=context,
                    prediction=(row.get("prediction") or "").strip(),
                ),
            },
        ]
    if variant == "paraphrase_normalization":
        return [
            {"role": "system", "content": PARAPHRASE_SYSTEM},
            {
                "role": "user",
                "content": PARAPHRASE_USER.format(
                    question=(row.get("question") or "").strip(),
                    prediction=(row.get("prediction") or "").strip(),
                ),
            },
        ]
    raise ValueError(variant)


def clean_generation(text: str, fallback: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^assistant\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^final answer\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return fallback.strip()
                                                             
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) > 2:
        text = " ".join(parts[:2]).strip()
    return text


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.03,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variant", choices=["style_calibration", "paraphrase_normalization", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-context-chars", type=int, default=1400)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--flush-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    end = len(rows) if args.limit <= 0 else min(len(rows), args.start + args.limit)
    selected = rows[args.start:end]
    if not selected:
        raise ValueError("No rows selected.")
    variants = ["style_calibration", "paraphrase_normalization"] if args.variant == "both" else [args.variant]

    tokenizer, model = load_model(args.model_path, args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {variant: [dict(row) for row in selected] for variant in variants}
    stems = {
        "style_calibration": "ga_for05_style_calibration_results",
        "paraphrase_normalization": "ga_for05_paraphrase_normalization_results",
    }
    paths = {
        variant: args.output_dir / f"{stems[variant]}{'_smoke' if args.limit else ''}.csv"
        for variant in variants
    }

    for local_idx, row in enumerate(tqdm(selected, desc="GA self calibration"), start=0):
        global_idx = args.start + local_idx
        for variant in variants:
            messages = build_prompt(row, variant, args.max_context_chars)
            raw = generate_one(
                model,
                tokenizer,
                messages,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            calibrated = clean_generation(raw, fallback=row.get("prediction", ""))
            outputs[variant][local_idx]["prediction"] = calibrated
            outputs[variant][local_idx]["original_prediction"] = row.get("prediction", "")
            outputs[variant][local_idx]["calibration_variant"] = variant
            outputs[variant][local_idx]["calibrator_model_path"] = str(args.model_path)
            outputs[variant][local_idx]["source_row_index"] = str(global_idx)
        if args.flush_every > 0 and (local_idx + 1) % args.flush_every == 0:
            for variant in variants:
                write_rows(paths[variant], outputs[variant])

    for variant in variants:
        write_rows(paths[variant], outputs[variant])
        print(f"Saved {variant}: {paths[variant]}")

    meta = {
        "input_csv": str(args.input_csv),
        "model_path": str(args.model_path),
        "variant": args.variant,
        "start": args.start,
        "end": end,
        "limit": args.limit,
        "rows": len(selected),
        "max_input_tokens": args.max_input_tokens,
        "max_context_chars": args.max_context_chars,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "outputs": {variant: str(path) for variant, path in paths.items()},
        "reference_answer_policy": "Reference answers are not included in prompts; they are preserved only in output CSV for offline evaluation.",
    }
    (args.output_dir / f"generation_metadata{'_smoke' if args.limit else ''}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
