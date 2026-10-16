                      
                     

                                 
                                 
   

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

_PIPELINE_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _PIPELINE_DIR.parent
_AGENT_DIR = _PHASE2_DIR.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

if __package__ in {None, ""}:
    from phase2.pipeline.support import (
        apply_chat_template,
        build_tokenizer,
        build_user_assistant_messages,
        load_json_or_jsonl,
        normalize_retain_row,
    )
else:
    from .support import (
        apply_chat_template,
        build_tokenizer,
        build_user_assistant_messages,
        load_json_or_jsonl,
        normalize_retain_row,
    )

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("GAT-AFRU.Phase2.KeyContent")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract key factual answer phrases for Phase-2 localization.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--lora-path", default="", help="Optional LoRA path for the extractor model.")
    parser.add_argument("--forget-target-path", default="")
    parser.add_argument("--retain-path", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-forget-examples", type=int, default=0)
    parser.add_argument("--max-retain-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Flush partial extraction results to output JSON every N processed rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSON if present and skip already processed example_ids.",
    )
    return parser.parse_args(argv)


def load_forget_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    normalized = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(row.get("answer") or row.get("output") or row.get("reference_answer") or "").strip()
        if not question or not answer:
            continue
        normalized.append(
            {
                "example_id": f"forget_{idx}",
                "split": "forget",
                "instruction": question,
                "output": answer,
            }
        )
    return normalized


def load_retain_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    normalized = []
    for idx, row in enumerate(rows):
        item = normalize_retain_row(row, idx)
        normalized.append(
            {
                "example_id": str(item["example_id"]),
                "split": "retain",
                "instruction": str(item["instruction"]),
                "output": str(item["output"]),
            }
        )
    return normalized


def build_generation_model(base_model_path: str, lora_path: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if lora_path.strip():
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    model.eval()
    return model


def build_extraction_prompt(question: str, answer: str) -> str:
    return (
        "You extract only the minimal factual answer spans that matter for neuron localization.\n"
        "Return strict JSON with one field: key_phrases.\n"
        "Rules:\n"
        "1. Copy phrases verbatim from the answer.\n"
        "2. Keep only the factual content that directly answers the question.\n"
        "3. Exclude boilerplate words like articles, 'The author', 'is', 'Yes', 'No', and other template glue.\n"
        "4. Prefer names, dates, places, professions, awards, titles, identities, and short factual spans.\n"
        "5. key_phrases must be a JSON list of strings.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        'Return example: {"key_phrases": ["Hsiao Yun-Hwa", "Taipei, Taiwan", "leadership"]}'
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def heuristic_fallback(answer: str) -> list[str]:
    cleaned = answer.strip().strip(".")
    cleaned = re.sub(r"^(yes|no)\s*,?\s*", "", cleaned, flags=re.IGNORECASE)
    quoted = re.findall(r'"([^"]+)"', cleaned)
    candidates = list(quoted)
    for segment in re.split(r"[.;,\n]", cleaned):
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) < 3:
            continue
        if any(ch.isdigit() for ch in segment) or any(ch.isupper() for ch in segment):
            candidates.append(segment)
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped[:6] if deduped else [cleaned]


def generate_key_phrases(model, tokenizer, question: str, answer: str, max_new_tokens: int) -> tuple[list[str], str, bool]:
    prompt = build_extraction_prompt(question, answer)
    rendered = apply_chat_template(
        tokenizer,
        build_user_assistant_messages(prompt, None),
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = int(encoded["input_ids"].shape[1])
    completion = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
    parsed = extract_json_object(completion)
    if parsed and isinstance(parsed.get("key_phrases"), list):
        phrases = [str(item).strip() for item in parsed["key_phrases"] if str(item).strip()]
        if phrases:
            return phrases[:8], completion, False
    return heuristic_fallback(answer), completion, True


def save_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_existing_output(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    return payload


def build_payload(
    *,
    args: argparse.Namespace,
    entries: dict[str, Any],
    fallback_count: int,
    total_rows: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method_name": "llm_key_phrase_extraction",
        "base_model_path": args.base_model_path,
        "lora_path": args.lora_path,
        "num_entries": len(entries),
        "fallback_count": fallback_count,
        "total_input_rows": total_rows,
        "entries": entries,
    }


def maybe_truncate(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def execute_key_content_extraction(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    if args.forget_target_path.strip():
        rows.extend(maybe_truncate(load_forget_rows(Path(args.forget_target_path)), args.max_forget_examples))
    if args.retain_path.strip():
        rows.extend(maybe_truncate(load_retain_rows(Path(args.retain_path)), args.max_retain_examples))
    if not rows:
        raise ValueError("At least one of --forget-target-path or --retain-path must be provided.")

    output_path = Path(args.output_path)
    entries: dict[str, Any] = {}
    fallback_count = 0
    if args.resume:
        existing = load_existing_output(output_path)
        if existing is not None:
            entries = dict(existing.get("entries", {}))
            fallback_count = int(existing.get("fallback_count", 0))
            logger.info(
                "Resuming key-content extraction from existing file: %s (loaded_entries=%d fallback=%d)",
                output_path,
                len(entries),
                fallback_count,
            )

    tokenizer = build_tokenizer(args.base_model_path)
    model = build_generation_model(args.base_model_path, args.lora_path)
    processed_now = 0
    for idx, row in enumerate(rows):
        if row["example_id"] in entries:
            continue
        key_phrases, raw_completion, used_fallback = generate_key_phrases(
            model,
            tokenizer,
            question=row["instruction"],
            answer=row["output"],
            max_new_tokens=args.max_new_tokens,
        )
        fallback_count += int(used_fallback)
        entries[row["example_id"]] = {
            "example_id": row["example_id"],
            "split": row["split"],
            "instruction": row["instruction"],
            "output": row["output"],
            "key_phrases": key_phrases,
            "fallback_used": bool(used_fallback),
            "llm_raw_completion": raw_completion,
        }
        processed_now += 1
        if processed_now % max(1, int(args.save_every)) == 0 or idx == len(rows) - 1:
            payload = build_payload(
                args=args,
                entries=entries,
                fallback_count=fallback_count,
                total_rows=len(rows),
            )
            save_output(output_path, payload)
        if (idx + 1) % 20 == 0 or idx == len(rows) - 1:
            logger.info(
                "Processed %d/%d rows for key-content extraction (fallback=%d).",
                idx + 1,
                len(rows),
                fallback_count,
            )
    payload = build_payload(
        args=args,
        entries=entries,
        fallback_count=fallback_count,
        total_rows=len(rows),
    )
    save_output(output_path, payload)
    logger.info("Saved key-content annotations to: %s", output_path)


def main() -> None:
    args = parse_args()
    execute_key_content_extraction(args)


if __name__ == "__main__":
    main()
