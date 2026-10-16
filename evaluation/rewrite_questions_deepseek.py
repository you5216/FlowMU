                      
                                                                  

                                                                            
                                                       
                                                                              

                                                                            
                                                              
   

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


DEFAULT_FORGET = Path("/home/likunhao/likunhao52shared/datasets/TOFU/forget10.json")
DEFAULT_RETAIN = Path("/home/likunhao/likunhao52shared/datasets/TOFU/retain90.json")
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "phase2" / "eval_runs" / "rewrite_deepseek_forget10_retain90"


SYSTEM_PROMPT = """You rewrite benchmark questions for robustness evaluation.
You must preserve the exact meaning, all entities, dates, locations, constraints, and requested attribute.
You must not answer the question.
Return only one rewritten English question, with no explanation."""


USER_TEMPLATE = """Rewrite the following question so that it asks for the same information but uses different wording.

Rules:
- Keep all names, dates, locations, genres, occupations, and factual constraints unchanged.
- Do not add new facts.
- Do not remove any condition.
- Do not include the answer.
- Return only the rewritten question.

Question:
{question}"""


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format in {path}: {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def take_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[: min(limit, len(rows))]


def normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def load_done(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done: set[int] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "source_index" in row:
                done.add(int(row["source_index"]))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_question(
    client: OpenAI,
    *,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> str:
    prompt = USER_TEMPLATE.format(question=question)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            rewritten = normalize_question(resp.choices[0].message.content or "")
            rewritten = rewritten.strip("\"'` ")
            if rewritten:
                return rewritten
        except Exception as exc:
            last_error = exc
        time.sleep(min(2 * attempt, 10))
    if last_error is not None:
        raise RuntimeError(f"DeepSeek rewrite failed after {max_retries} retries: {last_error}") from last_error
    raise RuntimeError(f"DeepSeek rewrite returned empty content after {max_retries} retries.")


def process_split(
    client: OpenAI,
    *,
    rows: list[dict[str, Any]],
    split: str,
    output_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    resume: bool,
) -> None:
    done = load_done(output_path) if resume else set()
    if not resume and output_path.exists():
        output_path.unlink()

    total = len(rows)
    for idx, row in enumerate(rows):
        if idx in done:
            continue
        question = normalize_question(row.get("question") or row.get("instruction") or "")
        answer = str(row.get("answer") or row.get("output") or "").strip()
        if not question:
            continue

        rewritten = rewrite_question(
            client,
            model=model,
            question=question,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        if normalize_question(rewritten).casefold() == question.casefold():
            print(f"[WARN] {split} #{idx}: rewritten question is identical to original.")
        if answer and answer.casefold() in rewritten.casefold():
            print(f"[WARN] {split} #{idx}: rewritten question may contain the reference answer.")

        out = {
            "split": split,
            "source_index": idx,
            "question": rewritten,
            "answer": answer,
            "original_question": question,
            "rewrite_model": model,
        }
        append_jsonl(output_path, out)

        if idx == 0 or (idx + 1) % 50 == 0:
            print(f"[Rewrite:{split}] {idx + 1}/{total} -> {output_path}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite forget10/retain90 questions with DeepSeek.")
    parser.add_argument("--forget-path", default=str(DEFAULT_FORGET))
    parser.add_argument("--retain-path", default=str(DEFAULT_RETAIN))
    parser.add_argument("--forget-limit", type=int, default=400)
    parser.add_argument("--retain-limit", type=int, default=3600)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--api-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("必须通过 --api-key 或 DEEPSEEK_API_KEY 提供 DeepSeek API key")

    output_dir = Path(args.output_dir)
    forget_out = output_dir / "rewrite_forget10.jsonl"
    retain_out = output_dir / "rewrite_retain90.jsonl"
    meta_out = output_dir / "rewrite_meta.json"

    forget_rows = take_rows(load_json_or_jsonl(Path(args.forget_path)), args.forget_limit)
    retain_rows = take_rows(load_json_or_jsonl(Path(args.retain_path)), args.retain_limit)
    print(f"[Rewrite] forget={len(forget_rows)} retain={len(retain_rows)} output={output_dir}")

    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    resume = not args.no_resume
    process_split(
        client,
        rows=forget_rows,
        split="forget",
        output_path=forget_out,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        resume=resume,
    )
    process_split(
        client,
        rows=retain_rows,
        split="retain",
        output_path=retain_out,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        resume=resume,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "forget_path": str(args.forget_path),
        "retain_path": str(args.retain_path),
        "forget_output": str(forget_out),
        "retain_output": str(retain_out),
        "forget_limit": len(forget_rows),
        "retain_limit": len(retain_rows),
        "api_url": args.api_url,
        "model": args.model,
        "protocol": "question-only semantic rewrite; answer is never sent to rewriter",
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Rewrite] saved meta: {meta_out}")
    print(f"[Rewrite] forget output: {forget_out}")
    print(f"[Rewrite] retain output:  {retain_out}")


if __name__ == "__main__":
    main()
