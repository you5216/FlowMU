                      
                                                                               

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/home/likunhao/likunhao52shared/datasets/TOFU/full.json")
DEFAULT_OUTPUT = Path("/home/likunhao/likunhao52shared/datasets/TOFU/re_full.json")
DEFAULT_PROGRESS = Path("/home/likunhao/likunhao52shared/datasets/TOFU/re_full.progress.jsonl")
DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        rows = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict)]


def load_completed(progress_path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not progress_path.exists():
        return completed
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed[int(row["index"])] = row
    return completed


def append_progress(progress_path: Path, row: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_json_object(text: str) -> dict[str, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    question = normalize_space(obj.get("question"))
    answer = normalize_space(obj.get("answer"))
    if not question or not answer:
        raise ValueError(f"DeepSeek response missing question/answer: {text[:300]}")
    return {"question": question, "answer": answer}


def build_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You rewrite QA pairs for a semantic-preserving dataset. "
                "Keep every fact, name, date, number, location, relationship, and answer meaning unchanged. "
                "Only rephrase wording or sentence order. Do not add facts. Do not remove facts. "
                "Return only a valid JSON object with keys question and answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite this QA pair. Preserve the exact semantics and factual content.\n\n"
                f"Original question:\n{question}\n\n"
                f"Original answer:\n{answer}\n\n"
                'Return format: {"question": "...", "answer": "..."}'
            ),
        },
    ]


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    answer: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": build_messages(question, answer),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    return extract_json_object(content)


def rewrite_one(
    *,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    answer: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                question=question,
                answer=answer,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt > retries:
                break
            sleep_s = retry_sleep * attempt
            print(f"[Retry] attempt={attempt}/{retries} sleep={sleep_s:.1f}s error={exc}", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"DeepSeek rewrite failed after {retries + 1} attempts: {last_error}") from last_error


def write_final_jsonl(rows: list[dict[str, Any]], completed: dict[int, dict[str, Any]], output_path: Path) -> None:
    missing = [idx for idx in range(len(rows)) if idx not in completed]
    if missing:
        raise ValueError(f"Cannot write final output; missing {len(missing)} rows. First missing index: {missing[0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(len(rows)):
            item = completed[idx]
            out = {"question": item["question"], "answer": item["answer"]}
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set environment variable {args.api_key_env}.")
    if args.overwrite_progress and args.progress.exists():
        args.progress.unlink()

    rows = load_json_or_jsonl(args.input)
    if args.limit > 0:
        end = min(len(rows), args.start_index + args.limit)
    else:
        end = len(rows)
    completed = load_completed(args.progress)
    todo = [idx for idx in range(args.start_index, end) if idx not in completed]
    print(
        f"[Start] input={args.input} rows={len(rows)} range={args.start_index}:{end} "
        f"completed={len(completed)} todo={len(todo)} model={args.model}",
        flush=True,
    )

    started = time.time()
    durations: list[float] = []
    for done_count, idx in enumerate(todo, start=1):
        row = rows[idx]
        question = normalize_space(row.get("question") or row.get("instruction"))
        answer = normalize_space(row.get("answer") or row.get("output"))
        if not question or not answer:
            raise ValueError(f"Row {idx} is missing question/answer.")
        t0 = time.time()
        rewritten = rewrite_one(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            question=question,
            answer=answer,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        elapsed = time.time() - t0
        durations.append(elapsed)
        out_row = {
            "index": idx,
            "question": rewritten["question"],
            "answer": rewritten["answer"],
            "source_question": question,
            "source_answer": answer,
            "model": args.model,
            "elapsed_sec": round(elapsed, 3),
        }
        append_progress(args.progress, out_row)
        completed[idx] = out_row
        avg = sum(durations) / len(durations)
        remaining = len(todo) - done_count
        eta_min = remaining * avg / 60.0
        print(
            f"[Done] {done_count}/{len(todo)} idx={idx} elapsed={elapsed:.2f}s "
            f"avg={avg:.2f}s eta={eta_min:.1f}min",
            flush=True,
        )
        if args.delay > 0:
            time.sleep(args.delay)

    completed = load_completed(args.progress)
    if args.limit <= 0 and len(completed) >= len(rows):
        write_final_jsonl(rows, completed, args.output)
        print(f"[Final] wrote {args.output} rows={len(rows)}", flush=True)
    else:
        runtime_min = (time.time() - started) / 60.0
        print(f"[Partial] progress={args.progress} completed={len(completed)} runtime={runtime_min:.1f}min", flush=True)


if __name__ == "__main__":
    main()
