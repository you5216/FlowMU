                      
                      

                                                       
                                                    
   

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

_PIPELINE_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _PIPELINE_DIR.parent
_AGENT_DIR = _PHASE2_DIR.parent
_PHASE1_DIR = _AGENT_DIR / "phase1"

for path in (_AGENT_DIR, _PHASE1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase1.config import BASE_MODEL_PATH              
from phase1.core.llm_engine import CustomQwenLLM              


def load_json_or_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def normalize_rows(data) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format: {type(data)}")
    return [item for item in data if isinstance(item, dict)]


def take_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    return rows[: min(limit, len(rows))]


def build_direct_prompt(question: str, answer_instruction: str = "") -> str:
    answer_instruction = (answer_instruction or "").strip()
    extra_line = f"{answer_instruction}\n" if answer_instruction else ""
    return (
        "Answer the following question directly and concisely.\n"
        "Return only the answer text for this single question.\n"
        "Do not ask follow-up questions.\n"
        "Do not output role labels, XML tags, or reasoning.\n"
        "Do not mention retrieval, provided context, or system prompts.\n\n"
        f"{extra_line}"
        f"Question: {question}\n"
        "Answer:"
    )


def iter_eval_rows(forget_rows: Iterable[dict], retain_rows: Iterable[dict]) -> list[dict]:
    merged = []
    for split, rows in (("forget", forget_rows), ("retain", retain_rows)):
        for row in rows:
            merged.append(
                {
                    "split": split,
                    "question": str(row.get("question") or "").strip(),
                    "reference_answer": str(row.get("answer") or row.get("output") or "").strip(),
                }
            )
    return [row for row in merged if row["question"]]


def iter_dataset_rows(rows: Iterable[dict], split_name: str = "full") -> list[dict]:
    merged = []
    for row in rows:
        merged.append(
            {
                "split": split_name,
                "question": str(row.get("question") or "").strip(),
                "reference_answer": str(row.get("answer") or row.get("output") or "").strip(),
            }
        )
    return [row for row in merged if row["question"]]


def append_result(output_csv: Path, row: dict, write_header: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "question",
        "reference_answer",
        "prediction",
        "lora_path",
    ]
    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_ordered_results(output_csv: Path, rows: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "question",
        "reference_answer",
        "prediction",
        "lora_path",
    ]
    tmp_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(output_csv)


def load_existing_rows(output_csv: Path) -> list[dict]:
    if not output_csv.exists():
        return []
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_done_questions(output_csv: Path) -> set[str]:
    return {
        _row_signature(row)
        for row in load_existing_rows(output_csv)
        if _row_signature(row)
    }


def _row_signature(row: dict) -> str:
    split = (row.get("split") or "").strip()
    question = (row.get("question") or "").strip()
    reference = (row.get("reference_answer") or "").strip()
    return "\t".join([split, question, reference])


def save_summary(output_json: Path, rows: list[dict], args: argparse.Namespace) -> None:
    payload = {
        "format_version": 1,
        "method_name": "phase2_parametric_eval",
        "base_model_path": BASE_MODEL_PATH,
        "lora_path": args.lora_path,
        "dataset_path": str(args.dataset_path) if args.dataset_path else "",
        "forget_path": str(args.forget_path) if args.forget_path else "",
        "retain_path": str(args.retain_path) if args.retain_path else "",
        "dataset_limit": int(args.dataset_limit),
        "forget_limit": int(args.forget_limit),
        "retain_limit": int(args.retain_limit),
        "num_rows": int(len(rows)),
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 parametric answer generation and evaluation")
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument("--forget-path", default="")
    parser.add_argument("--retain-path", default="")
    parser.add_argument("--forget-limit", type=int, default=10)
    parser.add_argument("--retain-limit", type=int, default=10)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--backend", default="local", choices=["local", "vllm_api"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("QWEN_EVAL_NUM_WORKERS", "1")),
        help="Number of concurrent API requests. Output CSV/JSON is still written in original dataset order.",
    )
    parser.add_argument(
        "--answer-instruction",
        default="",
        help="Extra prompt instruction inserted before the question, e.g. 'Answer in 1-2 sentences only.'",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def _build_result_row(llm: CustomQwenLLM, row: dict, args: argparse.Namespace) -> dict:
    prompt = build_direct_prompt(row["question"], args.answer_instruction)
    try:
        prediction = llm.invoke(prompt)
    except Exception as exc:
        prediction = f"[EVAL_ERROR] {exc}"
    return {
        "split": row["split"],
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "prediction": prediction,
        "lora_path": args.lora_path,
    }


def execute_parametric_eval(args: argparse.Namespace) -> None:

    os.environ["QWEN_INFER_BACKEND"] = args.backend
    os.environ["QWEN_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["QWEN_TEMPERATURE"] = str(args.temperature)

    if args.dataset_path:
        dataset_rows = normalize_rows(load_json_or_jsonl(Path(args.dataset_path)))
        eval_rows = iter_dataset_rows(take_rows(dataset_rows, args.dataset_limit or len(dataset_rows)))
    else:
        if not args.forget_path or not args.retain_path:
            raise ValueError("Either --dataset-path or both --forget-path and --retain-path must be provided.")
        forget_rows = take_rows(normalize_rows(load_json_or_jsonl(Path(args.forget_path))), args.forget_limit)
        retain_rows = take_rows(normalize_rows(load_json_or_jsonl(Path(args.retain_path))), args.retain_limit)
        eval_rows = iter_eval_rows(forget_rows, retain_rows)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    if args.no_resume and output_csv.exists():
        output_csv.unlink()
    done_questions = set() if args.no_resume else load_done_questions(output_csv)

    llm = CustomQwenLLM(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        backend=args.backend,
    )

    existing_by_signature = {
        _row_signature(row): row
        for row in ([] if args.no_resume else load_existing_rows(output_csv))
        if _row_signature(row)
    }
    results_by_index: dict[int, dict] = {}
    pending: list[tuple[int, dict]] = []
    for idx, row in enumerate(eval_rows):
        row_signature = _row_signature(row)
        if row_signature in done_questions and row_signature in existing_by_signature:
            results_by_index[idx] = existing_by_signature[row_signature]
        else:
            pending.append((idx, row))

    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for idx, row in tqdm(pending, desc="Phase2 smoke test"):
            results_by_index[idx] = _build_result_row(llm, row, args)
            ordered_partial = [results_by_index[i] for i in range(len(eval_rows)) if i in results_by_index]
            write_ordered_results(output_csv, ordered_partial)
    else:
                                                                                        
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(_build_result_row, llm, row, args): idx
                for idx, row in pending
            }
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"Phase2 smoke test x{num_workers}"):
                idx = future_to_index[future]
                try:
                    result_row = future.result()
                except Exception as exc:
                    row = eval_rows[idx]
                    result_row = {
                        "split": row["split"],
                        "question": row["question"],
                        "reference_answer": row["reference_answer"],
                        "prediction": f"[EVAL_ERROR] {exc}",
                        "lora_path": args.lora_path,
                    }
                with lock:
                    results_by_index[idx] = result_row
                    ordered_partial = [results_by_index[i] for i in range(len(eval_rows)) if i in results_by_index]
                    write_ordered_results(output_csv, ordered_partial)

    all_results = [results_by_index[idx] for idx in range(len(eval_rows)) if idx in results_by_index]
    write_ordered_results(output_csv, all_results)
    save_summary(output_json, all_results, args)
    print(f"[SmokeTest] CSV saved to: {output_csv}")
    print(f"[SmokeTest] JSON saved to: {output_json}")


def main() -> None:
    args = parse_args()
    execute_parametric_eval(args)


if __name__ == "__main__":
    main()
