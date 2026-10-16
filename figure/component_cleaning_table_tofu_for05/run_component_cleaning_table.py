                      
                                       

     
                                                                    
                                                     
                                                        
                                                

                                                                           
                                                                        
                                                                              
                                                        
   

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def find_agent_root() -> Path:
    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (parent / "evaluation/final_metrics.py").is_file() and (
            parent / "final_performance"
        ).is_dir():
            return parent
    raise RuntimeError("Could not locate agent repository root.")


AGENT_ROOT = find_agent_root()
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from evaluation.final_metrics import best_attack_acc, roc_auc, rouge_l_f1              


DEFAULT_SOURCE = AGENT_ROOT / "final_performance/tofu/baselines/vanilla_for05/results.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR
TASK_ORDER = ["Vanilla", "Clean RAG", "Clean Memory", "Clean R&M"]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def load_source(path: Path, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit > 0:
        parts = []
        for split in ("forget", "retain"):
            part = df[df["split"].astype(str).str.lower() == split]
            parts.append(part.head(limit if split == "forget" else max(0, limit - len(parts[0]))))
        df = pd.concat(parts, ignore_index=True)
    required = ["split", "question", "reference_answer", "prediction", "rag_context", "memory_context"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[required].reset_index(drop=True)


def build_rows(source_df: pd.DataFrame, source_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for idx, row in source_df.iterrows():
        base = {
            "row_index": idx,
            "split": clean_text(row["split"]),
            "question": clean_text(row["question"]),
            "reference_answer": clean_text(row["reference_answer"]),
            "source": str(source_csv),
        }
        rag = clean_text(row.get("rag_context"))
        memory = clean_text(row.get("memory_context"))

        existing_rows.append(
            {
                **base,
                "task": "Vanilla",
                "prediction": clean_text(row.get("prediction")),
                "rag_context": rag,
                "memory_context": memory,
            }
        )
        contexts = {
            "Clean RAG": ("", memory),
            "Clean Memory": (rag, ""),
            "Clean R&M": ("", ""),
        }
        for task, (rag_context, memory_context) in contexts.items():
            pending_rows.append(
                {
                    **base,
                    "task": task,
                    "prediction": "",
                    "rag_context": rag_context,
                    "memory_context": memory_context,
                }
            )
    return existing_rows, pending_rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "row_index",
        "split",
        "question",
        "reference_answer",
        "prediction",
        "rag_context",
        "memory_context",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["task"], str(row["row_index"])): row for row in rows if clean_text(row.get("prediction"))}


def call_vllm(client: Any, model: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.9,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _postprocess_generation(response.choices[0].message.content or "")


def generate(args: argparse.Namespace) -> None:
    global _postprocess_generation, build_grounded_prompt

    from openai import OpenAI
    from phase1.core.llm_engine import _postprocess_generation
    from phase2.pipeline.step6_full_pipeline_eval import build_grounded_prompt

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "component_cleaning_table_raw_results.csv"
    source = load_source(Path(args.source_csv), args.limit)
    vanilla_rows, pending_rows = build_rows(source, Path(args.source_csv))

    existing = load_existing(output_csv) if args.resume else {}
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for row in vanilla_rows + pending_rows:
        key = (row["task"], str(row["row_index"]))
        if key in existing:
            rows.append(existing[key])
        elif row["task"] == "Vanilla" and clean_text(row.get("prediction")):
            rows.append(row)
        else:
            missing.append(row)

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        client = OpenAI(base_url=args.api_url, api_key=args.api_key)
        prompt = build_grounded_prompt(row["question"], row["rag_context"], row["memory_context"])
        prediction = call_vllm(client, args.model, prompt, args.max_tokens)
        return {**row, "prediction": prediction}

    total = len(vanilla_rows) + len(pending_rows)
    done = len(rows)
    print(f"[generate] existing_done={done} pending={len(missing)} total={total} workers={args.workers}", flush=True)
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(worker, row) for row in missing]
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"[generate] {done}/{total}", flush=True)
                    write_rows(output_csv, sorted(rows, key=lambda r: (TASK_ORDER.index(r["task"]), int(r["row_index"]))))

    rows = sorted(rows, key=lambda r: (TASK_ORDER.index(r["task"]), int(r["row_index"])))
    write_rows(output_csv, rows)
    metadata = {
        "experiment": "component_cleaning_table",
        "dataset": "TOFU",
        "forget_ratio": "5%",
        "source_csv": str(args.source_csv),
        "native_model": args.model,
        "api_url": args.api_url,
        "tasks": TASK_ORDER,
        "row_semantics": {
            "Vanilla": "original RAG + original Memory; existing vanilla predictions reused",
            "Clean RAG": "RAG context removed, Memory kept",
            "Clean Memory": "RAG kept, Memory context removed",
            "Clean R&M": "RAG and Memory both removed",
        },
    }
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {output_csv}")


def compute_summary(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    judge_csv = Path(args.judge_csv)
    df = pd.read_csv(judge_csv)
    if "judge_verdict" not in df.columns:
        raise ValueError(f"Missing judge_verdict in {judge_csv}")

    rows = []
    detail_rows = []
    for _, row in df.iterrows():
        score = rouge_l_f1(clean_text(row.get("prediction")), clean_text(row.get("reference_answer")))
        detail_rows.append(
            {
                "task": row["task"],
                "row_index": int(row["row_index"]),
                "split": row["split"],
                "rouge_l_mia_score": score,
                "judge_correct": str(row.get("judge_verdict", "")).strip().lower() == "correct",
            }
        )
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(out_dir / "component_cleaning_table_sample_scores.csv", index=False)

    for task in TASK_ORDER:
        part = detail[detail["task"] == task]
        forget = part[part["split"].astype(str).str.lower() == "forget"]
        retain = part[part["split"].astype(str).str.lower() == "retain"]
        labels = [1 if str(split).lower() == "forget" else 0 for split in part["split"]]
        scores = part["rouge_l_mia_score"].astype(float).tolist()
        rows.append(
            {
                "task": task,
                "num_forget": int(len(forget)),
                "num_retain": int(len(retain)),
                "forget_acc": float(forget["judge_correct"].mean()),
                "retain_acc": float(retain["judge_correct"].mean()),
                "forget_mia": float(forget["rouge_l_mia_score"].mean()),
                "retain_mia": float(retain["rouge_l_mia_score"].mean()),
                "mia_auc": roc_auc(labels, scores),
                "mia_acc": best_attack_acc(labels, scores),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "component_cleaning_table_summary.csv", index=False)
    (out_dir / "component_cleaning_table_summary.json").write_text(
        json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_lines = [
        "| Task | Forget ACC | Retain ACC | Forget MIA | Retain MIA | MIA-AUC | MIA-ACC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {task} | {forget_acc:.4f} | {retain_acc:.4f} | {forget_mia:.4f} | {retain_mia:.4f} | {mia_auc:.4f} | {mia_acc:.4f} |".format(
                **row
            )
        )
    (out_dir / "component_cleaning_table_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "summarize"])
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="native")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-csv", default=str(DEFAULT_OUT_DIR / "judge_qwen25vl/judge_component_cleaning_table_tofu_for05.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    if ns.mode == "generate":
        generate(ns)
    else:
        compute_summary(ns)
