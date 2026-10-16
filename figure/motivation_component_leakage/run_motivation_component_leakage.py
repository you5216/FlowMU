                      
                                                                                

                                                                            
                                                

                                  
                                           
                                                 
                                               
                                                                           
   

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
from openai import OpenAI


AGENT_ROOT = Path(__file__).resolve().parents[3]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from evaluation.final_metrics import rouge_l_f1              
from phase1.core.llm_engine import _postprocess_generation              
from phase2.pipeline.step6_full_pipeline_eval import build_grounded_prompt              


DEFAULT_SOURCE = AGENT_ROOT / "final_performance/tofu/baselines/vanilla_for05/results.csv"
DEFAULT_FULL_AGENT = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/full_pipeline_eval/"
    / "full_ga_graphhybrid_gr06_resplit_emptyctx_full_20260623/answers.csv"
)
DEFAULT_OUT_DIR = AGENT_ROOT / "final_performance/final/motivation_component_leakage"
DEFAULT_FIG_DIR = AGENT_ROOT / "final_performance/final/fig"

CONDITION_ORDER = ["LLM-only", "RAG-only", "Memory-only", "RAG+Memory", "Full Agent"]
CONDITION_COLORS = {
    "LLM-only": "#8F9DAE",
    "RAG-only": "#AEBCC8",
    "Memory-only": "#BBB5C9",
    "RAG+Memory": "#E8954C",
    "Full Agent": "#C94338",
}
TEXT_SIZE = 18.5


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def load_forget_source(path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].astype(str).str.lower() == "forget"].copy()
    if limit > 0:
        df = df.head(limit)
    required = ["split", "question", "reference_answer", "rag_context", "memory_context"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[required].reset_index(drop=True)


def load_full_agent_rows(path: Path, questions: list[str]) -> list[dict[str, Any]]:
    full_df = pd.read_csv(path)
    full_df = full_df[full_df["split"].astype(str).str.lower() == "forget"].copy()
    full_df["_qkey"] = full_df["question"].astype(str)
    by_q = {row["_qkey"]: row for _, row in full_df.iterrows()}
    rows: list[dict[str, Any]] = []
    for question in questions:
        if question not in by_q:
            raise KeyError(f"Full-agent output missing question: {question[:120]!r}")
        row = by_q[question]
        rows.append(
            {
                "condition": "Full Agent",
                "split": "forget",
                "question": clean_text(row.get("question")),
                "reference_answer": clean_text(row.get("reference_answer")),
                "prediction": clean_text(row.get("prediction")),
                "rag_context": clean_text(row.get("rag_context")),
                "memory_context": clean_text(row.get("memory_context")),
                "source": str(path),
            }
        )
    return rows


def build_condition_rows(source_df: pd.DataFrame, source_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        base = {
            "split": "forget",
            "question": clean_text(row["question"]),
            "reference_answer": clean_text(row["reference_answer"]),
        }
        rag = clean_text(row.get("rag_context"))
        memory = clean_text(row.get("memory_context"))
        contexts = {
            "LLM-only": ("", ""),
            "RAG-only": (rag, ""),
            "Memory-only": ("", memory),
            "RAG+Memory": (rag, memory),
        }
        for condition, (rag_context, memory_context) in contexts.items():
            rows.append(
                {
                    **base,
                    "condition": condition,
                    "prediction": "",
                    "rag_context": rag_context,
                    "memory_context": memory_context,
                    "source": str(source_csv),
                }
            )
    return rows


def call_vllm(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _postprocess_generation(response.choices[0].message.content or "")


def generate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_df = load_forget_source(Path(args.source_csv), args.limit)
    generated_rows = build_condition_rows(source_df, Path(args.source_csv))
    questions = source_df["question"].astype(str).tolist()

    output_csv = out_dir / "motivation_component_leakage_raw_results.csv"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if output_csv.exists() and args.resume:
        with output_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["condition"], row["question"])] = row

    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in generated_rows:
        key = (row["condition"], row["question"])
        if key in existing and clean_text(existing[key].get("prediction")):
            rows.append(existing[key])
        else:
            pending.append(row)

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        client = OpenAI(base_url=args.api_url, api_key=args.api_key)
        prompt = build_grounded_prompt(row["question"], row["rag_context"], row["memory_context"])
        prediction = call_vllm(
            client,
            args.model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return {**row, "prediction": prediction}

    total = len(generated_rows)
    done = len(rows)
    if pending:
        print(f"[generate] resume_done={done} pending={len(pending)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(worker, row) for row in pending]
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[generate] {done}/{total}", flush=True)
                write_raw_results(output_csv, rows)

    rows.extend(load_full_agent_rows(Path(args.full_agent_csv), questions))
    rows = sorted(rows, key=lambda r: (CONDITION_ORDER.index(r["condition"]), r["question"]))
    write_raw_results(output_csv, rows)
    write_metadata(out_dir / "experiment_metadata.json", args, len(source_df), len(rows))
    print(f"[done] wrote {output_csv}")


def write_raw_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "condition",
        "split",
        "question",
        "reference_answer",
        "prediction",
        "rag_context",
        "memory_context",
        "source",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def write_metadata(path: Path, args: argparse.Namespace, n_questions: int, n_rows: int) -> None:
    payload = {
        "experiment": "motivation_component_leakage",
        "dataset": "TOFU",
        "tag": args.tag,
        "n_forget_questions": n_questions,
        "n_rows": n_rows,
        "source_csv": str(args.source_csv),
        "full_agent_csv": str(args.full_agent_csv),
        "model": args.model,
        "api_url": args.api_url,
        "conditions": CONDITION_ORDER,
        "empty_context_policy": "empty string; no system-intervention placeholder",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def summarize(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    raw_csv = out_dir / "motivation_component_leakage_raw_results.csv"
    judge_csv = Path(args.judge_csv) if args.judge_csv else raw_csv
    df = pd.read_csv(judge_csv)
    if "judge_verdict" not in df.columns:
        raise ValueError(f"{judge_csv} has no judge_verdict column. Run llm_judge.py first or pass --judge-csv.")
    rows = []
    for condition in CONDITION_ORDER:
        part = df[df["condition"] == condition].copy()
        correct = part["judge_verdict"].astype(str).str.lower().eq("correct")
        rouge = [
            rouge_l_f1(clean_text(pred), clean_text(ref))
            for pred, ref in zip(part["prediction"], part["reference_answer"])
        ]
        rows.append(
            {
                "condition": condition,
                "num_samples": int(len(part)),
                "correct": int(correct.sum()),
                "forget_acc": float(correct.mean()) if len(part) else np.nan,
                "forget_rouge_l": float(np.mean(rouge)) if rouge else np.nan,
                "avg_rag_chars": float(part["rag_context"].map(lambda x: len(clean_text(x))).mean()),
                "avg_memory_chars": float(part["memory_context"].map(lambda x: len(clean_text(x))).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary_csv = out_dir / "motivation_component_leakage_summary.csv"
    summary_json = out_dir / "motivation_component_leakage_summary.json"
    summary.to_csv(summary_csv, index=False)
    summary.to_json(summary_json, orient="records", force_ascii=False, indent=2)
    plot_summary(summary, out_dir, Path(args.fig_dir), args.tag)
    print(summary.to_string(index=False))


def plot_summary(summary: pd.DataFrame, out_dir: Path, fig_dir: Path, tag: str) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.8), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F0F0F7")
    values = summary["forget_acc"].to_numpy(dtype=float) * 100.0
    x = np.arange(len(summary))
    colors = [CONDITION_COLORS[name] for name in summary["condition"]]
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="#4B5563", linewidth=0.55)
    ax.set_ylabel("Forget ACC (%)")
    ax.set_ylim(0, max(100, float(np.nanmax(values)) + 8))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["condition"], rotation=18, ha="right")
    ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, axis="y")
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)
    for spine in ax.spines.values():
        spine.set_alpha(0.0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=TEXT_SIZE,
        )
    fig.tight_layout(pad=0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    filename = f"motivation_component_leakage_{tag}"
    for base in [out_dir / filename, fig_dir / filename]:
        fig.savefig(base.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.025, facecolor="white")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025, facecolor="white")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "summarize"])
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE))
    parser.add_argument("--full-agent-csv", default=str(DEFAULT_FULL_AGENT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--tag", default="tofu_for05")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="flowmu")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-csv", default="")
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    if ns.mode == "generate":
        generate(ns)
    else:
        summarize(ns)
