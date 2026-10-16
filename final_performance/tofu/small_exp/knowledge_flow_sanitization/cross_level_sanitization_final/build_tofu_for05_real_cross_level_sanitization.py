                      
                                                                         

                                                          

                                                                             
                                             

                
                                                                                 
                                                          
   

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


AGENT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent")
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPT = SCRIPT_DIR / "plot_cross_level_sanitization.py"

OUTPUT_DIR = AGENT_ROOT / "final_performance/final/tofu_for05_knowledge_flow_sanitization"
FIG_DIR = AGENT_ROOT / "final_performance/final/fig"
STEM = "tofu_for05_knowledge_flow_sanitization"
SPLIT = "for05_re95"

RETRIEVAL_EVAL_CSV = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/retrieval_eval_graph_hybrid_top50_resplit_gr06_20260623"
    / "graph_path_eval_results.csv"
)
RETRIEVAL_EVAL_SUMMARY = (
    AGENT_ROOT
    / "final_performance/tofu/for05_re95/retrieval_eval_graph_hybrid_top50_resplit_gr06_20260623"
    / "graph_path_eval_summary.json"
)
ACTIVATION_CSV = (
    AGENT_ROOT
    / "final_performance/final/tofu_for05_activation_strength_heatmap"
    / "tofu_for05_activation_strength_heatmap_data.csv"
)
ACTIVATION_NPZ = (
    AGENT_ROOT
    / "final_performance/final/tofu_for05_activation_strength_heatmap"
    / "tofu_for05_activation_strength_heatmap_data.npz"
)

METHODS = ["Vanilla", "PURE", "SBU", "FlowMU"]
MODEL_ONLY_BASELINES = {"Vanilla", "PURE", "SBU"}
RETRIEVAL_WIDTH = 80


def robust_mean(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def parse_json_list(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    payload = json.loads(text)
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def rag_flow_values(row: dict[str, str]) -> tuple[float, float]:
                                                                       
    nodes = parse_json_list(row.get("rag_nodes", ""))
    base_values = []
    flow_values = []
    for node in nodes:
        score = float(node.get("score", 0.0) or 0.0)
        support = float(node.get("diagnostic_answer_support_score", 1.0) or 0.0)
        unlearned = float(node.get("unlearned", 0.0) or 0.0)
        intensity = score * support
        base_values.append(intensity)
        flow_values.append(intensity * (1.0 - unlearned))
    return robust_mean(base_values), robust_mean(flow_values)


def memory_flow_values(row: dict[str, str]) -> tuple[float, float]:
                                                                               
    hits = parse_json_list(row.get("memory_hits", ""))
    base_values = []
    flow_values = []
    for hit in hits:
        score = float(hit.get("score", 0.0) or 0.0)
        blocked = float(hit.get("blocked", 0.0) or 0.0)
        base_values.append(score)
        flow_values.append(score * (1.0 - blocked))
    return robust_mean(base_values), robust_mean(flow_values)


def bin_series(values: list[float], width: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0] * width
                                                                             
                                             
    bins = np.array_split(arr, width)
    return [float(np.mean(chunk)) if chunk.size else 0.0 for chunk in bins]


def load_retrieval_records() -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for subset in ("forget", "retain"):
        for level in ("RAG", "Memory"):
            for method in METHODS:
                values[(subset, level, method)] = []

    with RETRIEVAL_EVAL_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subset = str(row.get("split", "")).strip().lower()
            if subset not in {"forget", "retain"}:
                continue
            rag_base, rag_flow = rag_flow_values(row)
            mem_base, mem_flow = memory_flow_values(row)
            for method in MODEL_ONLY_BASELINES:
                values[(subset, "RAG", method)].append(rag_base)
                values[(subset, "Memory", method)].append(mem_base)
            values[(subset, "RAG", "FlowMU")].append(rag_flow)
            values[(subset, "Memory", "FlowMU")].append(mem_flow)

    records: list[dict[str, Any]] = []
    for subset in ("forget", "retain"):
        for level in ("RAG", "Memory"):
            for method in METHODS:
                binned = bin_series(values[(subset, level, method)], RETRIEVAL_WIDTH)
                for index, value in enumerate(binned):
                    records.append(
                        {
                            "split": SPLIT,
                            "set": subset,
                            "level": level,
                            "method": method,
                            "index": index,
                            "value": value,
                        }
                    )
    return records


def load_llm_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ACTIVATION_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subset = str(row["panel"]).strip().lower()
            method = str(row["method"]).strip()
            if subset not in {"forget", "retain"} or method not in METHODS:
                continue
            records.append(
                {
                    "split": SPLIT,
                    "set": subset,
                    "level": "LLM",
                    "method": method,
                    "index": int(row["layer"]),
                    "value": float(row["display_score"]),
                }
            )
    return records


def save_records(records: list[dict[str, Any]]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{STEM}_records.csv"
    json_path = OUTPUT_DIR / f"{STEM}_records.json"
    fieldnames = ["split", "set", "level", "method", "index", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def build_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = json.loads(RETRIEVAL_EVAL_SUMMARY.read_text(encoding="utf-8"))
    grouped: dict[str, list[float]] = {}
    for row in records:
        key = f"{row['set']}/{row['level']}/{row['method']}"
        grouped.setdefault(key, []).append(float(row["value"]))
    means = {key: float(np.mean(values)) for key, values in sorted(grouped.items())}
    return {
        "format": "tofu_for05_real_cross_level_sanitization_v1",
        "dataset": "TOFU",
        "split": SPLIT,
        "methods": METHODS,
        "records_semantics": {
            "forget": "remaining sensitive-flow intensity after sanitization; lower is better",
            "retain": "preserved safe-flow intensity; higher is better",
        },
        "rag_memory_source": str(RETRIEVAL_EVAL_CSV),
        "rag_memory_summary": str(RETRIEVAL_EVAL_SUMMARY),
        "rag_unlearning_graph": summary.get("rag_unlearning_graph"),
        "memory_unlearning_graph": summary.get("memory_unlearning_graph"),
        "llm_activation_source": str(ACTIVATION_CSV),
        "llm_activation_npz": str(ACTIVATION_NPZ),
        "rag_memory_indexing": f"CSV-order equal-size binning into {RETRIEVAL_WIDTH} bins per split/method/level",
        "llm_indexing": "Transformer layer index 0-31 from the final activation-strength heatmap",
        "baseline_policy": "Vanilla/PURE/SBU are model-only baselines at RAG/Memory level, so their RAG/Memory flow uses the unfiltered retrieved flow; FlowMU uses real unlearned/blocked filtering.",
        "value_transform": "RAG baseline = mean(score * answer_support); RAG FlowMU = mean(score * answer_support * (1 - unlearned)); Memory baseline = mean(path score); Memory FlowMU = mean(path score * (1 - blocked)); LLM = final activation-strength display_score by layer.",
        "manual_point_editing": False,
        "mean_values": means,
        "retrieval_summary_metrics": {
            "forget": summary.get("forget"),
            "retain": summary.get("retain"),
        },
    }


def save_metadata(metadata: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / f"{STEM}_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_plot(csv_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--input-csv",
        str(csv_path),
        "--output-dir",
        str(OUTPUT_DIR),
        "--stem",
        STEM,
    ]
    subprocess.run(cmd, cwd=str(AGENT_ROOT), check=True)


def sync_final_figures() -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for subset in ("forget", "retain"):
        for suffix in ("png", "pdf", "svg"):
            src = OUTPUT_DIR / f"{STEM}_{subset}_cross_level_heatmap.{suffix}"
            if src.exists():
                dst = FIG_DIR / src.name
                shutil.copy2(src, dst)
                copied.append(dst)
    return copied


def main() -> None:
    if not RETRIEVAL_EVAL_CSV.exists():
        raise FileNotFoundError(RETRIEVAL_EVAL_CSV)
    if not ACTIVATION_CSV.exists():
        raise FileNotFoundError(ACTIVATION_CSV)

    records = [*load_retrieval_records(), *load_llm_records()]
    csv_path, json_path = save_records(records)
    metadata_path = save_metadata(build_metadata(records))
    run_plot(csv_path)
    copied_figs = sync_final_figures()

    print(f"Saved real records CSV: {csv_path}")
    print(f"Saved real records JSON: {json_path}")
    print(f"Saved metadata: {metadata_path}")
    for path in copied_figs:
        print(f"Copied final fig: {path}")
    print(f"Records: {len(records)}")


if __name__ == "__main__":
    main()
