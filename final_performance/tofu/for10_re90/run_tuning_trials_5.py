                      
                                                                            

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent")
PYTHON = Path("/home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python")
DATA = Path("/home/likunhao/likunhao52shared/datasets/TOFU")
BASE = ROOT / "final_performance/tofu/for10_re90"
LOG_BASE = ROOT / "logs/for10_re90/tuning_trials"
CACHE_DIR = BASE / "cache"
JSONL = BASE / "hyperparams/tuning_trials_5.jsonl"
BEST_JSON = BASE / "hyperparams/best_phase1_gat_tuning_trials_5.json"


TARGETS = {
    "rag_forget": 0.82,
    "rag_retain": 0.70,
    "memory_forget": 0.80,
    "memory_retain": 0.70,
}


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")


def write_skip_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if log_path.exists() else "w"
    with log_path.open(mode, encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def trial_score(metrics: dict[str, float]) -> float:
    return (
        min(metrics["rag_forget"], TARGETS["rag_forget"]) / TARGETS["rag_forget"]
        + min(metrics["rag_retain"], TARGETS["rag_retain"]) / TARGETS["rag_retain"]
        + min(metrics["memory_forget"], TARGETS["memory_forget"]) / TARGETS["memory_forget"]
        + min(metrics["memory_retain"], TARGETS["memory_retain"]) / TARGETS["memory_retain"]
    )


def next_params(prev: dict[str, float] | None, metrics: dict[str, float] | None) -> dict[str, float]:
    if prev is None or metrics is None:
        return {
            "rag_drop_class_weight_scale": 0.62,
            "rag_safe_loss_weight": 0.50,
            "rag_retrain_epochs": 35,
            "memory_background_relation_scale": 0.08,
            "memory_community_floor": 0.30,
            "memory_path_decay": 1.8,
            "memory_path_loss_weight": 0.50,
            "memory_safe_loss_weight": 0.18,
            "memory_toxic_loss_weight": 0.50,
            "memory_sparse_loss_weight": 0.025,
            "memory_consistency_loss_weight": 0.05,
            "memory_next_turn_loss_weight": 0.10,
            "memory_relation_loss_weight": 0.05,
            "memory_ranking_loss_weight": 0.20,
        }

    p = dict(prev)

                                                                               
                                                                               
    if metrics["rag_retain"] < 0.66:
        p["rag_drop_class_weight_scale"] = max(0.25, p["rag_drop_class_weight_scale"] - 0.12)
        p["rag_safe_loss_weight"] = min(0.90, p["rag_safe_loss_weight"] + 0.15)
    elif metrics["rag_retain"] < TARGETS["rag_retain"]:
        p["rag_drop_class_weight_scale"] = max(0.30, p["rag_drop_class_weight_scale"] - 0.06)
        p["rag_safe_loss_weight"] = min(0.80, p["rag_safe_loss_weight"] + 0.08)
    elif metrics["rag_forget"] < TARGETS["rag_forget"]:
        p["rag_drop_class_weight_scale"] = min(1.10, p["rag_drop_class_weight_scale"] + 0.08)
        p["rag_safe_loss_weight"] = max(0.15, p["rag_safe_loss_weight"] - 0.05)

    if metrics["rag_forget"] < 0.72 and metrics["rag_retain"] > 0.70:
        p["rag_drop_class_weight_scale"] = min(1.20, p["rag_drop_class_weight_scale"] + 0.12)

                                                                                
                                  
    if metrics["memory_forget"] < 0.72 and metrics["memory_retain"] > 0.78:
        p["memory_background_relation_scale"] = max(0.03, p["memory_background_relation_scale"] - 0.03)
        p["memory_community_floor"] = max(0.22, p["memory_community_floor"] - 0.04)
        p["memory_path_decay"] = min(2.2, p["memory_path_decay"] + 0.15)
        p["memory_path_loss_weight"] = min(0.65, p["memory_path_loss_weight"] + 0.08)
        p["memory_toxic_loss_weight"] = min(0.70, p["memory_toxic_loss_weight"] + 0.08)
        p["memory_safe_loss_weight"] = max(0.08, p["memory_safe_loss_weight"] - 0.04)
        p["memory_next_turn_loss_weight"] = min(0.20, p["memory_next_turn_loss_weight"] + 0.05)
        p["memory_relation_loss_weight"] = min(0.12, p["memory_relation_loss_weight"] + 0.03)
    elif metrics["memory_forget"] < TARGETS["memory_forget"] and metrics["memory_retain"] >= TARGETS["memory_retain"]:
        p["memory_background_relation_scale"] = max(0.04, p["memory_background_relation_scale"] - 0.02)
        p["memory_community_floor"] = max(0.24, p["memory_community_floor"] - 0.03)
        p["memory_path_loss_weight"] = min(0.60, p["memory_path_loss_weight"] + 0.05)
        p["memory_toxic_loss_weight"] = min(0.65, p["memory_toxic_loss_weight"] + 0.05)
    elif metrics["memory_retain"] < TARGETS["memory_retain"]:
        p["memory_background_relation_scale"] = min(0.18, p["memory_background_relation_scale"] + 0.03)
        p["memory_community_floor"] = min(0.38, p["memory_community_floor"] + 0.04)
        p["memory_safe_loss_weight"] = min(0.28, p["memory_safe_loss_weight"] + 0.04)
        p["memory_toxic_loss_weight"] = max(0.30, p["memory_toxic_loss_weight"] - 0.06)
        p["memory_path_loss_weight"] = max(0.35, p["memory_path_loss_weight"] - 0.05)

    p["rag_retrain_epochs"] = 35 if metrics["rag_forget"] < TARGETS["rag_forget"] else 30
    return p


def run_trial(trial_idx: int, params: dict[str, float]) -> dict[str, object]:
    run_name = f"trial_{trial_idx:02d}"
    phase1_dir = BASE / f"phase1_gat_{run_name}"
    eval_dir = BASE / f"retrieval_eval_{run_name}"
    log_dir = LOG_BASE / run_name
    query_cache = CACHE_DIR / "query_forget10_authorqa_qwen35_max128_mean_4bit.npz"
    question_cache = CACHE_DIR / "retrieval_forget10_retain90_questions_qwen35_max64_mean_4bit.npz"

    train_cmd = [
        str(PYTHON),
        "phase1/dual_graph_unlearning_gat/run_dual_graph_unlearning.py",
        "--rag-graph",
        "phase1/artifacts/rag/rag_graph.pkl",
        "--memory-graph",
        "phase1/artifacts/memory/generic_memory_graph.pkl",
        "--forget-queries-path",
        "phase1/author/forget10_author.json",
        "--forget-qa-query-path",
        str(DATA / "forget10.json"),
        "--output-dir",
        str(phase1_dir),
        "--skip-pretrain",
        "--rag-pretrain-model-path",
        "final_performance/tofu/for95_re05/phase1_gat/rag_pre_gat.pt",
        "--memory-pretrain-model-path",
        "final_performance/tofu/for95_re05/phase1_gat/memory_pre_gat.pt",
        "--query-embedding-cache",
        str(query_cache),
        "--embedding-batch-size",
        "4",
        "--rag-qa-topk",
        "3",
        "--rag-retrain-epochs",
        str(int(params["rag_retrain_epochs"])),
        "--rag-drop-class-weight-scale",
        str(params["rag_drop_class_weight_scale"]),
        "--rag-safe-loss-weight",
        str(params["rag_safe_loss_weight"]),
        "--memory-retrain-epochs",
        "30",
        "--memory-qa-topk",
        "5",
        "--memory-hard-threshold",
        "-1",
        "--memory-path-cutoff",
        "4",
        "--memory-mentioned-prior",
        "1.0",
        "--memory-relation-prior",
        "0.8",
        "--memory-next-turn-prior",
        "0.5",
        "--memory-background-relation-scale",
        str(params["memory_background_relation_scale"]),
        "--memory-community-floor",
        str(params["memory_community_floor"]),
        "--memory-path-decay",
        str(params["memory_path_decay"]),
        "--memory-path-loss-weight",
        str(params["memory_path_loss_weight"]),
        "--memory-safe-loss-weight",
        str(params["memory_safe_loss_weight"]),
        "--memory-toxic-loss-weight",
        str(params["memory_toxic_loss_weight"]),
        "--memory-sparse-loss-weight",
        str(params["memory_sparse_loss_weight"]),
        "--memory-consistency-loss-weight",
        str(params["memory_consistency_loss_weight"]),
        "--memory-next-turn-loss-weight",
        str(params["memory_next_turn_loss_weight"]),
        "--memory-relation-loss-weight",
        str(params["memory_relation_loss_weight"]),
        "--memory-ranking-loss-weight",
        str(params["memory_ranking_loss_weight"]),
    ]
    required_graph_outputs = [
        phase1_dir / "rag_unlearning_graph.pkl",
        phase1_dir / "memory_unlearning_graph.pkl",
        phase1_dir / "dual_graph_unlearning_report.json",
    ]
    if all(path.exists() for path in required_graph_outputs):
        write_skip_log(log_dir / "train.log", f"[skip] existing GAT outputs found in {phase1_dir}")
    else:
        run_command(train_cmd, log_dir / "train.log")

    eval_cmd = [
        str(PYTHON),
        "phase1/dual_graph_unlearning_gat/eval_unlearning_retrieval.py",
        "--rag-graph",
        "phase1/artifacts/rag/rag_graph.pkl",
        "--rag-unlearning-graph",
        str(phase1_dir / "rag_unlearning_graph.pkl"),
        "--memory-graph",
        "phase1/artifacts/memory/generic_memory_graph.pkl",
        "--memory-unlearning-graph",
        str(phase1_dir / "memory_unlearning_graph.pkl"),
        "--forget-path",
        str(DATA / "forget10.json"),
        "--retain-path",
        str(DATA / "retain90.json"),
        "--output-dir",
        str(eval_dir),
        "--forget-limit",
        "0",
        "--retain-limit",
        "0",
        "--rag-top-k",
        "3",
        "--memory-max-depth",
        "5",
        "--memory-max-seeds",
        "3",
        "--memory-max-hits",
        "3",
        "--memory-entity-sim-floor",
        "0.35",
        "--embedding-batch-size",
        "4",
        "--question-embedding-cache",
        str(question_cache),
        "--fast-memory-eval",
        "--memory-path-cache-dir",
        "final_performance/tofu/memory_path_cache",
        "--num-workers",
        "8",
        "--metrics-only",
    ]

    summary_path = eval_dir / "graph_path_eval_summary.json"
    if summary_path.exists():
        write_skip_log(log_dir / "eval.log", f"[skip] existing retrieval eval outputs found in {eval_dir}")
    else:
        run_command(eval_cmd, log_dir / "eval.log")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = {
        "rag_forget": float(summary["forget"]["rag_mask_hit_at_k"]),
        "rag_retain": float(summary["retain"]["rag_keep_hit_at_k"]),
        "memory_forget": float(summary["forget"]["memory_path_block_hit_at_k"]),
        "memory_retain": float(summary["retain"]["memory_path_safe_hit_at_k"]),
    }
    report = json.loads((phase1_dir / "dual_graph_unlearning_report.json").read_text(encoding="utf-8"))
    result = {
        "trial": trial_idx,
        "run_name": run_name,
        "time": time.strftime("%F %T"),
        "params": params,
        "metrics": metrics,
        "score": trial_score(metrics),
        "paths": {
            "phase1_dir": str(phase1_dir),
            "eval_dir": str(eval_dir),
            "summary": str(summary_path),
            "train_log": str(log_dir / "train.log"),
            "eval_log": str(log_dir / "eval.log"),
        },
        "training_summary": {
            "rag_unlearned_nodes": report["rag"].get("unlearned_count"),
            "memory_unlearned_edges": report["memory"].get("unlearned_edge_count"),
            "memory_unlearned_nodes": report["memory"].get("unlearned_node_count"),
            "rag_drop_threshold": report["rag"].get("drop_threshold"),
            "memory_punish_threshold": report["memory"].get("punish_threshold"),
        },
    }
    return result


def main() -> None:
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    LOG_BASE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    best: dict[str, object] | None = None
    prev_params: dict[str, float] | None = None
    prev_metrics: dict[str, float] | None = None

    with JSONL.open("a", encoding="utf-8") as jsonl:
        for trial_idx in range(1, 6):
            params = next_params(prev_params, prev_metrics)
            print(f"[trial {trial_idx}] params={json.dumps(params, sort_keys=True)}", flush=True)
            result = run_trial(trial_idx, params)
            jsonl.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            jsonl.flush()
            print(f"[trial {trial_idx}] metrics={json.dumps(result['metrics'], sort_keys=True)}", flush=True)
            print(f"[trial {trial_idx}] score={result['score']}", flush=True)
            if best is None or float(result["score"]) > float(best["score"]):
                best = result
                BEST_JSON.write_text(json.dumps(best, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                print(f"[trial {trial_idx}] new_best={BEST_JSON}", flush=True)
            prev_params = params
            prev_metrics = result["metrics"]                            

    print("[done] best=" + json.dumps(best, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
