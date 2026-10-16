                      
                                                                        

                                                                       
                                                                          
                                                      

                                                                           
                                                                             
   

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "run_lqa_pure_sbu_six_rows_20260706.py"
spec = importlib.util.spec_from_file_location("lqa_six", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_SCRIPT}")
lqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lqa)


ROOT = lqa.ROOT
BASE_MODEL = ROOT / "llms/Qwen3.5-9B"
ORIGIN_LORA = (
    lqa.AGENT_ROOT
    / "final_performance/literaryQA/base/lora/literaryqa_sft_lora_qwen35_9b_e5_ckpt596/checkpoint-2980"
)
MAX_STEPS = 350
SAVE_STEPS = 50
RUN_ROOT = lqa.OUT_ROOT / "sbu_lam001_agentrag_uniform_step350"
TRAIN_ROOT = RUN_ROOT / "training"
MERGED_ROOT = RUN_ROOT / "merged"
FINAL_ROOT = RUN_ROOT / "final"
LOG_ROOT = RUN_ROOT / "logs"


def _json_status_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "done"
    except Exception:
        return False


def train_ratio(ratio: str, *, force: bool = False) -> Path:
    cfg = lqa.RATIO_CFG[ratio]
    train_dir = TRAIN_ROOT / f"sbu_for{ratio}_lam001_uniform_step350_lora"
    if _json_status_done(train_dir / "sbu_train_metadata.json") and not force:
        lqa.log(f"train already done ratio={ratio}: {train_dir}")
        return train_dir
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        str(lqa.ENV_PY),
        "-m",
        "sbu.train_sbu_lora_tofu",
        "--forget-kl-target",
        "uniform",
        "--lambda-f",
        "0.01",
        "--temperature",
        "1.0",
        "--retain-weight",
        "1.0",
        "--output-dir",
        str(train_dir),
        "--base-model",
        str(BASE_MODEL),
        "--origin-lora",
        str(ORIGIN_LORA),
        "--forget-json",
        str(cfg["forget"]),
        "--retain-json",
        str(cfg["retain"]),
        "--max-steps",
        str(MAX_STEPS),
        "--save-steps",
        str(SAVE_STEPS),
        "--learning-rate",
        "1e-5",
    ]
    lqa.run(cmd, cwd=lqa.MY_SBU_ROOT, env=env, stdout=LOG_ROOT / f"sbu_for{ratio}_lam001_train.log")
    return train_dir


def merge_ratio(ratio: str, train_dir: Path, *, force: bool = False) -> Path:
    merged = MERGED_ROOT / f"sbu_for{ratio}_lam001_uniform_step350"
    if (merged / "config.json").exists() and not force:
        lqa.log(f"merge already exists ratio={ratio}: {merged}")
        return merged
    ckpt = train_dir / f"checkpoint-{MAX_STEPS}"
    lora_path = ckpt if ckpt.exists() else train_dir
    lqa.run(
        [
            str(lqa.ENV_PY),
            str(lqa.BASELINE_ROOT / "scripts/merge_lora_literaryqa.py"),
            "--base-model-path",
            str(lqa.NATIVE_LLM),
            "--lora-path",
            str(lora_path),
            "--output-dir",
            str(merged),
            "--torch-dtype",
            "float16",
        ],
        cwd=lqa.BASELINE_ROOT,
        stdout=LOG_ROOT / f"sbu_for{ratio}_lam001_merge.log",
    )
    return merged


def run_sbu_agent_eval(
    *,
    ratio: str,
    model_path: Path,
    cell_dir: Path,
    label: str,
    num_workers: int,
) -> None:
    cfg = lqa.RATIO_CFG[ratio]
    if lqa.row_count(cell_dir / "results.csv") >= cfg["fl"] + cfg["rl"]:
        lqa.log(f"step6 already complete {label}")
        return
    phase_a = lqa.BASELINE_ROOT / f"experiments/literaryqa/sbu_phase_a/sbu_for{ratio}"
    proc = None
    try:
        proc = lqa.start_vllm(
            model_path,
            port=lqa.MODEL_PORT,
            served_name="baseline_llm",
            max_num_seqs=num_workers,
            log_file=LOG_ROOT / f"{label}_vllm.log",
        )
        env = os.environ.copy()
        env.update(
            {
                "VLLM_API_BASE": f"http://127.0.0.1:{lqa.MODEL_PORT}/v1",
                "VLLM_MODEL_NAME": "baseline_llm",
                "QWEN_INFER_BACKEND": "vllm_api",
                "MY_SBU_ROOT": str(lqa.MY_SBU_ROOT),
            }
        )
        lqa.run(
            [
                str(lqa.ENV_PY),
                "-m",
                "phase2.pipeline.step6_sbu_eval",
                "--base-model-path",
                str(lqa.NATIVE_LLM),
                "--lora-path",
                str(model_path),
                "--forget-path",
                str(cfg["forget"]),
                "--retain-path",
                str(cfg["retain"]),
                "--forget-limit",
                str(cfg["fl"]),
                "--retain-limit",
                str(cfg["rl"]),
                "--phase-a-assets-dir",
                str(phase_a),
                "--output-csv",
                str(cell_dir / "results.csv"),
                "--output-json",
                str(cell_dir / "results.json"),
                "--backend",
                "vllm_api",
                "--num-workers",
                str(num_workers),
                "--memory-top-k",
                "5",
                "--id-prefix",
                "lqa",
                "--rag-mode",
                "clean_memory",
                "--checkpoint-every",
                "100",
                "--no-resume",
            ],
            cwd=lqa.BASELINE_ROOT / "agent",
            env=env,
            stdout=LOG_ROOT / f"{label}_step6.log",
        )
    finally:
        lqa.stop_proc(proc)


def run_all(args: argparse.Namespace) -> None:
    for path in [RUN_ROOT, TRAIN_ROOT, MERGED_ROOT, FINAL_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    for ratio in ["01", "05", "10"]:
        train_dir = train_ratio(ratio, force=args.force_train)
        merged = merge_ratio(ratio, train_dir, force=args.force_merge)
        cell = FINAL_ROOT / f"sbu_for{ratio}_lam001_uniform_step350_agentrag"
        label = f"sbu_for{ratio}_lam001_uniform_step350_agentrag"
        run_sbu_agent_eval(ratio=ratio, model_path=merged, cell_dir=cell, label=label, num_workers=args.num_workers)
        lqa.run_judge_and_metrics(cell, label, num_workers=args.judge_workers)
    write_table()


def write_table() -> None:
    rows = []
    for ratio in ["01", "05", "10"]:
        path = FINAL_ROOT / f"sbu_for{ratio}_lam001_uniform_step350_agentrag/final_metrics/final_metrics_summary.json"
        if not path.exists():
            continue
        m = lqa.load_metrics(path)
        rows.append(
            {
                "method": "SBU-AgentRAG-lam001-uniformKL-step350",
                "forget_ratio": ratio,
                "forget_acc": m["forget_acc"],
                "retain_acc": m["retain_acc"],
                "forget_rouge_l": m["forget_rouge_l"],
                "retain_rouge_l": m["retain_rouge_l"],
                "metrics_path": str(path),
            }
        )
    out = RUN_ROOT / "sbu_lam001_agentrag_step350_metrics.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lqa.log(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "all", "table"], default="all")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=16)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-merge", action="store_true")
    args = ap.parse_args()
    lqa.check_paths()
    if args.stage == "table":
        write_table()
    elif args.stage == "train":
        for path in [RUN_ROOT, TRAIN_ROOT, MERGED_ROOT, FINAL_ROOT, LOG_ROOT]:
            path.mkdir(parents=True, exist_ok=True)
        for ratio in ["01", "05", "10"]:
            train_ratio(ratio, force=args.force_train)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
