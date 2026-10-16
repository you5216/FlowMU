                      
                                                                 

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


AGENT_DIR = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent")
PY = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python")
VLLM = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm")

BASE_MODEL = AGENT_DIR / "merge_model/native_llm"
ORIGINAL_LORA = Path("/disk8T/lkh/python_projects/tofu_native_lora/checkpoint-2000")
JUDGE_MODEL = Path("/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct")

LOCALIZATION_PKL = AGENT_DIR / "final_performance/tofu/for10_re90/phase2_localization_dualview_trial16_top100_qaw2/localized_neurons.pkl"
FORGET_QA = AGENT_DIR / "final_performance/tofu/for10_re90/phase2_qa_trial16/graph_forget_qa.jsonl"
RETAIN_QA = AGENT_DIR / "final_performance/tofu/for10_re90/phase2_qa_trial16/graph_retain_qa.jsonl"
OFFICIAL_FORGET = Path("/disk8T/lkh/python_projects/datasets/TOFU/forget10.json")
OFFICIAL_RETAIN = Path("/disk8T/lkh/python_projects/datasets/TOFU/retain90.json")


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"run: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=AGENT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        tail = tail_text(log_path)
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}\n{tail}")


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_openai(port: int, proc: subprocess.Popen[str], log_path: Path, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before ready on {port}\n{tail_text(log_path)}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"timeout waiting for vLLM on {port}\n{tail_text(log_path)}")


def start_vllm(model_path: Path, served_name: str, port: int, log_path: Path, max_model_len: int) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        str(VLLM),
        "serve",
        str(model_path),
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.90",
        "--enforce-eager",
    ]
    log(f"start vLLM: served={served_name} port={port}")
    f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=AGENT_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    wait_openai(port, proc, log_path)
    return proc


def stop_proc(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def validate_generation(csv_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split_counts: dict[str, int] = {}
    for row in rows:
        split = row.get("split", "")
        split_counts[split] = split_counts.get(split, 0) + 1
    if len(rows) != 4000 or split_counts.get("forget") != 400 or split_counts.get("retain") != 3600:
        raise RuntimeError(f"bad generation counts: total={len(rows)} split_counts={split_counts}")
    empty = sum(1 for row in rows if not row.get("prediction", "").strip())
    errors = sum(1 for row in rows if "[EVAL_ERROR]" in row.get("prediction", ""))
    think = sum(1 for row in rows if "<think>" in row.get("prediction", "").lower())
    if empty or errors or think:
        raise RuntimeError(f"bad generation content: empty={empty} errors={errors} think={think}")


def fix_tokenizer_config(merged_dir: Path) -> None:
    path = merged_dir / "tokenizer_config.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("tokenizer_class") == "TokenizersBackend":
        obj["tokenizer_class"] = "Qwen2Tokenizer"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--localization-pkl", default=str(LOCALIZATION_PKL))
    p.add_argument("--forget-ce", type=float, required=True)
    p.add_argument("--param-rv", type=float, required=True)
    p.add_argument("--retain-kl", type=float, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--early-stop-patience", type=int, default=120)
    p.add_argument("--best-score-retain-alpha", type=float, default=5.5)
    p.add_argument("--retain-topk", type=int, default=32)
    p.add_argument("--forget-weight-max", type=float, default=12.0)
    p.add_argument("--random-vector-target-similarity", type=float, default=0.25)
    p.add_argument("--forget-anchor-target-similarity", type=float, default=-0.1)
    p.add_argument("--model-port", type=int, default=18080)
    p.add_argument("--judge-port", type=int, default=18081)
    p.add_argument("--eval-workers", type=int, default=8)
    p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    label = args.label
    localization_pkl = Path(args.localization_pkl).expanduser()
    if not localization_pkl.is_absolute():
        localization_pkl = AGENT_DIR / localization_pkl
    if not localization_pkl.exists():
        raise FileNotFoundError(f"localization pkl not found: {localization_pkl}")
    trial_root = AGENT_DIR / "final_performance/tofu/for10_re90/phase2_lora_trials" / label
    lora_dir = trial_root / "lora"
    merge_dir = AGENT_DIR / "merge_model" / f"for10_re90_{label}"
    eval_dir = AGENT_DIR / "final_performance/tofu/for10_re90/param_eval" / label
    judge_dir = eval_dir / "judge_qwen25"
    metrics_dir = eval_dir / "final_metrics"
    log_dir = AGENT_DIR / "logs/for10_re90" / label
    metrics_json = metrics_dir / "final_metrics_summary.json"

    if metrics_json.exists() and not args.force:
        log(f"final metrics already exist: {metrics_json}")
        print(metrics_json.read_text(encoding="utf-8"))
        return 0

    for d in [trial_root, lora_dir, eval_dir, judge_dir, metrics_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_code = (
        "import sys\n"
        "from phase2.pipeline.config import parse_args, build_config\n"
        "from phase2.pipeline.runner import execute_parameter_unlearning\n"
        "cfg = build_config(parse_args(sys.argv[1:]))\n"
        "execute_parameter_unlearning(cfg)\n"
    )
    train_args = [
        "--base-model-path",
        str(BASE_MODEL),
        "--original-lora-path",
        str(ORIGINAL_LORA),
        "--output-dir",
        str(lora_dir.relative_to(AGENT_DIR)),
        "--forget-target-path",
        str(FORGET_QA.relative_to(AGENT_DIR)),
        "--retain-path",
        str(RETAIN_QA.relative_to(AGENT_DIR)),
        "--init-new-lora",
        "--new-lora-r",
        "16",
        "--new-lora-alpha",
        "32",
        "--new-lora-dropout",
        "0.05",
        "--new-lora-target-modules",
        "gate_proj,up_proj,down_proj",
        "--per-device-train-batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        "1",
        "--seed",
        "42",
        "--learning-rate",
        str(args.lr),
        "--num-train-epochs",
        str(args.epochs),
        "--logging-steps",
        "5",
        "--max-length",
        "512",
        "--warmup-steps",
        "5",
        "--max-grad-norm",
        "0.3",
        "--reuse-localization-pkl",
        str(localization_pkl.relative_to(AGENT_DIR) if localization_pkl.is_relative_to(AGENT_DIR) else localization_pkl),
        "--forget-loss-strategy",
        "parameter_random_vector",
        "--forget-objective-weight",
        "1.0",
        "--forget-target-mode",
        "random_vector",
        "--forget-ce-weight",
        str(args.forget_ce),
        "--random-vector-forget-weight",
        "1.0",
        "--forget-anchor-weight",
        "1.0",
        "--forget-anchor-max-similarity",
        "0.0",
        "--random-vector-target-similarity",
        str(args.random_vector_target_similarity),
        "--forget-anchor-target-similarity",
        str(args.forget_anchor_target_similarity),
        "--forget-weight-max",
        str(args.forget_weight_max),
        "--random-vector-seed",
        "42",
        "--parameter-random-vector-weight",
        str(args.param_rv),
        "--parameter-random-vector-scale",
        "0.01",
        "--forget-loss-top-layers",
        "32",
        "--graph-vector-suppress-weight",
        "0.0",
        "--retain-ce-weight",
        "0.0",
        "--retain-kl-weight",
        str(args.retain_kl),
        "--disable-dual-kl-objective",
        "--retain-kl-target",
        "0.02",
        "--retain-topk",
        str(args.retain_topk),
        "--forget-nll-calibration-weight",
        "0.0",
        "--early-stop-min-steps",
        "200",
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--best-score-retain-alpha",
        str(args.best_score_retain_alpha),
        "--best-score-mia-alpha",
        "0.0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    run_cmd([str(PY), "-c", train_code, *train_args], log_dir / "train.log", env=env)

    run_cmd(
        [
            str(PY),
            "-m",
            "phase2.pipeline.merge_lora_for_vllm",
            "--base-model-path",
            str(BASE_MODEL),
            "--lora-path",
            str(lora_dir.relative_to(AGENT_DIR)),
            "--output-dir",
            str(merge_dir.relative_to(AGENT_DIR)),
            "--torch-dtype",
            "float16",
            "--device-map",
            "cpu",
            "--label",
            label,
            "--force",
        ],
        log_dir / "merge.log",
    )
    fix_tokenizer_config(merge_dir)

    model_proc: subprocess.Popen[str] | None = None
    judge_proc: subprocess.Popen[str] | None = None
    try:
        model_proc = start_vllm(merge_dir, "merged", args.model_port, log_dir / "vllm_eval.log", 2048)
        eval_env = os.environ.copy()
        eval_env.update(
            {
                "QWEN_INFER_BACKEND": "vllm_api",
                "VLLM_API_BASE": f"http://127.0.0.1:{args.model_port}/v1",
                "VLLM_MODEL_NAME": "merged",
                "QWEN_EVAL_NUM_WORKERS": str(args.eval_workers),
            }
        )
        run_cmd(
            [
                str(PY),
                "-m",
                "phase2.pipeline.parametric_eval",
                "--lora-path",
                str(lora_dir.relative_to(AGENT_DIR)),
                "--forget-path",
                str(OFFICIAL_FORGET),
                "--retain-path",
                str(OFFICIAL_RETAIN),
                "--forget-limit",
                "400",
                "--retain-limit",
                "3600",
                "--output-csv",
                str(eval_dir.relative_to(AGENT_DIR) / "results.csv"),
                "--output-json",
                str(eval_dir.relative_to(AGENT_DIR) / "results.json"),
                "--backend",
                "vllm_api",
                "--max-new-tokens",
                "128",
                "--temperature",
                "0.0",
                "--num-workers",
                str(args.eval_workers),
                "--no-resume",
            ],
            log_dir / "eval.log",
            env=eval_env,
        )
        validate_generation(eval_dir / "results.csv")
        stop_proc(model_proc)
        model_proc = None

        judge_proc = start_vllm(JUDGE_MODEL, "judge", args.judge_port, log_dir / "vllm_judge.log", 4096)
        run_cmd(
            [
                str(PY),
                "-m",
                "evaluation.llm_judge",
                "--files",
                str(eval_dir.relative_to(AGENT_DIR) / "results.csv"),
                "--labels",
                label,
                "--api-url",
                f"http://127.0.0.1:{args.judge_port}/v1",
                "--model",
                "judge",
                "--num-workers",
                str(args.judge_workers),
                "--output-dir",
                str(judge_dir.relative_to(AGENT_DIR)),
            ],
            log_dir / "judge.log",
        )
        stop_proc(judge_proc)
        judge_proc = None

        judge_csv = judge_dir / f"judge_{label}.csv"
        run_cmd(
            [
                str(PY),
                "-m",
                "evaluation.final_metrics",
                "--files",
                str(judge_csv.relative_to(AGENT_DIR)),
                "--labels",
                label,
                "--output-dir",
                str(metrics_dir.relative_to(AGENT_DIR)),
            ],
            log_dir / "final_metrics.log",
        )
    finally:
        stop_proc(model_proc)
        stop_proc(judge_proc)

    print(metrics_json.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
