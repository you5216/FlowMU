                      
                                                           

                                                                               
                                                                        
   

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/disk8T/lkh/python_projects")
BASELINE_ROOT = ROOT / "baseline_xzy(2)"
AGENT_ROOT = ROOT / "agent_unlearning_zhx/agent"
MY_SBU_ROOT = ROOT / "my_sbu_plansv2.0"
ENV_PY = ROOT / "agent_unlearning_zhx/environment/env52/bin/python"
VLLM = ROOT / "agent_unlearning_zhx/environment/52vllm/bin/vllm"
LQA_ROOT = ROOT / "datasets/LiteraryQA/literaryqa_annotations"
JUDGE_MODEL = ROOT / "llms/Qwen2.5-VL-7B-Instruct"
NATIVE_LLM = AGENT_ROOT / "final_performance/literaryQA/base/lora/merged/literary_neative_model"

OUT_ROOT = BASELINE_ROOT / "experiments/literaryqa/final_pure_sbu_20260706"
LOG_ROOT = OUT_ROOT / "logs"
EMPTY_RETAIN_JSON = OUT_ROOT / "empty_retain.json"
MODEL_PORT = 8100
JUDGE_PORT = 8101

RATIO_CFG = {
    "01": {
        "forget": LQA_ROOT / "memory/forget01.json",
        "retain": LQA_ROOT / "memory/retain99.json",
        "fl": 204,
        "rl": 18857,
        "pure_model": AGENT_ROOT / "final_performance/literaryQA/for01_re99/merged_model/ga_pure_weak/literaryQA_for01_re99_ga_ce20_weakpure_20260703",
    },
    "05": {
        "forget": LQA_ROOT / "memory/forget05.json",
        "retain": LQA_ROOT / "memory/retain95.json",
        "fl": 992,
        "rl": 18069,
        "pure_model": AGENT_ROOT / "final_performance/literaryQA/for05_re95/merged_model/ga_pure_weak/literaryQA_for05_re95_ga_ce25_weakpure_20260703",
    },
    "10": {
        "forget": LQA_ROOT / "memory/forget10.json",
        "retain": LQA_ROOT / "memory/retain90.json",
        "fl": 1970,
        "rl": 17091,
        "pure_model": AGENT_ROOT / "final_performance/literaryQA/for10_re90/merged_model/ga_pure_weak/literaryQA_for10_re90_ga_ce10_weakpure_20260703",
    },
}

SBU_SCREEN_STEPS = [100, 200, 300, 400, 500]


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, stdout: Path | None = None) -> None:
    log("$ " + " ".join(str(x) for x in cmd))
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("a", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def check_paths() -> None:
    for path in [BASELINE_ROOT, AGENT_ROOT, MY_SBU_ROOT, ENV_PY, VLLM, LQA_ROOT, JUDGE_MODEL, NATIVE_LLM]:
        if not path.exists():
            raise FileNotFoundError(path)
    for ratio, cfg in RATIO_CFG.items():
        for key in ["forget", "retain", "pure_model"]:
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"ratio={ratio} {key}: {cfg[key]}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EMPTY_RETAIN_JSON.exists():
        EMPTY_RETAIN_JSON.write_text("[]\n", encoding="utf-8")


def kill_vllm() -> None:
    patterns = ["vllm serve", "VLLM::EngineCore"]
    for pat in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    for pat in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


def wait_server(port: int, log_file: Path, timeout: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout:
        proc = subprocess.run(
            ["curl", "-fsS", f"http://127.0.0.1:{port}/v1/models"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            log(f"vLLM ready port={port}")
            return
        if not any("vllm" in p for p in subprocess.run(["pgrep", "-af", "vllm"], capture_output=True, text=True).stdout.splitlines()):
            if log_file.exists():
                log(log_file.read_text(errors="ignore")[-4000:])
            raise RuntimeError(f"vLLM exited before ready on port {port}")
        time.sleep(5)
    if log_file.exists():
        log(log_file.read_text(errors="ignore")[-4000:])
    raise TimeoutError(f"vLLM not ready on port {port}")


def start_vllm(model_path: Path, *, port: int, served_name: str, max_num_seqs: int, log_file: Path) -> subprocess.Popen:
    kill_vllm()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VLLM),
        "serve",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
        "--trust-remote-code",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("GPU_ID", "0")
    f = log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env, preexec_fn=os.setsid)
    try:
        wait_server(port, log_file)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        f.close()
        raise
    return proc


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        kill_vllm()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    time.sleep(5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
    kill_vllm()


def row_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def run_judge_and_metrics(cell_dir: Path, label: str, *, num_workers: int = 16) -> None:
    judge_csv = cell_dir / "judge_qwen25" / f"judge_{label}.csv"
    metrics_json = cell_dir / "final_metrics/final_metrics_summary.json"
    if judge_csv.exists() and metrics_json.exists() and row_count(judge_csv) == row_count(cell_dir / "results.csv"):
        log(f"judge/metrics already complete {label}")
        return
    proc = None
    try:
        proc = start_vllm(JUDGE_MODEL, port=JUDGE_PORT, served_name="judge", max_num_seqs=num_workers, log_file=LOG_ROOT / f"{label}_judge_vllm.log")
        run(
            [
                str(ENV_PY),
                "evaluation/llm_judge.py",
                "--files",
                str(cell_dir / "results.csv"),
                "--labels",
                label,
                "--api-url",
                f"http://127.0.0.1:{JUDGE_PORT}/v1",
                "--model",
                "judge",
                "--output-dir",
                str(cell_dir / "judge_qwen25"),
                "--num-workers",
                str(num_workers),
            ],
            cwd=BASELINE_ROOT / "agent",
            stdout=LOG_ROOT / f"{label}_judge.log",
        )
    finally:
        stop_proc(proc)
    run(
        [
            str(ENV_PY),
            "evaluation/final_metrics.py",
            "--files",
            str(judge_csv),
            "--labels",
            label,
            "--output-dir",
            str(cell_dir / "final_metrics"),
        ],
        cwd=BASELINE_ROOT / "agent",
        stdout=LOG_ROOT / f"{label}_metrics.log",
    )


def run_naive_step6(
    *,
    ratio: str,
    model_path: Path,
    cell_dir: Path,
    label: str,
    forget_limit: int,
    retain_limit: int,
    empty_forget_context: bool,
    require_complete: int,
    num_workers: int,
) -> None:
    if row_count(cell_dir / "results.csv") >= require_complete:
        log(f"step6 already complete {label}")
        return
    cfg = RATIO_CFG[ratio]
    assets_dir = BASELINE_ROOT / f"experiments/literaryqa/naive_faiss/pure_for{ratio}"
    proc = None
    try:
        proc = start_vllm(model_path, port=MODEL_PORT, served_name="baseline_llm", max_num_seqs=num_workers, log_file=LOG_ROOT / f"{label}_vllm.log")
        env = os.environ.copy()
        env.update(
            {
                "VLLM_API_BASE": f"http://127.0.0.1:{MODEL_PORT}/v1",
                "VLLM_MODEL_NAME": "baseline_llm",
                "QWEN_INFER_BACKEND": "vllm_api",
                "MY_SBU_ROOT": str(MY_SBU_ROOT),
            }
        )
        cmd = [
            str(ENV_PY),
            "-m",
            "phase2.pipeline.step6_naive_faiss_eval",
            "--assets-dir",
            str(assets_dir),
            "--base-model-path",
            str(model_path),
            "--lora-path",
            str(model_path),
            "--forget-path",
            str(cfg["forget"]),
            "--retain-path",
            str(cfg["retain"]),
            "--forget-limit",
            str(forget_limit),
            "--retain-limit",
            str(retain_limit),
            "--output-csv",
            str(cell_dir / "results.csv"),
            "--output-json",
            str(cell_dir / "results.json"),
            "--backend",
            "vllm_api",
            "--num-workers",
            str(num_workers),
            "--max-new-tokens",
            "128",
            "--rag-top-k",
            "8",
            "--memory-top-k",
            "4",
            "--embedding-device",
            "cpu",
            "--checkpoint-every",
            "100",
            "--no-resume",
        ]
        if empty_forget_context:
            cmd.append("--empty-forget-context")
        run(cmd, cwd=BASELINE_ROOT / "agent", env=env, stdout=LOG_ROOT / f"{label}_step6.log")
    finally:
        stop_proc(proc)


def combine_pure_full(ratio: str, forget_cell: Path, full_cell: Path) -> None:
    old_results = BASELINE_ROOT / f"experiments/literaryqa/baselines_tofu/pure_for{ratio}/results.csv"
    if not old_results.exists():
        raise FileNotFoundError(old_results)
    rows: list[dict[str, str]] = []
    with (forget_cell / "results.csv").open(newline="", encoding="utf-8") as f:
        rows.extend([row for row in csv.DictReader(f) if row.get("split") == "forget"])
    with old_results.open(newline="", encoding="utf-8") as f:
        rows.extend([row for row in csv.DictReader(f) if row.get("split") == "retain"])
    full_cell.mkdir(parents=True, exist_ok=True)
    out_csv = full_cell / "results.csv"
    fields = [
        "split",
        "question",
        "reference_answer",
        "prediction",
        "rag_context",
        "memory_context",
        "lora_path",
        "sfnl_prompt_mode",
        "unlearning_request",
        "memory_policy",
        "rag_policy",
        "model_path",
        "assets_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    (full_cell / "results.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "method_name": "pure_empty_forget_context_plus_existing_retain",
                "ratio": ratio,
                "forget_results_csv": str(forget_cell / "results.csv"),
                "retain_results_csv": str(old_results),
                "num_rows": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def filter_results_to_split(cell_dir: Path, split: str) -> None:
    csv_path = cell_dir / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("split") == split]
    if not rows:
        raise RuntimeError(f"no {split} rows in {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = cell_dir / "results.json"
    json_path.write_text(
        json.dumps({"format_version": 1, "filtered_split": split, "num_rows": len(rows), "rows": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def combine_pure_judge(ratio: str, forget_cell: Path, full_cell: Path, forget_label: str) -> None:
    forget_judge = forget_cell / "judge_qwen25" / f"judge_{forget_label}.csv"
    old_judge = BASELINE_ROOT / f"experiments/literaryqa/baselines_tofu/pure_for{ratio}/judge_qwen25/judge_pure_for{ratio}.csv"
    if not forget_judge.exists():
        raise FileNotFoundError(forget_judge)
    if not old_judge.exists():
        raise FileNotFoundError(old_judge)
    rows: list[dict[str, str]] = []
    with forget_judge.open(newline="", encoding="utf-8") as f:
        rows.extend([row for row in csv.DictReader(f) if row.get("split") == "forget"])
    with old_judge.open(newline="", encoding="utf-8") as f:
        rows.extend([row for row in csv.DictReader(f) if row.get("split") == "retain"])
    out_dir = full_cell / "judge_qwen25"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"judge_pure_for{ratio}.csv"
    preferred = [
        "split",
        "question",
        "reference_answer",
        "prediction",
        "rag_context",
        "memory_context",
        "lora_path",
        "sfnl_prompt_mode",
        "unlearning_request",
        "memory_policy",
        "rag_policy",
        "model_path",
        "assets_dir",
        "judge_verdict",
        "judge_raw_response",
        "extracted_answer",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    run(
        [
            str(ENV_PY),
            "evaluation/final_metrics.py",
            "--files",
            str(out_csv),
            "--labels",
            f"pure_for{ratio}",
            "--output-dir",
            str(full_cell / "final_metrics"),
        ],
        cwd=BASELINE_ROOT / "agent",
        stdout=LOG_ROOT / f"pure_for{ratio}_metrics.log",
    )


def run_pure(args: argparse.Namespace) -> None:
    for ratio, cfg in RATIO_CFG.items():
        label = f"pure_for{ratio}_empty_forget_context"
        forget_cell = OUT_ROOT / "pure_forget_only" / f"pure_for{ratio}"
        full_cell = OUT_ROOT / "pure_final" / f"pure_for{ratio}"
        run_naive_step6(
            ratio=ratio,
            model_path=Path(cfg["pure_model"]),
            cell_dir=forget_cell,
            label=label,
            forget_limit=cfg["fl"],
            retain_limit=1,
            empty_forget_context=True,
            require_complete=cfg["fl"],
            num_workers=args.num_workers,
        )
        filter_results_to_split(forget_cell, "forget")
        combine_pure_full(ratio, forget_cell, full_cell)
        run_judge_and_metrics(forget_cell, label, num_workers=args.judge_workers)
        combine_pure_judge(ratio, forget_cell, full_cell, label)


def merge_sbu_checkpoint(ratio: str, step: int) -> Path:
    ckpt = BASELINE_ROOT / f"experiments/literaryqa/training/sbu_for{ratio}_lora/checkpoint-{step}"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    merged = OUT_ROOT / "sbu_merged_ckpt" / f"sbu_for{ratio}_ckpt{step}"
    if (merged / "config.json").exists():
        return merged
    run(
        [
            str(ENV_PY),
            str(BASELINE_ROOT / "scripts/merge_lora_literaryqa.py"),
            "--base-model-path",
            str(NATIVE_LLM),
            "--lora-path",
            str(ckpt),
            "--output-dir",
            str(merged),
            "--torch-dtype",
            "float16",
        ],
        cwd=BASELINE_ROOT,
        stdout=LOG_ROOT / f"sbu_for{ratio}_ckpt{step}_merge.log",
    )
    return merged


def run_sbu_step6(
    *,
    ratio: str,
    model_path: Path,
    cell_dir: Path,
    label: str,
    forget_limit: int,
    retain_path: Path,
    retain_limit: int,
    require_complete: int,
    num_workers: int,
) -> None:
    if row_count(cell_dir / "results.csv") >= require_complete:
        log(f"step6 already complete {label}")
        return
    cfg = RATIO_CFG[ratio]
    phase_a = BASELINE_ROOT / f"experiments/literaryqa/sbu_phase_a/sbu_for{ratio}"
    proc = None
    try:
        proc = start_vllm(model_path, port=MODEL_PORT, served_name="baseline_llm", max_num_seqs=num_workers, log_file=LOG_ROOT / f"{label}_vllm.log")
        env = os.environ.copy()
        env.update(
            {
                "VLLM_API_BASE": f"http://127.0.0.1:{MODEL_PORT}/v1",
                "VLLM_MODEL_NAME": "baseline_llm",
                "QWEN_INFER_BACKEND": "vllm_api",
                "MY_SBU_ROOT": str(MY_SBU_ROOT),
            }
        )
        run(
            [
                str(ENV_PY),
                "-m",
                "phase2.pipeline.step6_sbu_eval",
                "--base-model-path",
                str(NATIVE_LLM),
                "--lora-path",
                str(model_path),
                "--forget-path",
                str(cfg["forget"]),
                "--retain-path",
                str(retain_path),
                "--forget-limit",
                str(forget_limit),
                "--retain-limit",
                str(retain_limit),
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
                "--checkpoint-every",
                "100",
                "--no-resume",
            ],
            cwd=BASELINE_ROOT / "agent",
            env=env,
            stdout=LOG_ROOT / f"{label}_step6.log",
        )
    finally:
        stop_proc(proc)


def screen_sbu_ratio(ratio: str, args: argparse.Namespace) -> tuple[int, dict]:
    summary_path = OUT_ROOT / "sbu_screening_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    cfg = RATIO_CFG[ratio]
    best: tuple[int, dict] | None = None
    for step in SBU_SCREEN_STEPS:
        key = f"for{ratio}_ckpt{step}"
        if key in summary and (OUT_ROOT / "sbu_screen" / key / "final_metrics/final_metrics_summary.json").exists():
            metrics = summary[key]
        else:
            model_path = merge_sbu_checkpoint(ratio, step)
            cell_dir = OUT_ROOT / "sbu_screen" / key
            label = f"sbu_for{ratio}_ckpt{step}_forget_screen"
            run_sbu_step6(
                ratio=ratio,
                model_path=model_path,
                cell_dir=cell_dir,
                label=label,
                forget_limit=cfg["fl"],
                retain_path=EMPTY_RETAIN_JSON,
                retain_limit=1,
                require_complete=cfg["fl"],
                num_workers=args.num_workers,
            )
                                                                        
            rows = []
            with (cell_dir / "results.csv").open(newline="", encoding="utf-8") as f:
                rows = [row for row in csv.DictReader(f) if row.get("split") == "forget"]
            fields = list(rows[0].keys()) if rows else []
            with (cell_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            run_judge_and_metrics(cell_dir, label, num_workers=args.judge_workers)
            metrics = load_metrics(cell_dir / "final_metrics/final_metrics_summary.json")
            summary[key] = {
                "ratio": ratio,
                "step": step,
                "model_path": str(model_path),
                "forget_acc": metrics["forget_acc"],
                "retain_acc": metrics["retain_acc"],
                "forget_rouge_l": metrics["forget_rouge_l"],
                "retain_rouge_l": metrics["retain_rouge_l"],
            }
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"SBU screen ratio={ratio} step={step} F-ACC={metrics['forget_acc']:.4f}")
        if best is None or abs(metrics["forget_acc"] - 0.45) < abs(best[1]["forget_acc"] - 0.45):
            best = (step, metrics)
        if metrics["forget_acc"] >= 0.45:
            return step, metrics
    assert best is not None
    return best


def run_sbu(args: argparse.Namespace) -> None:
    chosen: dict[str, dict] = {}
    for ratio, cfg in RATIO_CFG.items():
        step, screen_metrics = screen_sbu_ratio(ratio, args)
        model_path = merge_sbu_checkpoint(ratio, step)
        chosen[ratio] = {"step": step, "screen": screen_metrics, "model_path": str(model_path)}
        label = f"sbu_for{ratio}_ckpt{step}"
        cell_dir = OUT_ROOT / "sbu_final" / f"sbu_for{ratio}"
        run_sbu_step6(
            ratio=ratio,
            model_path=model_path,
            cell_dir=cell_dir,
            label=label,
            forget_limit=cfg["fl"],
            retain_path=cfg["retain"],
            retain_limit=cfg["rl"],
            require_complete=cfg["fl"] + cfg["rl"],
            num_workers=args.num_workers,
        )
        run_judge_and_metrics(cell_dir, f"sbu_for{ratio}", num_workers=args.judge_workers)
    (OUT_ROOT / "sbu_chosen_checkpoints.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_final_table() -> None:
    rows = []
    for method in ["pure", "sbu"]:
        root = OUT_ROOT / f"{method}_final"
        for ratio in ["01", "05", "10"]:
            cell = root / f"{method}_for{ratio}"
            metrics_path = cell / "final_metrics/final_metrics_summary.json"
            if not metrics_path.exists():
                continue
            m = load_metrics(metrics_path)
            rows.append(
                {
                    "method": method.upper(),
                    "forget_ratio": ratio,
                    "forget_acc": m["forget_acc"],
                    "retain_acc": m["retain_acc"],
                    "forget_rouge_l": m["forget_rouge_l"],
                    "retain_rouge_l": m["retain_rouge_l"],
                    "num_total": m["num_total"],
                    "metrics_path": str(metrics_path),
                }
            )
    out_csv = OUT_ROOT / "final_six_rows_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT_ROOT / "final_six_rows_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"saved final table -> {out_csv}")
    for row in rows:
        log(
            f"{row['method']} @{row['forget_ratio']}: "
            f"F-ACC={row['forget_acc']:.4f} R-ACC={row['retain_acc']:.4f} "
            f"F-RL={row['forget_rouge_l']:.4f} R-RL={row['retain_rouge_l']:.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pure", "sbu", "all", "table"], default="all")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=16)
    args = ap.parse_args()
    check_paths()
    if args.stage in {"pure", "all"}:
        run_pure(args)
    if args.stage in {"sbu", "all"}:
        run_sbu(args)
    write_final_table()


if __name__ == "__main__":
    main()
