                      
                                            

                                                                               
                                                                          
                                                               
   

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT.parent


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step([0-9]+)", path.name)
    return int(match.group(1)) if match else 0


def run_cmd(cmd: list[str], *, log_path: Path, cwd: Path, execute: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(q(part) for part in cmd)
    print(rendered, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n")
    if not execute:
        return

    start = time.time()
    max_gpu_memory_mib = 0
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in out.splitlines():
                    line = line.strip()
                    if line:
                        max_gpu_memory_mib = max(max_gpu_memory_mib, int(float(line)))
            except Exception:
                pass
            time.sleep(2)
        code = int(proc.returncode or 0)
        handle.write(f"\nEXIT_CODE={code}\nELAPSED_SEC={time.time() - start:.3f}\n")
        handle.write(f"MAX_GPU_MEMORY_MIB={max_gpu_memory_mib}\n")
    if code != 0:
        raise SystemExit(code)


def run_shell(script: str, *, log_path: Path, cwd: Path, execute: bool = True) -> None:
    run_cmd(["bash", "-lc", script], log_path=log_path, cwd=cwd, execute=execute)


def round_paths(round_idx: int) -> dict[str, Path]:
    base = ROOT / "artifacts" / "incremental" / "ga_ce10" / f"round_{round_idx:02d}"
    return {
        "base": base,
        "phase1": ROOT / "artifacts" / "incremental" / "phase1" / f"round_{round_idx:02d}",
        "phase2_qa": base / "phase2_qa",
        "localization": base / "localization",
        "localization_pkl": base / "localization" / "localized_neurons.pkl",
        "lora": base / "lora",
        "merged": base / "merged_model",
        "eval": base / "eval",
        "judge": base / "judge",
        "logs": base / "logs",
    }


def ce10_checkpoint(lora_dir: Path) -> Path:
    root = lora_dir / "ce_threshold_checkpoints"
    candidates = []
    if root.exists():
        for item in root.iterdir():
            if item.is_dir() and re.fullmatch(r"ce10_step[0-9]+", item.name):
                candidates.append(item)
    if not candidates:
        raise FileNotFoundError(f"CE10 checkpoint not found under {root}")
    return sorted(candidates, key=checkpoint_step)[-1]


def make_adapter_only_resume(prev_ce10: Path, dst_root: Path) -> Path:
    step = checkpoint_step(prev_ce10)
    dst = dst_root / f"resume_adapter_only_step{step}"
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"training_state.pt", "threshold_metadata.json"}}

    shutil.copytree(prev_ce10, dst, ignore=ignore)
    return dst


def run_round(
    *,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    round_idx: int,
    execute: bool,
    initial_max_steps: int,
    extra_steps: int,
    eval_forget_limit: int,
    eval_retain_limit: int,
) -> None:
    info = manifest["rounds"][round_idx - 1]
    paths = round_paths(round_idx)
    for directory in paths.values():
        if directory.suffix:
            continue
        directory.mkdir(parents=True, exist_ok=True)

    forget_cumulative = Path(info["cumulative_forget_path"])
    memory_cumulative = Path(info["memory_cumulative_forget_path"])

    if not (paths["phase1"] / "memory_unlearning_report.json").exists():
        rag_target = max(1, round(int(info["cumulative_forget_count"]) * 1.4))
        run_cmd(
            [
                cfg["env_python"],
                "-m",
                "phase1.dual_graph_unlearning_gat.run_dual_graph_unlearning",
                "--rag-graph",
                cfg["rag_base_graph"],
                "--memory-graph",
                cfg["memory_base_graph"],
                "--forget-authors-from",
                str(forget_cumulative),
                "--forget-qa-query-path",
                str(forget_cumulative),
                "--memory-forget-qa-query-path",
                str(memory_cumulative),
                "--encoder-model-path",
                cfg["embedding_model"],
                "--output-dir",
                str(paths["phase1"]),
                "--device",
                "cuda",
                "--embedding-device",
                "cuda",
                "--embedding-batch-size",
                "1",
                "--embedding-max-length",
                "128",
                "--query-embedding-cache",
                str(paths["phase1"] / "query_embeddings.npz"),
                "--skip-pretrain",
                "--rag-pretrain-model-path",
                cfg["rag_pretrain"],
                "--memory-pretrain-model-path",
                cfg["memory_pretrain"],
                "--rag-retrain-epochs",
                "36",
                "--memory-retrain-epochs",
                "36",
                "--rag-author-aux-weight",
                "0.0",
                "--rag-threshold-target-count",
                str(rag_target),
                "--rag-strong-hard-fraction",
                "0.925",
                "--memory-path-cutoff",
                "5",
                "--memory-qa-topk",
                "6",
                "--memory-relation-prior",
                "0.8",
                "--memory-mentioned-prior",
                "1.2",
                "--memory-qa-hard-fraction",
                "0.925",
            ],
            log_path=paths["logs"] / "phase1.log",
            cwd=Path(cfg["agent_dir"]),
            execute=execute,
        )

    if not (paths["phase2_qa"] / "graph_forget_qa.jsonl").exists():
        run_cmd(
            [
                cfg["env_python"],
                "-m",
                "phase2.pipeline.export_graph_phase2_qa",
                "--rag-graph",
                str(paths["phase1"] / "rag_unlearning_graph.pkl"),
                "--memory-graph",
                str(paths["phase1"] / "memory_unlearning_graph.pkl"),
                "--output-dir",
                str(paths["phase2_qa"]),
            ],
            log_path=paths["logs"] / "export_qa.log",
            cwd=Path(cfg["agent_dir"]),
            execute=execute,
        )

    if not paths["localization_pkl"].exists():
        run_cmd(
            [
                cfg["env_python"],
                "-m",
                "phase2.diagnostics.native_llm_graph_neuron_localizer",
                "--base-model-path",
                cfg["base_model"],
                "--rag-graph-path",
                str(paths["phase1"] / "rag_unlearning_graph.pkl"),
                "--memory-graph-path",
                str(paths["phase1"] / "memory_unlearning_graph.pkl"),
                "--memory-report-path",
                str(paths["phase1"] / "memory_unlearning_report.json"),
                "--output-dir",
                str(paths["localization"]),
                "--device",
                "cuda",
                "--seed",
                str(42 + round_idx),
                "--max-rag-forget",
                "0",
                "--max-memory-forget",
                "0",
                "--max-retain",
                "0",
                "--layer-range",
                "0-31",
                "--neurons-per-layer",
                "20",
                "--batch-size",
                "4",
                "--qa-forget-path",
                str(paths["phase2_qa"] / "graph_forget_qa.jsonl"),
                "--qa-retain-path",
                str(paths["phase2_qa"] / "graph_retain_qa.jsonl"),
                "--qa-max-forget",
                "512",
                "--qa-max-retain",
                "512",
                "--qa-batch-size",
                "1",
                "--graph-view-weight",
                "1.0",
                "--qa-view-weight",
                "2.0",
                "--ranking-mode",
                "positive_diff",
                "--new-lora-r",
                "16",
                "--new-lora-alpha",
                "32",
                "--new-lora-dropout",
                "0.05",
                "--new-lora-target-modules",
                "gate_proj,up_proj,down_proj",
            ],
            log_path=paths["logs"] / "localize.log",
            cwd=Path(cfg["agent_dir"]),
            execute=execute,
        )

    if not (paths["lora"] / "ce_threshold_checkpoints").exists() or not list(
        (paths["lora"] / "ce_threshold_checkpoints").glob("ce10_step*")
    ):
        resume_args: list[str] = []
        max_train_steps = initial_max_steps
        if round_idx > 1:
            previous_ce10 = ce10_checkpoint(round_paths(round_idx - 1)["lora"])
            resume_path = make_adapter_only_resume(previous_ce10, paths["base"] / "resume")
            max_train_steps = checkpoint_step(previous_ce10) + extra_steps
            resume_args = ["--resume-from-checkpoint", str(resume_path)]
        train_cmd = [
            cfg["env_python"],
            "-m",
            "phase2.pipeline.main",
            "--stage",
            "unlearning",
            "--base-model-path",
            cfg["base_model"],
            "--original-lora-path",
            cfg["native_lora"],
            "--forget-target-path",
            str(paths["phase2_qa"] / "graph_forget_qa.jsonl"),
            "--retain-path",
            str(ROOT / "splits" / "empty_retain_for_forget_only.json"),
            "--unlearning-output-dir",
            str(paths["lora"]),
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
            "2",
            "--gradient-accumulation-steps",
            "4",
            "--seed",
            str(42 + round_idx),
            "--max-length",
            "512",
            "--warmup-steps",
            "5",
            "--reuse-localization-pkl",
            str(paths["localization_pkl"]),
            "--forget-target-mode",
            "random_vector",
            "--random-vector-target-similarity",
            "0.25",
            "--forget-anchor-target-similarity",
            "-0.1",
            "--forget-weight-lr",
            "0.05",
            "--forget-weight-min",
            "0.0",
            "--forget-weight-max",
            "12.0",
            "--random-vector-seed",
            str(42 + round_idx),
            "--parameter-random-vector-scale",
            "0.01",
            "--forget-loss-top-layers",
            "32",
            "--retain-ce-weight",
            "0.0",
            "--retain-kl-target",
            "0.02",
            "--retain-topk",
            "32",
            "--early-stop-min-steps",
            "200",
            "--early-stop-patience",
            "200",
            "--best-score-retain-alpha",
            "2.0",
            "--best-score-mia-alpha",
            "0.0",
            "--forget-ce-checkpoint-thresholds",
            "10",
            "--learning-rate",
            "3e-4",
            "--num-train-epochs",
            "20",
            "--max-train-steps",
            str(max_train_steps),
            "--logging-steps",
            "5",
            "--max-grad-norm",
            "1.0",
            "--forget-loss-strategy",
            "gradient_ascent",
            "--forget-objective-weight",
            "1.0",
            "--forget-ce-weight",
            "0.0",
            "--answer-suppression-weight",
            "0.0",
            "--random-vector-forget-weight",
            "1.0",
            "--forget-anchor-weight",
            "1.0",
            "--parameter-random-vector-weight",
            "1.0",
            "--retain-kl-weight",
            "0.0",
            "--disable-dual-kl-objective",
            "--disable-early-stop",
            *resume_args,
        ]
        run_cmd(train_cmd, log_path=paths["logs"] / "train_ga_ce10.log", cwd=Path(cfg["agent_dir"]), execute=execute)

    selected_ce10 = ce10_checkpoint(paths["lora"])
    print(f"[Round {round_idx}] selected CE10: {selected_ce10}", flush=True)

    if not (paths["merged"] / "model.safetensors.index.json").exists():
        run_cmd(
            [
                cfg["env_python"],
                "-m",
                "phase2.pipeline.merge_lora_for_vllm",
                "--base-model-path",
                cfg["base_model"],
                "--lora-path",
                str(selected_ce10),
                "--output-dir",
                str(paths["merged"]),
                "--torch-dtype",
                "float16",
                "--device-map",
                "cpu",
                "--max-shard-size",
                "4GB",
                "--force",
                "--label",
                f"incremental_ga_ce10_round{round_idx:02d}",
                "--note",
                "strict sequential TOFU for10 GA-CE10",
            ],
            log_path=paths["logs"] / "merge.log",
            cwd=Path(cfg["agent_dir"]),
            execute=execute,
        )

    eval_csv = paths["eval"] / "results.csv"
    eval_json = paths["eval"] / "results.json"
    sanity_json = paths["eval"] / "semantic_sanity.json"
    model_name = f"incremental_ga_ce10_r{round_idx}"
    if not eval_csv.exists():
        eval_cmd = [
            cfg["env_python"],
            "-m",
            "phase2.pipeline.step6_full_pipeline_eval",
            "--base-model-path",
            cfg["base_model"],
            "--lora-path",
            str(paths["merged"]),
            "--rag-graph-path",
            str(paths["phase1"] / "rag_unlearning_graph.pkl"),
            "--memory-graph-path",
            str(paths["phase1"] / "memory_unlearning_graph.pkl"),
            "--forget-path",
            cfg["forget10"],
            "--retain-path",
            cfg["retain90"],
            "--forget-limit",
            str(eval_forget_limit),
            "--retain-limit",
            str(eval_retain_limit),
            "--output-csv",
            str(eval_csv),
            "--output-json",
            str(eval_json),
            "--backend",
            "vllm_api",
            "--max-new-tokens",
            "128",
            "--temperature",
            "0.0",
            "--num-workers",
            "8",
            "--rag-fallback-top-k",
            "3",
            "--memory-max-hits",
            "3",
            "--checkpoint-every",
            "50",
        ]
        server_cmd = (
            f"CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 {q(cfg['vllm_bin'])} serve {q(paths['merged'])} "
            f"--host 0.0.0.0 --port 8002 --served-model-name {q(model_name)} "
            "--max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code"
        )
        script = (
            "set -euo pipefail; "
            f"({server_cmd}) > {q(paths['logs'] / 'answer_vllm.log')} 2>&1 & "
            f"echo $! > {q(paths['logs'] / 'answer_vllm.pid')}; "
            f"trap 'kill $(cat {q(paths['logs'] / 'answer_vllm.pid')}) 2>/dev/null || true' EXIT; "
            "for i in $(seq 1 180); do curl -fsS http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && break; sleep 5; done; "
            "curl -fsS http://127.0.0.1:8002/v1/models >/dev/null; "
            f"{q(cfg['env_python'])} {q(ROOT / 'scripts' / 'semantic_sanity_check.py')} "
            f"--api-url http://127.0.0.1:8002/v1 --model-name {q(model_name)} --output-json {q(sanity_json)}; "
            f"VLLM_API_BASE=http://127.0.0.1:8002/v1 VLLM_MODEL_NAME={q(model_name)} QWEN_INFER_BACKEND=vllm_api "
            + " ".join(q(part) for part in eval_cmd)
        )
        run_shell(script, log_path=paths["logs"] / "generate.log", cwd=Path(cfg["agent_dir"]), execute=execute)

    judge_csv = paths["judge"] / f"judge_incremental_ga_ce10_round{round_idx:02d}.csv"
    if not judge_csv.exists():
        label = f"incremental_ga_ce10_round{round_idx:02d}"
        script = (
            "set -euo pipefail; "
            f"CUDA_VISIBLE_DEVICES=0 {q(cfg['vllm_bin'])} serve {q(cfg['judge_model'])} "
            "--host 0.0.0.0 --port 8003 --served-model-name judge --max-model-len 2048 "
            "--max-num-seqs 2 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code "
            f"--generation-config vllm > {q(paths['logs'] / 'judge_vllm.log')} 2>&1 & "
            f"echo $! > {q(paths['logs'] / 'judge_vllm.pid')}; "
            f"trap 'kill $(cat {q(paths['logs'] / 'judge_vllm.pid')}) 2>/dev/null || true' EXIT; "
            "for i in $(seq 1 180); do curl -fsS http://127.0.0.1:8003/v1/models >/dev/null 2>&1 && break; sleep 5; done; "
            "curl -fsS http://127.0.0.1:8003/v1/models >/dev/null; "
            f"{q(cfg['env_python'])} evaluation/llm_judge.py --files {q(eval_csv)} --labels {q(label)} "
            f"--api-url http://127.0.0.1:8003/v1 --model judge --output-dir {q(paths['judge'])} --num-workers 1"
        )
        run_shell(script, log_path=paths["logs"] / "judge.log", cwd=Path(cfg["agent_dir"]), execute=execute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default_paths.json"))
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--initial-max-steps", type=int, default=260)
    parser.add_argument("--extra-steps", type=int, default=120)
    parser.add_argument("--eval-forget-limit", type=int, default=400)
    parser.add_argument("--eval-retain-limit", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    for key in [
        "agent_dir",
        "env_python",
        "vllm_bin",
        "base_model",
        "native_lora",
        "judge_model",
        "embedding_model",
        "forget10",
        "retain90",
        "rag_base_graph",
        "memory_base_graph",
        "rag_pretrain",
        "memory_pretrain",
    ]:
        ensure(Path(cfg[key]), key)
    manifest_path = ROOT / "splits" / "round_manifest.json"
    ensure(manifest_path, "round manifest")
    manifest = load_json(manifest_path)

    for round_idx in args.rounds:
        print(f"\n========== GA-CE10 ROUND {round_idx:02d} ==========", flush=True)
        run_round(
            cfg=cfg,
            manifest=manifest,
            round_idx=round_idx,
            execute=not args.dry_run,
            initial_max_steps=args.initial_max_steps,
            extra_steps=args.extra_steps,
            eval_forget_limit=args.eval_forget_limit,
            eval_retain_limit=args.eval_retain_limit,
        )


if __name__ == "__main__":
    main()
