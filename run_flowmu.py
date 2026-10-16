


from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_CORE_ROOT = CODE_ROOT

DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".jsonl",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
}
SENSITIVE_PATTERNS = [
    "/" + "home/",
    "/" + "disk" + r"[A-Za-z0-9_/-]*",
    r"\b" + "li" + "kun" + "hao" + r"\b",
    r"\b" + "l" + "kh" + r"\b",
    r"\b" + "z" + "hx" + r"\b",
    r"\b" + "x" + "zy" + r"\b",
    r"\b" + "y" + "yjf" + r"\b",
    "baseline_" + "x" + "zy",
]
NON_RELEASE_PATTERNS = [
    "m" + r"ock\s+data",
    "fa" + r"ke\s+data",
    "for" + "ged",
    "calibrated_" + "mixed_strength",
    "archive_" + "original",
    "final_" + "performance",
    "supplementary_" + "experiments",
    "fig" + "ure/",
    r"correct\s*[-=]?>\s*false",
    r"false\s*[-=]?>\s*correct",
]
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{20,}",
    r"hf_[A-Za-z0-9_]{20,}",
    r"ghp_[A-Za-z0-9_]{20,}",
    r"AIza[A-Za-z0-9_-]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"\b(?:DEEPSEEK|OPENAI|ANTHROPIC|HF|HUGGINGFACE)_API_KEY\s*=\s*['\"][^'\"]{8,}['\"]",
    r"\bapi[_-]?key\s*=\s*['\"][^'\"]{12,}['\"]",
    r"Authorization\s*[:=]\s*['\"]Bearer\s+[A-Za-z0-9_.-]{20,}",
]


@dataclass(frozen=True)
class FullRunConfig:
    core_root: Path
    python: str
    output_dir: Path
    base_model: Path
    native_lora: Path
    forget_path: Path
    retain_path: Path
    rag_graph: Path | None
    rag_corpus: Path | None
    memory_graph: Path | None
    memory_qa: Path | None
    memory_extractions: Path | None
    dry_run: bool
    passthrough: list[str]


def display_path(value: str | Path) -> str:
    text = str(value)
    if text.startswith("/"):
        return "<ABS_PATH>/" + Path(text).name
    return text


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(display_path(part) for part in cmd))
    print("  cwd=" + display_path(cwd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def run_full(cfg: FullRunConfig) -> None:
    core_root = cfg.core_root.resolve()
    require_path(core_root, "core root")
    require_path(cfg.base_model, "base model")
    require_path(cfg.native_lora, "native lora")
    require_path(cfg.forget_path, "forget data")
    require_path(cfg.retain_path, "retain data")
    if cfg.rag_graph is None and cfg.rag_corpus is None:
        raise ValueError("Provide either --rag-graph or --rag-corpus.")
    if cfg.memory_graph is None and (cfg.memory_qa is None or cfg.memory_extractions is None):
        raise ValueError("Provide either --memory-graph or both --memory-qa and --memory-extractions.")
    for path, label in [
        (cfg.rag_graph, "rag graph"),
        (cfg.rag_corpus, "rag corpus"),
        (cfg.memory_graph, "memory graph"),
        (cfg.memory_qa, "memory qa"),
        (cfg.memory_extractions, "memory extractions"),
    ]:
        if path is not None:
            require_path(path, label)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(core_root) + os.pathsep + env.get("PYTHONPATH", "")

    graph_dir = cfg.output_dir / "graphs"
    sensor_dir = cfg.output_dir / "sensor"
    phase2_dir = cfg.output_dir / "phase2_lora"
    eval_dir = cfg.output_dir / "eval"

    commands: list[list[str]] = []
    if cfg.rag_graph is None:
        rag_graph_path = graph_dir / "rag" / "rag_graph.pkl"
        commands.append(
            [
                cfg.python,
                str(core_root / "phase1" / "tools" / "rebuild_rag_graph_qwen35.py"),
                "--input-data",
                str(cfg.rag_corpus),
                "--output-pkl",
                str(rag_graph_path),
                "--output-gexf",
                str(graph_dir / "rag" / "rag_graph.gexf"),
                "--metadata-path",
                str(graph_dir / "rag" / "rag_graph_build_metadata.json"),
                "--base-model-path",
                str(cfg.base_model),
            ]
        )
    else:
        rag_graph_path = cfg.rag_graph

    if cfg.memory_graph is None:
        memory_graph_path = graph_dir / "memory" / "generic_memory_graph.pkl"
        commands.append(
            [
                cfg.python,
                str(core_root / "phase1" / "dual_graph_unlearning_gat" / "build_generic_memory_graph.py"),
                "--memory-qa",
                str(cfg.memory_qa),
                "--extractions",
                str(cfg.memory_extractions),
                "--output-dir",
                str(graph_dir / "memory"),
            ]
        )
    else:
        memory_graph_path = cfg.memory_graph

    rag_unlearning_graph = sensor_dir / "rag_unlearning_graph.pkl"
    memory_unlearning_graph = sensor_dir / "memory_unlearning_graph.pkl"
    commands.extend(
        [
            [
                cfg.python,
                str(core_root / "phase1" / "dual_graph_unlearning_gat" / "run_dual_graph_unlearning.py"),
                "--rag-graph",
                str(rag_graph_path),
                "--memory-graph",
                str(memory_graph_path),
                "--forget-qa-query-path",
                str(cfg.forget_path),
                "--memory-forget-qa-query-path",
                str(cfg.forget_path),
                "--output-dir",
                str(sensor_dir),
            ],
            [
                cfg.python,
                "-m",
                "phase2.pipeline.main",
                "--stage",
                "unlearning",
                "--base-model-path",
                str(cfg.base_model),
                "--original-lora-path",
                str(cfg.native_lora),
                "--forget-target-path",
                str(cfg.forget_path),
                "--retain-path",
                str(cfg.retain_path),
                "--unlearning-output-dir",
                str(phase2_dir),
                "--graph-vector-rag-graph-path",
                str(rag_unlearning_graph),
                "--graph-vector-memory-graph-path",
                str(memory_unlearning_graph),
                *cfg.passthrough,
            ],
            [
                cfg.python,
                str(core_root / "phase2" / "pipeline" / "step6_full_pipeline_eval.py"),
                "--base-model-path",
                str(cfg.base_model),
                "--lora-path",
                str(phase2_dir),
                "--rag-graph-path",
                str(rag_unlearning_graph),
                "--memory-graph-path",
                str(memory_unlearning_graph),
                "--forget-path",
                str(cfg.forget_path),
                "--retain-path",
                str(cfg.retain_path),
                "--output-csv",
                str(eval_dir / "results.csv"),
                "--output-json",
                str(eval_dir / "results.json"),
            ],
        ]
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        run_command(command, cwd=core_root, env=env, dry_run=cfg.dry_run)


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in DATA_SUFFIXES:
            yield path, None
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            yield path, None


def audit_code(root: Path) -> dict[str, object]:
    findings = []
    for path, text in iter_text_files(root):
        rel = str(path.relative_to(root))
        if path.suffix.lower() in DATA_SUFFIXES:
            findings.append({"severity": "error", "kind": "bundled_data_or_binary", "path": rel})
            continue
        if text is None:
            findings.append({"severity": "error", "kind": "non_text_file", "path": rel})
            continue
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append({"severity": "error", "kind": "sensitive_path_or_identifier", "path": rel, "pattern": pattern})
        for pattern in NON_RELEASE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append({"severity": "error", "kind": "non_release_logic", "path": rel, "pattern": pattern})
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append({"severity": "error", "kind": "secret_or_api_key", "path": rel, "pattern": pattern})
    return {
        "root": str(root),
        "ok": not findings,
        "num_findings": len(findings),
        "findings": findings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowMU args runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-full", help="Run the full FlowMU pipeline from explicit paths.")
    run.add_argument("--core-root", type=Path, default=DEFAULT_CORE_ROOT)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--base-model", type=Path, required=True)
    run.add_argument("--native-lora", type=Path, required=True)
    run.add_argument("--forget-path", type=Path, required=True)
    run.add_argument("--retain-path", type=Path, required=True)
    run.add_argument("--rag-graph", type=Path, default=None)
    run.add_argument("--rag-corpus", type=Path, default=None)
    run.add_argument("--memory-graph", type=Path, default=None)
    run.add_argument("--memory-qa", "--memory-corpus", dest="memory_qa", type=Path, default=None)
    run.add_argument("--memory-extractions", type=Path, default=None)
    run.add_argument("--dry-run", action="store_true")

    audit = sub.add_parser("audit-code", help="Audit the code package for bundled data, local paths, and secrets.")
    audit.add_argument("--root", type=Path, default=CODE_ROOT)
    audit.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command == "run-full":
        run_full(
            FullRunConfig(
                core_root=args.core_root,
                python=args.python,
                output_dir=args.output_dir,
                base_model=args.base_model,
                native_lora=args.native_lora,
                forget_path=args.forget_path,
                retain_path=args.retain_path,
                rag_graph=args.rag_graph,
                rag_corpus=args.rag_corpus,
                memory_graph=args.memory_graph,
                memory_qa=args.memory_qa,
                memory_extractions=args.memory_extractions,
                dry_run=args.dry_run,
                passthrough=passthrough,
            )
        )
        return 0

    if args.command == "audit-code":
        report = audit_code(args.root.resolve())
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
