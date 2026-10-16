from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from ._shared import ARTIFACTS_DIR, PHASE1_DIR, PROJECT_ROOT, default_results_dir, ensure_phase1_on_path
except ImportError:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from evaluation._shared import ARTIFACTS_DIR, PHASE1_DIR, PROJECT_ROOT, default_results_dir, ensure_phase1_on_path

__all__ = [
    "ARTIFACTS_DIR",
    "PHASE1_DIR",
    "PROJECT_ROOT",
    "default_results_dir",
    "ensure_phase1_on_path",
    "main",
]


def _resolve_labels(files: list[Path], labels: list[str] | None) -> list[str]:
    resolved = labels or [path.stem for path in files]
    if len(resolved) != len(files):
        raise ValueError("--labels length must match --files length")
    return resolved


def _run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("\n[Eval] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main(argv: list[str] | None = None) -> None:

    parser = argparse.ArgumentParser(description="Run current full evaluation pipeline.")
    parser.add_argument("--files", nargs="+", required=True, help="Raw result CSVs, or judged CSVs with --skip-judge.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels matching --files.")
    parser.add_argument("--output-dir", default=str(default_results_dir() / "all_eval"))
    parser.add_argument("--skip-judge", action="store_true", help="Treat --files as already judged CSVs.")
    parser.add_argument("--judge-api-url", default="http://localhost:8000/v1")
    parser.add_argument("--judge-model", default="judge")
    args = parser.parse_args(argv)

    files = [Path(path).expanduser().resolve() for path in args.files]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Input CSV not found: " + ", ".join(missing))

    labels = _resolve_labels(files, args.labels)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_judge:
        judged_files = files
    else:
        judge_dir = output_dir / "judge"
        _run_module(
            "evaluation.llm_judge",
            [
                "--files",
                *[str(path) for path in files],
                "--labels",
                *labels,
                "--output-dir",
                str(judge_dir),
                "--api-url",
                args.judge_api_url,
                "--model",
                args.judge_model,
            ],
        )
        judged_files = [judge_dir / f"judge_{label}.csv" for label in labels]
        missing_judged = [str(path) for path in judged_files if not path.exists()]
        if missing_judged:
            raise FileNotFoundError("Judge output not found: " + ", ".join(missing_judged))

    final_metrics_dir = output_dir / "final_metrics"
    _run_module(
        "evaluation.final_metrics",
        [
            "--files",
            *[str(path) for path in judged_files],
            "--labels",
            *labels,
            "--output-dir",
            str(final_metrics_dir),
        ],
    )

    print(f"\n[Eval] Completed. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
