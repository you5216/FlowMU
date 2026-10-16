                      
                                                                  

                                                                    
                                                                        
                                                                  
   

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_LINE_RE = re.compile(r"\[GAT-AFRU\.Phase2\.Runner\]\s+-\s+([A-Za-z0-9_]+)=(.*)$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"None", "null"}:
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value) or re.fullmatch(
            r"[-+]?\d+[eE][-+]?\d+", value
        ):
            return float(value)
    except ValueError:
        pass
    return value


def load_metrics(path: Path) -> dict[str, Any]:
    metrics = load_json(path)
    if isinstance(metrics, list):
        if not metrics:
            raise ValueError(f"Empty final metrics file: {path}")
        metrics = metrics[0]
    if not isinstance(metrics, dict):
        raise ValueError(f"Unsupported final metrics format: {path}")
    for key in ("forget_acc", "retain_acc"):
        if key not in metrics:
            raise ValueError(f"Missing {key} in final metrics: {path}")
    return metrics


def load_stage2_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    metadata = load_json(path)
    if not isinstance(metadata, dict):
        return {}
    config = metadata.get("config_summary")
    if isinstance(config, dict):
        return config
    return {}


def parse_train_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    config: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = CONFIG_LINE_RE.search(line)
        if match:
            key, value = match.groups()
            config[key] = parse_scalar(value)
    return config


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "trial"


def trial_sort_key(trial: dict[str, Any]) -> tuple[float, float]:
    metrics = trial["metrics"]
    return (float(metrics["forget_acc"]), -float(metrics["retain_acc"]))


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return trial_sort_key(candidate) < trial_sort_key(incumbent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a Phase-2 hyperparameter trial.")
    parser.add_argument("--split-dir", required=True, help="Split result dir, e.g. final_performance/tofu/for95_re05")
    parser.add_argument("--trial-label", required=True)
    parser.add_argument("--final-metrics", required=True)
    parser.add_argument("--stage2-metadata", default="")
    parser.add_argument("--train-log", default="")
    parser.add_argument("--min-retain-acc", type=float, default=0.0)
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    output_dir = split_dir / "hyperparams"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_metrics_path = Path(args.final_metrics)
    stage2_metadata_path = Path(args.stage2_metadata) if args.stage2_metadata else None
    train_log_path = Path(args.train_log) if args.train_log else None

    metrics = load_metrics(final_metrics_path)
    hyperparameters = load_stage2_metadata(stage2_metadata_path)
    hyperparameters.update(parse_train_log(train_log_path))

    retain_acc = float(metrics["retain_acc"])
    trial = {
        "trial_label": args.trial_label,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "eligible_for_best": retain_acc >= float(args.min_retain_acc),
        "selection_rule": {
            "primary": "minimize forget_acc",
            "tie_breaker": "maximize retain_acc",
            "min_retain_acc": float(args.min_retain_acc),
        },
        "source_paths": {
            "final_metrics": str(final_metrics_path),
            "stage2_metadata": str(stage2_metadata_path) if stage2_metadata_path else "",
            "train_log": str(train_log_path) if train_log_path else "",
        },
        "note": args.note,
    }

    trial_path = output_dir / f"trial_{sanitize_label(args.trial_label)}.json"
    write_json(trial_path, trial)

    trials_path = output_dir / "trials.json"
    trials: list[dict[str, Any]] = []
    if trials_path.exists():
        existing = load_json(trials_path)
        if isinstance(existing, list):
            trials = existing
    trials = [item for item in trials if item.get("trial_label") != args.trial_label]
    trials.append(trial)
    trials.sort(key=lambda item: item.get("recorded_at_utc", ""))
    write_json(trials_path, trials)

    best_path = output_dir / "best_hyperparams.json"
    eligible_trials = [item for item in trials if item.get("eligible_for_best")]
    best = min(eligible_trials, key=trial_sort_key) if eligible_trials else None
    if best is not None:
        write_json(best_path, best)

    result = {
        "trial_path": str(trial_path),
        "trials_path": str(trials_path),
        "best_path": str(best_path) if best is not None else "",
        "is_current_best": bool(best is not None and best.get("trial_label") == args.trial_label),
        "best_trial": best.get("trial_label") if best is not None else "",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
