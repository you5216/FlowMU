                      
                                                                          

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text or "").lower())


def rouge_l(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    dp = [0] * (len(ref) + 1)
    for token in pred:
        prev = 0
        for j, ref_token in enumerate(ref, start=1):
            tmp = dp[j]
            if token == ref_token:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def safe_acc(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("judge_verdict") == "Correct") / len(rows)


def safe_rouge(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    return sum(rouge_l(row.get("prediction", ""), row.get("reference_answer", "")) for row in rows) / len(rows)


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def parse_log_cost(log_path: Path) -> tuple[float | None, int | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    elapsed = re.findall(r"ELAPSED_SEC=([0-9.]+)", text)
    mem = re.findall(r"MAX_GPU_MEMORY_MIB=([0-9]+)", text)
    return (float(elapsed[-1]) / 60.0 if elapsed else None, int(mem[-1]) if mem else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "splits" / "round_manifest.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    batch_size = int(manifest["batch_size"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for round_info in manifest["rounds"]:
        round_idx = int(round_info["round"])
        start = int(round_info["delta_start"])
        end = int(round_info["delta_end"])
        base = ROOT / "artifacts" / "incremental" / "ga_ce10" / f"round_{round_idx:02d}"
        judge_csv = base / "judge" / f"judge_incremental_ga_ce10_round{round_idx:02d}.csv"
        if not judge_csv.exists():
            continue
        with judge_csv.open("r", encoding="utf-8", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        forget_rows = [row for row in all_rows if (row.get("split_type") or row.get("split") or "").lower() == "forget"]
        retain_rows = [row for row in all_rows if (row.get("split_type") or row.get("split") or "").lower() == "retain"]
        current = forget_rows[start:end]
        historical = forget_rows[:start]
        future = forget_rows[end:]
        rows_out.append(
            {
                "protocol": "incremental",
                "method": "GA-CE10",
                "round": round_idx,
                "current_f_acc": safe_acc(current),
                "historical_f_acc": safe_acc(historical),
                "future_f_acc": safe_acc(future),
                "retain_acc": safe_acc(retain_rows),
                "current_f_rougel": safe_rouge(current),
                "historical_f_rougel": safe_rouge(historical),
                "retain_rougel": safe_rouge(retain_rows),
                "current_n": len(current),
                "historical_n": len(historical),
                "future_n": len(future),
                "retain_n": len(retain_rows),
            }
        )
        for stage in ["phase1", "export_qa", "localize", "train_ga_ce10", "merge", "generate", "judge"]:
            log_name = {
                "train_ga_ce10": "train_ga_ce10",
                "export_qa": "export_qa",
            }.get(stage, stage)
            elapsed_min, max_mem = parse_log_cost(base / "logs" / f"{log_name}.log")
            cost_rows.append(
                {
                    "round": round_idx,
                    "stage": stage,
                    "time_min": elapsed_min,
                    "max_gpu_mib": max_mem,
                }
            )

    with (output_dir / "sequential_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "protocol",
            "method",
            "round",
            "current_f_acc",
            "historical_f_acc",
            "future_f_acc",
            "retain_acc",
            "current_f_rougel",
            "historical_f_rougel",
            "retain_rougel",
            "current_n",
            "historical_n",
            "future_n",
            "retain_n",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    (output_dir / "sequential_metrics.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| Protocol | Method | Round | Current F-ACC ↓ | Historical F-ACC ↓ | Retain ACC ↑ | Current F-RougeL ↓ | Historical F-RougeL ↓ | Retain RougeL ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        lines.append(
            "| {protocol} | {method} | {round} | {current_f_acc} | {historical_f_acc} | {retain_acc} | {current_f_rougel} | {historical_f_rougel} | {retain_rougel} |".format(
                protocol=row["protocol"],
                method=row["method"],
                round=row["round"],
                current_f_acc=fmt(row["current_f_acc"]),
                historical_f_acc=fmt(row["historical_f_acc"]),
                retain_acc=fmt(row["retain_acc"]),
                current_f_rougel=fmt(row["current_f_rougel"]),
                historical_f_rougel=fmt(row["historical_f_rougel"]),
                retain_rougel=fmt(row["retain_rougel"]),
            )
        )
    (output_dir / "sequential_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output_dir / "incremental_cost.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "stage", "time_min", "max_gpu_mib"])
        writer.writeheader()
        writer.writerows(cost_rows)
    (output_dir / "incremental_cost.json").write_text(json.dumps(cost_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metric_rows": len(rows_out), "cost_rows": len(cost_rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
