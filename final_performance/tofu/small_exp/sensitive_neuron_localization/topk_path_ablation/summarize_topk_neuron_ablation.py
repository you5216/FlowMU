                      
                                                                            

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[5]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from evaluation.final_metrics import compute_metrics              


RUN_RE = re.compile(
    r"(?P<split>for\d+_re\d+)/"
    r"(?P<method>FlowMU|Random)(?:_(?P<mask_mode>drop|keep))?"
    r"_topk(?P<topk>\d+)_seed(?P<seed>\d+)/results\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Top-k ablation results.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def parse_run(csv_path: Path, run_root: Path) -> dict[str, str] | None:
    rel = csv_path.relative_to(run_root).as_posix()
    match = RUN_RE.match(rel)
    if not match:
        return None
    meta = match.groupdict()
    meta["mask_mode"] = meta.get("mask_mode") or "legacy_keep"
    return meta


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    rows: list[dict[str, object]] = []

    for csv_path in sorted(run_root.rglob("results.csv")):
        meta = parse_run(csv_path.resolve(), run_root)
        if meta is None:
            continue
        label = f"{meta['split']}_{meta['method']}_{meta['mask_mode']}_topk{meta['topk']}_seed{meta['seed']}"
        metrics = compute_metrics(csv_path, label)
        for panel, prefix in (("forget", "forget"), ("retain", "retain")):
            num_samples = int(metrics[f"num_{prefix}"])
            if num_samples <= 0:
                continue
            rows.append(
                {
                    "split": meta["split"],
                    "panel": panel,
                    "method": meta["method"],
                    "mask_mode": meta["mask_mode"],
                    "topk": int(meta["topk"]),
                    "rouge_l": metrics[f"{prefix}_rouge_l"],
                    "acc": metrics[f"{prefix}_acc"],
                    "num_samples": num_samples,
                    "seed": int(meta["seed"]),
                    "input_csv": str(csv_path),
                    "label": label,
                }
            )

    if not rows:
        raise ValueError(f"No matching results.csv found under {run_root}")

    rows.sort(
        key=lambda r: (
            str(r["split"]),
            str(r["panel"]),
            str(r["mask_mode"]),
            str(r["method"]),
            int(r["topk"]),
            int(r["seed"]),
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved csv: {args.output_csv}")
    print(f"saved json: {output_json}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
