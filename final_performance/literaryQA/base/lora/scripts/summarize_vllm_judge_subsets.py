                      
                                                                                   

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag")


def load_json_or_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected list data: {path}")
    return [row for row in data if isinstance(row, dict)]


def key_from_row(row: dict) -> tuple[str, str]:
    return (
        str(row.get("question") or "").strip(),
        str(row.get("reference_answer") or row.get("answer") or row.get("output") or "").strip(),
    )


def is_correct(row: dict) -> bool:
    return str(row.get("judge_verdict") or "").strip().lower() == "correct"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_subset(judged_rows: list[dict], target_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], deque[dict]] = defaultdict(deque)
    for row in judged_rows:
        buckets[key_from_row(row)].append(row)

    selected: list[dict] = []
    missing: list[tuple[str, str]] = []
    for target in target_rows:
        key = key_from_row(target)
        if buckets[key]:
            selected.append(dict(buckets[key].popleft()))
        else:
            missing.append(key)
    if missing:
        examples = missing[:5]
        raise RuntimeError(f"Missing {len(missing)} target rows in judged CSV. Examples: {examples}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LiteraryQA subset judge acc from forget10 judged CSV")
    parser.add_argument("--judged-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--splits", default="01,05,10")
    args = parser.parse_args()

    judged_csv = Path(args.judged_csv)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    judged_rows = read_csv(judged_csv)
    judged_counter = Counter(key_from_row(row) for row in judged_rows)

    summaries = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        target_rows = load_json_or_jsonl(data_root / f"forget{split}.json")
        target_counter = Counter(key_from_row(row) for row in target_rows)
        bad = [key for key, count in target_counter.items() if judged_counter[key] < count]
        if bad:
            raise RuntimeError(f"forget{split}: judged CSV does not cover {len(bad)} unique target keys")
        selected = select_subset(judged_rows, target_rows)
        correct = sum(1 for row in selected if is_correct(row))
        acc = correct / len(selected) if selected else 0.0
        subset_csv = output_dir / f"judge_forget{split}_from_forget10.csv"
        write_csv(subset_csv, selected)
        summaries.append(
            {
                "split": f"forget{split}",
                "n": len(selected),
                "correct": correct,
                "acc": acc,
                "source_judged_csv": str(judged_csv),
                "subset_csv": str(subset_csv),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "subset_acc_summary.json"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[Summary] saved: {summary_json}")


if __name__ == "__main__":
    main()
