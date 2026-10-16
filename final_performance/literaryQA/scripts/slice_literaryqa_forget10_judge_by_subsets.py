                      
                                                                           

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_json_or_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format: {path}")
    return [row for row in data if isinstance(row, dict)]


def row_key(row: dict) -> tuple[str, str]:
    return (
        str(row.get("question") or "").strip(),
        str(row.get("answer") or row.get("reference_answer") or row.get("output") or "").strip(),
    )


def load_key_counts(path: Path) -> Counter[tuple[str, str]]:
    return Counter(key for key in (row_key(row) for row in load_json_or_jsonl(path)) if key[0])


def is_correct(row: dict) -> bool:
    return str(row.get("judge_verdict") or "").strip().lower() == "correct"


def summarize_judged_csv(judged_csv: Path, data_root: Path) -> list[dict]:
    with judged_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty judged csv: {judged_csv}")

    subsets = {
        "forget01": load_key_counts(data_root / "forget01.json"),
        "forget05": load_key_counts(data_root / "forget05.json"),
        "forget10": load_key_counts(data_root / "forget10.json"),
    }
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = row_key(row)
        if key[0]:
            by_key.setdefault(key, []).append(row)

    out = []
    for subset, expected_counts in subsets.items():
        selected: list[dict] = []
        missing = 0
        for key, expected_count in expected_counts.items():
            available = by_key.get(key, [])
            selected.extend(available[:expected_count])
            missing += max(0, expected_count - len(available))
        correct = sum(1 for row in selected if is_correct(row))
        total = len(selected)
        out.append(
            {
                "label": judged_csv.stem.replace("judge_", "", 1),
                "subset": subset,
                "correct": correct,
                "total": total,
                "expected_total": sum(expected_counts.values()),
                "missing": missing,
                "acc": correct / total if total else 0.0,
                "judged_csv": str(judged_csv),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-csvs", nargs="+", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results: list[dict] = []
    for path in args.judge_csvs:
        results.extend(summarize_judged_csv(Path(path), data_root))

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "subset",
            "correct",
            "total",
            "expected_total",
            "missing",
            "acc",
            "judged_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{'label':<30} {'subset':<10} {'acc':>8} {'correct/total':>16} {'missing':>8}")
    for row in results:
        print(
            f"{row['label']:<30} {row['subset']:<10} {row['acc']:>8.4f} "
            f"{row['correct']:>6}/{row['total']:<6} {row['missing']:>8}"
        )
    print(f"saved: {output_csv}")
    print(f"saved: {output_json}")


if __name__ == "__main__":
    main()
