


from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from evaluation.llm_judge import extract_final_answer
except Exception:
    def extract_final_answer(text: str) -> str:
        return (text or "").strip()


def normalize_text(text: str) -> str:
    text = re.sub(r"\*+", "", str(text or ""))
    text = text.casefold()
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def safe_split(row: dict) -> str:
    value = (row.get("split") or row.get("split_type") or "").strip().lower()
    if value not in {"forget", "retain"}:
        raise ValueError(f"Expected forget/retain split for task metrics, got {value!r}")
    return value


def is_correct(row: dict) -> bool:
    verdict = (row.get("judge_verdict") or "").strip().lower()
    if verdict:
        return verdict == "correct"
    pred = normalize_text(extract_final_answer(row.get("prediction", "")))
    ref = normalize_text(row.get("reference_answer", ""))
    return bool(pred and ref and pred == ref)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def roc_auc(labels: list[int], scores: list[float]) -> float:
    n = len(labels)
    pos = sum(labels)
    neg = n - pos
    if n == 0 or pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(n), key=lambda i: scores[i])
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and math.isclose(scores[order[j]], scores[order[i]], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if labels[order[k]] == 1:
                rank_sum_pos += avg_rank
        i = j
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(max(0.0, min(1.0, auc)))


def best_attack_acc(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    candidates = sorted(set(scores))
    thresholds = [-float("inf")] + candidates + [float("inf")]
    best = 0.0
    for threshold in thresholds:
        correct_high = sum((1 if score >= threshold else 0) == label for label, score in zip(labels, scores))
        correct_low = sum((1 if score <= threshold else 0) == label for label, score in zip(labels, scores))
        best = max(best, correct_high / len(labels), correct_low / len(labels))
    return float(best)


def compute_metrics(input_csv: Path, label: str) -> dict:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {input_csv}")

    for row in rows:
        row["_split"] = safe_split(row)
        row["_prediction_final"] = extract_final_answer(row.get("prediction", ""))
        row["_rouge_l"] = rouge_l_f1(row["_prediction_final"], row.get("reference_answer", ""))
        row["_correct"] = is_correct(row)

    forget = [row for row in rows if row["_split"] == "forget"]
    retain = [row for row in rows if row["_split"] == "retain"]
    labels = [1 if row["_split"] == "forget" else 0 for row in rows]
    scores = [float(row["_rouge_l"]) for row in rows]

    return {
        "label": label,
        "input_csv": str(input_csv),
        "num_total": len(rows),
        "num_forget": len(forget),
        "num_retain": len(retain),
        "forget_acc": mean(1.0 if row["_correct"] else 0.0 for row in forget),
        "retain_acc": mean(1.0 if row["_correct"] else 0.0 for row in retain),
        "forget_rouge_l": mean(float(row["_rouge_l"]) for row in forget),
        "retain_rouge_l": mean(float(row["_rouge_l"]) for row in retain),
        "split_distinguishability_auc": roc_auc(labels, scores),
        "split_distinguishability_acc_in_sample": best_attack_acc(labels, scores),
        "split_distinguishability_score": "rouge_l(prediction, reference_answer)",
        "split_distinguishability_positive_class": "forget",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute task metrics and legacy forget/retain split diagnostics.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    files = [Path(path) for path in args.files]
    labels = args.labels or [path.stem for path in files]
    if len(files) != len(labels):
        raise ValueError("--labels length must match --files length")

    results = [compute_metrics(path, label) for path, label in zip(files, labels)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / "final_metrics_summary.json"
    out_csv = output_dir / "final_metrics_summary.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 110)
    print("Final Metrics Summary")
    print("=" * 110)
    print(f"{'label':<34} {'F-Acc':>8} {'R-Acc':>8} {'F-RL':>8} {'R-RL':>8} {'Split-AUC':>9} {'Split-ACC':>9}")
    print("-" * 110)
    for item in results:
        print(
            f"{item['label']:<34} "
            f"{item['forget_acc']:>8.4f} {item['retain_acc']:>8.4f} "
            f"{item['forget_rouge_l']:>8.4f} {item['retain_rouge_l']:>8.4f} "
            f"{item['split_distinguishability_auc']:>9.4f} {item['split_distinguishability_acc_in_sample']:>9.4f}"
        )
    print("=" * 110)
    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
