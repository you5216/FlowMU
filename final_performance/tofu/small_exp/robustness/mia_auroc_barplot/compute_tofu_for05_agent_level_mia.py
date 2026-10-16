                      
                                                          

                                                 
                                                         

                                                                             
                                                                        
   

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


AGENT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent")
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from evaluation.final_metrics import rouge_l_f1              
from evaluation.llm_judge import extract_final_answer              


MANIFEST_DIR = AGENT_ROOT / "final_performance/tofu/small_exp/robustness/smoke_v2"
OUTPUT_DIR = AGENT_ROOT / "final_performance/final/tofu_for05_agent_level_mia_auroc"
STEM = "tofu_for05_agent_level_mia_auroc"

METHOD_FILES = {
    "Vanilla": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_vanilla_for05.csv",
    "PURE": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_pure_for05.csv",
    "SBU": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_sbu_for05.csv",
    "SF-NL": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_sfnl_for05.csv",
    "FlowMU": AGENT_ROOT
    / "final_performance/tofu/for05_re95/full_pipeline_eval/"
    / "full_dpo_fullretain_lambda03top100_beta02_lr2e5_e02_cached_20260622/"
    / "judge_qwen25vl/judge_full_dpo_fullretain_lambda03top100_beta02_lr2e5_e02_cached_20260622.csv",
}
METHOD_ORDER = ["Vanilla", "PURE", "SBU", "SF-NL", "FlowMU"]
METHOD_COLORS = {
    "Vanilla": "#8d99ae",
    "PURE": "#6f9ec9",
    "SBU": "#7aa974",
    "SF-NL": "#c7a76c",
    "FlowMU": "#C43C2F",
}


@dataclass(frozen=True)
class MethodResult:
    method: str
    input_csv: str
    n_train: int
    n_val: int
    n_test: int
    train_forget: int
    train_retain: int
    val_forget: int
    val_retain: int
    test_forget: int
    test_retain: int
    classifier_raw_val_auc: float
    classifier_attack_val_auc: float
    classifier_raw_test_auc: float
    classifier_attack_test_auc: float
    classifier_test_acc: float
    rouge_raw_test_auc: float
    rouge_attack_test_auc: float
    correct_raw_test_auc: float
    correct_attack_test_auc: float


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def row_id(row: dict[str, Any], idx: int) -> str:
    raw = str(row.get("id") or row.get("question") or idx)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_manifest() -> dict[str, list[str]]:
    manifest = {}
    for split in ("train", "val", "test"):
        path = MANIFEST_DIR / f"mia_{split}_ids.json"
        if not path.exists():
            raise FileNotFoundError(path)
        ids = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Invalid manifest file: {path}")
        manifest[split] = [str(item) for item in ids]
    return manifest


def assign_manifest_splits(rows: list[dict[str, str]], manifest: dict[str, list[str]]) -> list[str]:
                                                                              
    counters = {split: Counter(ids) for split, ids in manifest.items()}
    row_ids = [row_id(row, idx) for idx, row in enumerate(rows)]
    csv_counts = Counter(row_ids)
    manifest_total = Counter()
    for counter in counters.values():
        manifest_total.update(counter)
    if csv_counts != manifest_total:
        missing = manifest_total - csv_counts
        extra = csv_counts - manifest_total
        raise ValueError(
            "CSV row IDs do not match the fixed MIA manifest. "
            f"missing={dict(missing.most_common(5))}, extra={dict(extra.most_common(5))}"
        )

    assigned = []
    for sid in row_ids:
        selected = None
        for split in ("train", "val", "test"):
            if counters[split][sid] > 0:
                selected = split
                counters[split][sid] -= 1
                break
        if selected is None:
            raise RuntimeError(f"Internal manifest assignment failure for id={sid}")
        assigned.append(selected)
    return assigned


def is_forget(row: dict[str, str]) -> int:
    split = (row.get("split") or row.get("split_type") or "").strip().lower()
    if split == "forget":
        return 1
    if split == "retain":
        return 0
    raise ValueError(f"Unknown row split: {split!r}")


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def refusal_flag(text: str) -> float:
    low = str(text or "").casefold()
    phrases = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not enough information",
        "no information",
        "not provided",
    ]
    return 1.0 if any(phrase in low for phrase in phrases) else 0.0


def correct_flag(row: dict[str, str]) -> float:
    verdict = str(row.get("judge_verdict") or "").strip().lower()
    if verdict:
        return 1.0 if verdict == "correct" else 0.0
    final = extract_final_answer(row.get("prediction", ""))
    return 1.0 if rouge_l_f1(final, row.get("reference_answer", "")) >= 0.95 else 0.0


def agent_features(row: dict[str, str]) -> tuple[list[float], dict[str, float]]:
    prediction = str(row.get("prediction") or "")
    final_answer = extract_final_answer(prediction)
    reference = str(row.get("reference_answer") or "")
    rouge_l = rouge_l_f1(final_answer, reference)
    correct = correct_flag(row)
    pred_len = token_count(prediction)
    final_len = token_count(final_answer)
    ref_len = max(1, token_count(reference))
    empty = 1.0 if not prediction.strip() else 0.0
    refusal = refusal_flag(prediction)
    length_ratio = min(final_len / ref_len, 5.0)
    features = [
        correct,
        rouge_l,
        np.log1p(pred_len),
        np.log1p(final_len),
        length_ratio,
        empty,
        refusal,
    ]
    scalars = {
        "rouge_l": rouge_l,
        "correct": correct,
    }
    return [float(x) for x in features], scalars


def attack_auc(raw_auc: float) -> float:
    return float(max(raw_auc, 1.0 - raw_auc))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.astype(int).tolist())) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def split_counts(labels: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    selected = labels[mask]
    forget = int(selected.sum())
    retain = int(selected.shape[0] - forget)
    return forget, retain


def compute_method(method: str, path: Path, manifest: dict[str, list[str]]) -> tuple[MethodResult, list[dict[str, Any]]]:
    rows = load_csv(path)
    manifest_splits = assign_manifest_splits(rows, manifest)

    features = []
    labels = []
    scalar_rows = []
    for idx, (row, mia_split) in enumerate(zip(rows, manifest_splits)):
        feat, scalars = agent_features(row)
        label = is_forget(row)
        features.append(feat)
        labels.append(label)
        scalar_rows.append(
            {
                "method": method,
                "sample_id": row_id(row, idx),
                "mia_split": mia_split,
                "member_label": label,
                "dataset_split": "forget" if label else "retain",
                **scalars,
            }
        )

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split_arr = np.asarray(manifest_splits, dtype=object)
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    )
    clf.fit(x[train_mask], y[train_mask])
    val_prob = clf.predict_proba(x[val_mask])[:, 1]
    test_prob = clf.predict_proba(x[test_mask])[:, 1]
    test_pred = (test_prob >= 0.5).astype(np.int64)

    raw_val_auc = safe_auc(y[val_mask], val_prob)
    raw_test_auc = safe_auc(y[test_mask], test_prob)
    rouge_scores = np.asarray([row["rouge_l"] for row in scalar_rows], dtype=np.float32)
    correct_scores = np.asarray([row["correct"] for row in scalar_rows], dtype=np.float32)
    rouge_raw_test_auc = safe_auc(y[test_mask], rouge_scores[test_mask])
    correct_raw_test_auc = safe_auc(y[test_mask], correct_scores[test_mask])
    train_forget, train_retain = split_counts(y, train_mask)
    val_forget, val_retain = split_counts(y, val_mask)
    test_forget, test_retain = split_counts(y, test_mask)

    result = MethodResult(
        method=method,
        input_csv=str(path),
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_test=int(test_mask.sum()),
        train_forget=train_forget,
        train_retain=train_retain,
        val_forget=val_forget,
        val_retain=val_retain,
        test_forget=test_forget,
        test_retain=test_retain,
        classifier_raw_val_auc=raw_val_auc,
        classifier_attack_val_auc=attack_auc(raw_val_auc),
        classifier_raw_test_auc=raw_test_auc,
        classifier_attack_test_auc=attack_auc(raw_test_auc),
        classifier_test_acc=float(accuracy_score(y[test_mask], test_pred)),
        rouge_raw_test_auc=rouge_raw_test_auc,
        rouge_attack_test_auc=attack_auc(rouge_raw_test_auc),
        correct_raw_test_auc=correct_raw_test_auc,
        correct_attack_test_auc=attack_auc(correct_raw_test_auc),
    )
    return result, scalar_rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.8)


def draw(results: list[MethodResult]) -> list[Path]:
    configure_matplotlib()
    methods = [item.method for item in results]
    values = [item.classifier_attack_test_auc for item in results]
    x = np.arange(len(methods), dtype=float)

    fig, ax = plt.subplots(figsize=(5.9, 2.75))
    ax.bar(
        x,
        values,
        width=0.58,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="#333333",
        linewidth=0.55,
        zorder=3,
    )
    ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.15, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Agent-level MIA AUROC")
    ax.set_ylim(0.48, max(0.62, max(values) + 0.035))
    ax.set_yticks(np.arange(0.50, ax.get_ylim()[1] + 1e-6, 0.05))
    style_axes(ax)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.96)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_DIR / f"{STEM}.svg",
        OUTPUT_DIR / f"{STEM}.png",
        OUTPUT_DIR / f"{STEM}.pdf",
    ]
    for path in outputs:
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return outputs


def write_outputs(results: list[MethodResult], sample_rows: list[dict[str, Any]], outputs: list[Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_dicts = [item.__dict__ for item in results]
    with (OUTPUT_DIR / f"{STEM}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(result_dicts)
    with (OUTPUT_DIR / f"{STEM}_sample_scores.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "sample_id", "mia_split", "member_label", "dataset_split", "rouge_l", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    payload = {
        "dataset": "TOFU",
        "split": "for05_re95",
        "task": "Agent-level membership inference attack",
        "member_class": "forget",
        "non_member_class": "retain",
        "manifest_dir": str(MANIFEST_DIR),
        "manifest_policy": "fixed project mia_train_ids/mia_val_ids/mia_test_ids; duplicate IDs are preserved by manifest counts",
        "primary_metric": "classifier_attack_test_auc = max(raw_test_auc, 1 - raw_test_auc)",
        "feature_set": [
            "judge_correct",
            "rouge_l(final_answer, reference_answer)",
            "log1p(prediction_length)",
            "log1p(final_answer_length)",
            "final_answer_length/reference_answer_length",
            "empty_prediction",
            "refusal_phrase_flag",
        ],
        "outputs": [str(path) for path in outputs],
        "results": result_dicts,
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    manifest = load_manifest()
    results = []
    sample_rows = []
    for method in METHOD_ORDER:
        result, rows = compute_method(method, METHOD_FILES[method], manifest)
        results.append(result)
        sample_rows.extend(rows)

    outputs = draw(results)
    write_outputs(results, sample_rows, outputs)
    print("Agent-level MIA AUROC")
    for item in results:
        print(
            f"{item.method:<8} test_attack_auc={item.classifier_attack_test_auc:.4f} "
            f"raw={item.classifier_raw_test_auc:.4f} "
            f"rouge_attack={item.rouge_attack_test_auc:.4f} "
            f"n_test={item.n_test} forget={item.test_forget} retain={item.test_retain}"
        )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
