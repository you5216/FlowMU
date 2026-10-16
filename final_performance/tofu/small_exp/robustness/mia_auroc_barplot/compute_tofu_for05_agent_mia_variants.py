                      
                                                   

         
                                                                             
                                                          
                                                                                
                                 

                                                                           
   

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
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
OUTPUT_DIR = AGENT_ROOT / "final_performance/final/tofu_for05_agent_mia_variants"
STEM = "tofu_for05_agent_mia_variants"

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
VARIANT_ORDER = ["reference_aware", "black_box", "utility_normalized"]
VARIANT_LABELS = {
    "reference_aware": "Reference-aware",
    "black_box": "Black-box",
    "utility_normalized": "Utility-normalized",
}
METHOD_COLORS = {
    "Vanilla": "#8d99ae",
    "PURE": "#6f9ec9",
    "SBU": "#7aa974",
    "SF-NL": "#c7a76c",
    "FlowMU": "#C43C2F",
}


@dataclass(frozen=True)
class VariantResult:
    method: str
    variant: str
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
    raw_val_auc: float
    attack_val_auc: float
    raw_test_auc: float
    attack_test_auc: float
    test_acc: float
    test_forget_utility_mean: float
    test_retain_utility_mean: float
    test_utility_gap_abs: float
    matched_policy: str


def load_manifest() -> dict[str, list[str]]:
    manifest = {}
    for split in ("train", "val", "test"):
        path = MANIFEST_DIR / f"mia_{split}_ids.json"
        manifest[split] = [str(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    return manifest


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def row_id(row: dict[str, Any], idx: int) -> str:
    raw = str(row.get("id") or row.get("question") or idx)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def assign_manifest_splits(rows: list[dict[str, str]], manifest: dict[str, list[str]]) -> list[str]:
    counters = {split: Counter(ids) for split, ids in manifest.items()}
    row_ids = [row_id(row, idx) for idx, row in enumerate(rows)]
    csv_counts = Counter(row_ids)
    manifest_counts = Counter()
    for counter in counters.values():
        manifest_counts.update(counter)
    if csv_counts != manifest_counts:
        missing = manifest_counts - csv_counts
        extra = csv_counts - manifest_counts
        raise ValueError(
            "CSV row IDs do not match fixed MIA manifest: "
            f"missing={dict(missing.most_common(5))}, extra={dict(extra.most_common(5))}"
        )

    assigned = []
    for sid in row_ids:
        selected = ""
        for split in ("train", "val", "test"):
            if counters[split][sid] > 0:
                counters[split][sid] -= 1
                selected = split
                break
        if not selected:
            raise RuntimeError(f"Could not assign row id {sid}")
        assigned.append(selected)
    return assigned


def is_forget(row: dict[str, str]) -> int:
    split = (row.get("split") or row.get("split_type") or "").strip().lower()
    if split == "forget":
        return 1
    if split == "retain":
        return 0
    raise ValueError(f"Unknown split {split!r}")


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or ""))


def token_count(text: str) -> int:
    return len(tokens(text))


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", str(text or ""))))


def ratio(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def text_ratio(text: str, pattern: str) -> float:
    text = str(text or "")
    return ratio(len(re.findall(pattern, text)), max(1, len(text)))


def flag_any(text: str, phrases: list[str]) -> float:
    low = str(text or "").casefold()
    return 1.0 if any(phrase in low for phrase in phrases) else 0.0


def correct_flag(row: dict[str, str], rouge_l: float) -> float:
    verdict = str(row.get("judge_verdict") or "").strip().lower()
    if verdict:
        return 1.0 if verdict == "correct" else 0.0
    return 1.0 if rouge_l >= 0.95 else 0.0


def black_box_features(prediction: str, final_answer: str) -> list[float]:
    pred = str(prediction or "")
    final = str(final_answer or "")
    pred_tokens = tokens(pred)
    final_tokens = tokens(final)
    pred_len = len(pred_tokens)
    final_len = len(final_tokens)
    pred_chars = len(pred)
    final_chars = len(final)
    unique_ratio = ratio(len(set(tok.casefold() for tok in final_tokens)), max(1, final_len))
    avg_tok_len = ratio(sum(len(tok) for tok in final_tokens), max(1, final_len))
    refusal = flag_any(
        pred,
        [
            "i don't know",
            "i do not know",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "not enough information",
            "no information",
            "not provided",
        ],
    )
    uncertainty = flag_any(pred, ["maybe", "possibly", "likely", "unclear", "unknown", "not sure"])
    apology = flag_any(pred, ["sorry", "apologize", "apologies"])
    return [
        math.log1p(pred_len),
        math.log1p(final_len),
        math.log1p(pred_chars),
        math.log1p(final_chars),
        math.log1p(sentence_count(pred)),
        math.log1p(pred.count("\n")),
        unique_ratio,
        avg_tok_len,
        text_ratio(pred, r"\d"),
        text_ratio(pred, r"[A-Z]"),
        text_ratio(pred, r"[^\w\s]"),
        1.0 if not pred.strip() else 0.0,
        refusal,
        uncertainty,
        apology,
    ]


def row_features(row: dict[str, str]) -> tuple[list[float], list[float], dict[str, float]]:
    prediction = str(row.get("prediction") or "")
    final_answer = extract_final_answer(prediction)
    reference = str(row.get("reference_answer") or "")
    rouge_l = rouge_l_f1(final_answer, reference)
    correct = correct_flag(row, rouge_l)
    text_features = black_box_features(prediction, final_answer)
    ref_len = max(1, token_count(reference))
    final_len = token_count(final_answer)
    length_ratio = min(final_len / ref_len, 5.0)
    reference_aware = [correct, rouge_l, length_ratio, *text_features]
    utility = float(correct + 0.25 * rouge_l)
    meta = {
        "rouge_l": float(rouge_l),
        "correct": float(correct),
        "utility_score": utility,
        "prediction_length": float(token_count(prediction)),
        "final_answer_length": float(final_len),
    }
    return reference_aware, text_features, meta


def attack_auc(raw_auc: float) -> float:
    return float(max(raw_auc, 1.0 - raw_auc))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.astype(int).tolist())) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def fit_eval(
    features: np.ndarray,
    labels: np.ndarray,
    split_arr: np.ndarray,
    utilities: np.ndarray,
    *,
    method: str,
    variant: str,
    input_csv: str,
    matched_indices: dict[str, np.ndarray] | None = None,
) -> VariantResult:
    if matched_indices is None:
        train_idx = np.where(split_arr == "train")[0]
        val_idx = np.where(split_arr == "val")[0]
        test_idx = np.where(split_arr == "test")[0]
        policy = "fixed manifest full split"
    else:
        train_idx = matched_indices["train"]
        val_idx = matched_indices["val"]
        test_idx = matched_indices["test"]
        policy = "1:1 nearest-neighbor matching by utility_score = judge_correct + 0.25*rouge_l within each manifest split"

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    )
    clf.fit(features[train_idx], labels[train_idx])
    val_prob = clf.predict_proba(features[val_idx])[:, 1]
    test_prob = clf.predict_proba(features[test_idx])[:, 1]
    test_pred = (test_prob >= 0.5).astype(np.int64)

    def count(idx: np.ndarray) -> tuple[int, int]:
        selected = labels[idx]
        forget = int(selected.sum())
        return forget, int(selected.shape[0] - forget)

    train_forget, train_retain = count(train_idx)
    val_forget, val_retain = count(val_idx)
    test_forget, test_retain = count(test_idx)
    test_forget_util = utilities[test_idx][labels[test_idx] == 1]
    test_retain_util = utilities[test_idx][labels[test_idx] == 0]
    forget_mean = float(test_forget_util.mean()) if test_forget_util.size else 0.0
    retain_mean = float(test_retain_util.mean()) if test_retain_util.size else 0.0
    raw_val_auc = safe_auc(labels[val_idx], val_prob)
    raw_test_auc = safe_auc(labels[test_idx], test_prob)
    return VariantResult(
        method=method,
        variant=variant,
        input_csv=input_csv,
        n_train=int(train_idx.shape[0]),
        n_val=int(val_idx.shape[0]),
        n_test=int(test_idx.shape[0]),
        train_forget=train_forget,
        train_retain=train_retain,
        val_forget=val_forget,
        val_retain=val_retain,
        test_forget=test_forget,
        test_retain=test_retain,
        raw_val_auc=raw_val_auc,
        attack_val_auc=attack_auc(raw_val_auc),
        raw_test_auc=raw_test_auc,
        attack_test_auc=attack_auc(raw_test_auc),
        test_acc=float(accuracy_score(labels[test_idx], test_pred)),
        test_forget_utility_mean=forget_mean,
        test_retain_utility_mean=retain_mean,
        test_utility_gap_abs=abs(forget_mean - retain_mean),
        matched_policy=policy,
    )


def matched_indices_for_split(labels: np.ndarray, split_arr: np.ndarray, utilities: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for split in ("train", "val", "test"):
        split_idx = np.where(split_arr == split)[0]
        forget_idx = split_idx[labels[split_idx] == 1]
        retain_idx = split_idx[labels[split_idx] == 0]
        if forget_idx.size == 0 or retain_idx.size < forget_idx.size:
            raise ValueError(f"Cannot utility-match split {split}: forget={forget_idx.size}, retain={retain_idx.size}")

        available = set(retain_idx.tolist())
        matched = []
        for f_idx in sorted(forget_idx.tolist(), key=lambda idx: float(utilities[idx])):
            best = min(available, key=lambda ridx: (abs(float(utilities[ridx]) - float(utilities[f_idx])), ridx))
            available.remove(best)
            matched.extend([f_idx, best])
        out[split] = np.asarray(sorted(matched), dtype=np.int64)
    return out


def compute_method(method: str, path: Path, manifest: dict[str, list[str]]) -> tuple[list[VariantResult], list[dict[str, Any]]]:
    rows = load_rows(path)
    splits = assign_manifest_splits(rows, manifest)
    reference_features = []
    text_features = []
    labels = []
    utilities = []
    sample_rows = []
    for idx, (row, mia_split) in enumerate(zip(rows, splits)):
        ref_feat, text_feat, meta = row_features(row)
        label = is_forget(row)
        reference_features.append(ref_feat)
        text_features.append(text_feat)
        labels.append(label)
        utilities.append(meta["utility_score"])
        sample_rows.append(
            {
                "method": method,
                "sample_id": row_id(row, idx),
                "mia_split": mia_split,
                "dataset_split": "forget" if label else "retain",
                "member_label": label,
                **meta,
            }
        )

    y = np.asarray(labels, dtype=np.int64)
    split_arr = np.asarray(splits, dtype=object)
    utilities_arr = np.asarray(utilities, dtype=np.float32)
    x_reference = np.asarray(reference_features, dtype=np.float32)
    x_text = np.asarray(text_features, dtype=np.float32)
    matched = matched_indices_for_split(y, split_arr, utilities_arr)

    results = [
        fit_eval(
            x_reference,
            y,
            split_arr,
            utilities_arr,
            method=method,
            variant="reference_aware",
            input_csv=str(path),
        ),
        fit_eval(
            x_text,
            y,
            split_arr,
            utilities_arr,
            method=method,
            variant="black_box",
            input_csv=str(path),
        ),
        fit_eval(
            x_text,
            y,
            split_arr,
            utilities_arr,
            method=method,
            variant="utility_normalized",
            input_csv=str(path),
            matched_indices=matched,
        ),
    ]
    return results, sample_rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
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


def draw(results: list[VariantResult]) -> list[Path]:
    configure_matplotlib()
    by_key = {(item.method, item.variant): item.attack_test_auc for item in results}
    max_value = max(by_key.values())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), sharey=True)
    x = np.arange(len(METHOD_ORDER), dtype=float)
    for ax, variant in zip(axes, VARIANT_ORDER):
        values = [by_key[(method, variant)] for method in METHOD_ORDER]
        ax.bar(
            x,
            values,
            width=0.58,
            color=[METHOD_COLORS[method] for method in METHOD_ORDER],
            edgecolor="#333333",
            linewidth=0.55,
            zorder=3,
        )
        ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.05, zorder=2)
        ax.set_title(VARIANT_LABELS[variant], fontsize=9.5, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right", rotation_mode="anchor")
        ax.set_ylim(0.48, max(0.62, max_value + 0.035))
        ax.set_yticks(np.arange(0.50, ax.get_ylim()[1] + 1e-6, 0.05))
        style_axes(ax)
    axes[0].set_ylabel("MIA AUROC")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.26, top=0.88, wspace=0.12)

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


def write_outputs(results: list[VariantResult], sample_rows: list[dict[str, Any]], outputs: list[Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_dicts = [item.__dict__ for item in results]
    with (OUTPUT_DIR / f"{STEM}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(result_dicts)
    with (OUTPUT_DIR / f"{STEM}_sample_scores.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "sample_id",
            "mia_split",
            "dataset_split",
            "member_label",
            "rouge_l",
            "correct",
            "utility_score",
            "prediction_length",
            "final_answer_length",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    payload = {
        "dataset": "TOFU",
        "split": "for05_re95",
        "manifest_dir": str(MANIFEST_DIR),
        "member_class": "forget",
        "metric": "attack_test_auc = max(raw_test_auc, 1 - raw_test_auc)",
        "variants": {
            "reference_aware": "Agent attacker can use correctness/Rouge-L plus text statistics.",
            "black_box": "Agent attacker uses generated text statistics only; no reference answer or judge verdict.",
            "utility_normalized": "Each manifest split is 1:1 matched by utility_score = judge_correct + 0.25*Rouge-L, then attacked with black-box text statistics.",
        },
        "outputs": [str(path) for path in outputs],
        "results": result_dicts,
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    manifest = load_manifest()
    all_results = []
    all_sample_rows = []
    for method in METHOD_ORDER:
        results, sample_rows = compute_method(method, METHOD_FILES[method], manifest)
        all_results.extend(results)
        all_sample_rows.extend(sample_rows)
    outputs = draw(all_results)
    write_outputs(all_results, all_sample_rows, outputs)

    print("TOFU for05 Agent-level MIA variants")
    for variant in VARIANT_ORDER:
        print(f"\n{VARIANT_LABELS[variant]}")
        for method in METHOD_ORDER:
            item = next(r for r in all_results if r.method == method and r.variant == variant)
            print(
                f"{method:<8} attack_auc={item.attack_test_auc:.4f} raw={item.raw_test_auc:.4f} "
                f"n_test={item.n_test} f/r={item.test_forget}/{item.test_retain} "
                f"utility_gap={item.test_utility_gap_abs:.4f}"
            )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
