from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .final_metrics import normalize_text, rouge_l_f1


def norm(text: str) -> str:
    return normalize_text(text)


def rouge_l(prediction: str, answer: str) -> float:
    return rouge_l_f1(prediction, answer)


def compare(
    rows: list[dict],
    positive_split: str,
    negative_split: str,
    feature: str,
    *,
    seed: int = 42,
    bootstrap_draws: int = 1000,
) -> dict:
    positive = np.asarray([float(row[feature]) for row in rows if row["split"] == positive_split], dtype=float)
    negative = np.asarray([float(row[feature]) for row in rows if row["split"] == negative_split], dtype=float)
    if not len(positive) or not len(negative):
        raise ValueError(f"Missing {positive_split} or {negative_split} scores for {feature}")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError(f"Non-finite {feature} score")
    labels = np.concatenate((np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)))
    scores = np.concatenate((positive, negative))
    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(bootstrap_draws):
        sampled_positive = rng.choice(positive, len(positive), replace=True)
        sampled_negative = rng.choice(negative, len(negative), replace=True)
        sampled_scores = np.concatenate((sampled_positive, sampled_negative))
        bootstrap.append(float(roc_auc_score(labels, sampled_scores)))
    interval = [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))] if bootstrap else [math.nan, math.nan]
    return {
        "positive_split": positive_split,
        "negative_split": negative_split,
        "feature": feature,
        "n_positive": len(positive),
        "n_negative": len(negative),
        "auroc": auroc,
        "auroc_flipped": max(auroc, 1.0 - auroc),
        "balanced_accuracy_best_threshold": float(np.max((1.0 + tpr - fpr) / 2.0)),
        "tpr_at_1pct_fpr": float(np.max(tpr[fpr <= 0.01])),
        "auroc_bootstrap_95ci": interval,
        "positive_mean_score": float(np.mean(positive)),
        "negative_mean_score": float(np.mean(negative)),
    }
