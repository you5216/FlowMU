                      
                                                                     

                                                                               
                                                                           
                                                    

                                                  

                                                                          
                                                                
   

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_tofu_for05_agent_mia_variants import (              
    MANIFEST_DIR,
    METHOD_COLORS,
    METHOD_FILES,
    METHOD_ORDER,
    OUTPUT_DIR as VARIANT_OUTPUT_DIR,
    assign_manifest_splits,
    fit_eval,
    is_forget,
    load_manifest,
    load_rows,
    row_features,
    row_id,
)


OUTPUT_DIR = VARIANT_OUTPUT_DIR.parent / "tofu_for05_agent_mia_repeated_utility"
STEM = "tofu_for05_agent_mia_repeated_utility"
METHOD_SEED_OFFSET = {method: idx * 100_003 for idx, method in enumerate(METHOD_ORDER)}


def prepare_method(method: str, path: Path, manifest: dict[str, list[str]]) -> dict[str, object]:
    rows = load_rows(path)
    splits = assign_manifest_splits(rows, manifest)
    text_features = []
    labels = []
    utilities = []
    sample_rows = []
    for idx, (row, mia_split) in enumerate(zip(rows, splits)):
        _reference_features, black_box_features, meta = row_features(row)
        label = is_forget(row)
        text_features.append(black_box_features)
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
    return {
        "method": method,
        "input_csv": str(path),
        "features": np.asarray(text_features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "split_arr": np.asarray(splits, dtype=object),
        "utilities": np.asarray(utilities, dtype=np.float32),
        "sample_rows": sample_rows,
    }


def randomized_match_split(
    labels: np.ndarray,
    split_arr: np.ndarray,
    utilities: np.ndarray,
    rng: np.random.Generator,
    *,
    k_neighbors: int,
    temperature: float,
) -> dict[str, np.ndarray]:
    matched: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        split_idx = np.where(split_arr == split)[0]
        forget_idx = split_idx[labels[split_idx] == 1]
        retain_idx = split_idx[labels[split_idx] == 0]
        if forget_idx.size == 0 or retain_idx.size < forget_idx.size:
            raise ValueError(f"Cannot match split={split}: forget={forget_idx.size}, retain={retain_idx.size}")

        available = retain_idx.copy()
        rng.shuffle(available)
        selected = []
                                                                               
        forget_order = forget_idx.copy()
        rng.shuffle(forget_order)
        for f_idx in forget_order:
            distances = np.abs(utilities[available] - utilities[f_idx])
            order = np.argsort(distances, kind="mergesort")
            pool_size = min(k_neighbors, order.size)
            pool_order = order[:pool_size]
            pool = available[pool_order]
            pool_dist = distances[pool_order].astype(np.float64)
            weights = np.exp(-pool_dist / max(temperature, 1e-8))
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                chosen_pos = int(pool_order[0])
            else:
                chosen_in_pool = int(rng.choice(np.arange(pool_size), p=weights / weights.sum()))
                chosen_pos = int(pool_order[chosen_in_pool])
            chosen = int(available[chosen_pos])
            selected.extend([int(f_idx), chosen])
            available = np.delete(available, chosen_pos)
        matched[split] = np.asarray(sorted(selected), dtype=np.int64)
    return matched


def compute_repeats(
    prepared: dict[str, object],
    *,
    repeats: int,
    seed: int,
    k_neighbors: int,
    temperature: float,
) -> list[dict[str, object]]:
    method = str(prepared["method"])
    features = prepared["features"]
    labels = prepared["labels"]
    split_arr = prepared["split_arr"]
    utilities = prepared["utilities"]
    input_csv = str(prepared["input_csv"])
    rows = []
    for repeat_idx in range(repeats):
        rng = np.random.default_rng(seed + repeat_idx * 1009 + METHOD_SEED_OFFSET[method])
        matched = randomized_match_split(
            labels,
            split_arr,
            utilities,
            rng,
            k_neighbors=k_neighbors,
            temperature=temperature,
        )
        result = fit_eval(
            features,
            labels,
            split_arr,
            utilities,
            method=method,
            variant="repeated_utility_normalized_black_box",
            input_csv=input_csv,
            matched_indices=matched,
        )
        rows.append(
            {
                "method": method,
                "repeat": repeat_idx,
                "attack_test_auc": result.attack_test_auc,
                "raw_test_auc": result.raw_test_auc,
                "attack_val_auc": result.attack_val_auc,
                "raw_val_auc": result.raw_val_auc,
                "test_acc": result.test_acc,
                "n_train": result.n_train,
                "n_val": result.n_val,
                "n_test": result.n_test,
                "train_forget": result.train_forget,
                "train_retain": result.train_retain,
                "val_forget": result.val_forget,
                "val_retain": result.val_retain,
                "test_forget": result.test_forget,
                "test_retain": result.test_retain,
                "test_forget_utility_mean": result.test_forget_utility_mean,
                "test_retain_utility_mean": result.test_retain_utility_mean,
                "test_utility_gap_abs": result.test_utility_gap_abs,
            }
        )
    return rows


def summarize_repeat_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for method in METHOD_ORDER:
        vals = np.asarray([float(row["attack_test_auc"]) for row in rows if row["method"] == method], dtype=np.float64)
        gaps = np.asarray([float(row["test_utility_gap_abs"]) for row in rows if row["method"] == method], dtype=np.float64)
        if vals.size == 0:
            raise ValueError(f"No repeat rows for method={method}")
        summaries.append(
            {
                "method": method,
                "repeats": int(vals.size),
                "attack_test_auc_mean": float(vals.mean()),
                "attack_test_auc_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "attack_test_auc_p05": float(np.quantile(vals, 0.05)),
                "attack_test_auc_median": float(np.quantile(vals, 0.50)),
                "attack_test_auc_p95": float(np.quantile(vals, 0.95)),
                "privacy_risk_mean": float(np.mean(2.0 * np.abs(vals - 0.5))),
                "test_utility_gap_abs_mean": float(gaps.mean()),
                "test_utility_gap_abs_std": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
            }
        )
    return summaries


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


def draw(summaries: list[dict[str, object]], repeat_rows: list[dict[str, object]]) -> list[Path]:
    configure_matplotlib()
    x = np.arange(len(METHOD_ORDER), dtype=float)
    means = np.asarray([float(row["attack_test_auc_mean"]) for row in summaries], dtype=np.float64)
    stds = np.asarray([float(row["attack_test_auc_std"]) for row in summaries], dtype=np.float64)
    p05 = np.asarray([float(row["attack_test_auc_p05"]) for row in summaries], dtype=np.float64)
    p95 = np.asarray([float(row["attack_test_auc_p95"]) for row in summaries], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.2, 2.85))
    ax.bar(
        x,
        means,
        width=0.58,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        edgecolor="#333333",
        linewidth=0.55,
        zorder=3,
    )
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        zorder=4,
        label="mean ± std",
    )
    ax.vlines(x + 0.20, p05, p95, color="#222222", linewidth=1.1, alpha=0.75, zorder=4, label="5-95%")

    rng = np.random.default_rng(123)
    for idx, method in enumerate(METHOD_ORDER):
        vals = [float(row["attack_test_auc"]) for row in repeat_rows if row["method"] == method]
        if len(vals) > 120:
            vals = list(rng.choice(np.asarray(vals), size=120, replace=False))
        jitter = rng.normal(0.0, 0.025, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx, dtype=float) + jitter,
            vals,
            s=4,
            color="#1f1f1f",
            alpha=0.14,
            linewidths=0,
            zorder=2,
        )

    ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.05, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel("MIA AUROC")
    ax.set_ylim(0.48, max(0.72, float(np.max(p95)) + 0.035))
    ax.set_yticks(np.arange(0.50, ax.get_ylim()[1] + 1e-6, 0.05))
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.93)

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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 1:
        raise ValueError("--repeats must be > 1")
    manifest = load_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_repeat_rows: list[dict[str, object]] = []
    all_sample_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        prepared = prepare_method(method, METHOD_FILES[method], manifest)
        all_sample_rows.extend(prepared["sample_rows"])
        method_rows = compute_repeats(
            prepared,
            repeats=args.repeats,
            seed=args.seed,
            k_neighbors=args.k_neighbors,
            temperature=args.temperature,
        )
        all_repeat_rows.extend(method_rows)
        vals = np.asarray([float(row["attack_test_auc"]) for row in method_rows], dtype=np.float64)
        print(f"{method:<8} mean={vals.mean():.4f} std={vals.std(ddof=1):.4f}")

    summaries = summarize_repeat_rows(all_repeat_rows)
    outputs = draw(summaries, all_repeat_rows)

    write_csv(OUTPUT_DIR / f"{STEM}_repeat_scores.csv", all_repeat_rows)
    write_csv(OUTPUT_DIR / f"{STEM}_summary.csv", summaries)
    write_csv(OUTPUT_DIR / f"{STEM}_sample_scores.csv", all_sample_rows)
    payload = {
        "dataset": "TOFU",
        "split": "for05_re95",
        "task": "Repeated utility-matched black-box Agent-level MIA",
        "manifest_dir": str(MANIFEST_DIR),
        "member_class": "forget",
        "non_member_class": "retain",
        "metric": "attack_test_auc = max(raw_test_auc, 1 - raw_test_auc)",
        "repeats": args.repeats,
        "seed": args.seed,
        "k_neighbors": args.k_neighbors,
        "temperature": args.temperature,
        "matching_policy": "Within each fixed MIA split, pair every forget sample with one retain sample sampled from the k nearest retain candidates by utility_score = judge_correct + 0.25*Rouge-L. The attacker sees only black-box output text features.",
        "outputs": [str(path) for path in outputs],
        "summary": summaries,
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
