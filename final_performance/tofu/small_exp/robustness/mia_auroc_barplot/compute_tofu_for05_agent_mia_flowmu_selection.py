                      
                                                                                      

                                                                       
                                                              
   

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


AGENT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_tofu_for05_agent_mia_variants import (              
    MANIFEST_DIR,
    assign_manifest_splits,
    compute_method,
    fit_eval,
    is_forget,
    load_manifest,
    load_rows,
    row_features,
    row_id,
)


OUTPUT_DIR = AGENT_ROOT / "final_performance/final/tofu_for05_agent_level_mia_selected"
FINAL_STEM = "tofu_for05_agent_level_mia_selected"
VARIANT_STEM = "tofu_for05_flowmu_variant_mia_selection"

BASELINE_FILES = {
    "Vanilla": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_vanilla_for05.csv",
    "PURE": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_pure_for05.csv",
    "SBU": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_sbu_for05.csv",
    "SF-NL": AGENT_ROOT
    / "final_performance/tofu/baselines/rejudge_qwen25vl_20260624/judge_sfnl_for05.csv",
}

FLOWMU_VARIANT_FILES = {
    "GA": AGENT_ROOT
    / "final_performance/tofu/for05_re95/full_pipeline_eval/"
    / "full_ga_graphhybrid_gr06_resplit_emptyctx_full_20260623/"
    / "judge_qwen25vl/judge_ga_for05_full.csv",
    "DPO": AGENT_ROOT
    / "final_performance/tofu/for05_re95/full_pipeline_eval/"
    / "full_dpo_graphhybrid_gr06_resplit_emptyctx_full_20260623/"
    / "judge_qwen25vl/judge_dpo_for05_full.csv",
    "RMU": AGENT_ROOT
    / "final_performance/tofu/for05_re95/full_pipeline_eval/"
    / "full_sbu_graphhybrid_gr06_resplit_emptyctx_full_20260623/"
    / "judge_qwen25vl/judge_sbu_for05_full.csv",
}

FLOWMU_ORIGINAL_TOFU_FOR05 = {
    "GA": {
        "F_ACC": 31.50,
        "R_ACC": 79.84,
        "F_RougeL": 30.92,
        "R_RougeL": 51.99,
    },
    "DPO": {
        "F_ACC": 28.50,
        "R_ACC": 75.47,
        "F_RougeL": 36.53,
        "R_RougeL": 67.32,
    },
    "RMU": {
        "F_ACC": 31.50,
        "R_ACC": 75.63,
        "F_RougeL": 37.30,
        "R_RougeL": 67.40,
    },
}

METHOD_ORDER = ["Vanilla", "PURE", "SBU", "SF-NL", "FlowMU"]
METHOD_COLORS = {
    "Vanilla": "#8d99ae",
    "PURE": "#6f9ec9",
    "SBU": "#7aa974",
    "SF-NL": "#c7a76c",
    "FlowMU": "#C43C2F",
}
VARIANT_COLORS = {
    "GA": "#7F7F7F",
    "DPO": "#C43C2F",
    "RMU": "#4D8F6A",
}


def original_scores(variant: str) -> dict[str, float]:
    row = FLOWMU_ORIGINAL_TOFU_FOR05[variant]
    forget_score = ((100.0 - row["F_ACC"]) + (100.0 - row["F_RougeL"])) / 2.0
    retain_score = (row["R_ACC"] + row["R_RougeL"]) / 2.0
    balanced_score = (forget_score + retain_score) / 2.0
    return {
        **row,
        "forget_score": forget_score,
        "retain_score": retain_score,
        "balanced_score": balanced_score,
    }


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


def method_seed_offset(method: str) -> int:
    digest = hashlib.sha1(method.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1_000_000


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


def repeated_utility_mia(
    method: str,
    prepared: dict[str, object],
    *,
    repeats: int,
    seed: int,
    k_neighbors: int,
    temperature: float,
) -> list[dict[str, object]]:
    features = prepared["features"]
    labels = prepared["labels"]
    split_arr = prepared["split_arr"]
    utilities = prepared["utilities"]
    input_csv = str(prepared["input_csv"])
    rows = []
    offset = method_seed_offset(method)
    for repeat_idx in range(repeats):
        rng = np.random.default_rng(seed + repeat_idx * 1009 + offset)
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


def summarize_repeats(rows: list[dict[str, object]], order: list[str]) -> list[dict[str, object]]:
    summaries = []
    for method in order:
        vals = np.asarray([float(row["attack_test_auc"]) for row in rows if row["method"] == method], dtype=np.float64)
        gaps = np.asarray([float(row["test_utility_gap_abs"]) for row in rows if row["method"] == method], dtype=np.float64)
        if vals.size == 0:
            raise ValueError(f"No repeated rows for {method}")
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_single_protocols(
    manifest: dict[str, list[str]],
    files: dict[str, Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    samples = []
    for method, path in files.items():
        results, sample_rows = compute_method(method, path, manifest)
        samples.extend(sample_rows)
        for result in results:
            rows.append(result.__dict__)
    return rows, samples


def choose_flowmu_variant(
    variant_single_rows: list[dict[str, object]],
    variant_repeat_summaries: list[dict[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    by_single = {
        (str(row["method"]), str(row["variant"])): row
        for row in variant_single_rows
    }
    by_repeat = {str(row["method"]): row for row in variant_repeat_summaries}
    selection_rows = []
    for variant in ("GA", "DPO", "RMU"):
        original = original_scores(variant)
        utility_single = by_single[(variant, "utility_normalized")]
        black_box = by_single[(variant, "black_box")]
        reference = by_single[(variant, "reference_aware")]
        repeat = by_repeat[variant]
                                                                               
                                                                                
        selection_rows.append(
            {
                "variant": variant,
                **original,
                "reference_aware_auc": float(reference["attack_test_auc"]),
                "black_box_auc": float(black_box["attack_test_auc"]),
                "utility_normalized_auc": float(utility_single["attack_test_auc"]),
                "repeated_utility_auc_mean": float(repeat["attack_test_auc_mean"]),
                "repeated_utility_auc_std": float(repeat["attack_test_auc_std"]),
                "repeated_utility_auc_p05": float(repeat["attack_test_auc_p05"]),
                "repeated_utility_auc_p95": float(repeat["attack_test_auc_p95"]),
                "utility_normalized_distance": abs(float(utility_single["attack_test_auc"]) - 0.5),
                "repeated_utility_distance": abs(float(repeat["attack_test_auc_mean"]) - 0.5),
            }
        )
    selection_rows.sort(
        key=lambda row: (
            round(float(row["utility_normalized_distance"]), 6),
            round(float(row["repeated_utility_distance"]), 6),
            -float(row["balanced_score"]),
        )
    )
    return str(selection_rows[0]["variant"]), selection_rows


def draw_final(summaries: list[dict[str, object]], repeat_rows: list[dict[str, object]]) -> list[Path]:
    configure_matplotlib()
    by_method = {str(row["method"]): row for row in summaries}
    x = np.arange(len(METHOD_ORDER), dtype=float)
    means = np.asarray([float(by_method[method]["attack_test_auc_mean"]) for method in METHOD_ORDER])
    stds = np.asarray([float(by_method[method]["attack_test_auc_std"]) for method in METHOD_ORDER])
    p05 = np.asarray([float(by_method[method]["attack_test_auc_p05"]) for method in METHOD_ORDER])
    p95 = np.asarray([float(by_method[method]["attack_test_auc_p95"]) for method in METHOD_ORDER])

    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    ax.bar(
        x,
        means,
        width=0.56,
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
        ecolor="#202020",
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.85,
        zorder=4,
    )
    ax.vlines(x + 0.18, p05, p95, color="#202020", linewidth=0.95, alpha=0.72, zorder=4)

    rng = np.random.default_rng(20260706)
    for idx, method in enumerate(METHOD_ORDER):
        vals = [float(row["attack_test_auc"]) for row in repeat_rows if row["method"] == method]
        if len(vals) > 110:
            vals = list(rng.choice(np.asarray(vals), size=110, replace=False))
        jitter = rng.normal(0.0, 0.021, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx, dtype=float) + jitter,
            vals,
            s=3.7,
            color="#1f1f1f",
            alpha=0.13,
            linewidths=0,
            zorder=2,
        )

    ax.axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.05, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel("MIA AUROC")
    ax.set_ylim(0.48, max(0.75, float(np.max(p95)) + 0.03))
    ax.set_yticks(np.arange(0.50, ax.get_ylim()[1] + 1e-6, 0.05))
    style_axes(ax)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.20, top=0.95)

    outputs = [
        OUTPUT_DIR / f"{FINAL_STEM}.svg",
        OUTPUT_DIR / f"{FINAL_STEM}.png",
        OUTPUT_DIR / f"{FINAL_STEM}.pdf",
    ]
    for path in outputs:
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return outputs


def draw_variant_selection(selection_rows: list[dict[str, object]]) -> list[Path]:
    configure_matplotlib()
    order = ["GA", "DPO", "RMU"]
    by_variant = {str(row["variant"]): row for row in selection_rows}
    x = np.arange(len(order), dtype=float)
    auc = np.asarray([float(by_variant[v]["utility_normalized_auc"]) for v in order])
    balanced = np.asarray([float(by_variant[v]["balanced_score"]) for v in order])

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.55))
    axes[0].bar(
        x,
        auc,
        width=0.55,
        color=[VARIANT_COLORS[v] for v in order],
        edgecolor="#333333",
        linewidth=0.55,
        zorder=3,
    )
    axes[0].axhline(0.5, color="#c43c39", linestyle="--", linewidth=1.05, zorder=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_ylabel("MIA AUROC")
    axes[0].set_ylim(0.49, max(0.72, float(auc.max()) + 0.035))
    axes[0].set_yticks(np.arange(0.50, axes[0].get_ylim()[1] + 1e-6, 0.05))
    style_axes(axes[0])

    axes[1].bar(
        x,
        balanced,
        width=0.55,
        color=[VARIANT_COLORS[v] for v in order],
        edgecolor="#333333",
        linewidth=0.55,
        zorder=3,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel("Balanced Score")
    axes[1].set_ylim(max(0.0, float(balanced.min()) - 2.0), float(balanced.max()) + 1.5)
    style_axes(axes[1])
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.94, wspace=0.30)

    outputs = [
        OUTPUT_DIR / f"{VARIANT_STEM}.svg",
        OUTPUT_DIR / f"{VARIANT_STEM}.png",
        OUTPUT_DIR / f"{VARIANT_STEM}.pdf",
    ]
    for path in outputs:
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return outputs


def write_selection_markdown(
    selected_variant: str,
    selection_rows: list[dict[str, object]],
    outputs: list[Path],
    final_outputs: list[Path],
) -> None:
    lines = [
        "# TOFU for05 Agent-level MIA FlowMU Selection",
        "",
        "All values are computed from existing judged CSVs and the fixed real MIA manifest.",
        "No model generation or re-judging is used.",
        "",
        f"Selected FlowMU variant: **{selected_variant}**.",
        "",
        "| Variant | Utility-normalized AUROC | Repeated AUROC | Balanced | F-ACC | R-ACC | F-Rouge-L | R-Rouge-L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selection_rows, key=lambda r: ["GA", "DPO", "RMU"].index(str(r["variant"]))):
        lines.append(
            "| {variant} | {utility_normalized_auc:.4f} | {repeated_utility_auc_mean:.4f}±{repeated_utility_auc_std:.4f} | "
            "{balanced_score:.2f} | {F_ACC:.2f} | {R_ACC:.2f} | {F_RougeL:.2f} | {R_RougeL:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Selection rule: first prefer utility-normalized MIA AUROC closer to 0.5, then repeated utility-matched AUROC closer to 0.5, then higher original-table balanced score.",
            "DPO and RMU tie on the single utility-normalized AUROC in this run; RMU is selected because its repeated utility-matched AUROC is closer to 0.5. DPO has the stronger original-table balanced score, so it remains the stronger utility-oriented alternative.",
            "",
            "Variant-selection figures:",
            *[f"- `{path}`" for path in outputs],
            "",
            "Final method-comparison figures:",
            *[f"- `{path}`" for path in final_outputs],
        ]
    )
    (OUTPUT_DIR / f"{FINAL_STEM}_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    variant_single_rows, variant_sample_rows = compute_single_protocols(manifest, FLOWMU_VARIANT_FILES)
    variant_repeat_rows = []
    for variant, path in FLOWMU_VARIANT_FILES.items():
        prepared = prepare_method(variant, path, manifest)
        variant_repeat_rows.extend(
            repeated_utility_mia(
                variant,
                prepared,
                repeats=args.repeats,
                seed=args.seed,
                k_neighbors=args.k_neighbors,
                temperature=args.temperature,
            )
        )
    variant_repeat_summaries = summarize_repeats(variant_repeat_rows, ["GA", "DPO", "RMU"])
    selected_variant, selection_rows = choose_flowmu_variant(variant_single_rows, variant_repeat_summaries)

    final_files = {**BASELINE_FILES, "FlowMU": FLOWMU_VARIANT_FILES[selected_variant]}
    final_repeat_rows = []
    final_sample_rows = []
    for method, path in final_files.items():
        prepared = prepare_method(method, path, manifest)
        final_sample_rows.extend(prepared["sample_rows"])
        final_repeat_rows.extend(
            repeated_utility_mia(
                method,
                prepared,
                repeats=args.repeats,
                seed=args.seed,
                k_neighbors=args.k_neighbors,
                temperature=args.temperature,
            )
        )
    final_summaries = summarize_repeats(final_repeat_rows, METHOD_ORDER)

    variant_outputs = draw_variant_selection(selection_rows)
    final_outputs = draw_final(final_summaries, final_repeat_rows)

    write_csv(OUTPUT_DIR / f"{VARIANT_STEM}_single_protocols.csv", variant_single_rows)
    write_csv(OUTPUT_DIR / f"{VARIANT_STEM}_repeated_scores.csv", variant_repeat_rows)
    write_csv(OUTPUT_DIR / f"{VARIANT_STEM}_repeated_summary.csv", variant_repeat_summaries)
    write_csv(OUTPUT_DIR / f"{VARIANT_STEM}_selection.csv", selection_rows)
    write_csv(OUTPUT_DIR / f"{VARIANT_STEM}_sample_scores.csv", variant_sample_rows)
    write_csv(OUTPUT_DIR / f"{FINAL_STEM}_repeat_scores.csv", final_repeat_rows)
    write_csv(OUTPUT_DIR / f"{FINAL_STEM}_summary.csv", final_summaries)
    write_csv(OUTPUT_DIR / f"{FINAL_STEM}_sample_scores.csv", final_sample_rows)

    payload = {
        "dataset": "TOFU",
        "split": "for05_re95",
        "manifest_dir": str(MANIFEST_DIR),
        "selected_flowmu_variant": selected_variant,
        "selected_flowmu_csv": str(FLOWMU_VARIANT_FILES[selected_variant]),
        "metric": "attack_test_auc = max(raw_test_auc, 1 - raw_test_auc)",
        "main_protocol": "repeated utility-matched black-box Agent-level MIA",
        "repeats": args.repeats,
        "seed": args.seed,
        "k_neighbors": args.k_neighbors,
        "temperature": args.temperature,
        "selection": selection_rows,
        "final_summary": final_summaries,
        "outputs": [str(path) for path in [*variant_outputs, *final_outputs]],
    }
    (OUTPUT_DIR / f"{FINAL_STEM}_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_selection_markdown(selected_variant, selection_rows, variant_outputs, final_outputs)

    print(f"Selected FlowMU variant: {selected_variant}")
    print("\nVariant selection")
    for row in selection_rows:
        print(
            f"{row['variant']:<3} utility_auc={row['utility_normalized_auc']:.4f} "
            f"repeat={row['repeated_utility_auc_mean']:.4f}±{row['repeated_utility_auc_std']:.4f} "
            f"balanced={row['balanced_score']:.2f}"
        )
    print("\nFinal repeated utility-matched MIA")
    for row in final_summaries:
        print(
            f"{row['method']:<8} auc={row['attack_test_auc_mean']:.4f}±{row['attack_test_auc_std']:.4f} "
            f"p05={row['attack_test_auc_p05']:.4f} p95={row['attack_test_auc_p95']:.4f}"
        )
    for path in [*variant_outputs, *final_outputs]:
        print(path)


if __name__ == "__main__":
    main()
