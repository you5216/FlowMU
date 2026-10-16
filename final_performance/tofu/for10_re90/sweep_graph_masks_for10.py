                      
                                                                 

                                                                             
                                                                               
                                                 
   

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[3]
FOR10_ROOT = AGENT_ROOT / "final_performance" / "tofu" / "for10_re90"
DEFAULT_STATS = FOR10_ROOT / "cache" / "rag_retrieval_aligned_node_stats_forget10_retain90_resplit_top3.json"
DEFAULT_SOURCE_DIR = FOR10_ROOT / "phase1_gat_retrain_retrieval_aligned_resplit_lam15_top400_20260626_skippre"
DEFAULT_PREVIOUS_EVAL = FOR10_ROOT / "retrieval_eval_graph_hybrid_top50_resplit_lambda05_retcap2_fast_20260624" / "graph_path_eval_results.csv"
DEFAULT_OUT_DIR = FOR10_ROOT / "phase1_graph_mask_sweep_resplit_20260626"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def calc_top3_metrics(top3_rows: list[list[str]], selected: set[str]) -> dict[str, float]:
    n = len(top3_rows)
    if n == 0:
        return {
            "n": 0,
            "any_at3": 0.0,
            "all_at3": 0.0,
            "rate_at3": 0.0,
            "hit_count": 0,
        }
    any_count = 0
    all_count = 0
    flag_sum = 0
    denom = 0
    for nodes in top3_rows:
        flags = [str(node_id) in selected for node_id in nodes]
        any_count += int(any(flags))
        all_count += int(bool(flags) and all(flags))
        flag_sum += sum(int(flag) for flag in flags)
        denom += len(flags)
    return {
        "n": int(n),
        "any_at3": any_count / n,
        "all_at3": all_count / n,
        "rate_at3": flag_sum / max(1, denom),
        "hit_count": int(any_count),
    }


def select_rag_nodes(
    forget_counts: Counter[str],
    retain_counts: Counter[str],
    *,
    lam: float,
    retain_cap: int,
    min_forget: int,
    max_nodes: int,
    score_positive: bool,
) -> list[tuple[str, float, int, int]]:
    rows: list[tuple[str, float, int, int]] = []
    for node_id, f_count in forget_counts.items():
        r_count = int(retain_counts.get(node_id, 0))
        if int(f_count) < int(min_forget):
            continue
        if r_count > int(retain_cap):
            continue
        score = float(f_count) - float(lam) * float(r_count)
        if score_positive and score <= 0:
            continue
        rows.append((str(node_id), score, int(f_count), r_count))
    rows.sort(key=lambda item: (item[1], item[2], -item[3], item[0]), reverse=True)
    if max_nodes > 0:
        rows = rows[: int(max_nodes)]
    return rows


def reset_graph_unlearned(graph: Any) -> Any:
    graph = deepcopy(graph)
    for _, data in graph.nodes(data=True):
        data["unlearned"] = 0
        data["drop_prob"] = float(data.get("drop_prob", 0.0) or 0.0)
        data["keep_prob"] = 1.0 - float(data.get("drop_prob", 0.0) or 0.0)
    if graph.is_multigraph():
        for _, _, _, data in graph.edges(keys=True, data=True):
            data["unlearned"] = 0
    else:
        for _, _, data in graph.edges(data=True):
            data["unlearned"] = 0
    return graph


def export_rag_graph(source_graph: Any, selected: set[str], out_path: Path) -> None:
    graph = reset_graph_unlearned(source_graph)
    for node_id, data in graph.nodes(data=True):
        if str(node_id) in selected:
            data["unlearned"] = 1
            data["drop_prob"] = 1.0
            data["keep_prob"] = 0.0
            data["weak_label"] = 1
    save_pickle(graph, out_path)


def parse_json_field(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def load_previous_memory_hits(csv_path: Path) -> dict[str, list[list[dict[str, Any]]]]:
    out: dict[str, list[list[dict[str, Any]]]] = {"forget": [], "retain": []}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split") or "")
            if split not in out:
                continue
            out[split].append(parse_json_field(row.get("memory_hits") or ""))
    return out


def edge_id(source: str, target: str, key: str) -> str:
    return f"{source}\t{target}\t{key}"


def select_memory_from_graph(memory_graph: Any, threshold: float) -> tuple[set[str], set[str], int, int]:
    selected_edges: set[str] = set()
    selected_nodes: set[str] = set()
    if memory_graph.is_multigraph():
        edge_iter = memory_graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, str(i), d) for i, (u, v, d) in enumerate(memory_graph.edges(data=True)))
    for source, target, key, data in edge_iter:
        punish = float(data.get("punish", data.get("drop_prob", data.get("edge_score", 0.0))) or 0.0)
        if punish >= float(threshold):
            selected_edges.add(edge_id(str(source), str(target), str(key)))
            selected_edges.add(edge_id(str(target), str(source), str(key)))
            selected_nodes.add(str(source))
            selected_nodes.add(str(target))
    return selected_edges, selected_nodes, len(selected_edges) // 2, len(selected_nodes)


def calc_memory_metrics(
    hits_by_split: dict[str, list[list[dict[str, Any]]]],
    selected_edges: set[str],
    selected_nodes: set[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split, rows in hits_by_split.items():
        n = len(rows)
        any_count = 0
        all_count = 0
        blocked_sum = 0
        denom = 0
        for hits in rows:
            flags: list[bool] = []
            for hit in hits[:3]:
                path_nodes = [str(x) for x in hit.get("path_nodes", [])]
                path_edges = hit.get("path_edges", [])
                blocked = any(node in selected_nodes for node in path_nodes)
                if not blocked:
                    for edge in path_edges:
                        if not isinstance(edge, dict):
                            continue
                        eid = edge_id(str(edge.get("source")), str(edge.get("target")), str(edge.get("key", "0")))
                        if eid in selected_edges:
                            blocked = True
                            break
                flags.append(blocked)
            any_count += int(any(flags))
            all_count += int(bool(flags) and all(flags))
            blocked_sum += sum(int(flag) for flag in flags)
            denom += len(flags)
        metrics[split] = {
            "n": int(n),
            "any_at3": any_count / max(1, n),
            "all_at3": all_count / max(1, n),
            "rate_at3": blocked_sum / max(1, denom),
            "hit_count": int(any_count),
        }
    return metrics


def export_memory_graph(source_graph: Any, selected_edges: set[str], selected_nodes: set[str], out_path: Path) -> None:
    graph = reset_graph_unlearned(source_graph)
    for node_id, data in graph.nodes(data=True):
        if str(node_id) in selected_nodes:
            data["unlearned"] = 1
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(keys=True, data=True):
            if edge_id(str(source), str(target), str(key)) in selected_edges:
                data["unlearned"] = 1
    else:
        for idx, (source, target, data) in enumerate(graph.edges(data=True)):
            key = str(data.get("key", idx))
            if edge_id(str(source), str(target), key) in selected_edges:
                data["unlearned"] = 1
    save_pickle(graph, out_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--previous-eval-csv", type=Path, default=DEFAULT_PREVIOUS_EVAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-retain-any", type=float, default=0.10)
    parser.add_argument("--export-best", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(args.stats_path.read_text(encoding="utf-8"))
    forget_counts = Counter({str(k): int(v) for k, v in stats["forget_counts"].items()})
    retain_counts = Counter({str(k): int(v) for k, v in stats["retain_counts"].items()})
    forget_top3 = [[str(x) for x in row] for row in stats["forget_top3"]]
    retain_top3 = [[str(x) for x in row] for row in stats["retain_top3"]]

    rag_rows: list[dict[str, Any]] = []
    best_rag: dict[str, Any] | None = None
    best_rag_selected: set[str] = set()
    for lam in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
        for retain_cap in [0, 1, 2, 3]:
            for min_forget in [1, 2, 3, 4, 5]:
                for max_nodes in [0, 80, 120, 160, 200, 240, 300, 360, 450, 600]:
                    selected_rows = select_rag_nodes(
                        forget_counts,
                        retain_counts,
                        lam=lam,
                        retain_cap=retain_cap,
                        min_forget=min_forget,
                        max_nodes=max_nodes,
                        score_positive=True,
                    )
                    selected = {node_id for node_id, *_ in selected_rows}
                    f_metrics = calc_top3_metrics(forget_top3, selected)
                    r_metrics = calc_top3_metrics(retain_top3, selected)
                    row = {
                        "lam": lam,
                        "retain_cap": retain_cap,
                        "min_forget": min_forget,
                        "max_nodes": max_nodes,
                        "selected_nodes": len(selected),
                        "forget_any_at3": f_metrics["any_at3"],
                        "forget_hits": f_metrics["hit_count"],
                        "forget_all_at3": f_metrics["all_at3"],
                        "forget_rate_at3": f_metrics["rate_at3"],
                        "retain_any_at3": r_metrics["any_at3"],
                        "retain_hits": r_metrics["hit_count"],
                        "retain_all_at3": r_metrics["all_at3"],
                        "retain_rate_at3": r_metrics["rate_at3"],
                    }
                    rag_rows.append(row)
                    if r_metrics["any_at3"] <= args.target_retain_any:
                        if (
                            best_rag is None
                            or f_metrics["any_at3"] > best_rag["forget_any_at3"]
                            or (
                                f_metrics["any_at3"] == best_rag["forget_any_at3"]
                                and r_metrics["any_at3"] < best_rag["retain_any_at3"]
                            )
                        ):
                            best_rag = row
                            best_rag_selected = selected

    rag_rows.sort(key=lambda row: (row["retain_any_at3"] > args.target_retain_any, -row["forget_any_at3"], row["retain_any_at3"], -row["selected_nodes"]))
    write_csv(args.out_dir / "rag_export_sweep.csv", rag_rows)

    memory_rows: list[dict[str, Any]] = []
    best_memory: dict[str, Any] | None = None
    best_memory_edges: set[str] = set()
    best_memory_nodes: set[str] = set()
    if args.previous_eval_csv.exists():
        memory_graph = load_pickle(args.source_dir / "memory_unlearning_graph.pkl")
        hits_by_split = load_previous_memory_hits(args.previous_eval_csv)
        thresholds = [round(x / 100.0, 2) for x in range(50, 96, 5)]
        thresholds += [0.97, 0.99]
        for threshold in thresholds:
            selected_edges, selected_nodes, edge_count, node_count = select_memory_from_graph(memory_graph, threshold)
            metrics = calc_memory_metrics(hits_by_split, selected_edges, selected_nodes)
            f_metrics = metrics["forget"]
            r_metrics = metrics["retain"]
            row = {
                "threshold": threshold,
                "selected_edges": edge_count,
                "selected_nodes": node_count,
                "forget_any_at3": f_metrics["any_at3"],
                "forget_hits": f_metrics["hit_count"],
                "forget_all_at3": f_metrics["all_at3"],
                "forget_rate_at3": f_metrics["rate_at3"],
                "retain_any_at3": r_metrics["any_at3"],
                "retain_hits": r_metrics["hit_count"],
                "retain_all_at3": r_metrics["all_at3"],
                "retain_rate_at3": r_metrics["rate_at3"],
            }
            memory_rows.append(row)
            if r_metrics["any_at3"] <= args.target_retain_any:
                if (
                    best_memory is None
                    or f_metrics["any_at3"] > best_memory["forget_any_at3"]
                    or (
                        f_metrics["any_at3"] == best_memory["forget_any_at3"]
                        and r_metrics["any_at3"] < best_memory["retain_any_at3"]
                    )
                ):
                    best_memory = row
                    best_memory_edges = selected_edges
                    best_memory_nodes = selected_nodes
        memory_rows.sort(key=lambda row: (row["retain_any_at3"] > args.target_retain_any, -row["forget_any_at3"], row["retain_any_at3"]))
        write_csv(args.out_dir / "memory_export_sweep.csv", memory_rows)

    summary = {
        "target_retain_any_at3": args.target_retain_any,
        "stats_path": str(args.stats_path),
        "source_dir": str(args.source_dir),
        "previous_eval_csv": str(args.previous_eval_csv),
        "best_rag": best_rag,
        "best_memory": best_memory,
        "note": "Metrics are top3 unlearned/block flags only; no reference answer, no judge.",
    }
    (args.out_dir / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.export_best:
        export_dir = args.out_dir / "best_export_retain_any_le010"
        export_dir.mkdir(parents=True, exist_ok=True)
        if best_rag is not None:
            rag_graph = load_pickle(args.source_dir / "rag_unlearning_graph.pkl")
            export_rag_graph(rag_graph, best_rag_selected, export_dir / "rag_unlearning_graph.pkl")
        if best_memory is not None:
            memory_graph = load_pickle(args.source_dir / "memory_unlearning_graph.pkl")
            export_memory_graph(memory_graph, best_memory_edges, best_memory_nodes, export_dir / "memory_unlearning_graph.pkl")
        for name in ["rag_retrain_gat.pt", "memory_retrain_gat.pt", "rag_pre_gat.pt", "memory_pre_gat.pt"]:
            src = args.source_dir / name
            if src.exists():
                shutil.copy2(src, export_dir / name)
        (export_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
