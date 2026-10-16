                      
                                                                   

                                                                              
                                                                              
                                                                                   
                                                             
   

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.dual_graph_unlearning_gat.run_dual_graph_unlearning import (
    normalize_key,
    qa_key,
    rag_node_question_answer,
)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected list JSON in {path}")
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def norm(value: Any) -> str:
    return normalize_key(str(value or ""))


def match_rag_strong_nodes(graph: Any, rows: list[dict[str, Any]]) -> list[str]:
    exact_index: dict[str, list[str]] = {}
    question_index: dict[str, list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        question, answer = rag_node_question_answer(dict(data))
        qk = norm(question)
        if qk:
            question_index.setdefault(qk, []).append(str(node_id))
        if qk and answer:
            exact_index.setdefault(qa_key(question, answer), []).append(str(node_id))

    matched: list[str] = []
    missing: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or row.get("output") or "").strip()
        node_ids = exact_index.get(qa_key(question, answer), []) if question and answer else []
        if not node_ids and question:
            node_ids = question_index.get(norm(question), [])
        if not node_ids:
            missing.append({"idx": idx, "question": question})
            continue
        matched.extend(node_ids)
    unique = list(dict.fromkeys(matched))
    return unique, missing


def match_memory_chunks(graph: Any, rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    chunk_index: dict[tuple[str, str], list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type") or "") != "chunk":
            continue
        question = norm(data.get("question"))
        answer = norm(data.get("answer"))
        if question and answer:
            chunk_index.setdefault((question, answer), []).append(str(node_id))

    matched: list[str] = []
    missing: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = norm(row.get("question"))
        answer = norm(row.get("answer") or row.get("output"))
        node_ids = chunk_index.get((question, answer), [])
        if not node_ids:
            missing.append({"idx": idx, "question": str(row.get("question") or "")})
            continue
        matched.extend(node_ids)
    unique = list(dict.fromkeys(matched))
    return unique, missing


def anchor_by_chunk(graph: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type") or "") == "anchor" and data.get("chunk_id") is not None:
            out[str(data.get("chunk_id"))] = str(node_id)
    return out


def export(args: argparse.Namespace) -> None:
    trial_dir = Path(args.trial_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    train_artifacts_dir = output_dir / "train_artifacts"
    checkpoints_dir.mkdir(exist_ok=True)
    train_artifacts_dir.mkdir(exist_ok=True)

    rag_graph = load_pickle(trial_dir / "rag_unlearning_graph.pkl")
    memory_graph = load_pickle(trial_dir / "memory_unlearning_graph.pkl")
    rag_rows = load_json_or_jsonl(Path(args.rag_forget_path))
    memory_rows = load_json_or_jsonl(Path(args.memory_forget_path))

    rag_strong_nodes, rag_missing = match_rag_strong_nodes(rag_graph, rag_rows)
    rag_target_count = min(len(rag_strong_nodes), round(len(rag_strong_nodes) * float(args.rag_coverage)))
    rag_ranked = sorted(
        rag_strong_nodes,
        key=lambda node_id: float(rag_graph.nodes[node_id].get("drop_prob", 0.0) or 0.0),
        reverse=True,
    )
    rag_selected = set(rag_ranked[:rag_target_count])
    rag_pre_existing = {
        str(node_id)
        for node_id, data in rag_graph.nodes(data=True)
        if int(data.get("unlearned", 0)) == 1
    }
    for node_id, data in rag_graph.nodes(data=True):
        node_key = str(node_id)
        data["unlearned"] = int(node_key in rag_pre_existing or node_key in rag_selected)
        data["final_rag_export_selected_strong"] = int(node_key in rag_selected)

    memory_chunks, memory_missing = match_memory_chunks(memory_graph, memory_rows)
    memory_target_count = min(len(memory_chunks), round(len(memory_chunks) * float(args.memory_coverage)))
    memory_ranked = sorted(
        memory_chunks,
        key=lambda node_id: float(memory_graph.nodes[node_id].get("punish", 0.0) or 0.0),
        reverse=True,
    )
    selected_chunks = set(memory_ranked[:memory_target_count])
    chunk_to_anchor = anchor_by_chunk(memory_graph)
    selected_anchors = {chunk_to_anchor[c] for c in selected_chunks if c in chunk_to_anchor}

                                                                                   
    for _, data in memory_graph.nodes(data=True):
        data["unlearned"] = 0
        data["final_memory_export_selected_chunk"] = 0
    for _, _, data in memory_graph.edges(data=True):
        data["unlearned"] = 0
        data["final_memory_export_local_edge"] = 0

    for chunk_id in selected_chunks:
        if chunk_id in memory_graph.nodes:
            memory_graph.nodes[chunk_id]["unlearned"] = 1
            memory_graph.nodes[chunk_id]["final_memory_export_selected_chunk"] = 1
    for anchor_id in selected_anchors:
        if anchor_id in memory_graph.nodes:
            memory_graph.nodes[anchor_id]["unlearned"] = 1
    for u, v, data in memory_graph.edges(data=True):
        if str(u) in selected_anchors or str(v) in selected_anchors:
            data["unlearned"] = 1
            data["final_memory_export_local_edge"] = 1
            if str(u) in memory_graph.nodes:
                memory_graph.nodes[str(u)]["unlearned"] = 1
            if str(v) in memory_graph.nodes:
                memory_graph.nodes[str(v)]["unlearned"] = 1

    selected_memory_records = []
    for rank, chunk_id in enumerate(memory_ranked[:memory_target_count], start=1):
        data = memory_graph.nodes[chunk_id]
        selected_memory_records.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "anchor_id": chunk_to_anchor.get(chunk_id),
                "punish": float(data.get("punish", 0.0) or 0.0),
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
            }
        )

    save_pickle(output_dir / "rag_unlearning_graph.pkl", rag_graph)
    save_pickle(output_dir / "memory_unlearning_graph.pkl", memory_graph)
    shutil.copy2(trial_dir / "rag_retrain_gat.pt", checkpoints_dir / "rag_retrain_gat.pt")
    shutil.copy2(trial_dir / "memory_retrain_gat.pt", checkpoints_dir / "memory_retrain_gat.pt")
    for name in ["rag_pre_gat.pt", "memory_pre_gat.pt"]:
        src = Path(args.pretrain_dir) / name
        if src.exists():
            shutil.copy2(src, checkpoints_dir / name)
    for name in [
        "rag_unlearning_report.json",
        "memory_unlearning_report.json",
        "dual_graph_unlearning_report.json",
        "memory_graph_with_louvain.pkl",
        "memory_louvain_report.json",
    ]:
        src = trial_dir / name
        if src.exists():
            shutil.copy2(src, train_artifacts_dir / name)

    (output_dir / "selected_memory_forget_chunks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected_memory_records),
        encoding="utf-8",
    )

    rag_final_unlearned = sum(1 for _, d in rag_graph.nodes(data=True) if int(d.get("unlearned", 0)) == 1)
    memory_unlearned_nodes = sum(1 for _, d in memory_graph.nodes(data=True) if int(d.get("unlearned", 0)) == 1)
    memory_unlearned_edges = sum(1 for _, _, d in memory_graph.edges(data=True) if int(d.get("unlearned", 0)) == 1)
    memory_unlearned_chunks = sum(
        1
        for _, d in memory_graph.nodes(data=True)
        if str(d.get("node_type") or "") == "chunk" and int(d.get("unlearned", 0)) == 1
    )
    memory_unlearned_anchors = sum(
        1
        for _, d in memory_graph.nodes(data=True)
        if str(d.get("node_type") or "") == "anchor" and int(d.get("unlearned", 0)) == 1
    )
    memory_unlearned_entities = sum(
        1
        for _, d in memory_graph.nodes(data=True)
        if str(d.get("node_type") or "") == "entity" and int(d.get("unlearned", 0)) == 1
    )
    report = {
        "purpose": "LiteraryQA for10_re90 final RAG + Memory unlearning graph artifacts",
        "source_trial_dir": str(trial_dir),
        "legal_supervision": "official forget split exact QA/chunk match plus GAT score sorting; no retrieval eval CSV",
        "rag": {
            "forget_rows": len(rag_rows),
            "matched_strong_nodes": len(rag_strong_nodes),
            "missing_rows": len(rag_missing),
            "target_strong_coverage": float(args.rag_coverage),
            "selected_strong_nodes": len(rag_selected),
            "selected_strong_coverage": len(rag_selected) / len(rag_strong_nodes) if rag_strong_nodes else 0.0,
            "pre_existing_unlearned": len(rag_pre_existing),
            "final_unlearned_count": rag_final_unlearned,
            "selection_rule": "keep existing GAT unlearned nodes OR top drop_prob strong QA nodes up to target coverage",
        },
        "memory": {
            "forget_rows": len(memory_rows),
            "matched_chunks": len(memory_chunks),
            "missing_rows": len(memory_missing),
            "target_chunk_coverage": float(args.memory_coverage),
            "selected_chunks": len(selected_chunks),
            "selected_chunk_coverage": len(selected_chunks) / len(memory_chunks) if memory_chunks else 0.0,
            "selected_anchors": len(selected_anchors),
            "final_unlearned_edges": memory_unlearned_edges,
            "final_unlearned_nodes": memory_unlearned_nodes,
            "final_unlearned_chunks": memory_unlearned_chunks,
            "final_unlearned_anchors": memory_unlearned_anchors,
            "final_unlearned_entities": memory_unlearned_entities,
            "selection_rule": "top punish exact memory chunks plus anchors and local incident edges",
        },
        "usable_graphs": {
            "rag_unlearning_graph": str((output_dir / "rag_unlearning_graph.pkl").resolve()),
            "memory_unlearning_graph": str((output_dir / "memory_unlearning_graph.pkl").resolve()),
        },
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# LiteraryQA for10_re90 final graph artifacts\n\n"
        "RAG is capped to partial strong-QA coverage; Memory is capped to chunk coverage.\n"
        "This export uses only official forget split matching and GAT scores.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", default="final_performance/literaryQA/for10_re90/phase1_gat_trial_02_memory_chunk70")
    parser.add_argument("--output-dir", default="final_performance/literaryQA/for10_re90/rag_memory_unlearning_graph")
    parser.add_argument("--rag-forget-path", default="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/forget10.json")
    parser.add_argument("--memory-forget-path", default="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/memory/forget10.json")
    parser.add_argument("--pretrain-dir", default="final_performance/literaryQA/base/phase1_gat_pretrain")
    parser.add_argument("--rag-coverage", type=float, default=0.85)
    parser.add_argument("--memory-coverage", type=float, default=0.725)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
