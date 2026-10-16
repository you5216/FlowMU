                      
                                                            

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def qa_key(question: str, answer: str) -> str:
    return f"{question.strip()}\n{answer.strip()}"


def add_row(
    rows: list[dict[str, Any]],
    seen: set[str],
    *,
    question: str,
    answer: str,
    source: str,
    source_id: str,
    score: float,
) -> None:
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    if not question or not answer:
        return
    key = qa_key(question, answer)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "question": question,
            "answer": answer,
            "source": source,
            "source_id": source_id,
            "score": float(score),
        }
    )


def collect_rag_rows(graph: nx.Graph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forget_rows: list[dict[str, Any]] = []
    retain_rows: list[dict[str, Any]] = []
    forget_seen: set[str] = set()
    retain_seen: set[str] = set()

    for node_id, data in graph.nodes(data=True):
        score = float(data.get("drop_prob", data.get("unlearned", 0.0)) or 0.0)
        target_rows = forget_rows if int(data.get("unlearned", 0) or 0) == 1 else retain_rows
        target_seen = forget_seen if target_rows is forget_rows else retain_seen
        add_row(
            target_rows,
            target_seen,
            question=str(data.get("question") or ""),
            answer=str(data.get("answer") or ""),
            source="rag_node",
            source_id=str(node_id),
            score=score,
        )

    forget_rows.sort(key=lambda row: row["score"], reverse=True)
    retain_rows.sort(key=lambda row: row["score"])
    return forget_rows, retain_rows


def chunk_node_for(graph: nx.MultiDiGraph, node_id: str) -> tuple[str, dict[str, Any]] | None:
    if node_id not in graph.nodes:
        return None
    data = graph.nodes[node_id]
    if str(data.get("type") or data.get("node_type") or "") == "chunk":
        return node_id, data
    chunk_id = str(data.get("chunk_id") or "").strip()
    if chunk_id and chunk_id in graph.nodes:
        chunk_data = graph.nodes[chunk_id]
        if str(chunk_data.get("type") or chunk_data.get("node_type") or "") == "chunk":
            return chunk_id, chunk_data
    return None


def collect_memory_rows(graph: nx.MultiDiGraph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forget_rows: list[dict[str, Any]] = []
    retain_rows: list[dict[str, Any]] = []
    forget_seen: set[str] = set()
    retain_seen: set[str] = set()
    risky_chunks: dict[str, float] = {}
    safe_chunks: dict[str, float] = {}

    for source, target, _key, edge_data in graph.edges(keys=True, data=True):
        edge_unlearned = int(edge_data.get("unlearned", 0) or 0)
        punish = float(edge_data.get("punish", edge_data.get("drop_prob", 0.0)) or 0.0)
        for endpoint in (str(source), str(target)):
            chunk = chunk_node_for(graph, endpoint)
            if chunk is None:
                continue
            chunk_id, _ = chunk
            if edge_unlearned == 1:
                risky_chunks[chunk_id] = max(risky_chunks.get(chunk_id, 0.0), punish)
            else:
                safe_chunks[chunk_id] = max(safe_chunks.get(chunk_id, 0.0), 1.0 - punish)

    for node_id, data in graph.nodes(data=True):
        node_type = str(data.get("type") or data.get("node_type") or "")
        if node_type != "chunk":
            continue
        question = str(data.get("question") or "")
        answer = str(data.get("answer") or data.get("text") or data.get("content") or "")
        node_unlearned = int(data.get("unlearned", 0) or 0)
        node_punish = float(data.get("punish", data.get("drop_prob", 0.0)) or 0.0)
        if node_unlearned == 1:
            risky_chunks[str(node_id)] = max(risky_chunks.get(str(node_id), 0.0), node_punish)
        elif str(node_id) not in risky_chunks:
            safe_chunks[str(node_id)] = max(safe_chunks.get(str(node_id), 0.0), 1.0 - node_punish)
        if str(node_id) in risky_chunks:
            add_row(
                forget_rows,
                forget_seen,
                question=question,
                answer=answer,
                source="memory_chunk",
                source_id=str(node_id),
                score=risky_chunks[str(node_id)],
            )
        elif str(node_id) in safe_chunks:
            add_row(
                retain_rows,
                retain_seen,
                question=question,
                answer=answer,
                source="memory_chunk",
                source_id=str(node_id),
                score=safe_chunks[str(node_id)],
            )

    forget_rows.sort(key=lambda row: row["score"], reverse=True)
    retain_rows.sort(key=lambda row: row["score"], reverse=True)
    return forget_rows, retain_rows


def merge_with_source_coverage(
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    *,
    descending: bool,
) -> list[dict[str, Any]]:
                                                                           
    rows = list(primary_rows) + list(secondary_rows)
    rows.sort(key=lambda row: row["score"], reverse=descending)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export graph-derived forget/retain QA JSONL files.")
    parser.add_argument("--rag-graph", required=True)
    parser.add_argument("--memory-graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-forget",
        type=int,
        default=0,
        help="Deprecated compatibility argument. Graph-derived forget QA is no longer truncated.",
    )
    parser.add_argument(
        "--max-retain",
        type=int,
        default=0,
        help="Deprecated compatibility argument. Graph-derived retain QA is no longer truncated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rag_graph = load_pickle(Path(args.rag_graph))
    memory_graph = load_pickle(Path(args.memory_graph))

    rag_forget, rag_retain = collect_rag_rows(rag_graph)
    mem_forget, mem_retain = collect_memory_rows(memory_graph)

    forget_rows = merge_with_source_coverage(
        rag_forget,
        mem_forget,
        descending=True,
    )
    retain_rows = merge_with_source_coverage(
        rag_retain,
        mem_retain,
        descending=False,
    )
    write_jsonl(output_dir / "graph_forget_qa.jsonl", forget_rows)
    write_jsonl(output_dir / "graph_retain_qa.jsonl", retain_rows)
    report = {
        "rag_forget": len(rag_forget),
        "rag_retain": len(rag_retain),
        "memory_forget": len(mem_forget),
        "memory_retain": len(mem_retain),
        "export_forget": len(forget_rows),
        "export_retain": len(retain_rows),
        "export_limit_policy": "no_truncation_all_graph_qa",
        "deprecated_max_forget_arg_ignored": int(args.max_forget),
        "deprecated_max_retain_arg_ignored": int(args.max_retain),
        "export_forget_sources": dict(Counter(row["source"] for row in forget_rows)),
        "export_retain_sources": dict(Counter(row["source"] for row in retain_rows)),
        "forget_path": str(output_dir / "graph_forget_qa.jsonl"),
        "retain_path": str(output_dir / "graph_retain_qa.jsonl"),
    }
    (output_dir / "graph_phase2_qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
