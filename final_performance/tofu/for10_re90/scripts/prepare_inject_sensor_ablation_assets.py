                      
                                                                 

                                                                            

                                                         
                                                   
                                                             

                                              

                                                                               
                                                                               
                                                                         
                                                                     
   

from __future__ import annotations

import argparse
import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return [dict(row) for row in rows if isinstance(row, dict)]
    if isinstance(obj, list):
        return [dict(row) for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_official_qa_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(load_json_or_jsonl(path)):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(
            row.get("answer")
            or row.get("output")
            or row.get("original_output")
            or row.get("reference_answer")
            or ""
        ).strip()
        if not question or not answer:
            continue
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "source": "official_forget10_raw_qa",
                "source_id": str(row.get("id") or row.get("example_id") or idx),
                "score": 1.0,
            }
        )
    return rows


def reset_rag_graph(graph: Any) -> Any:
    graph = deepcopy(graph)
    for _, data in graph.nodes(data=True):
        data["unlearned"] = 0
        data["drop_prob"] = 0.0
        data["keep_prob"] = 1.0
        data["weak_label"] = 0
    return graph


def reset_memory_graph(graph: Any) -> Any:
    graph = deepcopy(graph)
    for _, data in graph.nodes(data=True):
        data["unlearned"] = 0
        data["punish"] = 0.0
        data["keep_prob"] = 1.0
        data["weak_label"] = 0.0
    if graph.is_multigraph():
        for _, _, _, data in graph.edges(keys=True, data=True):
            data["unlearned"] = 0
            data["punish"] = 0.0
            data["keep_prob"] = 1.0
            data["weak_label"] = 0.0
    else:
        for _, _, data in graph.edges(data=True):
            data["unlearned"] = 0
            data["punish"] = 0.0
            data["keep_prob"] = 1.0
            data["weak_label"] = 0.0
    return graph


def node_type(data: dict[str, Any]) -> str:
    return str(data.get("node_type") or data.get("type") or "")


def chunk_for_node(graph: Any, node_id: str) -> str | None:
    if node_id not in graph.nodes:
        return None
    data = graph.nodes[node_id]
    if node_type(data) == "chunk":
        return str(node_id)
    chunk_id = str(data.get("chunk_id") or "").strip()
    if chunk_id and chunk_id in graph.nodes and node_type(graph.nodes[chunk_id]) == "chunk":
        return chunk_id
    return None


def anchor_for_chunk(graph: Any, chunk_id: str) -> str | None:
    suffix = chunk_id.split(":", 1)[1] if ":" in chunk_id else ""
    candidate = f"anchor:{suffix}" if suffix else ""
    if candidate and candidate in graph.nodes:
        return candidate
    for node_id, data in graph.nodes(data=True):
        if str(data.get("chunk_id") or "") == chunk_id:
            return str(node_id)
    return None


def edge_iter(graph: Any):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for idx, (source, target, data) in enumerate(graph.edges(data=True)):
            yield source, target, str(data.get("key", idx)), data


def build_similarity_rag_graph(source_graph: Any, *, target_nodes: int) -> tuple[Any, dict[str, Any]]:
    graph = reset_rag_graph(source_graph)
    ranked = []
    for node_id, data in graph.nodes(data=True):
        score = float(data.get("query_similarity", 0.0) or 0.0)
        ranked.append((score, str(node_id)))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = {node_id for _, node_id in ranked[:target_nodes]}
    for node_id, data in graph.nodes(data=True):
        if str(node_id) not in selected:
            continue
        score = float(data.get("query_similarity", 0.0) or 0.0)
        data["unlearned"] = 1
        data["drop_prob"] = score
        data["keep_prob"] = 1.0 - min(1.0, max(0.0, score))
        data["weak_label"] = 1
        data["similarity_sensor_selected"] = 1
    threshold = ranked[target_nodes - 1][0] if ranked and target_nodes > 0 else 0.0
    report = {
        "method": "similarity_only_sensor_ablation",
        "selection_field": "query_similarity",
        "target_nodes": int(target_nodes),
        "selected_nodes": int(len(selected)),
        "selection_threshold": float(threshold),
        "gat_scores_used_for_selection": False,
    }
    return graph, report


def edge_similarity_score(graph: Any, source: str, target: str, data: dict[str, Any]) -> float:
    source_chunk = chunk_for_node(graph, str(source))
    target_chunk = chunk_for_node(graph, str(target))
    source_rel = float(graph.nodes[source_chunk].get("query_relevance", 0.0) or 0.0) if source_chunk else 0.0
    target_rel = float(graph.nodes[target_chunk].get("query_relevance", 0.0) or 0.0) if target_chunk else 0.0
    semantic = float(data.get("similarity", 0.0) or 0.0)
    return 0.9 * max(source_rel, target_rel) + 0.1 * semantic


def build_similarity_memory_graph(
    source_graph: Any,
    *,
    target_chunks: int,
    target_edges: int,
) -> tuple[Any, dict[str, Any]]:
    graph = reset_memory_graph(source_graph)
    chunks = []
    for node_id, data in graph.nodes(data=True):
        if node_type(data) != "chunk":
            continue
        score = float(data.get("query_relevance", 0.0) or 0.0)
        chunks.append((score, str(node_id)))
    chunks.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_chunks = {node_id for _, node_id in chunks[:target_chunks]}

    selected_nodes: set[str] = set(selected_chunks)
    for chunk_id in selected_chunks:
        anchor_id = anchor_for_chunk(graph, chunk_id)
        if anchor_id:
            selected_nodes.add(anchor_id)

    edge_rows = []
    for source, target, key, data in edge_iter(graph):
        source_chunk = chunk_for_node(graph, str(source))
        target_chunk = chunk_for_node(graph, str(target))
        in_selected_region = bool(
            (source_chunk and source_chunk in selected_chunks)
            or (target_chunk and target_chunk in selected_chunks)
        )
        if not in_selected_region:
            continue
        score = edge_similarity_score(graph, str(source), str(target), dict(data))
        edge_rows.append((score, str(source), str(target), str(key)))
    edge_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    selected_edges = {(source, target, key) for _, source, target, key in edge_rows[:target_edges]}

    for node_id, data in graph.nodes(data=True):
        if str(node_id) not in selected_nodes:
            continue
        chunk_id = chunk_for_node(graph, str(node_id)) or str(node_id)
        rel = float(graph.nodes[chunk_id].get("query_relevance", 0.0) or 0.0) if chunk_id in graph.nodes else 0.0
        data["unlearned"] = 1
        data["punish"] = rel
        data["keep_prob"] = 1.0 - min(1.0, max(0.0, rel))
        data["weak_label"] = 1.0
        data["similarity_sensor_selected"] = 1

    selected_edge_count = 0
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(keys=True, data=True):
            if (str(source), str(target), str(key)) not in selected_edges:
                continue
            score = edge_similarity_score(graph, str(source), str(target), dict(data))
            data["unlearned"] = 1
            data["punish"] = score
            data["keep_prob"] = 1.0 - min(1.0, max(0.0, score))
            data["weak_label"] = 1.0
            data["similarity_sensor_selected"] = 1
            selected_edge_count += 1
    else:
        for source, target, data in graph.edges(data=True):
            key = str(data.get("key", "0"))
            if (str(source), str(target), key) not in selected_edges:
                continue
            score = edge_similarity_score(graph, str(source), str(target), dict(data))
            data["unlearned"] = 1
            data["punish"] = score
            data["keep_prob"] = 1.0 - min(1.0, max(0.0, score))
            data["weak_label"] = 1.0
            data["similarity_sensor_selected"] = 1
            selected_edge_count += 1

    threshold = chunks[target_chunks - 1][0] if chunks and target_chunks > 0 else 0.0
    report = {
        "method": "similarity_only_sensor_ablation",
        "node_selection_field": "query_relevance",
        "edge_selection_field": "0.9*max(endpoint_chunk_query_relevance)+0.1*edge_similarity",
        "target_chunks": int(target_chunks),
        "selected_chunks": int(len(selected_chunks)),
        "selected_nodes": int(len(selected_nodes)),
        "target_edges": int(target_edges),
        "selected_edges": int(selected_edge_count),
        "chunk_selection_threshold": float(threshold),
        "gat_scores_used_for_selection": False,
    }
    return graph, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-forget", type=Path, required=True)
    parser.add_argument("--source-rag-graph", type=Path, required=True)
    parser.add_argument("--source-memory-graph", type=Path, required=True)
    parser.add_argument("--inject-qa-dir", type=Path, required=True)
    parser.add_argument("--sensor-graph-dir", type=Path, required=True)
    parser.add_argument("--rag-target-nodes", type=int, default=560)
    parser.add_argument("--memory-target-chunks", type=int, default=1255)
    parser.add_argument("--memory-target-edges", type=int, default=1135)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_rows = normalize_official_qa_rows(args.official_forget)
    raw_count = write_jsonl(args.inject_qa_dir / "raw_official_forget_qa.jsonl", raw_rows)
    save_json(
        args.inject_qa_dir / "raw_official_forget_qa_report.json",
        {
            "source": str(args.official_forget),
            "output": str(args.inject_qa_dir / "raw_official_forget_qa.jsonl"),
            "rows": int(raw_count),
        },
    )

    rag_graph = load_pickle(args.source_rag_graph)
    memory_graph = load_pickle(args.source_memory_graph)
    sensor_rag, rag_report = build_similarity_rag_graph(rag_graph, target_nodes=args.rag_target_nodes)
    sensor_memory, memory_report = build_similarity_memory_graph(
        memory_graph,
        target_chunks=args.memory_target_chunks,
        target_edges=args.memory_target_edges,
    )

    save_pickle(args.sensor_graph_dir / "rag_unlearning_graph.pkl", sensor_rag)
    save_pickle(args.sensor_graph_dir / "memory_unlearning_graph.pkl", sensor_memory)
    save_json(args.sensor_graph_dir / "rag_unlearning_report.json", rag_report)
    save_json(args.sensor_graph_dir / "memory_unlearning_report.json", memory_report)
    save_json(
        args.sensor_graph_dir / "similarity_sensor_report.json",
        {
            "method": "similarity_only_sensor_ablation",
            "source_rag_graph": str(args.source_rag_graph),
            "source_memory_graph": str(args.source_memory_graph),
            "rag": rag_report,
            "memory": memory_report,
        },
    )

    print(
        json.dumps(
            {
                "raw_forget_qa_rows": raw_count,
                "raw_forget_qa": str(args.inject_qa_dir / "raw_official_forget_qa.jsonl"),
                "sensor_graph_dir": str(args.sensor_graph_dir),
                "sensor_rag_selected": rag_report["selected_nodes"],
                "sensor_memory_chunks": memory_report["selected_chunks"],
                "sensor_memory_edges": memory_report["selected_edges"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
