                      
                                                                               

                                                                  

      
                                                
                                                 
                               

      
                                  
                              
                                                                        

                                                                     
   

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase1.dual_graph_unlearning_gat.graph_utils import (              
    add_louvain_communities_to_memory,
    json_ready,
    l2_normalize,
    save_json,
    save_pickle,
    short_hash,
    slugify,
)


DEFAULT_MEMORY_QA = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/MemoryGenerate/answer_gen/artifacts/memory_qa.json"
)
DEFAULT_EXTRACTIONS = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/MemoryGenerate/graph_gen/artifacts_no_merge/extractions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/phase1/artifacts/memory"
)

FORBIDDEN_NODE_ATTRS = {
    "author_index",
    "turn",
    "position",
    "raw",
    "aliases",
    "entity_type",
    "turns",
    "mention_positions",
}
FORBIDDEN_EDGE_ATTRS = {
    "author_index",
    "turn",
    "position",
    "source_position",
    "target_position",
}


@dataclass
class QAChunk:
    index: int
    question: str
    answer: str


@dataclass
class EntityRecord:
    canonical_name: str
    mentions: set[int] = field(default_factory=set)


def normalize_entity_name(name: Any) -> str:
    text = str(name or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:()[]{}")
    lowered = text.lower()
    if lowered in {"", "he", "she", "it", "they", "him", "her", "them", "this", "that", "unknown", "n/a", "none"}:
        return ""
    return text


def entity_key(name: str) -> str:
    normalized = normalize_entity_name(name).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" \t\r\n\"'`.,;:()[]{}")
    return normalized


def clean_predicate(value: Any) -> str:
    predicate = str(value or "related_to").strip().lower()
    predicate = re.sub(r"[^a-z0-9]+", "_", predicate)
    predicate = re.sub(r"_+", "_", predicate).strip("_")
    return predicate or "related_to"


def safe_entity_slug(name: str) -> str:
    slug = slugify(name)
    slug = re.sub(r"author_?", "entity_", slug, flags=re.IGNORECASE)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "entity"


def qa_text(chunk: QAChunk) -> str:
    return f"Question: {chunk.question}\nAnswer: {chunk.answer}"


def load_qa_chunks(path: Path, limit: int = 0) -> list[QAChunk]:
    rows: list[QAChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            question = str(obj.get("question") or "").strip()
            answer = str(obj.get("answer") or "").strip()
            if not question and not answer:
                continue
            rows.append(QAChunk(index=len(rows) + 1, question=question, answer=answer))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No QA chunks loaded from {path}")
    return rows


def load_extractions(path: Path, valid_count: int) -> tuple[dict[int, dict[str, Any]], list[tuple[int, int]]]:
    by_index: dict[int, dict[str, Any]] = {}
    document_ranges: list[tuple[int, int]] = []
    document_offset = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            is_document_local = bool(
                obj.get("document_signature")
                or obj.get("source") == "literaryqa_memory_full"
                or (obj.get("document_id") and obj.get("turn_count") and obj.get("title"))
            )
            try:
                turn_count = int(obj.get("turn_count") or 0)
            except (TypeError, ValueError):
                turn_count = 0
            for item in obj.get("turn_extractions", []) or []:
                try:
                    local_idx = int(item.get("turn"))
                except (TypeError, ValueError):
                    continue
                idx = document_offset + local_idx if is_document_local else local_idx
                if 1 <= idx <= valid_count:
                    by_index[idx] = item
            if is_document_local:
                start = document_offset + 1
                document_offset += turn_count
                end = min(document_offset, valid_count)
                if start <= end:
                    document_ranges.append((start, end))
    return by_index, document_ranges


def build_embeddings(texts: list[str], *, batch_size: int, cache_path: Path) -> np.ndarray:
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)

    missing = [text for text in texts if text not in cache]
    if missing:
        from procedure.runtime_support import build_shared_embeddings

        embedder = build_shared_embeddings()
        for start in range(0, len(missing), max(1, batch_size)):
            batch = missing[start : start + max(1, batch_size)]
            vectors = embedder.embed_documents(batch)
            for text, vector in zip(batch, vectors):
                cache[text] = [float(x) for x in vector]
            print(f"[Embedding] cached={len(cache)} missing_left={len(missing) - min(len(missing), start + len(batch))}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(cache, f)

    arr = np.asarray([cache[text] for text in texts], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Embedding cache produced invalid matrix.")
    return arr


def collect_entities_and_relations(
    extractions: dict[int, dict[str, Any]]
) -> tuple[dict[str, EntityRecord], dict[int, set[str]], list[dict[str, Any]]]:
    entities: dict[str, EntityRecord] = {}
    mentions_by_chunk: dict[int, set[str]] = {}
    relations: list[dict[str, Any]] = []

    def add_entity(name: Any, idx: int | None = None) -> str:
        canonical = normalize_entity_name(name)
        if not canonical:
            return ""
        key = entity_key(canonical)
        if not key:
            return ""
        record = entities.setdefault(key, EntityRecord(canonical_name=canonical))
        if idx is not None:
            record.mentions.add(int(idx))
        return key

    for idx, extraction in extractions.items():
        mentioned_keys = mentions_by_chunk.setdefault(int(idx), set())
        for item in extraction.get("entities", []) or []:
            if isinstance(item, dict):
                key = add_entity(item.get("name"), idx)
            else:
                key = add_entity(item, idx)
            if key:
                mentioned_keys.add(key)
        for name in extraction.get("mentioned_entities", []) or []:
            key = add_entity(name, idx)
            if key:
                mentioned_keys.add(key)
        for relation in extraction.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            subj_key = add_entity(relation.get("subject"), idx)
            obj_key = add_entity(relation.get("object"), idx)
            if not subj_key or not obj_key:
                continue
            mentioned_keys.add(subj_key)
            mentioned_keys.add(obj_key)
            try:
                confidence = float(relation.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            relations.append({
                "source_key": subj_key,
                "target_key": obj_key,
                "relation": clean_predicate(relation.get("predicate")),
                "confidence": max(0.0, min(1.0, confidence)),
            })
    return entities, mentions_by_chunk, relations


def build_generic_graph(
    chunks: list[QAChunk],
    extractions: dict[int, dict[str, Any]],
    *,
    output_dir: Path,
    embedding_batch_size: int,
    document_ranges: list[tuple[int, int]] | None = None,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entities, mentions_by_chunk, relations = collect_entities_and_relations(extractions)

    chunk_texts = [qa_text(chunk) for chunk in chunks]
    entity_keys = sorted(entities)
    entity_texts = [entities[key].canonical_name for key in entity_keys]
    all_texts = chunk_texts + entity_texts
    embeddings = build_embeddings(
        all_texts,
        batch_size=embedding_batch_size,
        cache_path=output_dir / "embedding_cache.pkl",
    )
    chunk_embeddings = embeddings[: len(chunk_texts)]
    entity_embeddings = embeddings[len(chunk_texts) :]

    graph = nx.MultiDiGraph()
    graph.graph.update({
        "format_version": 3,
        "graph_type": "generic_memory_graph",
        "node_types": ["entity", "chunk", "anchor"],
        "edge_types": ["relation", "mentioned_in", "next_turn"],
        "entity_merge_rule": "exact_after_normalization",
        "next_turn_rule": "global_i_to_i_plus_1_with_chunk_similarity",
    })

    chunk_id_by_index: dict[int, str] = {}
    anchor_id_by_index: dict[int, str] = {}
    for pos, chunk in enumerate(chunks, start=1):
        chunk_id = f"chunk:{pos:06d}"
        anchor_id = f"anchor:{pos:06d}"
        chunk_id_by_index[chunk.index] = chunk_id
        anchor_id_by_index[chunk.index] = anchor_id
        graph.add_node(
            chunk_id,
            node_type="chunk",
            question=chunk.question,
            answer=chunk.answer,
            embedding=chunk_embeddings[pos - 1].astype(float).tolist(),
        )
        graph.add_node(
            anchor_id,
            node_type="anchor",
            chunk_id=chunk_id,
        )

    entity_node_by_key: dict[str, str] = {}
    for idx, key in enumerate(entity_keys):
        record = entities[key]
        node_id = f"entity:{safe_entity_slug(record.canonical_name)}:{short_hash(key, 10)}"
        entity_node_by_key[key] = node_id
        graph.add_node(
            node_id,
            node_type="entity",
            canonical_name=record.canonical_name,
            embedding=entity_embeddings[idx].astype(float).tolist(),
        )

    mentioned_edge_count = 0
    for chunk_index, keys in mentions_by_chunk.items():
        anchor_id = anchor_id_by_index.get(chunk_index)
        if anchor_id is None:
            continue
        for key in sorted(keys):
            entity_id = entity_node_by_key.get(key)
            if entity_id is None:
                continue
            graph.add_edge(
                entity_id,
                anchor_id,
                edge_type="mentioned_in",
                relation="mentioned_in",
            )
            mentioned_edge_count += 1

    relation_edge_count = 0
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in relations:
        source = entity_node_by_key.get(relation["source_key"])
        target = entity_node_by_key.get(relation["target_key"])
        if source is None or target is None:
            continue
        dedupe_key = (source, target, relation["relation"])
        if dedupe_key in seen_relations:
            continue
        seen_relations.add(dedupe_key)
        graph.add_edge(
            source,
            target,
            edge_type="relation",
            relation=relation["relation"],
            confidence=float(relation["confidence"]),
        )
        relation_edge_count += 1

    normalized_chunks = l2_normalize(chunk_embeddings)
    next_turn_edge_count = 0
    for chunk_index in range(1, len(chunks)):
        source = anchor_id_by_index[chunk_index]
        target = anchor_id_by_index[chunk_index + 1]
        similarity = float(np.dot(normalized_chunks[chunk_index - 1], normalized_chunks[chunk_index]))
        graph.add_edge(
            source,
            target,
            edge_type="next_turn",
            relation="semantic_next",
            similarity=similarity,
        )
        next_turn_edge_count += 1

    graph, louvain_report = add_louvain_communities_to_memory(graph)
    report = validate_generic_graph(graph)
    report.update({
        "chunk_count": len(chunks),
        "entity_count": len(entity_keys),
        "anchor_count": len(chunks),
        "extraction_count": len(extractions),
        "document_range_count": len(document_ranges or []),
        "mentioned_edge_count": mentioned_edge_count,
        "relation_edge_count": relation_edge_count,
        "next_turn_edge_count": next_turn_edge_count,
        "louvain": louvain_report,
    })
    return graph, report


def validate_generic_graph(graph: nx.MultiDiGraph) -> dict[str, Any]:
    node_forbidden = {
        attr: sum(1 for _, data in graph.nodes(data=True) if attr in data)
        for attr in sorted(FORBIDDEN_NODE_ATTRS)
    }
    edge_forbidden = {
        attr: sum(1 for *_, data in graph.edges(keys=True, data=True) if attr in data)
        for attr in sorted(FORBIDDEN_EDGE_ATTRS)
    }
    id_author_count = sum(1 for node_id in graph.nodes if "author_" in str(node_id).lower())
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    undirected.add_edges_from((source, target) for source, target in graph.edges())
    components = list(nx.connected_components(undirected))
    node_type_counts: dict[str, int] = {}
    edge_type_counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        node_type = str(data.get("node_type") or "unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    for _, _, data in graph.edges(data=True):
        edge_type = str(data.get("edge_type") or "unknown")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
        "forbidden_node_attr_counts": node_forbidden,
        "forbidden_edge_attr_counts": edge_forbidden,
        "id_contains_author_count": id_author_count,
        "connected_components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
    }


def graph_to_json_payload(graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "format_version": graph.graph.get("format_version", 3),
        "graph_type": graph.graph.get("graph_type", "generic_memory_graph"),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "nodes": [
            {"id": node_id, **{key: json_ready(value) for key, value in data.items() if key != "embedding"}}
            for node_id, data in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "key": key,
                **{attr_key: json_ready(attr_value) for attr_key, attr_value in data.items()},
            }
            for source, target, key, data in graph.edges(keys=True, data=True)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-qa", type=Path, default=DEFAULT_MEMORY_QA)
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 88)
    print("Generic Memory Graph Builder")
    print("=" * 88)
    print(f"[Input] memory_qa={args.memory_qa}")
    print(f"[Input] extractions={args.extractions}")
    print(f"[Output] {args.output_dir}")
    chunks = load_qa_chunks(args.memory_qa, limit=max(0, int(args.limit)))
    extractions, document_ranges = load_extractions(args.extractions, valid_count=len(chunks))
    print(f"[Load] chunks={len(chunks)} extractions={len(extractions)} document_ranges={len(document_ranges)}")
    graph, report = build_generic_graph(
        chunks,
        extractions,
        output_dir=args.output_dir,
        embedding_batch_size=max(1, int(args.embedding_batch_size)),
        document_ranges=document_ranges,
    )

    graph_path = args.output_dir / "generic_memory_graph.pkl"
    json_path = args.output_dir / "generic_memory_graph.json"
    report_path = args.output_dir / "generic_memory_graph_report.json"
    save_pickle(graph_path, graph)
    json_path.write_text(json.dumps(graph_to_json_payload(graph), ensure_ascii=False, indent=2), encoding="utf-8")
    save_json(report_path, report)

    print(f"[Save] {graph_path}")
    print(f"[Save] {report_path}")
    print(
        "[Summary] nodes={node_count} edges={edge_count} components={connected_components} "
        "communities={communities}".format(
            node_count=report["node_count"],
            edge_count=report["edge_count"],
            connected_components=report["connected_components"],
            communities=report["louvain"]["community_count"],
        )
    )
    print("[Forbidden]", {
        "node": report["forbidden_node_attr_counts"],
        "edge": report["forbidden_edge_attr_counts"],
        "id_author": report["id_contains_author_count"],
    })


if __name__ == "__main__":
    main()
