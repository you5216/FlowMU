                      
                                                                                    

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase1.core.qwen_embeddings import build_qwen_embeddings              
from phase1.dual_graph_unlearning_gat.graph_utils import (              
    MEMORY_EDGE_TYPES,
    MEMORY_NODE_TYPES,
    add_louvain_communities_to_memory,
    build_memory_batch,
    build_rag_batch,
    infer_embedding_dim,
    json_ready,
    l2_normalize,
    load_pickle,
    save_json,
    save_pickle,
)
from phase1.dual_graph_unlearning_gat.models import MemoryGAT, RAGGAT              


DEFAULT_RAG_GRAPH = Path("/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/phase1/artifacts/rag/rag_graph.pkl")
DEFAULT_MEMORY_GRAPH = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/phase1/artifacts/memory/generic_memory_graph.pkl"
)
DEFAULT_ENCODER = Path("/disk8T/lkh/python_projects/llms/Qwen3.5-9B")
DEFAULT_OUTPUT_DIR = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/phase1/dual_graph_unlearning_gat/artifacts_split_unlearning"
)

                  
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA unavailable; falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(text: str) -> str:
    folded = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(text or ""))
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def title_alias_keys(text: str) -> set[str]:
                                                                                        
    raw = normalize_text(text)
    aliases = {normalize_key(raw)}
    no_parenthetical = normalize_text(re.sub(r"\([^)]*\)", " ", raw))
    aliases.add(normalize_key(no_parenthetical))
    aliases.add(normalize_key(raw.replace("_", " ")))
    return {key for key in aliases if key}


def extract_target_text(forget_query: str) -> str:
                                                                   

                                                                             
                                                                     
                                                                       
       
    text = normalize_text(forget_query)
    patterns = [
        r"(?i)^\s*forget\s+all\s+about\s+(.+?)\s*$",
        r"(?i)^\s*forget\s+about\s+(.+?)\s*$",
        r"(?i)^\s*forget\s+(.+?)\s*$",
        r"(?i)^\s*remove\s+all\s+information\s+about\s+(.+?)\s*$",
        r"(?i)^\s*unlearn\s+all\s+about\s+(.+?)\s*$",
        r"(?i)^\s*unlearn\s+(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            candidate = normalize_text(match.group(1).strip(" .,:;\"'"))
            if candidate:
                return candidate
    return text


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_queries_file(path: Path) -> list[str]:
    obj = load_json_or_jsonl(path)
    if isinstance(obj, list):
        queries: list[str] = []
        for item in obj:
            if isinstance(item, str):
                queries.append(item)
            elif isinstance(item, dict):
                value = item.get("forget_query") or item.get("query") or item.get("target") or item.get("author")
                if value:
                    queries.append(str(value))
        return queries
    if isinstance(obj, dict):
        value = obj.get("forget_queries") or obj.get("queries") or obj.get("targets") or obj.get("authors")
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    raise ValueError(f"Unsupported forget queries file format: {path}")

                                
def hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b"\0")
        digest.update(encoded)
        digest.update(b"\0")
    return digest.hexdigest()


def load_cached_embeddings(cache_path: Path | None, metadata: dict[str, Any]) -> np.ndarray | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data["metadata_json"].item()))
            vectors = np.asarray(data["vectors"], dtype=np.float32)
    except Exception as exc:
        print(f"[Query] ignoring invalid embedding cache {cache_path}: {exc}")
        return None
    if cached_metadata != metadata:
        print(f"[Query] ignoring stale embedding cache: {cache_path}")
        return None
    print(f"[Query] loaded embedding cache: {cache_path} shape={tuple(vectors.shape)}")
    return vectors


def save_cached_embeddings(cache_path: Path | None, metadata: dict[str, Any], vectors: np.ndarray) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        vectors=np.asarray(vectors, dtype=np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    print(f"[Query] saved embedding cache: {cache_path} shape={tuple(vectors.shape)}")


def format_qa_query(row: dict[str, Any]) -> str:
    question = normalize_text(row.get("question") or row.get("instruction") or "")
    answer = normalize_text(row.get("answer") or row.get("output") or "")
    if not question or not answer:
        return ""
    return f"Question: {question}\nAnswer: {answer}"


def load_qa_queries_file(path: Path) -> list[str]:
    obj = load_json_or_jsonl(path)
    rows = obj if isinstance(obj, list) else [obj]
    queries: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        query = format_qa_query(row)
        if query:
            queries.append(query)
    if not queries:
        raise ValueError(f"Could not load question-answer forget queries from {path}")
    return queries


def prepare_forget_query_text(query: str) -> str:
    text = str(query).strip()
    if re.match(r"(?is)^\s*Question\s*:", text):
        return text
    return extract_target_text(text)


def extract_author_candidates_from_text(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"(?i)\b(?:author|writer|novelist|poet)\s+(?:named|called)?\s*([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:[\s-]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,4})",
        r"(?i)\bfull name is\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:[\s-]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,4})",
        r"(?i)\bname is\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:[\s-]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,4})",
        r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:[\s-]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,4})'s\b",
        r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+(?:[\s-]+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,4})\b",
    ]
    stop = {
        "What",
        "Who",
        "When",
        "Where",
        "How",
        "Can",
        "Could",
        "The",
        "Yes",
        "No",
        "Answer",
        "Question",
        "Taipei Taiwan",
        "Santiago Chile",
    }
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = normalize_text(match.group(1).strip(" .,:;\"'’"))
            if not name or name in stop:
                continue
            if any(token.lower() in {"author", "writer", "genre", "book", "father", "mother"} for token in name.split()):
                continue
            candidates.append(name)
    return candidates


def load_author_queries_from_forget_dataset(path: Path, *, max_authors: int = 0) -> list[str]:
    obj = load_json_or_jsonl(path)
    rows = obj if isinstance(obj, list) else [obj]
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = f"{row.get('question', '')}\n{row.get('answer', '')}"
        for candidate in extract_author_candidates_from_text(text):
            counts[candidate] += 1
            first_seen.setdefault(candidate, idx)
    if not counts:
        raise ValueError(f"Could not extract author names from {path}")
    ordered = sorted(counts, key=lambda name: (-counts[name], first_seen[name], name))
    if max_authors > 0:
        ordered = ordered[:max_authors]
    return [f"forget all about {name}" for name in ordered]


def infer_query_type(query: str) -> str:
    return "qa" if re.match(r"(?is)^\s*Question\s*:", str(query).strip()) else "author"


def normalize_percentile(value: float) -> float:
                                                                           
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def expand_node_ranges(ranges: list[str], node_to_idx: dict[str, int]) -> set[str]:
                                                                                
    selected: set[str] = set()
    for raw in ranges:
        text = str(raw or "").strip()
        if not text:
            continue
        if ":" not in text:
            node_id = text if text.startswith("NODE_") else f"NODE_{text}"
            if node_id in node_to_idx:
                selected.add(node_id)
            continue
        left, right = [part.strip() for part in text.split(":", 1)]
        left = left.removeprefix("NODE_")
        right = right.removeprefix("NODE_")
        start = int(left)
        end = int(right)
        step = 1 if end >= start else -1
        for idx in range(start, end + step, step):
            node_id = f"NODE_{idx}"
            if node_id in node_to_idx:
                selected.add(node_id)
    return selected


def qa_key(question: Any, answer: Any) -> str:
    return normalize_key(f"{question} {answer}")


def split_qa_target(text: str) -> tuple[str, str]:
                                                                        
    match = re.match(r"(?is)^\s*Question\s*:\s*(.*?)\s*Answer\s*:\s*(.*?)\s*$", str(text or "").strip())
    if not match:
        return "", ""
    return normalize_text(match.group(1)), normalize_text(match.group(2))


def rag_node_question_answer(data: dict[str, Any]) -> tuple[str, str]:
    question = normalize_text(data.get("question") or data.get("instruction") or data.get("query") or "")
    answer = normalize_text(data.get("answer") or data.get("output") or data.get("response") or "")
    return question, answer


def build_rag_exact_qa_index(
    rag_graph: nx.Graph,
    node_ids: list[str],
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
                                                                           
    exact_index: dict[str, list[int]] = defaultdict(list)
    question_index: dict[str, list[int]] = defaultdict(list)
    answer_index: dict[str, list[int]] = defaultdict(list)
    for idx, node_id in enumerate(node_ids):
        question, answer = rag_node_question_answer(dict(rag_graph.nodes[node_id]))
        if not question:
            continue
        question_key = normalize_key(question)
        if question_key:
            question_index[question_key].append(idx)
        answer_key = normalize_key(answer)
        if answer_key:
            answer_index[answer_key].append(idx)
        full_key = qa_key(question, answer)
        if answer and full_key:
            exact_index[full_key].append(idx)
    return exact_index, question_index, answer_index


def build_memory_exact_qa_index(
    memory_graph: nx.Graph,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
                                                                          
    exact_index: dict[str, list[str]] = defaultdict(list)
    question_index: dict[str, list[str]] = defaultdict(list)
    answer_index: dict[str, list[str]] = defaultdict(list)
    for node_id, data in memory_graph.nodes(data=True):
        if str(data.get("node_type") or "") != "chunk":
            continue
        question = normalize_text(data.get("question") or "")
        answer = normalize_text(data.get("answer") or "")
        if question:
            question_index[normalize_key(question)].append(str(node_id))
        if answer:
            answer_index[normalize_key(answer)].append(str(node_id))
        if question and answer:
            exact_index[qa_key(question, answer)].append(str(node_id))
    return exact_index, question_index, answer_index


def match_memory_qa_targets_to_chunks(
    memory_graph: nx.Graph,
    qa_targets: list[str] | None,
) -> list[list[str]]:
                                                                  
    if not qa_targets:
        return []
    exact_index, question_index, answer_index = build_memory_exact_qa_index(memory_graph)
    matches: list[list[str]] = []
    for target in qa_targets:
        question, answer = split_qa_target(target)
        chunk_ids: list[str] = []
        if question and answer:
            chunk_ids = list(exact_index.get(qa_key(question, answer), []))
        if not chunk_ids and question:
            chunk_ids = list(question_index.get(normalize_key(question), []))
        if not chunk_ids and answer:
            chunk_ids = list(answer_index.get(normalize_key(answer), []))
        matches.append(chunk_ids)
    return matches


def memory_entity_keys(data: dict[str, Any], node_id: str) -> set[str]:
    keys: set[str] = set()
    for field in ("canonical_name", "name", "title", "entity", "label", "book_title", "author"):
        value = data.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            key = normalize_key(str(item))
            if key:
                keys.add(key)
    node_key = normalize_key(node_id)
    if node_key:
        keys.add(node_key)
    return keys


def load_rag_nodes_from_qa_path(
    path: Path,
    node_ids: list[Any],
    rag_graph: nx.Graph,
    query_vecs: np.ndarray,
    query_types: list[str],
) -> tuple[set[str], dict[str, Any]]:
                                                                           
    rows = load_json_or_jsonl(path)
    qa_query_indices = [idx for idx, qtype in enumerate(query_types) if qtype == "qa"]
    node_id_set = {str(node_id) for node_id in node_ids}
    rag_node_ids: list[str] = []
    rag_vecs: list[np.ndarray] = []
    for node_id, data in rag_graph.nodes(data=True):
        node_key = str(node_id)
        if node_key not in node_id_set:
            continue
        if data.get("type") != "static_knowledge":
            continue
        embedding = data.get("embedding")
        if embedding is None:
            continue
        rag_node_ids.append(node_key)
        rag_vecs.append(np.asarray(embedding, dtype=np.float32))

    selected: set[str] = set()
    ranks: list[int] = []
    if len(qa_query_indices) != len(rows):
        raise ValueError(
            f"QA force-positive path has {len(rows)} rows, but query types contain {len(qa_query_indices)} QA queries."
        )
    if not rag_vecs:
        raise ValueError("RAG graph has no static_knowledge embeddings for force-positive QA alignment.")

    node_matrix = l2_normalize(np.stack(rag_vecs))
    qa_matrix = l2_normalize(np.asarray(query_vecs[np.asarray(qa_query_indices, dtype=np.int64)], dtype=np.float32))
    scores = qa_matrix @ node_matrix.T
    order = np.argsort(-scores, axis=1)
    selected_by_row: list[str] = []
    for row_order in order:
        chosen = str(rag_node_ids[int(row_order[0])])
        chosen_rank = 1
        for rank, node_idx in enumerate(row_order[:50], start=1):
            candidate = str(rag_node_ids[int(node_idx)])
            if candidate not in selected:
                chosen = candidate
                chosen_rank = rank
                break
        selected.add(chosen)
        selected_by_row.append(chosen)
        ranks.append(chosen_rank)

    report = {
        "path": str(path),
        "row_count": int(len(rows)),
        "matched_node_count": int(len(selected)),
        "alignment": "qa_query_embedding_to_rag_node_embedding_unique_nearest",
        "top1_rows": int(sum(1 for rank in ranks if rank == 1)),
        "max_rank_used": int(max(ranks) if ranks else 0),
        "rank_histogram": {str(rank): int(ranks.count(rank)) for rank in sorted(set(ranks))},
        "selected_nodes_first20": selected_by_row[:20],
    }
    return selected, report


def collect_forget_queries(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    raw_queries: list[str] = []
    source_counts: dict[str, int] = {}
    raw_queries.extend(args.forget_query or [])
    source_counts["inline_query"] = len(args.forget_query or [])
    if args.forget_queries_path is not None:
        file_queries = load_queries_file(args.forget_queries_path)
        raw_queries.extend(file_queries)
        source_counts["query_file"] = len(file_queries)
    if args.forget_authors_from is not None:
        author_queries = load_author_queries_from_forget_dataset(args.forget_authors_from, max_authors=args.max_forget_authors)
        raw_queries.extend(author_queries)
        source_counts["authors_from_dataset"] = len(author_queries)
    if args.forget_qa_query_path is not None:
        qa_queries = load_qa_queries_file(args.forget_qa_query_path)
        raw_queries.extend(qa_queries)
        source_counts["qa_query_file"] = len(qa_queries)

    targets: list[str] = []
    query_types: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        target = prepare_forget_query_text(str(query))
        key = normalize_key(target)
        if target and key not in seen:
            seen.add(key)
            targets.append(target)
            query_types.append(infer_query_type(target))
    if not targets:
        raise ValueError(
            "No forget queries were provided. Use --forget-query, --forget-queries-path, "
            "--forget-qa-query-path, or --forget-authors-from."
        )
    source_counts["raw_total"] = len(raw_queries)
    source_counts["dedup_total"] = len(targets)
    source_counts["dedup_author"] = sum(1 for item in query_types if item == "author")
    source_counts["dedup_qa"] = sum(1 for item in query_types if item == "qa")
    return raw_queries, targets, query_types, source_counts


def sample_non_edges(
    *,
    node_count: int,
    existing: set[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
    left_pool: np.ndarray | None = None,
    right_pool: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left_pool = np.arange(node_count) if left_pool is None else left_pool
    right_pool = np.arange(node_count) if right_pool is None else right_pool
    left: list[int] = []
    right: list[int] = []
    max_tries = max(count * 30, 1000)
    tries = 0
    while len(left) < count and tries < max_tries:
        tries += 1
        u = int(rng.choice(left_pool))
        v = int(rng.choice(right_pool))
        if u == v or (u, v) in existing:
            continue
        left.append(u)
        right.append(v)
    if not left:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def sample_pairs_by_community(
    community: np.ndarray,
    *,
    count: int,
    positive: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, comm in enumerate(community.tolist()):
        if int(comm) >= 0:
            buckets[int(comm)].append(idx)
    valid = [nodes for nodes in buckets.values() if len(nodes) >= 2]
    if not valid:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    left: list[int] = []
    right: list[int] = []
    if positive:
        for _ in range(count):
            nodes = valid[int(rng.integers(0, len(valid)))]
            u, v = rng.choice(nodes, size=2, replace=False)
            left.append(int(u))
            right.append(int(v))
    else:
        comm_ids = list(buckets.keys())
        if len(comm_ids) < 2:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        for _ in range(count):
            cu, cv = rng.choice(comm_ids, size=2, replace=False)
            left.append(int(rng.choice(buckets[int(cu)])))
            right.append(int(rng.choice(buckets[int(cv)])))
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def remap_communities(values: torch.Tensor) -> tuple[torch.Tensor, dict[int, int]]:
    raw = [int(v) for v in values.detach().cpu().tolist()]
    uniq = sorted({v for v in raw if v >= 0})
    if not uniq:
        return torch.zeros_like(values), {-1: 0}
    mapping = {v: i for i, v in enumerate(uniq)}
    mapped = torch.tensor([mapping.get(v, 0) for v in raw], dtype=torch.long)
    return mapped, mapping


def choose_binary_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    positive_cutoff: float = 0.5,
    fallback: float = 0.5,
) -> tuple[float, dict[str, float]]:
    probs = np.asarray(probs, dtype=np.float32)
    y = (np.asarray(labels, dtype=np.float32) >= float(positive_cutoff)).astype(np.int32)
    if probs.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float(fallback), {
            "mode": "fallback",
            "positive_labels": float(y.sum()),
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    candidates = np.unique(np.quantile(probs, np.linspace(0.01, 0.99, 99)))
    candidates = np.unique(np.concatenate([candidates, np.asarray([fallback], dtype=np.float32)]))
    best = {"threshold": float(fallback), "f1": -1.0, "precision": 0.0, "recall": 0.0, "pred_rate": 0.0}
    for threshold in candidates.tolist():
        pred = (probs >= float(threshold)).astype(np.int32)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
        pred_rate = float(pred.mean())
        if f1 > best["f1"] or (abs(f1 - best["f1"]) < 1e-8 and pred_rate < best["pred_rate"]):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "pred_rate": pred_rate,
            }
    return float(best["threshold"]), {
        "mode": "auto_f1_from_query_weak_labels",
        "positive_labels": float(y.sum()),
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "pred_rate": best["pred_rate"],
    }


def choose_cardinality_regularized_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    target_count: int,
    penalty: float,
    positive_cutoff: float = 0.5,
    fallback: float = 0.5,
) -> tuple[float, dict[str, float]]:
    probs = np.asarray(probs, dtype=np.float32)
    y = (np.asarray(labels, dtype=np.float32) >= float(positive_cutoff)).astype(np.int32)
    target_rate = min(1.0, max(0.0, float(target_count) / max(1, int(probs.size))))
    if probs.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float(fallback), {
            "mode": "cardinality_regularized_f1_fallback",
            "positive_labels": float(y.sum()),
            "target_count": float(target_count),
            "target_rate": float(target_rate),
            "penalty": float(penalty),
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pred_rate": 0.0,
            "score": 0.0,
        }

    candidates = np.unique(np.quantile(probs, np.linspace(0.01, 0.99, 99)))
                                                                         
                                                                    
                                                        
    if 0 < int(target_count) <= int(probs.size):
        sorted_probs = np.sort(probs)[::-1]
        candidates = np.concatenate(
            [candidates, np.asarray([sorted_probs[int(target_count) - 1]], dtype=np.float32)]
        )
    candidates = np.unique(np.concatenate([candidates, np.asarray([fallback], dtype=np.float32)]))
    best = {
        "threshold": float(fallback),
        "score": -float("inf"),
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "pred_rate": 0.0,
    }
    for threshold in candidates.tolist():
        pred = (probs >= float(threshold)).astype(np.int32)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
        pred_rate = float(pred.mean())
        score = float(f1 - float(penalty) * abs(pred_rate - target_rate))
        if score > best["score"] or (abs(score - best["score"]) < 1e-8 and f1 > best["f1"]):
            best = {
                "threshold": float(threshold),
                "score": score,
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "pred_rate": pred_rate,
            }
    return float(best["threshold"]), {
        "mode": "cardinality_regularized_f1_from_query_labels",
        "positive_labels": float(y.sum()),
        "target_count": float(target_count),
        "target_rate": float(target_rate),
        "penalty": float(penalty),
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "pred_rate": best["pred_rate"],
        "score": best["score"],
    }


def choose_count_threshold(probs: np.ndarray, target_count: int) -> tuple[float, dict[str, float]]:
    probs = np.asarray(probs, dtype=np.float32)
    target_count = int(target_count)
    if probs.size == 0 or target_count <= 0:
        return 1.0, {
            "mode": "target_count",
            "target_count": float(max(0, target_count)),
            "actual_count": 0.0,
            "pred_rate": 0.0,
        }
    target_count = min(target_count, int(probs.size))
    sorted_probs = np.sort(probs)[::-1]
    threshold = float(sorted_probs[target_count - 1])
    actual_count = int((probs >= threshold).sum())
    return threshold, {
        "mode": "target_count",
        "target_count": float(target_count),
        "actual_count": float(actual_count),
        "pred_rate": float(actual_count / max(1, int(probs.size))),
    }


def class_weights_from_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=2).float()
    weights = labels.numel() / counts.clamp(min=1.0)
    return weights / weights.mean().clamp(min=1e-6)


def build_rag_query_labels(
    rag_graph: nx.Graph,
    query_vec: np.ndarray,
    query_target: str,
    *,
    query_type: str,
    seed_topk: int,
    sim_percentile: float,
    qa_topk: int,
) -> dict[str, Any]:
    node_ids = [str(n) for n in rag_graph.nodes]
    x = np.stack([np.asarray(rag_graph.nodes[node_id]["embedding"], dtype=np.float32) for node_id in node_ids])
    x_norm = l2_normalize(x)
    q = l2_normalize(np.asarray(query_vec, dtype=np.float32))
    similarities = x_norm @ q
    is_qa_query = query_type == "qa"
    k = min(max(1, int(qa_topk if is_qa_query else seed_topk)), len(node_ids))
    seed_indices = np.argsort(-similarities)[:k].astype(np.int64)
    target_key = normalize_key(query_target)
    lexical_indices = [
        idx
        for idx, node_id in enumerate(node_ids)
        if target_key
        and target_key
        in normalize_key(f"{rag_graph.nodes[node_id].get('question', '')} {rag_graph.nodes[node_id].get('answer', '')}")
    ]
    lexical_mask = np.zeros(len(node_ids), dtype=np.int64)
    if lexical_indices:
        lexical_mask[np.asarray(lexical_indices, dtype=np.int64)] = 1
    if lexical_indices and not is_qa_query:
        seed_indices = np.unique(np.concatenate([seed_indices, np.asarray(lexical_indices, dtype=np.int64)]))
    raw_communities = np.asarray([int(rag_graph.nodes[node_id].get("community", -1)) for node_id in node_ids])
    seed_communities = {int(raw_communities[idx]) for idx in seed_indices.tolist() if raw_communities[idx] >= 0}
    sim_percentile = normalize_percentile(sim_percentile)
    sim_threshold = float(np.percentile(similarities, float(sim_percentile)))

    labels = np.zeros(len(node_ids), dtype=np.int64)
    if is_qa_query:
        labels[seed_indices] = 1
    else:
        for idx in range(len(node_ids)):
            if int(raw_communities[idx]) in seed_communities and float(similarities[idx]) >= sim_threshold:
                labels[idx] = 1
    if lexical_indices and not is_qa_query:
        labels[np.asarray(lexical_indices, dtype=np.int64)] = 1
    if labels.sum() == 0:
        labels[seed_indices] = 1

    return {
        "query_vec": q.astype(np.float32),
        "similarities": similarities.astype(np.float32),
        "labels": labels,
        "lexical_mask": lexical_mask,
        "seed_indices": seed_indices.tolist(),
        "seed_node_ids": [node_ids[idx] for idx in seed_indices.tolist()],
        "lexical_seed_count": len(lexical_indices),
        "query_type": query_type,
        "seed_communities": sorted(seed_communities),
        "sim_threshold": sim_threshold,
        "positive_count": int(labels.sum()),
    }


def build_rag_multi_query_labels(
    rag_graph: nx.Graph,
    query_vecs: np.ndarray,
    query_targets: list[str],
    query_types: list[str],
    *,
    seed_topk: int,
    sim_percentile: float,
    qa_topk: int,
    author_aux_weight: float,
    community_top_m: int,
    disable_weak_labels: bool = False,
) -> dict[str, Any]:
    query_vecs = l2_normalize(np.asarray(query_vecs, dtype=np.float32))
    sim_percentile = normalize_percentile(sim_percentile)
    author_aux_weight = float(author_aux_weight)
    community_top_m = int(community_top_m)
    node_ids = [str(n) for n in rag_graph.nodes]
    raw_communities = np.asarray([int(rag_graph.nodes[node_id].get("community", -1)) for node_id in node_ids])
    exact_qa_index, question_index, answer_index = build_rag_exact_qa_index(rag_graph, node_ids)
    per_query = [
        build_rag_query_labels(
            rag_graph,
            query_vecs[idx],
            query_targets[idx],
            query_type=query_types[idx],
            seed_topk=seed_topk,
            sim_percentile=sim_percentile,
            qa_topk=qa_topk,
        )
        for idx in range(query_vecs.shape[0])
    ]
    qa_indices = [idx for idx, qtype in enumerate(query_types) if qtype == "qa"]
    author_indices = [idx for idx, qtype in enumerate(query_types) if qtype == "author"]
    if not qa_indices:
        raise ValueError("RAG dual-channel labeling requires QA queries; provide --forget-qa-query-path.")

    similarities = np.stack([item["similarities"] for item in per_query], axis=0)
    lexical_masks = np.stack([item["lexical_mask"] for item in per_query], axis=0)

    author_keys = {aidx: normalize_key(query_targets[aidx]) for aidx in author_indices}
    qa_seed_communities: set[int] = set()
    author_aux_positive_counts: dict[int, int] = {aidx: 0 for aidx in author_indices}
    merged_labels = np.zeros(len(node_ids), dtype=np.int64)
    merged_strong_mask = np.zeros(len(node_ids), dtype=np.int64)
    merged_weak_mask = np.zeros(len(node_ids), dtype=np.int64)
    merged_similarities = np.full(len(node_ids), -np.inf, dtype=np.float32)
    best_query_idx = np.full(len(node_ids), qa_indices[0], dtype=np.int64)
    exact_match_counts: list[int] = []
    exact_missing_queries: list[int] = []

    for aidx in author_indices:
                                                                            
                                                                                 
        per_query[aidx]["labels"] = np.zeros_like(per_query[aidx]["labels"], dtype=np.int64)
        per_query[aidx]["seed_indices"] = []
        per_query[aidx]["seed_node_ids"] = []
        per_query[aidx]["seed_communities"] = []
        per_query[aidx]["positive_count"] = 0
        per_query[aidx]["author_aux_allowed_communities"] = []
        per_query[aidx]["author_aux_sim_percentile"] = float(sim_percentile)

    for qidx in qa_indices:
        qa_item = per_query[qidx]
        q_seed_indices = np.asarray(qa_item["seed_indices"], dtype=np.int64)
        question, answer = split_qa_target(query_targets[qidx])
        exact_indices = exact_qa_index.get(qa_key(question, answer), []) if question and answer else []
        exact_match_type = "question_answer"
        if not exact_indices and question:
            exact_indices = question_index.get(normalize_key(question), [])
            exact_match_type = "question_only"
        if not exact_indices and answer:
            exact_indices = answer_index.get(normalize_key(answer), [])
            exact_match_type = "answer_only"
        exact_match_counts.append(int(len(exact_indices)))
        if not exact_indices:
            exact_missing_queries.append(int(qidx))
        exact_indices_arr = np.asarray(exact_indices, dtype=np.int64)

        q_seed_communities = {
            int(raw_communities[node_idx])
            for node_idx in q_seed_indices.tolist()
            if int(raw_communities[node_idx]) >= 0
        }
        if exact_indices_arr.size:
            q_seed_communities.update(
                int(raw_communities[node_idx])
                for node_idx in exact_indices_arr.tolist()
                if int(raw_communities[node_idx]) >= 0
            )
        if not q_seed_communities:
            continue
        qa_seed_communities.update(q_seed_communities)
        community_mask = np.isin(raw_communities, list(q_seed_communities))
        qa_target_key = normalize_key(query_targets[qidx])
        matched_authors = [
            aidx
            for aidx, author_key in author_keys.items()
            if author_key and author_key in qa_target_key
        ]

        qa_scores = similarities[qidx]
        if matched_authors and author_aux_weight > 0.0:
            author_scores = similarities[np.asarray(matched_authors, dtype=np.int64)].max(axis=0)
            combined_scores = qa_scores + author_aux_weight * np.maximum(author_scores, 0.0)
        else:
            combined_scores = qa_scores
        q_labels = np.zeros(len(node_ids), dtype=np.int64)
        candidate_node_indices = np.where(community_mask)[0]
        if community_top_m > 0 and len(candidate_node_indices):
            take = min(int(community_top_m), len(candidate_node_indices))
            selected = candidate_node_indices[np.argsort(-combined_scores[candidate_node_indices])[:take]]
            q_labels[selected] = 1
            threshold = float(combined_scores[selected[-1]]) if len(selected) else float("nan")
        else:
            threshold = float(np.percentile(combined_scores, float(sim_percentile)))
            q_labels[community_mask & (combined_scores >= threshold)] = 1
        if disable_weak_labels:
            q_labels[:] = 0
        elif q_seed_indices.size:
            q_labels[q_seed_indices] = 1
        if exact_indices_arr.size:
            q_labels[exact_indices_arr] = 1
            merged_strong_mask[exact_indices_arr] = 1
        qa_item["labels"] = q_labels
        qa_item["strong_indices"] = exact_indices
        qa_item["exact_match_type"] = exact_match_type if exact_indices else "missing"
        qa_item["exact_match_count"] = int(len(exact_indices))
        qa_item["positive_count"] = int(q_labels.sum())
        qa_item["sim_threshold"] = threshold

        positives = q_labels.astype(bool)
        merged_labels[positives] = 1
        weak_only = positives & (merged_strong_mask == 0)
        merged_weak_mask[weak_only] = 1

        better = combined_scores[candidate_node_indices] > merged_similarities[candidate_node_indices]
        update_indices = candidate_node_indices[better]
        if len(update_indices):
            merged_similarities[update_indices] = combined_scores[update_indices]
            best_query_idx[update_indices] = qidx

        for aidx in matched_authors:
            per_query[aidx]["author_aux_allowed_communities"] = sorted(q_seed_communities)
            author_scores = similarities[aidx]
            author_threshold = float(np.percentile(author_scores, float(sim_percentile)))
            author_aux_positive_counts[aidx] += int((community_mask & (author_scores >= author_threshold)).sum())

    fallback_scores = similarities[np.asarray(qa_indices, dtype=np.int64)]
    fallback_best_offsets = fallback_scores.argmax(axis=0)
    fallback_best_query_idx = np.asarray(qa_indices, dtype=np.int64)[fallback_best_offsets]
    fallback_similarities = fallback_scores[fallback_best_offsets, np.arange(fallback_scores.shape[1])]
    missing = ~np.isfinite(merged_similarities)
    if missing.any():
        merged_similarities[missing] = fallback_similarities[missing]
        best_query_idx[missing] = fallback_best_query_idx[missing]

    safe_mask = np.zeros_like(merged_labels, dtype=np.int64)
    negative_indices = np.where(merged_labels == 0)[0]
    if len(negative_indices):
        safe_count = min(len(negative_indices), max(1, int(merged_labels.sum())))
        ranked_safe = negative_indices[np.argsort(-merged_similarities[negative_indices])[:safe_count]]
        safe_mask[ranked_safe] = 1
    merged_weak_mask = np.where((merged_labels == 1) & (merged_strong_mask == 0), 1, 0).astype(np.int64)
    node_query_vecs = query_vecs[best_query_idx]
    return {
        "query_vecs": query_vecs.astype(np.float32),
        "node_query_vecs": node_query_vecs.astype(np.float32),
        "best_query_idx": best_query_idx,
        "similarities": merged_similarities.astype(np.float32),
        "labels": merged_labels,
        "lexical_mask": merged_strong_mask,
        "strong_mask": merged_strong_mask,
        "weak_mask": merged_weak_mask,
        "safe_mask": safe_mask,
        "positive_count": int(merged_labels.sum()),
        "strong_positive_count": int(merged_strong_mask.sum()),
        "weak_positive_count": int(merged_weak_mask.sum()),
        "lexical_positive_count": int(merged_strong_mask.sum()),
        "safe_negative_count": int(safe_mask.sum()),
        "per_query": per_query,
        "per_query_positive_counts": [int(item["positive_count"]) for item in per_query],
        "per_query_seed_counts": [len(item["seed_indices"]) for item in per_query],
        "per_query_lexical_seed_counts": [int(item["lexical_seed_count"]) for item in per_query],
        "per_query_exact_match_counts": exact_match_counts,
        "exact_qa_match_count": int(sum(1 for count in exact_match_counts if count > 0)),
        "exact_qa_missing_count": int(sum(1 for count in exact_match_counts if count == 0)),
        "exact_qa_missing_query_indices": exact_missing_queries[:50],
        "query_type_counts": dict(Counter(query_types)),
        "qa_topk": int(qa_topk),
        "community_source": "qa_topk_seed_communities_only",
        "author_aux_rule": "low_weight_author_similarity_inside_matching_qa_seed_communities_no_author_community_seeding_or_query_conditioning",
        "strong_label_source": "exact_question_answer_match_with_question_only_fallback",
        "weak_label_source": "embedding_topk_community_author_aux",
        "weak_labels_disabled": bool(disable_weak_labels),
        "author_aux_weight": author_aux_weight,
        "community_top_m": int(community_top_m),
        "community_expansion_rule": "per_qa_top_m_in_qa_seed_communities" if community_top_m > 0 else "percentile_in_qa_seed_communities",
        "author_aux_positive_counts": [
            int(author_aux_positive_counts.get(idx, 0))
            for idx, qtype in enumerate(query_types)
            if qtype == "author"
        ],
        "seed_communities_union": sorted(qa_seed_communities),
        "sim_percentile": float(sim_percentile),
        "seed_topk": int(seed_topk),
        "author_seed_topk_ignored_for_rag": True,
        "node_query_source": "qa_query_only",
    }


def apply_rag_retrieval_aligned_labels(
    query_data: dict[str, Any],
    node_ids: list[Any],
    stats_path: Path,
    *,
    positive_count: int,
    retain_penalty: float,
    force_positive_node_ranges: list[str] | None = None,
    force_positive_nodes: set[str] | None = None,
    force_positive_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    forget_counts = Counter({str(k): int(v) for k, v in stats.get("forget_counts", {}).items()})
    retain_counts = Counter({str(k): int(v) for k, v in stats.get("retain_counts", {}).items()})
    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_ids)}
    candidates = [node_id for node_id in forget_counts if node_id in node_to_idx]
    if not candidates:
        raise ValueError(f"No retrieval-aligned RAG candidates from {stats_path}")
    take = min(max(1, int(positive_count)), len(candidates))
    ranked = sorted(
        candidates,
        key=lambda node_id: (
            float(forget_counts[node_id]) - float(retain_penalty) * float(retain_counts[node_id]),
            float(forget_counts[node_id]),
            -float(retain_counts[node_id]),
        ),
        reverse=True,
    )
    positive_nodes = set(ranked[:take])
    forced_positive_nodes = expand_node_ranges(force_positive_node_ranges or [], node_to_idx)
    forced_positive_nodes.update(str(node_id) for node_id in (force_positive_nodes or set()) if str(node_id) in node_to_idx)
    positive_nodes.update(forced_positive_nodes)
    safe_nodes = {node_id for node_id in retain_counts if node_id in node_to_idx and node_id not in positive_nodes}

    labels = np.zeros(len(node_ids), dtype=np.int64)
    safe_mask = np.zeros(len(node_ids), dtype=np.int64)
    for node_id in positive_nodes:
        labels[node_to_idx[node_id]] = 1
    for node_id in safe_nodes:
        safe_mask[node_to_idx[node_id]] = 1

    query_data["labels"] = labels
    query_data["safe_mask"] = safe_mask
    query_data["strong_mask"] = labels.copy()
    query_data["weak_mask"] = np.zeros(len(node_ids), dtype=np.int64)
    query_data["lexical_mask"] = labels.copy()
    query_data["positive_count"] = int(labels.sum())
    query_data["strong_positive_count"] = int(labels.sum())
    query_data["weak_positive_count"] = 0
    query_data["safe_negative_count"] = int(safe_mask.sum())
    query_data["lexical_positive_count"] = int(labels.sum())
    query_data["retrieval_aligned"] = {
        "enabled": True,
        "stats_path": str(stats_path),
        "candidate_count": int(len(candidates)),
        "target_positive_count": int(positive_count),
        "forced_positive_count": int(len(forced_positive_nodes)),
        "forced_positive_report": force_positive_report or {},
        "actual_positive_count": int(labels.sum()),
        "safe_negative_count": int(safe_mask.sum()),
        "retain_penalty": float(retain_penalty),
        "positive_forget_count_sum": int(sum(forget_counts[node_id] for node_id in positive_nodes)),
        "positive_retain_count_sum": int(sum(retain_counts[node_id] for node_id in positive_nodes)),
    }
    return query_data


def attach_rag_retrieval_score_supervision(
    query_data: dict[str, Any],
    node_ids: list[Any],
    stats_path: Path,
    *,
    retain_penalty: float,
    temperature: float,
) -> dict[str, Any]:
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    forget_counts = Counter({str(k): int(v) for k, v in stats.get("forget_counts", {}).items()})
    retain_counts = Counter({str(k): int(v) for k, v in stats.get("retain_counts", {}).items()})
    target_scores = np.zeros(len(node_ids), dtype=np.float32)
    forget_arr = np.zeros(len(node_ids), dtype=np.float32)
    retain_arr = np.zeros(len(node_ids), dtype=np.float32)
    for idx, node_id in enumerate(node_ids):
        key = str(node_id)
        forget_arr[idx] = float(forget_counts.get(key, 0))
        retain_arr[idx] = float(retain_counts.get(key, 0))
        target_scores[idx] = forget_arr[idx] - float(retain_penalty) * retain_arr[idx]

    temp = max(1e-6, float(temperature))
    soft_targets = np.zeros(len(node_ids), dtype=np.float32)
    positive_scores = target_scores[target_scores > 0]
    if positive_scores.size:
        scaled_positive = np.clip(target_scores[target_scores > 0] / temp, -60.0, 60.0)
        soft_targets[target_scores > 0] = 1.0 / (1.0 + np.exp(-scaled_positive))
    positive_pool = np.where(target_scores > 0)[0]
    negative_pool = np.where(target_scores < 0)[0]
    if positive_pool.size == 0:
        positive_pool = np.argsort(-target_scores)[: max(1, min(200, len(target_scores)))]
    if negative_pool.size == 0:
        negative_pool = np.argsort(target_scores)[: max(1, min(200, len(target_scores)))]

    query_data["retrieval_score_targets"] = soft_targets.astype(np.float32)
    query_data["retrieval_score_raw"] = target_scores.astype(np.float32)
    query_data["retrieval_score_arrays"] = {
        "forget_counts": forget_arr.astype(np.float32),
        "retain_counts": retain_arr.astype(np.float32),
    }
    query_data["retrieval_score_rank_positive_indices"] = positive_pool.astype(np.int64)
    query_data["retrieval_score_rank_negative_indices"] = negative_pool.astype(np.int64)
    query_data["retrieval_score_supervision"] = {
        "enabled": True,
        "stats_path": str(stats_path),
        "retain_penalty": float(retain_penalty),
        "temperature": float(temp),
        "score_min": float(target_scores.min()) if target_scores.size else 0.0,
        "score_max": float(target_scores.max()) if target_scores.size else 0.0,
        "score_mean": float(target_scores.mean()) if target_scores.size else 0.0,
        "soft_target_min": float(soft_targets.min()) if soft_targets.size else 0.0,
        "soft_target_max": float(soft_targets.max()) if soft_targets.size else 0.0,
        "soft_target_mean": float(soft_targets.mean()) if soft_targets.size else 0.0,
        "rank_positive_count": int(len(positive_pool)),
        "rank_negative_count": int(len(negative_pool)),
        "forget_count_sum": int(forget_arr.sum()),
        "retain_count_sum": int(retain_arr.sum()),
    }
    return query_data


def build_rag_aux_features(query_data: dict[str, Any], *, entity_weight: float) -> np.ndarray | None:
    if "retrieval_score_raw" not in query_data:
        return None
    score = np.asarray(query_data["retrieval_score_raw"], dtype=np.float32)
    targets = np.asarray(query_data.get("retrieval_score_targets", np.zeros_like(score)), dtype=np.float32)
    similarities = np.asarray(query_data.get("similarities", np.zeros_like(score)), dtype=np.float32)
    lexical = np.asarray(query_data.get("lexical_mask", np.zeros_like(score)), dtype=np.float32)
    labels = np.asarray(query_data.get("labels", np.zeros_like(score)), dtype=np.float32)
    meta = query_data.get("retrieval_score_arrays", {})
    forget_counts = np.asarray(meta.get("forget_counts", np.zeros_like(score)), dtype=np.float32)
    retain_counts = np.asarray(meta.get("retain_counts", np.zeros_like(score)), dtype=np.float32)

    log_forget = np.log1p(forget_counts)
    log_retain = np.log1p(retain_counts)
    max_abs_score = max(1.0, float(np.max(np.abs(score))) if score.size else 1.0)
    score_norm = score / max_abs_score
    ratio = forget_counts / (retain_counts + 1.0)
    ratio_norm = ratio / max(1.0, float(np.max(ratio)) if ratio.size else 1.0)
    sim_min = float(np.min(similarities)) if similarities.size else 0.0
    sim_max = float(np.max(similarities)) if similarities.size else 1.0
    sim_norm = (similarities - sim_min) / max(1e-6, sim_max - sim_min)
    positive_specific = ((score > 0).astype(np.float32) * labels) * float(entity_weight)
    features = np.stack(
        [
            log_forget,
            log_retain,
            score_norm,
            ratio_norm,
            targets,
            sim_norm,
            lexical * float(entity_weight),
            positive_specific,
        ],
        axis=1,
    ).astype(np.float32)
    return features


def train_rag(
    rag_graph: nx.Graph,
    query_vecs: np.ndarray,
    query_targets: list[str],
    query_types: list[str],
    *,
    output_dir: Path,
    device: torch.device,
    hidden_dim: int,
    heads: int,
    dropout: float,
    pretrain_epochs: int,
    retrain_epochs: int,
    lr: float,
    weight_decay: float,
    edge_sample_size: int,
    rag_seed_topk: int,
    rag_sim_percentile: float,
    rag_qa_topk: int,
    rag_author_aux_weight: float,
    rag_community_top_m: int,
    rag_disable_weak_labels: bool,
    rag_drop_class_weight_scale: float,
    rag_lexical_loss_weight: float,
    rag_safe_loss_weight: float,
    rag_cardinality_target_count: int,
    rag_cardinality_loss_weight: float,
    rag_threshold_target_count: int,
    rag_threshold_cardinality_penalty: float,
    rag_include_strong_in_hard: bool,
    rag_strong_hard_fraction: float,
    rag_retrieval_aligned_stats: Path | None,
    rag_retrieval_aligned_positive_count: int,
    rag_retrieval_aligned_retain_penalty: float,
    rag_force_positive_node_range: list[str],
    rag_force_positive_qa_path: Path | None,
    rag_retrieval_score_supervision_stats: Path | None,
    rag_retrieval_score_retain_penalty: float,
    rag_retrieval_score_temperature: float,
    rag_retrieval_score_loss_weight: float,
    rag_retrieval_ranking_loss_weight: float,
    rag_retrieval_ranking_margin: float,
    rag_retrieval_ranking_pairs: int,
    rag_strong_ranking_loss_weight: float,
    rag_strong_ranking_margin: float,
    rag_strong_ranking_pairs: int,
    rag_aux_feature_head: bool,
    rag_aux_logit_weight: float,
    rag_gat_logit_weight: float,
    rag_entity_aux_feature_weight: float,
    skip_pretrain: bool,
    pretrain_model_path: Path | None,
    pretrain_only: bool,
    seed: int,
) -> tuple[nx.Graph, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    batch = build_rag_batch(rag_graph)
    batch.community, community_mapping = remap_communities(batch.community)
    community_count = int(batch.community.max().item()) + 1
    batch = batch.to(device)

    rag_aux_feature_dim = 8 if bool(rag_aux_feature_head) else 0
    model = RAGGAT(
        input_dim=batch.x.size(1),
        hidden_dim=hidden_dim,
        community_count=community_count,
        aux_feature_dim=rag_aux_feature_dim,
        aux_logit_weight=rag_aux_logit_weight,
        gat_logit_weight=rag_gat_logit_weight,
        heads=heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    node_count = len(batch.node_ids)
    existing = {(int(u), int(v)) for u, v in batch.edge_index.detach().cpu().T.tolist()}
    pos_left_all = batch.edge_index[0].detach().cpu().numpy()
    pos_right_all = batch.edge_index[1].detach().cpu().numpy()
    pos_weight_all = batch.edge_weight.squeeze(-1).detach().cpu().numpy().astype(np.float32)
    comm_np = batch.community.detach().cpu().numpy()

    pre_model_path = pretrain_model_path or (output_dir / "rag_pre_gat.pt")
    pretrain_log: list[dict[str, float]] = []
    if skip_pretrain:
        if not pre_model_path.exists():
            raise FileNotFoundError(f"RAG pretrain checkpoint not found: {pre_model_path}")
        load_result = model.load_state_dict(
            torch.load(pre_model_path, map_location=device),
            strict=not bool(rag_aux_feature_head),
        )
        if bool(rag_aux_feature_head):
            print(
                f"[RAG:pretrain] skipped; partially loaded {pre_model_path} "
                f"missing={list(load_result.missing_keys)} unexpected={list(load_result.unexpected_keys)}"
            )
        else:
            print(f"[RAG:pretrain] skipped; loaded {pre_model_path}")
    else:
        for epoch in range(1, pretrain_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(batch)
            loss_comm = F.cross_entropy(model.community_logits(z), batch.community)

            take = min(int(edge_sample_size), len(pos_left_all))
            pos_idx = rng.choice(len(pos_left_all), size=take, replace=False)
            neg_left, neg_right = sample_non_edges(node_count=node_count, existing=existing, count=take, rng=rng)
            edge_left = np.concatenate([pos_left_all[pos_idx], neg_left])
            edge_right = np.concatenate([pos_right_all[pos_idx], neg_right])
            edge_targets = np.concatenate([pos_weight_all[pos_idx], np.zeros(len(neg_left), dtype=np.float32)])
            edge_left_t = torch.tensor(edge_left, dtype=torch.long, device=device)
            edge_right_t = torch.tensor(edge_right, dtype=torch.long, device=device)
            edge_targets_t = torch.tensor(edge_targets, dtype=torch.float32, device=device)
            loss_edge = F.mse_loss(torch.sigmoid(model.edge_weight_logits(z, edge_left_t, edge_right_t)), edge_targets_t)

            con_count = min(int(edge_sample_size) // 2, max(256, node_count))
            p_left, p_right = sample_pairs_by_community(comm_np, count=con_count, positive=True, rng=rng)
            n_left, n_right = sample_pairs_by_community(comm_np, count=con_count, positive=False, rng=rng)
            con_left = np.concatenate([p_left, n_left])
            con_right = np.concatenate([p_right, n_right])
            con_labels = np.concatenate([np.ones(len(p_left), dtype=np.float32), np.zeros(len(n_left), dtype=np.float32)])
            if len(con_left):
                loss_con = F.binary_cross_entropy_with_logits(
                    model.contrast_logits(
                        z,
                        torch.tensor(con_left, dtype=torch.long, device=device),
                        torch.tensor(con_right, dtype=torch.long, device=device),
                    ),
                    torch.tensor(con_labels, dtype=torch.float32, device=device),
                )
            else:
                loss_con = z.sum() * 0.0

            loss = loss_comm + 0.7 * loss_edge + 0.3 * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pretrain_log.append({
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "community": float(loss_comm.detach().cpu()),
                "edge_weight": float(loss_edge.detach().cpu()),
                "contrast": float(loss_con.detach().cpu()),
            })
            if epoch == 1 or epoch == pretrain_epochs or epoch % max(1, pretrain_epochs // 5) == 0:
                print(f"[RAG:pretrain] epoch={epoch}/{pretrain_epochs} loss={pretrain_log[-1]['loss']:.4f}")
        torch.save(model.state_dict(), pre_model_path)

    if pretrain_only:
        report = {
            "pre_model_path": str(pre_model_path),
            "model_path": str(pre_model_path),
            "skip_pretrain": bool(skip_pretrain),
            "pretrain_only": True,
            "node_count": node_count,
            "edge_count_directed": int(batch.edge_index.size(1)),
            "community_count": community_count,
            "community_mapping_size": len(community_mapping),
            "pretrain_loss": "community_ce + 0.7*edge_weight_mse + 0.3*community_contrast_bce",
            "pretrain_log": pretrain_log,
        }
        return rag_graph.copy(), report

    query_data = build_rag_multi_query_labels(
        rag_graph,
        query_vecs,
        query_targets,
        query_types,
        seed_topk=rag_seed_topk,
        sim_percentile=rag_sim_percentile,
        qa_topk=rag_qa_topk,
        author_aux_weight=rag_author_aux_weight,
        community_top_m=rag_community_top_m,
        disable_weak_labels=rag_disable_weak_labels,
    )
    if rag_retrieval_aligned_stats is not None:
        force_positive_nodes: set[str] = set()
        force_positive_report: dict[str, Any] = {}
        if rag_force_positive_qa_path is not None:
            force_positive_nodes, force_positive_report = load_rag_nodes_from_qa_path(
                rag_force_positive_qa_path,
                batch.node_ids,
                rag_graph,
                query_vecs,
                query_types,
            )
            if force_positive_report.get("missing_count", 0):
                print(f"[RAG:labels] force-positive QA missing rows: {force_positive_report}")
            else:
                print(
                    "[RAG:labels] force-positive QA matched "
                    f"{force_positive_report.get('matched_node_count', 0)}/{force_positive_report.get('row_count', 0)} nodes"
                )
        query_data = apply_rag_retrieval_aligned_labels(
            query_data,
            batch.node_ids,
            rag_retrieval_aligned_stats,
            positive_count=rag_retrieval_aligned_positive_count,
            retain_penalty=rag_retrieval_aligned_retain_penalty,
            force_positive_node_ranges=rag_force_positive_node_range,
            force_positive_nodes=force_positive_nodes,
            force_positive_report=force_positive_report,
        )
    if rag_retrieval_score_supervision_stats is not None:
        query_data = attach_rag_retrieval_score_supervision(
            query_data,
            batch.node_ids,
            rag_retrieval_score_supervision_stats,
            retain_penalty=rag_retrieval_score_retain_penalty,
            temperature=rag_retrieval_score_temperature,
        )
    node_query_t = torch.tensor(query_data["node_query_vecs"], dtype=torch.float32, device=device)
    labels_t = torch.tensor(query_data["labels"], dtype=torch.long, device=device)
    strong_mask_t = torch.tensor(
        query_data.get("strong_mask", query_data.get("lexical_mask", np.zeros(node_count, dtype=np.int64))),
        dtype=torch.bool,
        device=device,
    ) & labels_t.bool()
    weak_mask_t = torch.tensor(
        query_data.get("weak_mask", query_data["labels"]),
        dtype=torch.bool,
        device=device,
    ) & labels_t.bool() & (~strong_mask_t)
    lexical_mask_t = strong_mask_t
    safe_mask_t = torch.tensor(query_data["safe_mask"], dtype=torch.bool, device=device)
    retrieval_score_targets_t = (
        torch.tensor(query_data["retrieval_score_targets"], dtype=torch.float32, device=device)
        if "retrieval_score_targets" in query_data
        else None
    )
    retrieval_rank_pos_t = (
        torch.tensor(query_data["retrieval_score_rank_positive_indices"], dtype=torch.long, device=device)
        if "retrieval_score_rank_positive_indices" in query_data
        else None
    )
    retrieval_rank_neg_t = (
        torch.tensor(query_data["retrieval_score_rank_negative_indices"], dtype=torch.long, device=device)
        if "retrieval_score_rank_negative_indices" in query_data
        else None
    )
    rag_aux_features_np = (
        build_rag_aux_features(query_data, entity_weight=rag_entity_aux_feature_weight)
        if bool(rag_aux_feature_head)
        else None
    )
    rag_aux_features_t = (
        torch.tensor(rag_aux_features_np, dtype=torch.float32, device=device)
        if rag_aux_features_np is not None
        else None
    )
    class_weights = class_weights_from_binary_labels(labels_t).to(device)
    class_weights[1] = class_weights[1] * float(rag_drop_class_weight_scale)

    retrain_log: list[dict[str, float]] = []
    for epoch in range(1, retrain_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model.encode(batch, node_query_t)
        logits = model.drop_logits(z, node_query_t, rag_aux_features_t)
        loss_ce = F.cross_entropy(logits, labels_t, weight=class_weights)
        loss_strong = (
            F.cross_entropy(logits[strong_mask_t], torch.ones(int(strong_mask_t.sum().item()), dtype=torch.long, device=device))
            if strong_mask_t.any()
            else logits.sum() * 0.0
        )
        loss_safe = (
            F.cross_entropy(logits[safe_mask_t], torch.zeros(int(safe_mask_t.sum().item()), dtype=torch.long, device=device))
            if safe_mask_t.any()
            else logits.sum() * 0.0
        )
        drop_probs = torch.softmax(logits, dim=-1)[:, 1]
        if int(rag_cardinality_target_count) > 0 and float(rag_cardinality_loss_weight) > 0:
            target_rate = min(1.0, max(0.0, float(rag_cardinality_target_count) / max(1, node_count)))
            target_rate_t = torch.tensor(target_rate, dtype=drop_probs.dtype, device=device)
            loss_cardinality = (drop_probs.mean() - target_rate_t).pow(2)
        else:
            target_rate = 0.0
            loss_cardinality = logits.sum() * 0.0
        if retrieval_score_targets_t is not None and float(rag_retrieval_score_loss_weight) > 0:
            loss_retrieval_score = F.binary_cross_entropy(drop_probs, retrieval_score_targets_t)
        else:
            loss_retrieval_score = logits.sum() * 0.0
        if (
            retrieval_rank_pos_t is not None
            and retrieval_rank_neg_t is not None
            and retrieval_rank_pos_t.numel() > 0
            and retrieval_rank_neg_t.numel() > 0
            and int(rag_retrieval_ranking_pairs) > 0
            and float(rag_retrieval_ranking_loss_weight) > 0
        ):
            pair_count = min(
                int(rag_retrieval_ranking_pairs),
                int(retrieval_rank_pos_t.numel()),
                int(retrieval_rank_neg_t.numel()),
            )
            pos_perm = torch.randperm(int(retrieval_rank_pos_t.numel()), device=device)[:pair_count]
            neg_perm = torch.randperm(int(retrieval_rank_neg_t.numel()), device=device)[:pair_count]
            pos_idx = retrieval_rank_pos_t[pos_perm]
            neg_idx = retrieval_rank_neg_t[neg_perm]
            loss_retrieval_ranking = F.relu(
                float(rag_retrieval_ranking_margin) - (drop_probs[pos_idx] - drop_probs[neg_idx])
            ).mean()
        else:
            loss_retrieval_ranking = logits.sum() * 0.0
        if (
            strong_mask_t.any()
            and safe_mask_t.any()
            and int(rag_strong_ranking_pairs) > 0
            and float(rag_strong_ranking_loss_weight) > 0
        ):
            strong_idx_all = torch.nonzero(strong_mask_t, as_tuple=False).flatten()
            safe_idx_all = torch.nonzero(safe_mask_t, as_tuple=False).flatten()
            pair_count = min(
                int(rag_strong_ranking_pairs),
                int(strong_idx_all.numel()),
                int(safe_idx_all.numel()),
            )
            strong_perm = torch.randperm(int(strong_idx_all.numel()), device=device)[:pair_count]
            safe_perm = torch.randperm(int(safe_idx_all.numel()), device=device)[:pair_count]
            pos_idx = strong_idx_all[strong_perm]
            neg_idx = safe_idx_all[safe_perm]
            loss_strong_ranking = F.relu(
                float(rag_strong_ranking_margin) - (drop_probs[pos_idx] - drop_probs[neg_idx])
            ).mean()
        else:
            loss_strong_ranking = logits.sum() * 0.0
        loss = (
            loss_ce
            + float(rag_lexical_loss_weight) * loss_strong
            + float(rag_safe_loss_weight) * loss_safe
            + float(rag_cardinality_loss_weight) * loss_cardinality
            + float(rag_retrieval_score_loss_weight) * loss_retrieval_score
            + float(rag_retrieval_ranking_loss_weight) * loss_retrieval_ranking
            + float(rag_strong_ranking_loss_weight) * loss_strong_ranking
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        retrain_log.append({
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "cross_entropy": float(loss_ce.detach().cpu()),
            "strong_positive": float(loss_strong.detach().cpu()),
            "safe_negative": float(loss_safe.detach().cpu()),
            "cardinality": float(loss_cardinality.detach().cpu()),
            "retrieval_score": float(loss_retrieval_score.detach().cpu()),
            "retrieval_ranking": float(loss_retrieval_ranking.detach().cpu()),
            "strong_ranking": float(loss_strong_ranking.detach().cpu()),
            "drop_prob_mean": float(drop_probs.detach().mean().cpu()),
            "cardinality_target_rate": float(target_rate),
            "strong_positive_count": float(strong_mask_t.sum().detach().cpu()),
            "weak_positive_count": float(weak_mask_t.sum().detach().cpu()),
        })
        if epoch == 1 or epoch == retrain_epochs or epoch % max(1, retrain_epochs // 5) == 0:
            print(f"[RAG:retrain] epoch={epoch}/{retrain_epochs} loss={retrain_log[-1]['loss']:.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(batch, node_query_t)
        probs = torch.softmax(model.drop_logits(z, node_query_t, rag_aux_features_t), dim=-1).detach().cpu().numpy()
    if int(rag_threshold_target_count) > 0 and float(rag_strong_hard_fraction) > 0.0:
        strong_np = np.asarray(query_data.get("strong_mask", np.zeros(node_count, dtype=np.int64)), dtype=np.int64).astype(bool)
        hard = np.zeros(node_count, dtype=np.int64)
        strong_indices = np.where(strong_np)[0]
        strong_take = min(
            int(strong_indices.size),
            max(1, int(math.ceil(float(rag_strong_hard_fraction) * float(strong_indices.size)))),
        )
        if strong_take > 0 and strong_indices.size:
            selected_strong = strong_indices[np.argsort(-probs[strong_indices, 1])[:strong_take]]
            hard[selected_strong] = 1
        remaining_target = max(0, int(rag_threshold_target_count) - int(hard.sum()))
        candidate_indices = np.where(hard == 0)[0]
        if remaining_target > 0 and candidate_indices.size:
            take = min(int(remaining_target), int(candidate_indices.size))
            ranked = candidate_indices[np.argsort(-probs[candidate_indices, 1])[:take]]
            hard[ranked] = 1
            rag_threshold = float(probs[ranked, 1].min()) if ranked.size else 1.0
        else:
            rag_threshold = 1.0
        rag_threshold_report = {
            "mode": "target_count_with_strong_labels_forced",
            "target_count": float(rag_threshold_target_count),
            "strong_count": float(strong_np.sum()),
            "strong_hard_fraction": float(rag_strong_hard_fraction),
            "strong_selected": float(strong_take),
            "remaining_target": float(remaining_target),
            "actual_count": float(hard.sum()),
            "pred_rate": float(hard.mean()),
        }
    elif int(rag_threshold_target_count) > 0 and bool(rag_include_strong_in_hard):
        strong_np = np.asarray(query_data.get("strong_mask", np.zeros(node_count, dtype=np.int64)), dtype=np.int64).astype(bool)
        hard = strong_np.astype(np.int64)
        remaining_target = max(0, int(rag_threshold_target_count) - int(hard.sum()))
        candidate_indices = np.where(~strong_np)[0]
        if remaining_target > 0 and candidate_indices.size:
            take = min(int(remaining_target), int(candidate_indices.size))
            ranked = candidate_indices[np.argsort(-probs[candidate_indices, 1])[:take]]
            hard[ranked] = 1
            rag_threshold = float(probs[ranked, 1].min()) if ranked.size else 1.0
        else:
            rag_threshold = 1.0
        rag_threshold_report = {
            "mode": "target_count_with_all_strong_labels_forced",
            "target_count": float(rag_threshold_target_count),
            "strong_count": float(strong_np.sum()),
            "remaining_target": float(remaining_target),
            "actual_count": float(hard.sum()),
            "pred_rate": float(hard.mean()),
        }
    elif int(rag_threshold_target_count) > 0 and float(rag_threshold_cardinality_penalty) > 0:
        rag_threshold, rag_threshold_report = choose_cardinality_regularized_threshold(
            probs[:, 1],
            query_data["labels"],
            target_count=int(rag_threshold_target_count),
            penalty=float(rag_threshold_cardinality_penalty),
            positive_cutoff=0.5,
            fallback=0.5,
        )
        hard = (probs[:, 1] >= rag_threshold).astype(np.int64)
    elif int(rag_threshold_target_count) > 0:
        rag_threshold, rag_threshold_report = choose_count_threshold(probs[:, 1], int(rag_threshold_target_count))
        hard = (probs[:, 1] >= rag_threshold).astype(np.int64)
    else:
        rag_threshold, rag_threshold_report = choose_binary_threshold(
            probs[:, 1],
            query_data["labels"],
            positive_cutoff=0.5,
            fallback=0.5,
        )
        hard = (probs[:, 1] >= rag_threshold).astype(np.int64)
        if bool(rag_include_strong_in_hard):
            strong_np = np.asarray(query_data.get("strong_mask", np.zeros(node_count, dtype=np.int64)), dtype=np.int64).astype(bool)
            hard = np.where(strong_np, 1, hard).astype(np.int64)
    out_graph = rag_graph.copy()
    for idx, node_id in enumerate(batch.node_ids):
        out_graph.nodes[node_id]["keep_prob"] = float(probs[idx, 0])
        out_graph.nodes[node_id]["drop_prob"] = float(probs[idx, 1])
        out_graph.nodes[node_id]["unlearned"] = int(hard[idx])
        out_graph.nodes[node_id]["query_similarity"] = float(query_data["similarities"][idx])
        out_graph.nodes[node_id]["best_query_index"] = int(query_data["best_query_idx"][idx])
        out_graph.nodes[node_id]["weak_label"] = int(query_data["labels"][idx])
        out_graph.nodes[node_id]["strong_label"] = int(np.asarray(query_data.get("strong_mask", np.zeros(node_count, dtype=np.int64)))[idx])
        out_graph.nodes[node_id]["weak_aux_label"] = int(np.asarray(query_data.get("weak_mask", np.zeros(node_count, dtype=np.int64)))[idx])

    retrain_model_path = output_dir / "rag_retrain_gat.pt"
    torch.save(model.state_dict(), retrain_model_path)
    report = {
        "pre_model_path": str(pre_model_path),
        "retrain_model_path": str(retrain_model_path),
        "model_path": str(retrain_model_path),
        "skip_pretrain": bool(skip_pretrain),
        "node_count": node_count,
        "edge_count_directed": int(batch.edge_index.size(1)),
        "community_count": community_count,
        "community_mapping_size": len(community_mapping),
        "query_label_source": "strong_exact_qa_plus_weak_embedding_community_author_aux",
        "community_source": query_data.get("community_source", ""),
        "node_query_source": query_data.get("node_query_source", ""),
        "author_seed_topk_ignored_for_rag": bool(query_data.get("author_seed_topk_ignored_for_rag", False)),
        "query_count": int(np.asarray(query_data.get("query_vecs", query_vecs)).shape[0]),
        "query_type_counts": query_data["query_type_counts"],
        "seed_topk": int(rag_seed_topk),
        "qa_topk": int(rag_qa_topk),
        "author_aux_weight": float(query_data.get("author_aux_weight", rag_author_aux_weight)),
        "community_top_m": int(query_data.get("community_top_m", rag_community_top_m)),
        "community_expansion_rule": query_data.get("community_expansion_rule", ""),
        "per_query_seed_counts": query_data["per_query_seed_counts"],
        "per_query_lexical_seed_counts": query_data["per_query_lexical_seed_counts"],
        "per_query_exact_match_counts": query_data.get("per_query_exact_match_counts", []),
        "exact_qa_match_count": int(query_data.get("exact_qa_match_count", 0)),
        "exact_qa_missing_count": int(query_data.get("exact_qa_missing_count", 0)),
        "exact_qa_missing_query_indices": query_data.get("exact_qa_missing_query_indices", []),
        "author_aux_rule": query_data.get("author_aux_rule", ""),
        "author_aux_positive_counts": query_data.get("author_aux_positive_counts", []),
        "seed_communities_union": query_data["seed_communities_union"],
        "sim_percentile_input": float(rag_sim_percentile),
        "sim_percentile": float(query_data.get("sim_percentile", rag_sim_percentile)),
        "per_query_positive_counts": query_data["per_query_positive_counts"],
        "positive_count": int(query_data["positive_count"]),
        "strong_positive_count": int(query_data.get("strong_positive_count", query_data["lexical_positive_count"])),
        "weak_positive_count": int(query_data.get("weak_positive_count", query_data["positive_count"])),
        "lexical_positive_count": int(query_data["lexical_positive_count"]),
        "safe_negative_count": int(query_data["safe_negative_count"]),
        "strong_label_source": query_data.get("strong_label_source", "exact_question_answer_match"),
        "weak_label_source": query_data.get("weak_label_source", "embedding_topk_community_author_aux"),
        "weak_labels_disabled": bool(query_data.get("weak_labels_disabled", False)),
        "retrieval_aligned": query_data.get("retrieval_aligned", {"enabled": False}),
        "retrieval_score_supervision": query_data.get("retrieval_score_supervision", {"enabled": False}),
        "aux_feature_head": {
            "enabled": bool(rag_aux_feature_head),
            "feature_dim": int(rag_aux_feature_dim),
            "aux_logit_weight": float(rag_aux_logit_weight),
            "gat_logit_weight": float(rag_gat_logit_weight),
            "entity_aux_feature_weight": float(rag_entity_aux_feature_weight),
            "feature_names": [
                "log1p_forget_count",
                "log1p_retain_count",
                "score_norm",
                "forget_retain_ratio_norm",
                "positive_soft_target",
                "query_similarity_norm",
                "lexical_entity_aux",
                "strong_label_entity_aux",
            ],
        },
        "query_pooling": "per_node_argmax_cosine_assignment",
        "decision_rule": (
            "cardinality_regularized_f1_threshold_on_drop_prob"
            if int(rag_threshold_target_count) > 0 and float(rag_threshold_cardinality_penalty) > 0
            else "target_count_threshold_on_drop_prob"
            if int(rag_threshold_target_count) > 0
            else "auto_f1_threshold_on_drop_prob"
        ),
        "drop_threshold": float(rag_threshold),
        "threshold_report": rag_threshold_report,
        "unlearned_count": int(hard.sum()),
        "drop_prob_mean": float(probs[:, 1].mean()),
        "drop_prob_max": float(probs[:, 1].max()),
        "drop_prob_min": float(probs[:, 1].min()),
        "class_weight_keep": float(class_weights[0].detach().cpu()),
        "class_weight_drop": float(class_weights[1].detach().cpu()),
        "drop_class_weight_scale": float(rag_drop_class_weight_scale),
        "loss_weights": {
            "strong_positive": float(rag_lexical_loss_weight),
            "safe_negative": float(rag_safe_loss_weight),
            "cardinality": float(rag_cardinality_loss_weight),
            "retrieval_score": float(rag_retrieval_score_loss_weight),
            "retrieval_ranking": float(rag_retrieval_ranking_loss_weight),
            "retrieval_ranking_margin": float(rag_retrieval_ranking_margin),
            "retrieval_ranking_pairs": int(rag_retrieval_ranking_pairs),
            "strong_ranking": float(rag_strong_ranking_loss_weight),
            "strong_ranking_margin": float(rag_strong_ranking_margin),
            "strong_ranking_pairs": int(rag_strong_ranking_pairs),
        },
        "cardinality_target_count": int(rag_cardinality_target_count),
        "cardinality_target_rate": float(
            min(1.0, max(0.0, float(rag_cardinality_target_count) / max(1, node_count)))
            if int(rag_cardinality_target_count) > 0
            else 0.0
        ),
        "threshold_target_count": int(rag_threshold_target_count),
        "threshold_cardinality_penalty": float(rag_threshold_cardinality_penalty),
        "include_strong_in_hard": bool(rag_include_strong_in_hard),
        "strong_hard_fraction": float(rag_strong_hard_fraction),
        "pretrain_loss": "community_ce + 0.7*edge_weight_mse + 0.3*community_contrast_bce",
        "retrain_loss": "query_aware_keep_drop_ce_plus_strong_exact_qa_positive_safe_negative_optional_cardinality_and_optional_retrieval_score_ranking",
        "pretrain_log": pretrain_log,
        "retrain_log": retrain_log,
    }
    return out_graph, report


def build_original_memory_edges(graph: nx.MultiDiGraph, node_to_idx: dict[str, int]) -> dict[str, Any]:
    left: list[int] = []
    right: list[int] = []
    edge_type: list[int] = []
    edge_score: list[float] = []
    edge_refs: list[tuple[str, str, int]] = []
    existing: set[tuple[int, int]] = set()
    by_type = Counter()
    for source, target, key, data in graph.edges(keys=True, data=True):
        u = node_to_idx[str(source)]
        v = node_to_idx[str(target)]
        edge_name = str(data.get("edge_type") or "no_edge")
        et = MEMORY_EDGE_TYPES.get(edge_name, MEMORY_EDGE_TYPES["no_edge"])
        score = float(data.get("confidence", data.get("similarity", 1.0)) or 1.0)
        left.append(u)
        right.append(v)
        edge_type.append(et)
        edge_score.append(max(0.0, min(1.0, score)))
        edge_refs.append((str(source), str(target), int(key)))
        existing.add((u, v))
        by_type[edge_name] += 1
    return {
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "edge_type": np.asarray(edge_type, dtype=np.int64),
        "edge_score": np.asarray(edge_score, dtype=np.float32),
        "edge_refs": edge_refs,
        "existing": existing,
        "by_type": dict(by_type),
    }


def sample_memory_negatives(
    batch,
    existing: set[tuple[int, int]],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    node_types = batch.node_type.detach().cpu().numpy()
    pools: dict[int, np.ndarray] = {
        type_id: np.where(node_types == type_id)[0].astype(np.int64)
        for type_id in np.unique(node_types).tolist()
    }
    pairs = [
        (MEMORY_NODE_TYPES["entity"], MEMORY_NODE_TYPES["anchor"]),
        (MEMORY_NODE_TYPES["anchor"], MEMORY_NODE_TYPES["anchor"]),
        (MEMORY_NODE_TYPES["entity"], MEMORY_NODE_TYPES["entity"]),
    ]
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    each = max(1, int(count) // len(pairs))
    for lt, rt in pairs:
        if lt not in pools or rt not in pools or len(pools[lt]) == 0 or len(pools[rt]) == 0:
            continue
        l, r = sample_non_edges(
            node_count=len(node_types),
            existing=existing,
            count=each,
            rng=rng,
            left_pool=pools[lt],
            right_pool=pools[rt],
        )
        left_parts.append(l)
        right_parts.append(r)
    if not left_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(left_parts), np.concatenate(right_parts)


def build_memory_query_labels(
    graph: nx.MultiDiGraph,
    batch,
    query_vec: np.ndarray,
    query_target: str,
    original_edges: dict[str, Any],
    *,
    path_cutoff: int,
    relation_prior: float,
    mentioned_prior: float,
    next_turn_prior: float,
    background_relation_scale: float,
    community_floor: float,
    path_decay: float,
) -> dict[str, Any]:
    node_ids = batch.node_ids
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    x = batch.x.detach().cpu().numpy().astype(np.float32)
    q = l2_normalize(np.asarray(query_vec, dtype=np.float32))
    norms = np.linalg.norm(x, axis=1)
    similarities = np.zeros(len(node_ids), dtype=np.float32)
    valid = norms > 1e-8
    if valid.any():
        similarities[valid] = l2_normalize(x[valid]) @ q

    target_key = normalize_key(query_target)
    target_entity_indices = {
        idx
        for idx, node_id in enumerate(node_ids)
        if graph.nodes[node_id].get("node_type") == "entity"
        and target_key
        and normalize_key(str(graph.nodes[node_id].get("canonical_name") or node_id)) == target_key
    }

    chunk_by_anchor = {
        str(node_id): str(data.get("chunk_id"))
        for node_id, data in graph.nodes(data=True)
        if str(data.get("node_type") or "") == "anchor" and data.get("chunk_id") is not None
    }

    target_anchor_ids: set[str] = set()
    for idx in target_entity_indices:
        entity_id = node_ids[idx]
        incident_edges = []
        if graph.is_directed():
            incident_edges.extend(graph.out_edges(entity_id, keys=True, data=True))
            incident_edges.extend(graph.in_edges(entity_id, keys=True, data=True))
        else:
            incident_edges.extend(graph.edges(entity_id, keys=True, data=True))
        for source, target, _key, data in incident_edges:
            if str(data.get("edge_type") or "") != "mentioned_in":
                continue
            other = str(target) if str(source) == entity_id else str(source)
            if other in graph.nodes and str(graph.nodes[other].get("node_type") or "") == "anchor":
                target_anchor_ids.add(other)

    target_chunk_ids = {
        chunk_id
        for anchor_id in target_anchor_ids
        if (chunk_id := chunk_by_anchor.get(anchor_id)) is not None and chunk_id in node_to_idx
    }
    primary_seed_ids = (
        {node_ids[idx] for idx in target_entity_indices}
        | target_anchor_ids
        | target_chunk_ids
    )
    seed_communities = {
        int(graph.nodes[node_id].get("community_id", -1))
        for node_id in primary_seed_ids
        if node_id in graph.nodes and int(graph.nodes[node_id].get("community_id", -1)) >= 0
    }

    path_graph = nx.Graph()
    path_graph.add_nodes_from(graph.nodes)
    edge_cost = {
        "mentioned_in": 1.2,
        "next_turn": 1.0,
        "relation": 1.5,
    }
    for source, target, _key, data in graph.edges(keys=True, data=True):
        cost = float(edge_cost.get(str(data.get("edge_type") or ""), 2.0))
        if path_graph.has_edge(source, target):
            path_graph[source][target]["weight"] = min(float(path_graph[source][target]["weight"]), cost)
        else:
            path_graph.add_edge(source, target, weight=cost)
    path_seed_ids = {node_id for node_id in primary_seed_ids if node_id in path_graph}
    if path_seed_ids:
        distances = dict(
            nx.multi_source_dijkstra_path_length(
                path_graph,
                list(path_seed_ids),
                cutoff=float(path_cutoff),
                weight="weight",
            )
        )
    else:
        distances = {}

    node_relevance = np.zeros(len(node_ids), dtype=np.float32)
    sim_scaled_all = np.zeros(len(node_ids), dtype=np.float32)
    if valid.any():
        sim_min = float(similarities[valid].min())
        sim_max = float(similarities[valid].max())
        sim_scaled = (similarities - sim_min) / max(1e-6, sim_max - sim_min)
        sim_scaled_all[valid] = np.clip(sim_scaled[valid], 0.0, 1.0)
    for node_id in primary_seed_ids:
        if node_id in node_to_idx:
            node_relevance[node_to_idx[node_id]] = 1.0
    for idx, node_id in enumerate(node_ids):
        comm = int(graph.nodes[node_id].get("community_id", -1))
        if comm in seed_communities:
            node_relevance[idx] = max(float(node_relevance[idx]), float(community_floor) * float(sim_scaled_all[idx]))
        if node_id in distances:
            node_relevance[idx] = max(
                float(node_relevance[idx]),
                float(math.exp(-distances[node_id] / max(1e-6, float(path_decay)))),
            )
    for anchor_id, chunk_id in chunk_by_anchor.items():
        if anchor_id in node_to_idx and chunk_id in node_to_idx:
            node_relevance[node_to_idx[chunk_id]] = max(
                float(node_relevance[node_to_idx[chunk_id]]),
                float(node_relevance[node_to_idx[anchor_id]]),
            )
    node_relevance = np.clip(node_relevance, 0.0, 1.0)
    target_chunk_indices = {
        node_to_idx[chunk_id]
        for chunk_id in target_chunk_ids
        if chunk_id in node_to_idx
    }

    prior = {
        MEMORY_EDGE_TYPES["relation"]: float(relation_prior),
        MEMORY_EDGE_TYPES["mentioned_in"]: float(mentioned_prior),
        MEMORY_EDGE_TYPES["next_turn"]: float(next_turn_prior),
        MEMORY_EDGE_TYPES["no_edge"]: 0.0,
    }
    left = original_edges["left"]
    right = original_edges["right"]
    etypes = original_edges["edge_type"]
    edge_scores = original_edges["edge_score"]
    labels = np.zeros(len(left), dtype=np.float32)
    next_turn_type = MEMORY_EDGE_TYPES["next_turn"]
    mentioned_type = MEMORY_EDGE_TYPES["mentioned_in"]
    relation_type = MEMORY_EDGE_TYPES["relation"]
    target_anchor_indices = {node_to_idx[node_id] for node_id in target_anchor_ids if node_id in node_to_idx}
    target_entity_indices = {int(idx) for idx in target_entity_indices}
    for idx, (u, v, et) in enumerate(zip(left.tolist(), right.tolist(), etypes.tolist())):
        endpoint = max(float(node_relevance[u]), float(node_relevance[v]))
        edge_prior = prior.get(int(et), 0.2)
        score_factor = float(edge_scores[idx]) if int(et) == next_turn_type else 1.0
        labels[idx] = min(1.0, endpoint * edge_prior * score_factor)
        if int(et) == mentioned_type and not (u in target_entity_indices or v in target_entity_indices):
            labels[idx] = min(float(labels[idx]), float(background_relation_scale))
        if int(et) == mentioned_type and (
            u in target_entity_indices
            or v in target_entity_indices
        ):
            labels[idx] = max(float(labels[idx]), min(1.0, float(mentioned_prior)))
        if int(et) == next_turn_type and (u in target_anchor_indices or v in target_anchor_indices):
            labels[idx] = max(float(labels[idx]), min(1.0, float(next_turn_prior)))
        if int(et) == relation_type:
            if u in target_entity_indices or v in target_entity_indices:
                labels[idx] = max(float(labels[idx]), min(1.0, float(relation_prior)))
            else:
                labels[idx] = min(float(labels[idx]), float(background_relation_scale))

    return {
        "query_vec": q.astype(np.float32),
        "similarities": similarities,
        "node_relevance": node_relevance,
        "edge_labels": labels,
        "edge_scores": edge_scores,
        "seed_node_ids": sorted(primary_seed_ids),
        "entity_seed_count": len(target_entity_indices),
        "target_entity_seed_count": len(target_entity_indices),
        "target_chunk_seed_count": len(target_chunk_indices),
        "target_anchor_seed_count": len(target_anchor_ids),
        "path_seed_count": len(path_seed_ids),
        "anchor_seed_count": len(target_anchor_ids),
        "seed_communities": sorted(seed_communities),
    }


def build_memory_multi_query_labels(
    graph: nx.MultiDiGraph,
    batch,
    query_vecs: np.ndarray,
    query_targets: list[str],
    original_edges: dict[str, Any],
    *,
    path_cutoff: int,
    relation_prior: float,
    mentioned_prior: float,
    next_turn_prior: float,
    background_relation_scale: float,
    community_floor: float,
    path_decay: float,
) -> dict[str, Any]:
    query_vecs = l2_normalize(np.asarray(query_vecs, dtype=np.float32))
    per_query = [
        build_memory_query_labels(
            graph,
            batch,
            query_vecs[idx],
            query_targets[idx],
            original_edges,
            path_cutoff=path_cutoff,
            relation_prior=relation_prior,
            mentioned_prior=mentioned_prior,
            next_turn_prior=next_turn_prior,
            background_relation_scale=background_relation_scale,
            community_floor=community_floor,
            path_decay=path_decay,
        )
        for idx in range(query_vecs.shape[0])
    ]
    similarities = np.stack([item["similarities"] for item in per_query], axis=0)
    node_relevance = np.stack([item["node_relevance"] for item in per_query], axis=0)
    edge_labels = np.stack([item["edge_labels"] for item in per_query], axis=0)
    best_query_idx = np.argmax(similarities, axis=0).astype(np.int64)
    merged_similarities = similarities[best_query_idx, np.arange(similarities.shape[1])]
    merged_node_relevance = node_relevance.max(axis=0).astype(np.float32)
    merged_edge_labels = edge_labels.max(axis=0).astype(np.float32)
    node_query_vecs = query_vecs[best_query_idx]
    seed_node_ids = sorted({node_id for item in per_query for node_id in item["seed_node_ids"]})
    seed_communities = sorted({comm for item in per_query for comm in item["seed_communities"]})
    return {
        "query_vecs": query_vecs.astype(np.float32),
        "node_query_vecs": node_query_vecs.astype(np.float32),
        "best_query_idx": best_query_idx,
        "similarities": merged_similarities.astype(np.float32),
        "node_relevance": merged_node_relevance,
        "edge_labels": merged_edge_labels,
        "seed_node_ids": seed_node_ids,
        "seed_communities": seed_communities,
        "per_query": per_query,
        "per_query_seed_counts": [len(item["seed_node_ids"]) for item in per_query],
        "per_query_edge_high_counts": [int((item["edge_labels"] >= 0.5).sum()) for item in per_query],
        "entity_seed_count": int(sum(item["entity_seed_count"] for item in per_query)),
        "target_entity_seed_count": int(sum(item["target_entity_seed_count"] for item in per_query)),
        "target_chunk_seed_count": int(sum(item["target_chunk_seed_count"] for item in per_query)),
        "target_anchor_seed_count": int(sum(item["target_anchor_seed_count"] for item in per_query)),
        "path_seed_count": int(sum(item["path_seed_count"] for item in per_query)),
        "anchor_seed_count": int(sum(item["anchor_seed_count"] for item in per_query)),
    }


def build_memory_author_qa_path_labels(
    graph: nx.MultiDiGraph,
    batch,
    author_vecs: np.ndarray,
    author_targets: list[str],
    qa_vecs: np.ndarray,
    qa_targets: list[str],
    original_edges: dict[str, Any],
    *,
    qa_topk: int,
    path_cutoff: int,
    relation_prior: float,
    mentioned_prior: float,
    next_turn_prior: float,
    path_decay: float,
) -> dict[str, Any]:
                                                                                      
    node_ids = batch.node_ids
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    x = batch.x.detach().cpu().numpy().astype(np.float32)
    x_norm = l2_normalize(x)
    author_vecs = l2_normalize(np.asarray(author_vecs, dtype=np.float32))
    qa_vecs = l2_normalize(np.asarray(qa_vecs, dtype=np.float32))
    combined_query_vecs = l2_normalize(np.concatenate([author_vecs, qa_vecs], axis=0))

    author_key_sets = [title_alias_keys(target) for target in author_targets]
    author_keys = [next(iter(keys), "") for keys in author_key_sets]
    qa_keys = [normalize_key(target) for target in qa_targets]
    qa_by_author: dict[int, list[int]] = {idx: [] for idx in range(len(author_targets))}
    global_qa_indices: list[int] = []
    for qidx, qa_key in enumerate(qa_keys):
        matched = [aidx for aidx, author_key in enumerate(author_keys) if author_key and author_key in qa_key]
        if matched:
            for aidx in matched:
                qa_by_author[aidx].append(qidx)
        else:
            global_qa_indices.append(qidx)

    entity_indices = [
        idx
        for idx, node_id in enumerate(node_ids)
        if str(graph.nodes[node_id].get("node_type") or "") == "entity"
    ]
    chunk_indices = [
        idx
        for idx, node_id in enumerate(node_ids)
        if str(graph.nodes[node_id].get("node_type") or "") == "chunk"
    ]
    chunk_to_anchor: dict[str, str] = {
        str(data.get("chunk_id")): str(node_id)
        for node_id, data in graph.nodes(data=True)
        if str(data.get("node_type") or "") == "anchor" and data.get("chunk_id") is not None
    }

    author_entity_indices: dict[int, list[int]] = {}
    author_entity_strength: dict[int, float] = {}
    author_entity_match_type: dict[int, str] = {}
    entity_seed_ids: set[str] = set()
    entity_matrix = x_norm[np.asarray(entity_indices, dtype=np.int64)] if entity_indices else np.empty((0, x.shape[1]), dtype=np.float32)
    for aidx, author_key in enumerate(author_keys):
        exact = [
            idx
            for idx in entity_indices
            if author_key_sets[aidx]
            and bool(author_key_sets[aidx] & memory_entity_keys(dict(graph.nodes[node_ids[idx]]), str(node_ids[idx])))
        ]
        strength = 1.0
        match_type = "exact_entity"
        if not exact and len(entity_indices):
            sims = entity_matrix @ author_vecs[aidx]
            exact = [entity_indices[int(np.argmax(sims))]]
            strength = 0.55
            match_type = "embedding_fallback_entity"
        author_entity_indices[aidx] = exact
        author_entity_strength[aidx] = float(strength)
        author_entity_match_type[aidx] = match_type if exact else "missing"
        entity_seed_ids.update(node_ids[idx] for idx in exact)

    chunk_matrix = x_norm[np.asarray(chunk_indices, dtype=np.int64)] if chunk_indices else np.empty((0, x.shape[1]), dtype=np.float32)
    qa_evidence: dict[int, list[tuple[str, str, float, int]]] = {idx: [] for idx in range(len(author_targets))}
    qa_topk = max(1, int(qa_topk))
    if len(chunk_indices):
        qa_chunk_sims = qa_vecs @ chunk_matrix.T
        for aidx in range(len(author_targets)):
            assigned_qas = qa_by_author[aidx] or global_qa_indices
            for qidx in assigned_qas:
                order = np.argsort(-qa_chunk_sims[qidx])[: min(qa_topk, len(chunk_indices))]
                for rank, local_idx in enumerate(order.tolist()):
                    chunk_idx = chunk_indices[int(local_idx)]
                    chunk_id = node_ids[chunk_idx]
                    anchor_id = chunk_to_anchor.get(chunk_id)
                    if anchor_id is None:
                        continue
                    strength = float(max(0.55, 1.0 - 0.2 * rank))
                    qa_evidence[aidx].append((anchor_id, chunk_id, strength, qidx))

    path_graph = nx.Graph()
    path_graph.add_nodes_from(graph.nodes)
    edge_cost = {
        "mentioned_in": 1.0,
        "next_turn": 1.1,
        "relation": 1.6,
    }
    edge_by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for edge_idx, (source, target, key) in enumerate(original_edges["edge_refs"]):
        data = graph.edges[source, target, key]
        edge_type = str(data.get("edge_type") or "")
        cost = float(edge_cost.get(edge_type, 2.0))
        pair_key = tuple(sorted((str(source), str(target))))
        edge_by_pair[pair_key].append(edge_idx)
        if path_graph.has_edge(source, target):
            path_graph[source][target]["weight"] = min(float(path_graph[source][target]["weight"]), cost)
        else:
            path_graph.add_edge(source, target, weight=cost)

    prior = {
        MEMORY_EDGE_TYPES["relation"]: float(relation_prior),
        MEMORY_EDGE_TYPES["mentioned_in"]: float(mentioned_prior),
        MEMORY_EDGE_TYPES["next_turn"]: float(next_turn_prior),
        MEMORY_EDGE_TYPES["no_edge"]: 0.0,
    }
    labels = np.zeros(len(original_edges["edge_refs"]), dtype=np.float32)
    node_relevance = np.zeros(len(node_ids), dtype=np.float32)
    evidence_anchor_ids: set[str] = set()
    evidence_chunk_ids: set[str] = set()
    path_seed_ids: set[str] = set()
    path_count = 0
    strong_path_count = 0
    weak_path_count = 0
    touched_edge_indices: set[int] = set()
    strong_edge_indices: set[int] = set()
    weak_edge_indices: set[int] = set()

    for aidx, entity_idxs in author_entity_indices.items():
        evidence = qa_evidence.get(aidx, [])
        if not entity_idxs or not evidence:
            continue
        entity_strength = float(author_entity_strength.get(aidx, 0.35))
        is_strong_entity = entity_strength >= 0.99
        entity_ids = [node_ids[idx] for idx in entity_idxs]
        for entity_id in entity_ids:
            if entity_id in node_to_idx:
                node_relevance[node_to_idx[entity_id]] = max(float(node_relevance[node_to_idx[entity_id]]), entity_strength)
                path_seed_ids.add(entity_id)
        best_evidence_by_anchor: dict[str, tuple[str, float]] = {}
        for anchor_id, chunk_id, strength, _qidx in evidence:
            current = best_evidence_by_anchor.get(anchor_id)
            if current is None or strength > current[1]:
                best_evidence_by_anchor[anchor_id] = (chunk_id, strength)
        for anchor_id, (chunk_id, strength) in best_evidence_by_anchor.items():
            evidence_anchor_ids.add(anchor_id)
            evidence_chunk_ids.add(chunk_id)
            evidence_strength = float(strength) * entity_strength
            if anchor_id in node_to_idx:
                node_relevance[node_to_idx[anchor_id]] = max(float(node_relevance[node_to_idx[anchor_id]]), evidence_strength)
            if chunk_id in node_to_idx:
                node_relevance[node_to_idx[chunk_id]] = max(float(node_relevance[node_to_idx[chunk_id]]), evidence_strength)
            if anchor_id not in path_graph:
                continue
            for entity_id in entity_ids:
                if entity_id not in path_graph:
                    continue
                try:
                    path = nx.shortest_path(path_graph, entity_id, anchor_id, weight="weight")
                    distance = nx.path_weight(path_graph, path, weight="weight")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if float(distance) > float(path_cutoff):
                    continue
                path_count += 1
                if is_strong_entity:
                    strong_path_count += 1
                else:
                    weak_path_count += 1
                path_weight = float(evidence_strength * math.exp(-float(distance) / max(1e-6, float(path_decay))))
                for node_id in path:
                    if node_id in node_to_idx:
                        node_relevance[node_to_idx[node_id]] = max(float(node_relevance[node_to_idx[node_id]]), path_weight)
                for source, target in zip(path[:-1], path[1:]):
                    pair_key = tuple(sorted((str(source), str(target))))
                    for edge_idx in edge_by_pair.get(pair_key, []):
                        et = int(original_edges["edge_type"][edge_idx])
                        score_factor = float(original_edges["edge_score"][edge_idx])
                        value = min(1.0, path_weight * prior.get(et, 0.2) * max(0.3, score_factor))
                        labels[edge_idx] = max(float(labels[edge_idx]), value)
                        touched_edge_indices.add(edge_idx)
                        if is_strong_entity:
                            strong_edge_indices.add(edge_idx)
                        else:
                            weak_edge_indices.add(edge_idx)

    assignment = build_memory_node_query_assignment(batch, combined_query_vecs)
    seed_node_ids = sorted(entity_seed_ids | evidence_anchor_ids | evidence_chunk_ids | path_seed_ids)
    return {
        "query_vecs": combined_query_vecs.astype(np.float32),
        "node_query_vecs": assignment["node_query_vecs"],
        "best_query_idx": assignment["best_query_idx"],
        "similarities": assignment["similarities"],
        "node_relevance": np.clip(node_relevance, 0.0, 1.0),
        "edge_labels": labels,
        "seed_node_ids": seed_node_ids,
        "seed_communities": sorted(
            {
                int(graph.nodes[node_id].get("community_id", -1))
                for node_id in seed_node_ids
                if node_id in graph.nodes and int(graph.nodes[node_id].get("community_id", -1)) >= 0
            }
        ),
        "per_query": [],
        "per_query_seed_counts": [],
        "per_query_edge_high_counts": [],
        "entity_seed_count": int(sum(len(items) for items in author_entity_indices.values())),
        "target_entity_seed_count": int(sum(len(items) for items in author_entity_indices.values())),
        "target_chunk_seed_count": len(evidence_chunk_ids),
        "target_anchor_seed_count": len(evidence_anchor_ids),
        "path_seed_count": len(path_seed_ids),
        "anchor_seed_count": len(evidence_anchor_ids),
        "author_query_count": len(author_targets),
        "qa_query_count": len(qa_targets),
        "qa_topk": int(qa_topk),
        "author_qa_matched_count": int(sum(len(items) for items in qa_by_author.values())),
        "author_qa_unmatched_count": int(len(global_qa_indices)),
        "path_count": int(path_count),
        "strong_entity_seed_count": int(sum(1 for v in author_entity_strength.values() if v >= 0.99)),
        "weak_entity_seed_count": int(sum(1 for v in author_entity_strength.values() if 0.0 < v < 0.99)),
        "entity_match_type_counts": dict(Counter(author_entity_match_type.values())),
        "strong_path_count": int(strong_path_count),
        "weak_path_count": int(weak_path_count),
        "strong_edge_label_count": int(len(strong_edge_indices)),
        "weak_edge_label_count": int(len(weak_edge_indices - strong_edge_indices)),
        "touched_edge_count": int(len(touched_edge_indices)),
        "strong_label_source": "exact_author_or_book_entity_to_qa_anchor_path",
        "weak_label_source": "embedding_fallback_entity_and_qa_chunk_path",
        "label_source": "strong_exact_entity_plus_weak_embedding_qa_path_supervision",
    }


def build_memory_ranking_pairs(
    labels: np.ndarray,
    edge_types: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pos_parts: list[np.ndarray] = []
    safe_parts: list[np.ndarray] = []
    type_ids = sorted(int(type_id) for type_id in np.unique(edge_types).tolist())
    each = max(1, int(count) // max(1, len(type_ids)))
    for type_id in type_ids:
        pos_pool = np.where((edge_types == type_id) & (labels >= 0.65))[0].astype(np.int64)
        safe_pool = np.where((edge_types == type_id) & (labels < 0.15))[0].astype(np.int64)
        if len(pos_pool) == 0 or len(safe_pool) == 0:
            continue
        take = min(each, len(pos_pool), len(safe_pool))
        pos_parts.append(rng.choice(pos_pool, size=take, replace=False))
        safe_parts.append(rng.choice(safe_pool, size=take, replace=False))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(safe_parts)


def build_memory_node_query_assignment(batch, query_vecs: np.ndarray) -> dict[str, np.ndarray]:
    query_vecs = l2_normalize(np.asarray(query_vecs, dtype=np.float32))
    x = batch.x.detach().cpu().numpy().astype(np.float32)
    similarities = np.zeros((query_vecs.shape[0], x.shape[0]), dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    valid = norms > 1e-8
    if valid.any():
        similarities[:, valid] = query_vecs @ l2_normalize(x[valid]).T
    best_query_idx = np.argmax(similarities, axis=0).astype(np.int64)
    merged_similarities = similarities[best_query_idx, np.arange(x.shape[0])]
    return {
        "node_query_vecs": query_vecs[best_query_idx].astype(np.float32),
        "best_query_idx": best_query_idx,
        "similarities": merged_similarities.astype(np.float32),
    }


def build_memory_path_supervision_labels(
    graph: nx.MultiDiGraph,
    batch,
    original_edges: dict[str, Any],
    *,
    query_vecs: np.ndarray,
    forget_questions: list[str],
    forget_vectors: np.ndarray,
    retain_questions: list[str],
    retain_vectors: np.ndarray,
    max_depth: int,
    max_seeds: int,
    max_hits: int,
    entity_sim_floor: float,
    retain_weight: float,
) -> dict[str, Any]:
    from phase1.dual_graph_unlearning_gat.eval_unlearning_retrieval import MemoryPathEvaluator

    node_to_idx = {node_id: idx for idx, node_id in enumerate(batch.node_ids)}
    edge_to_idx: dict[tuple[str, str, str], int] = {}
    for idx, (source, target, key) in enumerate(original_edges["edge_refs"]):
        edge_to_idx[(str(source), str(target), str(key))] = idx
        edge_to_idx[(str(target), str(source), str(key))] = idx

    evaluator = MemoryPathEvaluator(
        graph,
        graph,
        max_depth=max_depth,
        max_seeds=max_seeds,
        max_hits=max_hits,
        entity_sim_floor=entity_sim_floor,
    )
    forget_counts = np.zeros(len(original_edges["edge_refs"]), dtype=np.float32)
    retain_counts = np.zeros(len(original_edges["edge_refs"]), dtype=np.float32)
    node_relevance = np.zeros(len(batch.node_ids), dtype=np.float32)
    forget_path_count = 0
    retain_path_count = 0

    def accumulate(question: str, query_vec: np.ndarray, *, split: str) -> None:
        nonlocal forget_path_count, retain_path_count
        result = evaluator.evaluate(question, query_vec)
        hits = result.get("memory_hits", [])
        if split == "forget":
            forget_path_count += len(hits)
        else:
            retain_path_count += len(hits)
        for rank, hit in enumerate(hits):
            path_weight = float(1.0 / (1.0 + 0.15 * rank))
            if split == "forget":
                for node_id in hit.get("path_nodes", []):
                    idx = node_to_idx.get(str(node_id))
                    if idx is not None:
                        node_relevance[idx] = max(float(node_relevance[idx]), path_weight)
            for edge in hit.get("path_edges", []):
                ref = (str(edge.get("source")), str(edge.get("target")), str(edge.get("key")))
                edge_idx = edge_to_idx.get(ref)
                if edge_idx is None:
                    continue
                if split == "forget":
                    forget_counts[edge_idx] += path_weight
                else:
                    retain_counts[edge_idx] += path_weight

    for question, query_vec in zip(forget_questions, np.asarray(forget_vectors, dtype=np.float32)):
        accumulate(question, query_vec, split="forget")
    for question, query_vec in zip(retain_questions, np.asarray(retain_vectors, dtype=np.float32)):
        accumulate(question, query_vec, split="retain")

    weighted_retain = float(retain_weight) * retain_counts
    labels = np.divide(
        forget_counts,
        forget_counts + weighted_retain + 1e-6,
        out=np.zeros_like(forget_counts),
        where=forget_counts > 0,
    ).astype(np.float32)
    assignment = build_memory_node_query_assignment(batch, query_vecs)
    seed_node_ids = sorted(
        {
            node_id
            for node_id, relevance in zip(batch.node_ids, node_relevance.tolist())
            if float(relevance) > 0.0
        }
    )
    return {
        "query_vecs": l2_normalize(np.asarray(query_vecs, dtype=np.float32)),
        "node_query_vecs": assignment["node_query_vecs"],
        "best_query_idx": assignment["best_query_idx"],
        "similarities": assignment["similarities"],
        "node_relevance": node_relevance,
        "edge_labels": labels,
        "seed_node_ids": seed_node_ids,
        "seed_communities": sorted(
            {
                int(graph.nodes[node_id].get("community_id", -1))
                for node_id in seed_node_ids
                if node_id in graph.nodes and int(graph.nodes[node_id].get("community_id", -1)) >= 0
            }
        ),
        "per_query": [],
        "per_query_seed_counts": [],
        "per_query_edge_high_counts": [],
        "entity_seed_count": 0,
        "target_entity_seed_count": 0,
        "target_chunk_seed_count": 0,
        "target_anchor_seed_count": int(sum(1 for node_id in seed_node_ids if str(graph.nodes[node_id].get("node_type") or "") == "anchor")),
        "path_seed_count": len(seed_node_ids),
        "anchor_seed_count": int(sum(1 for node_id in seed_node_ids if str(graph.nodes[node_id].get("node_type") or "") == "anchor")),
        "forget_path_count": int(forget_path_count),
        "retain_path_count": int(retain_path_count),
        "forget_edge_touch_count": int((forget_counts > 0).sum()),
        "retain_edge_touch_count": int((retain_counts > 0).sum()),
        "path_supervision_retain_weight": float(retain_weight),
        "label_source": "base_memory_retrieval_topk_path_forget_retain_ratio",
    }


def build_memory_path_supervision_from_eval_rows(
    graph: nx.MultiDiGraph,
    batch,
    original_edges: dict[str, Any],
    *,
    query_vecs: np.ndarray,
    eval_rows: list[dict[str, Any]],
    retain_weight: float,
) -> dict[str, Any]:
    node_to_idx = {node_id: idx for idx, node_id in enumerate(batch.node_ids)}
    edge_to_idx: dict[tuple[str, str, str], int] = {}
    for idx, (source, target, key) in enumerate(original_edges["edge_refs"]):
        edge_to_idx[(str(source), str(target), str(key))] = idx
        edge_to_idx[(str(target), str(source), str(key))] = idx

    forget_counts = np.zeros(len(original_edges["edge_refs"]), dtype=np.float32)
    retain_counts = np.zeros(len(original_edges["edge_refs"]), dtype=np.float32)
    node_relevance = np.zeros(len(batch.node_ids), dtype=np.float32)
    forget_path_count = 0
    retain_path_count = 0

    for row in eval_rows:
        split = str(row.get("split") or "")
        hits = row.get("memory_hits") or []
        if isinstance(hits, str):
            hits = json.loads(hits)
        if split == "forget":
            forget_path_count += len(hits)
        elif split == "retain":
            retain_path_count += len(hits)
        else:
            continue
        for rank, hit in enumerate(hits):
            path_weight = float(1.0 / (1.0 + 0.15 * rank))
            if split == "forget":
                for node_id in hit.get("path_nodes", []):
                    idx = node_to_idx.get(str(node_id))
                    if idx is not None:
                        node_relevance[idx] = max(float(node_relevance[idx]), path_weight)
            for edge in hit.get("path_edges", []):
                ref = (str(edge.get("source")), str(edge.get("target")), str(edge.get("key")))
                edge_idx = edge_to_idx.get(ref)
                if edge_idx is None:
                    continue
                if split == "forget":
                    forget_counts[edge_idx] += path_weight
                else:
                    retain_counts[edge_idx] += path_weight

    weighted_retain = float(retain_weight) * retain_counts
    labels = np.divide(
        forget_counts,
        forget_counts + weighted_retain + 1e-6,
        out=np.zeros_like(forget_counts),
        where=forget_counts > 0,
    ).astype(np.float32)
    assignment = build_memory_node_query_assignment(batch, query_vecs)
    seed_node_ids = sorted(
        {
            node_id
            for node_id, relevance in zip(batch.node_ids, node_relevance.tolist())
            if float(relevance) > 0.0
        }
    )
    return {
        "query_vecs": l2_normalize(np.asarray(query_vecs, dtype=np.float32)),
        "node_query_vecs": assignment["node_query_vecs"],
        "best_query_idx": assignment["best_query_idx"],
        "similarities": assignment["similarities"],
        "node_relevance": node_relevance,
        "edge_labels": labels,
        "seed_node_ids": seed_node_ids,
        "seed_communities": sorted(
            {
                int(graph.nodes[node_id].get("community_id", -1))
                for node_id in seed_node_ids
                if node_id in graph.nodes and int(graph.nodes[node_id].get("community_id", -1)) >= 0
            }
        ),
        "per_query": [],
        "per_query_seed_counts": [],
        "per_query_edge_high_counts": [],
        "entity_seed_count": 0,
        "target_entity_seed_count": 0,
        "target_chunk_seed_count": 0,
        "target_anchor_seed_count": int(sum(1 for node_id in seed_node_ids if str(graph.nodes[node_id].get("node_type") or "") == "anchor")),
        "path_seed_count": len(seed_node_ids),
        "anchor_seed_count": int(sum(1 for node_id in seed_node_ids if str(graph.nodes[node_id].get("node_type") or "") == "anchor")),
        "forget_path_count": int(forget_path_count),
        "retain_path_count": int(retain_path_count),
        "forget_edge_touch_count": int((forget_counts > 0).sum()),
        "retain_edge_touch_count": int((retain_counts > 0).sum()),
        "path_supervision_retain_weight": float(retain_weight),
        "label_source": "cached_base_memory_retrieval_path_forget_retain_ratio",
    }


def train_memory(
    memory_graph: nx.MultiDiGraph,
    query_vecs: np.ndarray,
    query_targets: list[str],
    *,
    memory_author_query_vecs: np.ndarray | None = None,
    memory_author_query_targets: list[str] | None = None,
    memory_qa_query_vecs: np.ndarray | None = None,
    memory_qa_query_targets: list[str] | None = None,
    memory_qa_topk: int = 3,
    embedding_dim: int,
    output_dir: Path,
    device: torch.device,
    hidden_dim: int,
    heads: int,
    dropout: float,
    pretrain_epochs: int,
    retrain_epochs: int,
    lr: float,
    weight_decay: float,
    edge_sample_size: int,
    memory_path_cutoff: int,
    memory_hard_threshold: float,
    memory_relation_prior: float,
    memory_mentioned_prior: float,
    memory_next_turn_prior: float,
    memory_background_relation_scale: float,
    memory_community_floor: float,
    memory_path_decay: float,
    memory_path_loss_weight: float,
    memory_safe_loss_weight: float,
    memory_toxic_loss_weight: float,
    memory_sparse_loss_weight: float,
    memory_consistency_loss_weight: float,
    memory_next_turn_loss_weight: float,
    memory_relation_loss_weight: float,
    memory_next_turn_safe_loss_weight: float,
    memory_relation_safe_loss_weight: float,
    memory_ranking_loss_weight: float,
    memory_ranking_margin: float,
    memory_qa_hard_fraction: float,
    memory_path_forget_questions: list[str] | None = None,
    memory_path_forget_vecs: np.ndarray | None = None,
    memory_path_retain_questions: list[str] | None = None,
    memory_path_retain_vecs: np.ndarray | None = None,
    memory_path_eval_rows: list[dict[str, Any]] | None = None,
    memory_retrieval_max_depth: int = 5,
    memory_retrieval_max_seeds: int = 3,
    memory_retrieval_max_hits: int = 3,
    memory_retrieval_entity_sim_floor: float = 0.35,
    memory_path_retain_weight: float = 2.0,
    skip_pretrain: bool = False,
    pretrain_model_path: Path | None = None,
    pretrain_only: bool = False,
    seed: int = 20260608,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    rng = np.random.default_rng(seed + 97)
    batch = build_memory_batch(memory_graph, embedding_dim=embedding_dim)
    batch.community, community_mapping = remap_communities(batch.community)
    community_count = int(batch.community.max().item()) + 1
    node_to_idx = {node_id: idx for idx, node_id in enumerate(batch.node_ids)}
    original_edges = build_original_memory_edges(memory_graph, node_to_idx)
    batch = batch.to(device)

    model = MemoryGAT(
        input_dim=batch.x.size(1),
        hidden_dim=hidden_dim,
        community_count=community_count,
        node_type_count=len(MEMORY_NODE_TYPES),
        edge_type_count=len(MEMORY_EDGE_TYPES),
        heads=heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_left_all = original_edges["left"]
    pos_right_all = original_edges["right"]
    pos_type_all = original_edges["edge_type"]
    pos_score_all = original_edges["edge_score"]
    next_turn_type = MEMORY_EDGE_TYPES["next_turn"]
    comm_np = batch.community.detach().cpu().numpy()

    pre_model_path = pretrain_model_path or (output_dir / "memory_pre_gat.pt")
    pretrain_log: list[dict[str, float]] = []
    if skip_pretrain:
        if not pre_model_path.exists():
            raise FileNotFoundError(f"Memory pretrain checkpoint not found: {pre_model_path}")
        model.load_state_dict(torch.load(pre_model_path, map_location=device))
        print(f"[Memory:pretrain] skipped; loaded {pre_model_path}")
    else:
        for epoch in range(1, pretrain_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(batch)
            loss_node = F.cross_entropy(model.node_type_logits(z), batch.node_type)
            loss_comm = F.cross_entropy(model.community_logits(z), batch.community)

            take = min(int(edge_sample_size), len(pos_left_all))
            pos_idx = rng.choice(len(pos_left_all), size=take, replace=False)
            neg_left, neg_right = sample_memory_negatives(batch, original_edges["existing"], count=take, rng=rng)
            e_left = np.concatenate([pos_left_all[pos_idx], neg_left])
            e_right = np.concatenate([pos_right_all[pos_idx], neg_right])
            e_type = np.concatenate([pos_type_all[pos_idx], np.zeros(len(neg_left), dtype=np.int64)])
            e_score = np.concatenate([pos_score_all[pos_idx], np.zeros(len(neg_left), dtype=np.float32)])
            e_left_t = torch.tensor(e_left, dtype=torch.long, device=device)
            e_right_t = torch.tensor(e_right, dtype=torch.long, device=device)
            e_type_t = torch.tensor(e_type, dtype=torch.long, device=device)
            e_score_t = torch.tensor(e_score, dtype=torch.float32, device=device)
            loss_edge = F.cross_entropy(model.edge_class_logits(z, e_left_t, e_right_t), e_type_t)
            loss_score = F.mse_loss(torch.sigmoid(model.edge_score_logits(z, e_left_t, e_right_t)), e_score_t)

            next_indices = np.where(pos_type_all == next_turn_type)[0]
            if len(next_indices):
                take_next = min(int(edge_sample_size) // 2, len(next_indices))
                pidx = rng.choice(next_indices, size=take_next, replace=False)
                nleft, nright = sample_memory_negatives(batch, original_edges["existing"], count=take_next, rng=rng)
                t_left = np.concatenate([pos_left_all[pidx], nleft])
                t_right = np.concatenate([pos_right_all[pidx], nright])
                t_labels = np.concatenate([np.ones(len(pidx), dtype=np.float32), np.zeros(len(nleft), dtype=np.float32)])
                loss_temp = F.binary_cross_entropy_with_logits(
                    model.temporal_logits(
                        z,
                        torch.tensor(t_left, dtype=torch.long, device=device),
                        torch.tensor(t_right, dtype=torch.long, device=device),
                    ),
                    torch.tensor(t_labels, dtype=torch.float32, device=device),
                )
            else:
                loss_temp = z.sum() * 0.0

            con_count = min(int(edge_sample_size) // 2, max(512, len(batch.node_ids) // 4))
            p_left, p_right = sample_pairs_by_community(comm_np, count=con_count, positive=True, rng=rng)
            n_left, n_right = sample_pairs_by_community(comm_np, count=con_count, positive=False, rng=rng)
            c_left = np.concatenate([p_left, n_left])
            c_right = np.concatenate([p_right, n_right])
            c_labels = np.concatenate([np.ones(len(p_left), dtype=np.float32), np.zeros(len(n_left), dtype=np.float32)])
            if len(c_left):
                loss_con = F.binary_cross_entropy_with_logits(
                    model.contrast_logits(
                        z,
                        torch.tensor(c_left, dtype=torch.long, device=device),
                        torch.tensor(c_right, dtype=torch.long, device=device),
                    ),
                    torch.tensor(c_labels, dtype=torch.float32, device=device),
                )
            else:
                loss_con = z.sum() * 0.0

            loss = 0.1 * loss_node + 0.7 * loss_edge + 0.35 * loss_temp + 0.5 * loss_comm + 0.35 * loss_score + 0.25 * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pretrain_log.append({
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "node_type": float(loss_node.detach().cpu()),
                "edge_type": float(loss_edge.detach().cpu()),
                "temporal": float(loss_temp.detach().cpu()),
                "community": float(loss_comm.detach().cpu()),
                "edge_score": float(loss_score.detach().cpu()),
                "contrast": float(loss_con.detach().cpu()),
            })
            if epoch == 1 or epoch == pretrain_epochs or epoch % max(1, pretrain_epochs // 5) == 0:
                print(f"[Memory:pretrain] epoch={epoch}/{pretrain_epochs} loss={pretrain_log[-1]['loss']:.4f}")
        torch.save(model.state_dict(), pre_model_path)

    if pretrain_only:
        report = {
            "pre_model_path": str(pre_model_path),
            "model_path": str(pre_model_path),
            "skip_pretrain": bool(skip_pretrain),
            "pretrain_only": True,
            "node_count": len(batch.node_ids),
            "edge_count_original": int(len(original_edges["edge_refs"])),
            "edge_type_counts": original_edges["by_type"],
            "community_count": community_count,
            "community_mapping_size": len(community_mapping),
            "pretrain_loss": "0.1*node_type_ce + 0.7*edge_type_ce + 0.35*temporal_bce + 0.5*community_ce + 0.35*edge_score_mse + 0.25*community_contrast_bce",
            "pretrain_log": pretrain_log,
        }
        return memory_graph.copy(), report

    use_path_supervision = (
        memory_path_eval_rows is not None
        or (
            memory_path_forget_questions is not None
            and memory_path_forget_vecs is not None
            and memory_path_retain_questions is not None
            and memory_path_retain_vecs is not None
        )
    )
    if use_path_supervision:
        if memory_path_eval_rows is not None:
            print("[Memory:labels] building path supervision labels from cached eval rows")
            query_data = build_memory_path_supervision_from_eval_rows(
                memory_graph,
                batch,
                original_edges,
                query_vecs=query_vecs,
                eval_rows=memory_path_eval_rows,
                retain_weight=memory_path_retain_weight,
            )
        else:
            print("[Memory:labels] building base-retrieval path supervision labels")
            query_data = build_memory_path_supervision_labels(
                memory_graph,
                batch,
                original_edges,
                query_vecs=query_vecs,
                forget_questions=memory_path_forget_questions or [],
                forget_vectors=np.asarray(memory_path_forget_vecs, dtype=np.float32),
                retain_questions=memory_path_retain_questions or [],
                retain_vectors=np.asarray(memory_path_retain_vecs, dtype=np.float32),
                max_depth=memory_retrieval_max_depth,
                max_seeds=memory_retrieval_max_seeds,
                max_hits=memory_retrieval_max_hits,
                entity_sim_floor=memory_retrieval_entity_sim_floor,
                retain_weight=memory_path_retain_weight,
            )
    else:
        has_author_qa = (
            memory_author_query_vecs is not None
            and memory_author_query_targets is not None
            and memory_qa_query_vecs is not None
            and memory_qa_query_targets is not None
            and len(memory_author_query_targets) > 0
            and len(memory_qa_query_targets) > 0
        )
        if has_author_qa:
            query_data = build_memory_author_qa_path_labels(
                memory_graph,
                batch,
                np.asarray(memory_author_query_vecs, dtype=np.float32),
                memory_author_query_targets,
                np.asarray(memory_qa_query_vecs, dtype=np.float32),
                memory_qa_query_targets,
                original_edges,
                qa_topk=memory_qa_topk,
                path_cutoff=memory_path_cutoff,
                relation_prior=memory_relation_prior,
                mentioned_prior=memory_mentioned_prior,
                next_turn_prior=memory_next_turn_prior,
                path_decay=memory_path_decay,
            )
        else:
            query_data = build_memory_multi_query_labels(
                memory_graph,
                batch,
                query_vecs,
                query_targets,
                original_edges,
                path_cutoff=memory_path_cutoff,
                relation_prior=memory_relation_prior,
                mentioned_prior=memory_mentioned_prior,
                next_turn_prior=memory_next_turn_prior,
                background_relation_scale=memory_background_relation_scale,
                community_floor=memory_community_floor,
                path_decay=memory_path_decay,
            )
    node_query_t = torch.tensor(query_data["node_query_vecs"], dtype=torch.float32, device=device)
    labels_np = query_data["edge_labels"]
    labels_t = torch.tensor(labels_np, dtype=torch.float32, device=device)
    label_source = str(query_data.get("label_source", ""))
    label_positive_cutoff = (
        0.1
        if label_source
        in {
            "author_entity_to_qa_matched_anchor_path_supervision",
            "strong_exact_entity_plus_weak_embedding_qa_path_supervision",
        }
        else 0.5
    )
    positive_count = float((labels_np >= label_positive_cutoff).sum())
    negative_count = float((labels_np < label_positive_cutoff).sum())
    if positive_count > 0.0 and negative_count > 0.0:
        pos_weight_cap = 80.0 if label_positive_cutoff < 0.5 else 30.0
        pos_weight_value = min(float(pos_weight_cap), negative_count / max(1.0, positive_count))
        bce_pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    else:
        pos_weight_value = 1.0
        bce_pos_weight = None
    node_relevance_t = torch.tensor(query_data["node_relevance"], dtype=torch.float32, device=device)
    left_t = torch.tensor(original_edges["left"], dtype=torch.long, device=device)
    right_t = torch.tensor(original_edges["right"], dtype=torch.long, device=device)
    et_t = torch.tensor(original_edges["edge_type"], dtype=torch.long, device=device)
    next_turn_type_id = MEMORY_EDGE_TYPES["next_turn"]
    relation_type_id = MEMORY_EDGE_TYPES["relation"]
    rank_pos_np, rank_safe_np = build_memory_ranking_pairs(
        labels_np,
        original_edges["edge_type"],
        count=int(edge_sample_size),
        rng=rng,
    )
    rank_pos_t = torch.tensor(rank_pos_np, dtype=torch.long, device=device)
    rank_safe_t = torch.tensor(rank_safe_np, dtype=torch.long, device=device)

    retrain_log: list[dict[str, float]] = []
    for epoch in range(1, retrain_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model.encode(batch, node_query_t)
        logits = model.punish_logits(z, left_t, right_t, et_t)
        probs_t = torch.sigmoid(logits)
        loss_bce = F.binary_cross_entropy_with_logits(logits, labels_t, pos_weight=bce_pos_weight)
        safe = labels_t < 0.15
        toxic = labels_t > 0.65
        loss_safe = probs_t[safe].mean() if safe.any() else probs_t.mean() * 0.0
        loss_toxic = (1.0 - probs_t[toxic]).mean() if toxic.any() else probs_t.mean() * 0.0
        loss_sparse = probs_t.mean()
        endpoint_relevance = torch.maximum(node_relevance_t[left_t], node_relevance_t[right_t])
        loss_path = ((1.0 - probs_t) * labels_t * endpoint_relevance).mean()
        loss_consistency = F.mse_loss(probs_t, torch.clamp(endpoint_relevance, 0.0, 1.0))
        next_mask = (et_t == next_turn_type_id) & (labels_t >= 0.5)
        relation_mask = (et_t == relation_type_id) & (labels_t >= 0.5)
        next_safe_mask = (et_t == next_turn_type_id) & (labels_t < 0.15)
        relation_safe_mask = (et_t == relation_type_id) & (labels_t < 0.15)
        loss_next_turn = (
            F.binary_cross_entropy_with_logits(logits[next_mask], labels_t[next_mask])
            if next_mask.any()
            else probs_t.mean() * 0.0
        )
        loss_relation = (
            F.binary_cross_entropy_with_logits(logits[relation_mask], labels_t[relation_mask])
            if relation_mask.any()
            else probs_t.mean() * 0.0
        )
        loss_next_turn_safe = probs_t[next_safe_mask].mean() if next_safe_mask.any() else probs_t.mean() * 0.0
        loss_relation_safe = probs_t[relation_safe_mask].mean() if relation_safe_mask.any() else probs_t.mean() * 0.0
        loss_ranking = (
            F.relu(float(memory_ranking_margin) - (logits[rank_pos_t] - logits[rank_safe_t])).mean()
            if rank_pos_t.numel() > 0
            else probs_t.mean() * 0.0
        )
        loss = (
            loss_bce
            + float(memory_path_loss_weight) * loss_path
            + float(memory_safe_loss_weight) * loss_safe
            + float(memory_toxic_loss_weight) * loss_toxic
            + float(memory_sparse_loss_weight) * loss_sparse
            + float(memory_consistency_loss_weight) * loss_consistency
            + float(memory_next_turn_loss_weight) * loss_next_turn
            + float(memory_relation_loss_weight) * loss_relation
            + float(memory_next_turn_safe_loss_weight) * loss_next_turn_safe
            + float(memory_relation_safe_loss_weight) * loss_relation_safe
            + float(memory_ranking_loss_weight) * loss_ranking
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        retrain_log.append({
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "bce": float(loss_bce.detach().cpu()),
            "path": float(loss_path.detach().cpu()),
            "safe": float(loss_safe.detach().cpu()),
            "toxic": float(loss_toxic.detach().cpu()),
            "sparse": float(loss_sparse.detach().cpu()),
            "consistency": float(loss_consistency.detach().cpu()),
            "next_turn": float(loss_next_turn.detach().cpu()),
            "relation": float(loss_relation.detach().cpu()),
            "next_turn_safe": float(loss_next_turn_safe.detach().cpu()),
            "relation_safe": float(loss_relation_safe.detach().cpu()),
            "ranking": float(loss_ranking.detach().cpu()),
        })
        if epoch == 1 or epoch == retrain_epochs or epoch % max(1, retrain_epochs // 5) == 0:
            print(f"[Memory:retrain] epoch={epoch}/{retrain_epochs} loss={retrain_log[-1]['loss']:.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(batch, node_query_t)
        probs = torch.sigmoid(model.punish_logits(z, left_t, right_t, et_t)).detach().cpu().numpy()

    if float(memory_hard_threshold) < 0:
        threshold, threshold_report = choose_binary_threshold(
            probs,
            labels_np,
            positive_cutoff=label_positive_cutoff,
            fallback=0.5,
        )
    else:
        threshold = float(memory_hard_threshold)
        threshold_report = {"mode": "manual", "threshold": threshold}
    out_graph = memory_graph.copy()
    hard = probs >= threshold
    for idx, (source, target, key) in enumerate(original_edges["edge_refs"]):
        edge_data = out_graph.edges[source, target, key]
        edge_data["punish"] = float(probs[idx])
        edge_data["keep_prob"] = float(1.0 - probs[idx])
        edge_data["unlearned"] = int(hard[idx])
        edge_data["weak_label"] = float(labels_np[idx])

    node_punish = {node_id: 0.0 for node_id in batch.node_ids}
    for idx, (source, target, _key) in enumerate(original_edges["edge_refs"]):
        node_punish[source] = max(node_punish[source], float(probs[idx]))
        node_punish[target] = max(node_punish[target], float(probs[idx]))
    for node_id, data in out_graph.nodes(data=True):
        if str(data.get("node_type") or "") != "anchor":
            continue
        chunk_id = str(data.get("chunk_id") or "")
        if chunk_id in node_punish:
            node_punish[chunk_id] = max(float(node_punish[chunk_id]), float(node_punish.get(str(node_id), 0.0)))
    for idx, node_id in enumerate(batch.node_ids):
        out_graph.nodes[node_id]["punish"] = float(node_punish[node_id])
        out_graph.nodes[node_id]["query_relevance"] = float(query_data["node_relevance"][idx])
        out_graph.nodes[node_id]["best_query_index"] = int(query_data["best_query_idx"][idx])
        out_graph.nodes[node_id]["unlearned"] = int(node_punish[node_id] >= threshold)
    memory_qa_hard_report: dict[str, Any] = {"enabled": False}
    if float(memory_qa_hard_fraction) > 0.0 and memory_qa_query_targets:
        matched_chunks_by_row = match_memory_qa_targets_to_chunks(out_graph, memory_qa_query_targets)
        row_scores: list[tuple[float, int, list[str]]] = []
        for row_idx, chunk_ids in enumerate(matched_chunks_by_row):
            score = max((float(out_graph.nodes[cid].get("punish", 0.0)) for cid in chunk_ids if cid in out_graph), default=-1.0)
            row_scores.append((score, row_idx, [cid for cid in chunk_ids if cid in out_graph]))
        matched_rows = [(score, idx, chunks) for score, idx, chunks in row_scores if chunks]
        take_rows = min(
            len(matched_rows),
            max(1, int(math.ceil(float(memory_qa_hard_fraction) * float(len(matched_rows))))),
        )
        selected_rows = sorted(matched_rows, key=lambda item: item[0], reverse=True)[:take_rows]
        selected_chunk_ids: set[str] = set()
        selected_anchor_ids: set[str] = set()
        for _score, _idx, chunks in selected_rows:
            for chunk_id in chunks:
                selected_chunk_ids.add(chunk_id)
                suffix = str(chunk_id).split(":", 1)[1] if ":" in str(chunk_id) else ""
                anchor_id = f"anchor:{suffix}" if suffix else ""
                if anchor_id in out_graph:
                    selected_anchor_ids.add(anchor_id)
        for node_id in selected_chunk_ids | selected_anchor_ids:
            out_graph.nodes[node_id]["unlearned"] = 1
        memory_qa_hard_report = {
            "enabled": True,
            "fraction": float(memory_qa_hard_fraction),
            "qa_row_count": int(len(memory_qa_query_targets or [])),
            "matched_rows": int(len(matched_rows)),
            "selected_rows": int(len(selected_rows)),
            "selected_chunk_count": int(len(selected_chunk_ids)),
            "selected_anchor_count": int(len(selected_anchor_ids)),
            "selection_rule": "top_fraction_by_gat_node_punish_over_matched_memory_qa_chunks",
        }

    retrain_model_path = output_dir / "memory_retrain_gat.pt"
    torch.save(model.state_dict(), retrain_model_path)
    report = {
        "pre_model_path": str(pre_model_path),
        "retrain_model_path": str(retrain_model_path),
        "model_path": str(retrain_model_path),
        "skip_pretrain": bool(skip_pretrain),
        "node_count": len(batch.node_ids),
        "edge_count_original": int(len(original_edges["edge_refs"])),
        "edge_type_counts": original_edges["by_type"],
        "community_count": community_count,
        "community_mapping_size": len(community_mapping),
        "query_label_source": query_data.get(
            "label_source",
            "multi_query_author_entity_grounding_plus_infomap_community_path_edge_type_max_merge",
        ),
        "query_count": int(np.asarray(query_vecs).shape[0]),
        "author_query_count": int(query_data.get("author_query_count", 0)),
        "qa_query_count": int(query_data.get("qa_query_count", 0)),
        "memory_qa_topk": int(query_data.get("qa_topk", memory_qa_topk)),
        "author_qa_matched_count": int(query_data.get("author_qa_matched_count", 0)),
        "author_qa_unmatched_count": int(query_data.get("author_qa_unmatched_count", 0)),
        "author_to_qa_path_count": int(query_data.get("path_count", 0)),
        "strong_entity_seed_count": int(query_data.get("strong_entity_seed_count", 0)),
        "weak_entity_seed_count": int(query_data.get("weak_entity_seed_count", 0)),
        "entity_match_type_counts": query_data.get("entity_match_type_counts", {}),
        "strong_path_count": int(query_data.get("strong_path_count", 0)),
        "weak_path_count": int(query_data.get("weak_path_count", 0)),
        "strong_edge_label_count": int(query_data.get("strong_edge_label_count", 0)),
        "weak_edge_label_count": int(query_data.get("weak_edge_label_count", 0)),
        "strong_label_source": query_data.get("strong_label_source", ""),
        "weak_label_source": query_data.get("weak_label_source", ""),
        "author_to_qa_touched_edge_count": int(query_data.get("touched_edge_count", 0)),
        "seed_node_count": len(query_data["seed_node_ids"]),
        "entity_seed_count": query_data["entity_seed_count"],
        "target_entity_seed_count": query_data["target_entity_seed_count"],
        "target_chunk_seed_count": query_data["target_chunk_seed_count"],
        "target_anchor_seed_count": query_data["target_anchor_seed_count"],
        "path_seed_count": query_data["path_seed_count"],
        "anchor_seed_count": query_data["anchor_seed_count"],
        "seed_communities": query_data["seed_communities"],
        "per_query_seed_counts": query_data["per_query_seed_counts"],
        "per_query_edge_high_counts": query_data["per_query_edge_high_counts"],
        "query_pooling": "per_node_argmax_cosine_assignment_and_label_max_merge",
        "punish_threshold": float(threshold),
        "threshold_report": threshold_report,
        "memory_qa_hard_selection": memory_qa_hard_report,
        "edge_label_mean": float(labels_np.mean()) if len(labels_np) else 0.0,
        "edge_label_high_count": int((labels_np >= 0.5).sum()),
        "edge_label_positive_cutoff": float(label_positive_cutoff),
        "edge_label_positive_count": int((labels_np >= label_positive_cutoff).sum()),
        "path_supervision_bce_pos_weight": float(pos_weight_value),
        "unlearned_edge_count": int(hard.sum()),
        "unlearned_node_count": int(sum(1 for _, d in out_graph.nodes(data=True) if int(d.get("unlearned", 0)) == 1)),
        "punish_mean": float(probs.mean()) if len(probs) else 0.0,
        "punish_max": float(probs.max()) if len(probs) else 0.0,
        "punish_min": float(probs.min()) if len(probs) else 0.0,
        "label_params": {
            "seed_source": "base_memory_retrieval_path_supervision"
            if use_path_supervision
            else query_data.get("label_source", "exact_author_entity_to_mentioned_anchor_chunk"),
            "path_cutoff": int(memory_path_cutoff),
            "relation_prior": float(memory_relation_prior),
            "mentioned_prior": float(memory_mentioned_prior),
            "next_turn_prior": float(memory_next_turn_prior),
            "background_relation_scale": float(memory_background_relation_scale),
            "community_floor": float(memory_community_floor),
            "path_decay": float(memory_path_decay),
            "retrieval_max_depth": int(memory_retrieval_max_depth),
            "retrieval_max_seeds": int(memory_retrieval_max_seeds),
            "retrieval_max_hits": int(memory_retrieval_max_hits),
            "retrieval_entity_sim_floor": float(memory_retrieval_entity_sim_floor),
            "path_retain_weight": float(memory_path_retain_weight),
        },
        "path_supervision": {
            "enabled": bool(use_path_supervision),
            "forget_question_count": int(len(memory_path_forget_questions or [])),
            "retain_question_count": int(len(memory_path_retain_questions or [])),
            "forget_path_count": int(query_data.get("forget_path_count", 0)),
            "retain_path_count": int(query_data.get("retain_path_count", 0)),
            "forget_edge_touch_count": int(query_data.get("forget_edge_touch_count", 0)),
            "retain_edge_touch_count": int(query_data.get("retain_edge_touch_count", 0)),
        },
        "loss_weights": {
            "path": float(memory_path_loss_weight),
            "safe": float(memory_safe_loss_weight),
            "toxic": float(memory_toxic_loss_weight),
            "sparse": float(memory_sparse_loss_weight),
            "consistency": float(memory_consistency_loss_weight),
            "next_turn": float(memory_next_turn_loss_weight),
            "relation": float(memory_relation_loss_weight),
            "next_turn_safe": float(memory_next_turn_safe_loss_weight),
            "relation_safe": float(memory_relation_safe_loss_weight),
            "ranking": float(memory_ranking_loss_weight),
            "ranking_margin": float(memory_ranking_margin),
        },
        "ranking_pair_count": int(len(rank_pos_np)),
        "pretrain_loss": "0.1*node_type_ce + 0.7*edge_type_ce + 0.35*temporal_bce + 0.5*community_ce + 0.35*edge_score_mse + 0.25*community_contrast_bce",
        "retrain_loss": "query_aware_path_supervised_edge_punish_bce_plus_path_safe_toxic_sparse_consistency_edge_type_and_same_type_ranking"
        if use_path_supervision
        else "query_aware_strong_exact_entity_plus_weak_path_edge_punish_bce_plus_path_safe_toxic_sparse_consistency_edge_type_and_same_type_ranking",
        "pretrain_log": pretrain_log,
        "retrain_log": retrain_log,
    }
    return out_graph, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-graph", type=Path, default=DEFAULT_RAG_GRAPH)
    parser.add_argument("--memory-graph", type=Path, default=DEFAULT_MEMORY_GRAPH)
    parser.add_argument("--forget-query", action="append", default=[], help="Natural forget query. Can be repeated.")
    parser.add_argument("--forget-queries-path", type=Path, default=None, help="JSON/JSONL file containing queries or target names.")
    parser.add_argument(
        "--forget-qa-query-path",
        type=Path,
        default=None,
        help="Optional QA JSON/JSONL file whose question-answer pairs are added as forget queries.",
    )
    parser.add_argument(
        "--memory-forget-qa-query-path",
        type=Path,
        default=None,
        help=(
            "Optional QA JSON/JSONL file used only as Memory QA evidence. "
            "When set, RAG still uses --forget-qa-query-path, while Memory uses this file."
        ),
    )
    parser.add_argument(
        "--forget-authors-from",
        type=Path,
        default=None,
        help="Optional QA JSON/JSONL file used only to extract target author names into natural forget queries.",
    )
    parser.add_argument("--max-forget-authors", type=int, default=0)
    parser.add_argument("--encoder-model-path", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--embedding-device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--embedding-hidden-layer", type=int, default=-1)
    parser.add_argument("--embedding-pooling", type=str, default="mean", choices=["mean", "last_token"])
    parser.add_argument("--query-embedding-cache", type=Path, default=None)
    parser.add_argument("--no-embedding-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rag-pretrain-epochs", type=int, default=12)
    parser.add_argument("--rag-retrain-epochs", type=int, default=18)
    parser.add_argument("--memory-pretrain-epochs", type=int, default=12)
    parser.add_argument("--memory-retrain-epochs", type=int, default=18)
    parser.add_argument("--skip-pretrain", action="store_true", help="Load existing pre-GAT checkpoints and run retraining only.")
    parser.add_argument("--rag-pretrain-model-path", type=Path, default=None)
    parser.add_argument("--memory-pretrain-model-path", type=Path, default=None)
    parser.add_argument("--pretrain-only", action="store_true", help="Only run structural RAG/Memory GAT pretraining and save checkpoints.")
    parser.add_argument("--edge-sample-size", type=int, default=32768)
    parser.add_argument("--rag-seed-topk", type=int, default=64)
    parser.add_argument("--rag-sim-percentile", type=float, default=88.0)
    parser.add_argument("--rag-qa-topk", type=int, default=1)
    parser.add_argument("--rag-author-aux-weight", type=float, default=0.03)
    parser.add_argument(
        "--rag-community-top-m",
        type=int,
        default=0,
        help="If >0, label only the top-M nodes inside each QA seed community instead of percentile expansion.",
    )
    parser.add_argument(
        "--rag-disable-weak-labels",
        action="store_true",
        help="Use only exact QA matches as RAG positive labels; disable embedding/community weak positives.",
    )
    parser.add_argument("--rag-drop-class-weight-scale", type=float, default=1.0)
    parser.add_argument("--rag-lexical-loss-weight", type=float, default=0.0)
    parser.add_argument("--rag-safe-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--rag-cardinality-target-count",
        type=int,
        default=0,
        help="If >0, add a RAG retrain loss that pulls mean drop probability toward target_count / node_count.",
    )
    parser.add_argument("--rag-cardinality-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--rag-threshold-target-count",
        type=int,
        default=0,
        help="If >0, choose the final RAG threshold from drop_prob so about this many nodes are marked unlearned.",
    )
    parser.add_argument(
        "--rag-threshold-cardinality-penalty",
        type=float,
        default=0.0,
        help="If >0 with --rag-threshold-target-count, choose threshold by F1 minus cardinality deviation penalty.",
    )
    parser.add_argument(
        "--rag-include-strong-in-hard",
        action="store_true",
        help="Force exact QA strong-label RAG nodes into the final unlearned mask before adding threshold/top-count extras.",
    )
    parser.add_argument(
        "--rag-strong-hard-fraction",
        type=float,
        default=0.0,
        help="If >0, select only this fraction of exact QA strong-label RAG nodes by learned drop_prob into the final hard mask.",
    )
    parser.add_argument(
        "--rag-retrieval-aligned-stats",
        type=Path,
        default=None,
        help="Optional JSON containing forget/retain RAG top-k node counts for retrieval-aligned strong labels.",
    )
    parser.add_argument("--rag-retrieval-aligned-positive-count", type=int, default=200)
    parser.add_argument("--rag-retrieval-aligned-retain-penalty", type=float, default=0.2)
    parser.add_argument(
        "--rag-force-positive-node-range",
        action="append",
        default=[],
        help="Force RAG node ids/ranges into retrieval-aligned positive labels, e.g. 3600:3999.",
    )
    parser.add_argument(
        "--rag-force-positive-qa-path",
        type=Path,
        default=None,
        help="Force RAG nodes matching this QA JSON/JSONL into retrieval-aligned positive labels.",
    )
    parser.add_argument(
        "--rag-retrieval-score-supervision-stats",
        type=Path,
        default=None,
        help="Optional JSON containing forget/retain RAG top-k counts for soft score supervision.",
    )
    parser.add_argument("--rag-retrieval-score-retain-penalty", type=float, default=1.0)
    parser.add_argument("--rag-retrieval-score-temperature", type=float, default=1.0)
    parser.add_argument("--rag-retrieval-score-loss-weight", type=float, default=0.0)
    parser.add_argument("--rag-retrieval-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--rag-retrieval-ranking-margin", type=float, default=0.05)
    parser.add_argument("--rag-retrieval-ranking-pairs", type=int, default=512)
    parser.add_argument("--rag-strong-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--rag-strong-ranking-margin", type=float, default=0.05)
    parser.add_argument("--rag-strong-ranking-pairs", type=int, default=512)
    parser.add_argument("--rag-aux-feature-head", action="store_true")
    parser.add_argument("--rag-aux-logit-weight", type=float, default=1.0)
    parser.add_argument("--rag-gat-logit-weight", type=float, default=1.0)
    parser.add_argument("--rag-entity-aux-feature-weight", type=float, default=0.1)
    parser.add_argument("--memory-path-cutoff", type=int, default=3)
    parser.add_argument("--memory-qa-topk", type=int, default=3)
    parser.add_argument("--memory-hard-threshold", type=float, default=-1.0)
    parser.add_argument("--memory-relation-prior", type=float, default=1.0)
    parser.add_argument("--memory-mentioned-prior", type=float, default=0.9)
    parser.add_argument("--memory-next-turn-prior", type=float, default=0.35)
    parser.add_argument("--memory-background-relation-scale", type=float, default=0.15)
    parser.add_argument("--memory-community-floor", type=float, default=0.35)
    parser.add_argument("--memory-path-decay", type=float, default=1.6)
    parser.add_argument("--memory-path-loss-weight", type=float, default=0.25)
    parser.add_argument("--memory-safe-loss-weight", type=float, default=0.2)
    parser.add_argument("--memory-toxic-loss-weight", type=float, default=0.15)
    parser.add_argument("--memory-sparse-loss-weight", type=float, default=0.05)
    parser.add_argument("--memory-consistency-loss-weight", type=float, default=0.08)
    parser.add_argument("--memory-next-turn-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-relation-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-next-turn-safe-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-relation-safe-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-ranking-margin", type=float, default=1.0)
    parser.add_argument(
        "--memory-qa-hard-fraction",
        type=float,
        default=0.0,
        help="If >0, mark only this fraction of matched Memory forget QA chunks by learned punish score as final hard nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[Load] RAG graph:", args.rag_graph)
    rag_graph = load_pickle(args.rag_graph)
    print("[Load] Memory graph:", args.memory_graph)
    memory_graph = load_pickle(args.memory_graph)

    if args.pretrain_only:
        print("[Mode] pretrain-only: skip query collection and Qwen embeddings.")
        print("[Step1] Adding Louvain communities to Memory graph...")
        memory_graph, louvain_report = add_louvain_communities_to_memory(memory_graph)
        memory_graph_path = args.output_dir / "memory_graph_with_louvain.pkl"
        save_pickle(memory_graph_path, memory_graph)
        save_json(args.output_dir / "memory_louvain_report.json", louvain_report)
        print(
            "[Step1] memory nodes={} edges={} communities={}".format(
                memory_graph.number_of_nodes(),
                memory_graph.number_of_edges(),
                louvain_report["community_count"],
            )
        )

        dummy_vec = np.zeros((1, infer_embedding_dim(rag_graph, memory_graph)), dtype=np.float32)
        print("[Step2] Pretraining RAG-GAT from graph structure...")
        rag_pre_graph, rag_report = train_rag(
            rag_graph,
            dummy_vec,
            ["pretrain only"],
            ["author"],
            output_dir=args.output_dir,
            device=device,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            dropout=args.dropout,
            pretrain_epochs=args.rag_pretrain_epochs,
            retrain_epochs=0,
            lr=args.lr,
            weight_decay=args.weight_decay,
            edge_sample_size=args.edge_sample_size,
            rag_seed_topk=args.rag_seed_topk,
            rag_sim_percentile=args.rag_sim_percentile,
            rag_qa_topk=args.rag_qa_topk,
            rag_author_aux_weight=args.rag_author_aux_weight,
            rag_community_top_m=args.rag_community_top_m,
            rag_disable_weak_labels=args.rag_disable_weak_labels,
            rag_drop_class_weight_scale=args.rag_drop_class_weight_scale,
            rag_lexical_loss_weight=args.rag_lexical_loss_weight,
            rag_safe_loss_weight=args.rag_safe_loss_weight,
            rag_cardinality_target_count=args.rag_cardinality_target_count,
            rag_cardinality_loss_weight=args.rag_cardinality_loss_weight,
            rag_threshold_target_count=args.rag_threshold_target_count,
            rag_threshold_cardinality_penalty=args.rag_threshold_cardinality_penalty,
            rag_include_strong_in_hard=args.rag_include_strong_in_hard,
            rag_strong_hard_fraction=args.rag_strong_hard_fraction,
            rag_retrieval_aligned_stats=args.rag_retrieval_aligned_stats,
            rag_retrieval_aligned_positive_count=args.rag_retrieval_aligned_positive_count,
            rag_retrieval_aligned_retain_penalty=args.rag_retrieval_aligned_retain_penalty,
            rag_force_positive_node_range=args.rag_force_positive_node_range,
            rag_force_positive_qa_path=args.rag_force_positive_qa_path,
            rag_retrieval_score_supervision_stats=args.rag_retrieval_score_supervision_stats,
            rag_retrieval_score_retain_penalty=args.rag_retrieval_score_retain_penalty,
            rag_retrieval_score_temperature=args.rag_retrieval_score_temperature,
            rag_retrieval_score_loss_weight=args.rag_retrieval_score_loss_weight,
            rag_retrieval_ranking_loss_weight=args.rag_retrieval_ranking_loss_weight,
            rag_retrieval_ranking_margin=args.rag_retrieval_ranking_margin,
            rag_retrieval_ranking_pairs=args.rag_retrieval_ranking_pairs,
            rag_strong_ranking_loss_weight=args.rag_strong_ranking_loss_weight,
            rag_strong_ranking_margin=args.rag_strong_ranking_margin,
            rag_strong_ranking_pairs=args.rag_strong_ranking_pairs,
            rag_aux_feature_head=args.rag_aux_feature_head,
            rag_aux_logit_weight=args.rag_aux_logit_weight,
            rag_gat_logit_weight=args.rag_gat_logit_weight,
            rag_entity_aux_feature_weight=args.rag_entity_aux_feature_weight,
            skip_pretrain=False,
            pretrain_model_path=args.rag_pretrain_model_path,
            pretrain_only=True,
            seed=args.seed,
        )
        save_pickle(args.output_dir / "rag_pretrain_graph_snapshot.pkl", rag_pre_graph)
        save_json(args.output_dir / "rag_pretrain_report.json", rag_report)

        print("[Step3] Pretraining Memory-GAT from graph structure...")
        memory_pre_graph, memory_report = train_memory(
            memory_graph,
            dummy_vec,
            ["pretrain only"],
            embedding_dim=infer_embedding_dim(rag_graph, memory_graph),
            output_dir=args.output_dir,
            device=device,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            dropout=args.dropout,
            pretrain_epochs=args.memory_pretrain_epochs,
            retrain_epochs=0,
            lr=args.lr,
            weight_decay=args.weight_decay,
            edge_sample_size=args.edge_sample_size,
            memory_path_cutoff=args.memory_path_cutoff,
            memory_hard_threshold=args.memory_hard_threshold,
            memory_relation_prior=args.memory_relation_prior,
            memory_mentioned_prior=args.memory_mentioned_prior,
            memory_next_turn_prior=args.memory_next_turn_prior,
            memory_background_relation_scale=args.memory_background_relation_scale,
            memory_community_floor=args.memory_community_floor,
            memory_path_decay=args.memory_path_decay,
            memory_path_loss_weight=args.memory_path_loss_weight,
            memory_safe_loss_weight=args.memory_safe_loss_weight,
            memory_toxic_loss_weight=args.memory_toxic_loss_weight,
            memory_sparse_loss_weight=args.memory_sparse_loss_weight,
            memory_consistency_loss_weight=args.memory_consistency_loss_weight,
            memory_next_turn_loss_weight=args.memory_next_turn_loss_weight,
            memory_relation_loss_weight=args.memory_relation_loss_weight,
            memory_next_turn_safe_loss_weight=args.memory_next_turn_safe_loss_weight,
            memory_relation_safe_loss_weight=args.memory_relation_safe_loss_weight,
            memory_ranking_loss_weight=args.memory_ranking_loss_weight,
            memory_ranking_margin=args.memory_ranking_margin,
            memory_qa_hard_fraction=args.memory_qa_hard_fraction,
            skip_pretrain=False,
            pretrain_model_path=args.memory_pretrain_model_path,
            pretrain_only=True,
            seed=args.seed,
        )
        save_pickle(args.output_dir / "memory_pretrain_graph_snapshot.pkl", memory_pre_graph)
        save_json(args.output_dir / "memory_pretrain_report.json", memory_report)

        final_path = args.output_dir / "pretrain_report.json"
        save_json(
            final_path,
            json_ready(
                {
                    "method": "split_query_aware_rag_memory_gat_pretrain_only",
                    "rag_graph": str(args.rag_graph),
                    "memory_graph": str(args.memory_graph),
                    "memory_graph_with_louvain": str(memory_graph_path),
                    "rag": rag_report,
                    "memory": memory_report,
                }
            ),
        )
        print("=" * 88)
        print("Split Query-Aware Dual Graph GAT Pretrain Only")
        print("=" * 88)
        print(f"rag_pre_gat={rag_report['pre_model_path']}")
        print(f"memory_pre_gat={memory_report['pre_model_path']}")
        print(f"saved_report={final_path}")
        return

    raw_queries, target_texts, query_types, query_source_counts = collect_forget_queries(args)
    memory_extra_qa_target_texts: list[str] = []
    if args.memory_forget_qa_query_path is not None:
        memory_extra_qa_target_texts = load_qa_queries_file(args.memory_forget_qa_query_path)
        query_source_counts["memory_qa_query_file"] = len(memory_extra_qa_target_texts)
    print("[Query] raw_query_count:", len(raw_queries))
    print("[Query] target_count:", len(target_texts))
    if memory_extra_qa_target_texts:
        print("[Query] memory_only_qa_target_count:", len(memory_extra_qa_target_texts))
    print("[Query] source_counts:", json.dumps(query_source_counts, ensure_ascii=False, sort_keys=True))
    for idx, target in enumerate(target_texts[:30], start=1):
        print(f"[Query] target[{idx:02d}]: {target}")
    if len(target_texts) > 30:
        print(f"[Query] target[...] omitted={len(target_texts) - 30}")
    for idx, target in enumerate(memory_extra_qa_target_texts[:10], start=1):
        print(f"[Query] memory_only_target[{idx:02d}]: {target}")
    if len(memory_extra_qa_target_texts) > 10:
        print(f"[Query] memory_only_target[...] omitted={len(memory_extra_qa_target_texts) - 10}")
    print("[Query] encoder:", args.encoder_model_path)
    embedding_texts = target_texts + memory_extra_qa_target_texts
    cache_metadata = {
        "texts_sha256": hash_texts(embedding_texts),
        "text_count": len(embedding_texts),
        "main_text_count": len(target_texts),
        "memory_only_qa_text_count": len(memory_extra_qa_target_texts),
        "encoder_model_path": str(args.encoder_model_path),
        "embedding_max_length": int(args.embedding_max_length),
        "embedding_hidden_layer": int(args.embedding_hidden_layer),
        "embedding_pooling": str(args.embedding_pooling),
        "embedding_4bit": bool(not args.no_embedding_4bit),
    }
    all_query_vecs = load_cached_embeddings(args.query_embedding_cache, cache_metadata)
    if all_query_vecs is None:
        embeddings = build_qwen_embeddings(
            model_path=str(args.encoder_model_path),
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            hidden_layer=args.embedding_hidden_layer,
            pooling=args.embedding_pooling,
            load_in_4bit=not args.no_embedding_4bit,
        )
        all_query_vecs = np.asarray(embeddings.embed_documents(embedding_texts), dtype=np.float32)
        save_cached_embeddings(args.query_embedding_cache, cache_metadata, all_query_vecs)
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_query_vecs = l2_normalize(all_query_vecs)
    query_vecs = all_query_vecs[: len(target_texts)]
    memory_extra_qa_query_vecs = (
        all_query_vecs[len(target_texts) :]
        if memory_extra_qa_target_texts
        else None
    )

    print("[Step1] Adding Louvain communities to Memory graph...")
    memory_graph, louvain_report = add_louvain_communities_to_memory(memory_graph)
    memory_graph_path = args.output_dir / "memory_graph_with_louvain.pkl"
    save_pickle(memory_graph_path, memory_graph)
    save_json(args.output_dir / "memory_louvain_report.json", louvain_report)
    print(
        "[Step1] memory nodes={} edges={} communities={}".format(
            memory_graph.number_of_nodes(),
            memory_graph.number_of_edges(),
            louvain_report["community_count"],
        )
    )

    print("[Step2] Training RAG-GAT from graph + dual-channel forget queries...")
    rag_query_vecs = query_vecs
    rag_target_texts = target_texts
    rag_query_types = query_types
    print(
        "[RAG:query] using author+QA dual channel: "
        f"{dict(Counter(rag_query_types))} / total={len(rag_target_texts)}"
    )
    rag_unlearning_graph, rag_report = train_rag(
        rag_graph,
        rag_query_vecs,
        rag_target_texts,
        rag_query_types,
        output_dir=args.output_dir,
        device=device,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        pretrain_epochs=args.rag_pretrain_epochs,
        retrain_epochs=args.rag_retrain_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        edge_sample_size=args.edge_sample_size,
        rag_seed_topk=args.rag_seed_topk,
        rag_sim_percentile=args.rag_sim_percentile,
        rag_qa_topk=args.rag_qa_topk,
        rag_author_aux_weight=args.rag_author_aux_weight,
        rag_community_top_m=args.rag_community_top_m,
        rag_disable_weak_labels=args.rag_disable_weak_labels,
        rag_drop_class_weight_scale=args.rag_drop_class_weight_scale,
        rag_lexical_loss_weight=args.rag_lexical_loss_weight,
        rag_safe_loss_weight=args.rag_safe_loss_weight,
        rag_cardinality_target_count=args.rag_cardinality_target_count,
        rag_cardinality_loss_weight=args.rag_cardinality_loss_weight,
        rag_threshold_target_count=args.rag_threshold_target_count,
        rag_threshold_cardinality_penalty=args.rag_threshold_cardinality_penalty,
        rag_include_strong_in_hard=args.rag_include_strong_in_hard,
        rag_strong_hard_fraction=args.rag_strong_hard_fraction,
        rag_retrieval_aligned_stats=args.rag_retrieval_aligned_stats,
        rag_retrieval_aligned_positive_count=args.rag_retrieval_aligned_positive_count,
        rag_retrieval_aligned_retain_penalty=args.rag_retrieval_aligned_retain_penalty,
        rag_force_positive_node_range=args.rag_force_positive_node_range,
        rag_force_positive_qa_path=args.rag_force_positive_qa_path,
        rag_retrieval_score_supervision_stats=args.rag_retrieval_score_supervision_stats,
        rag_retrieval_score_retain_penalty=args.rag_retrieval_score_retain_penalty,
        rag_retrieval_score_temperature=args.rag_retrieval_score_temperature,
        rag_retrieval_score_loss_weight=args.rag_retrieval_score_loss_weight,
        rag_retrieval_ranking_loss_weight=args.rag_retrieval_ranking_loss_weight,
        rag_retrieval_ranking_margin=args.rag_retrieval_ranking_margin,
        rag_retrieval_ranking_pairs=args.rag_retrieval_ranking_pairs,
        rag_strong_ranking_loss_weight=args.rag_strong_ranking_loss_weight,
        rag_strong_ranking_margin=args.rag_strong_ranking_margin,
        rag_strong_ranking_pairs=args.rag_strong_ranking_pairs,
        rag_aux_feature_head=args.rag_aux_feature_head,
        rag_aux_logit_weight=args.rag_aux_logit_weight,
        rag_gat_logit_weight=args.rag_gat_logit_weight,
        rag_entity_aux_feature_weight=args.rag_entity_aux_feature_weight,
        skip_pretrain=args.skip_pretrain,
        pretrain_model_path=args.rag_pretrain_model_path,
        pretrain_only=False,
        seed=args.seed,
    )
    rag_out = args.output_dir / "rag_unlearning_graph.pkl"
    save_pickle(rag_out, rag_unlearning_graph)
    save_json(args.output_dir / "rag_unlearning_report.json", rag_report)
    print(f"[Save] {rag_out}")

    print("[Step3] Training Memory-GAT from graph + forget query...")
    author_indices = [idx for idx, qtype in enumerate(query_types) if qtype == "author"]
    qa_indices = [idx for idx, qtype in enumerate(query_types) if qtype == "qa"]
    if author_indices:
        memory_query_vecs = query_vecs[np.asarray(author_indices, dtype=np.int64)]
        memory_target_texts = [target_texts[idx] for idx in author_indices]
        print(f"[Memory:query] using author queries as entity channel: {len(memory_target_texts)} / {len(target_texts)}")
    else:
        memory_query_vecs = query_vecs
        memory_target_texts = target_texts
        print("[Memory:query] no author queries found; using all queries")
    if memory_extra_qa_query_vecs is not None:
        memory_qa_query_vecs = memory_extra_qa_query_vecs
        memory_qa_target_texts = memory_extra_qa_target_texts
        print(
            "[Memory:labels] using memory-only QA evidence file: "
            f"{args.memory_forget_qa_query_path} rows={len(memory_qa_target_texts)}"
        )
    else:
        memory_qa_query_vecs = query_vecs[np.asarray(qa_indices, dtype=np.int64)] if qa_indices else None
        memory_qa_target_texts = [target_texts[idx] for idx in qa_indices] if qa_indices else None
    if author_indices and qa_indices:
        print(
            "[Memory:labels] dual-channel author entity + QA evidence: "
            f"authors={len(memory_target_texts)} qa={len(memory_qa_target_texts or [])}"
        )
    memory_unlearning_graph, memory_report = train_memory(
        memory_graph,
        memory_query_vecs,
        memory_target_texts,
        memory_author_query_vecs=memory_query_vecs if author_indices else None,
        memory_author_query_targets=memory_target_texts if author_indices else None,
        memory_qa_query_vecs=memory_qa_query_vecs,
        memory_qa_query_targets=memory_qa_target_texts,
        memory_qa_topk=args.memory_qa_topk,
        embedding_dim=infer_embedding_dim(rag_graph, memory_graph),
        output_dir=args.output_dir,
        device=device,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        pretrain_epochs=args.memory_pretrain_epochs,
        retrain_epochs=args.memory_retrain_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        edge_sample_size=args.edge_sample_size,
        memory_path_cutoff=args.memory_path_cutoff,
        memory_hard_threshold=args.memory_hard_threshold,
        memory_relation_prior=args.memory_relation_prior,
        memory_mentioned_prior=args.memory_mentioned_prior,
        memory_next_turn_prior=args.memory_next_turn_prior,
        memory_background_relation_scale=args.memory_background_relation_scale,
        memory_community_floor=args.memory_community_floor,
        memory_path_decay=args.memory_path_decay,
        memory_path_loss_weight=args.memory_path_loss_weight,
        memory_safe_loss_weight=args.memory_safe_loss_weight,
        memory_toxic_loss_weight=args.memory_toxic_loss_weight,
        memory_sparse_loss_weight=args.memory_sparse_loss_weight,
        memory_consistency_loss_weight=args.memory_consistency_loss_weight,
        memory_next_turn_loss_weight=args.memory_next_turn_loss_weight,
        memory_relation_loss_weight=args.memory_relation_loss_weight,
        memory_next_turn_safe_loss_weight=args.memory_next_turn_safe_loss_weight,
        memory_relation_safe_loss_weight=args.memory_relation_safe_loss_weight,
        memory_ranking_loss_weight=args.memory_ranking_loss_weight,
        memory_ranking_margin=args.memory_ranking_margin,
        memory_qa_hard_fraction=args.memory_qa_hard_fraction,
        skip_pretrain=args.skip_pretrain,
        pretrain_model_path=args.memory_pretrain_model_path,
        pretrain_only=False,
        seed=args.seed,
    )
    memory_out = args.output_dir / "memory_unlearning_graph.pkl"
    save_pickle(memory_out, memory_unlearning_graph)
    save_json(args.output_dir / "memory_unlearning_report.json", memory_report)
    print(f"[Save] {memory_out}")

    final_report = {
        "method": "split_query_aware_rag_memory_gat",
        "raw_forget_queries": raw_queries,
        "target_texts": target_texts,
        "query_source_counts": query_source_counts,
        "query_type_counts": dict(Counter(query_types)),
        "query_count": len(target_texts),
        "memory_only_qa_query_path": str(args.memory_forget_qa_query_path) if args.memory_forget_qa_query_path else "",
        "memory_only_qa_query_count": len(memory_extra_qa_target_texts),
        "rag_query_count": len(rag_target_texts),
        "rag_query_type_counts": dict(Counter(rag_query_types)),
        "query_encoder": str(args.encoder_model_path),
        "rag_graph": str(args.rag_graph),
        "memory_graph": str(args.memory_graph),
        "memory_graph_with_louvain": str(memory_graph_path),
        "skip_pretrain": bool(args.skip_pretrain),
        "rag_pretrain_model_input": str(args.rag_pretrain_model_path) if args.rag_pretrain_model_path else "",
        "memory_pretrain_model_input": str(args.memory_pretrain_model_path) if args.memory_pretrain_model_path else "",
        "rag_unlearning_graph": str(rag_out),
        "memory_unlearning_graph": str(memory_out),
        "memory_query_count": int(len(memory_target_texts) + len(memory_qa_target_texts or []))
        if author_indices and qa_indices
        else len(memory_target_texts),
        "memory_author_query_count": len(memory_target_texts) if author_indices else 0,
        "memory_qa_query_count": len(memory_qa_target_texts or []),
        "memory_query_type_counts": dict(Counter("author" for _ in memory_target_texts))
        if author_indices and not qa_indices
        else dict(Counter(["author"] * len(memory_target_texts) + ["qa"] * len(memory_qa_target_texts or [])))
        if author_indices and qa_indices
        else dict(Counter(query_types)),
        "memory_louvain": louvain_report,
        "rag": rag_report,
        "memory": memory_report,
    }
    final_path = args.output_dir / "dual_graph_unlearning_report.json"
    save_json(final_path, json_ready(final_report))
    print("=" * 88)
    print("Split Query-Aware Dual Graph Unlearning GAT")
    print("=" * 88)
    print(f"target_count={len(target_texts)}")
    print(f"rag_unlearned_nodes={rag_report['unlearned_count']} / {rag_report['node_count']}")
    print(f"memory_unlearned_edges={memory_report['unlearned_edge_count']} / {memory_report['edge_count_original']}")
    print(f"saved_report={final_path}")


if __name__ == "__main__":
    main()
