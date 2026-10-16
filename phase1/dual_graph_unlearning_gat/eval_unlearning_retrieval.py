                      
                                                                              

                                                                              
                                                                            
                                                                               
                                 
   

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parents[1]
for path in (AGENT_ROOT, AGENT_ROOT / "phase1"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase1.core.qwen_embeddings import build_qwen_embeddings              


DEFAULT_RAG_GRAPH = AGENT_ROOT / "phase1" / "artifacts" / "rag" / "rag_graph.pkl"
DEFAULT_MEMORY_GRAPH = AGENT_ROOT / "phase1" / "artifacts" / "memory" / "generic_memory_graph.pkl"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts_query_unlearning_forget10_authors"
DEFAULT_RAG_UNLEARNING_GRAPH = DEFAULT_ARTIFACT_DIR / "rag_unlearning_graph.pkl"
DEFAULT_MEMORY_UNLEARNING_GRAPH = DEFAULT_ARTIFACT_DIR / "memory_unlearning_graph.pkl"
DEFAULT_FORGET = Path("/home/likunhao/likunhao52shared/datasets/TOFU/forget10.json")
DEFAULT_RETAIN = Path("/home/likunhao/likunhao52shared/datasets/TOFU/retain90.json")
DEFAULT_ENCODER = Path("/disk8T/lkh/python_projects/llms/Qwen3.5-9B")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "retrieval_eval_path_metrics_full"

NAME_PATTERN = re.compile(r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,4})\b")
NUMBER_PATTERN = re.compile(r"\b\d{2,4}(?:[/-]\d{1,2})?(?:[/-]\d{1,4})?\b")
STOPWORDS = {
    "about",
    "after",
    "also",
    "answer",
    "author",
    "because",
    "before",
    "being",
    "book",
    "could",
    "does",
    "from",
    "have",
    "known",
    "question",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

RAG_GRAPH_HYBRID_WEIGHTS = {
    "semantic": 0.58,
    "lexical": 0.24,
    "graph_risk": 0.06,
    "query_similarity": 0.12,
}


@dataclass(frozen=True)
class MemoryEdgeRef:
    source: str
    target: str
    key: str
    relation: str
    unlearned: int

    @property
    def edge_id(self) -> str:
        return f"{self.source}\t{self.target}\t{self.key}"


@dataclass(frozen=True)
class MemoryPathHit:
    chunk_id: str
    score: float
    depth: int
    question: str
    answer: str
    path_nodes: tuple[str, ...]
    path_edges: tuple[MemoryEdgeRef, ...]
    blocked_edges: tuple[str, ...]
    blocked_nodes: tuple[str, ...]

    @property
    def blocked(self) -> int:
        return int(bool(self.blocked_edges or self.blocked_nodes))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset in {path}")
    return [row for row in data if isinstance(row, dict)]


def hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b"\0")
        digest.update(encoded)
        digest.update(b"\0")
    return digest.hexdigest()


def load_cached_vectors(cache_path: Path | None, metadata: dict[str, Any]) -> np.ndarray | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data["metadata_json"].item()))
            vectors = np.asarray(data["vectors"], dtype=np.float32)
    except Exception as exc:
        print(f"[Eval] ignoring invalid question embedding cache {cache_path}: {exc}")
        return None
    if cached_metadata != metadata:
        print(f"[Eval] ignoring stale question embedding cache: {cache_path}")
        return None
    print(f"[Eval] loaded question embedding cache: {cache_path} shape={tuple(vectors.shape)}")
    return vectors


def save_cached_vectors(cache_path: Path | None, metadata: dict[str, Any], vectors: np.ndarray) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        vectors=np.asarray(vectors, dtype=np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    print(f"[Eval] saved question embedding cache: {cache_path} shape={tuple(vectors.shape)}")


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def tokens(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").casefold())
        if len(token) >= 3 and token not in STOPWORDS
    }


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(left_tokens))


def named_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in NAME_PATTERN.findall(text or ""):
        key = normalize_key(match)
        if key and key not in STOPWORDS:
            phrases.append(key)
    return phrases


def number_tokens(text: Any) -> set[str]:
    return set(NUMBER_PATTERN.findall(str(text or "")))


def answer_support_terms(reference_answer: str) -> set[str]:
                                                                             
    answer = str(reference_answer or "")
    terms = set(number_tokens(answer))
    for phrase in named_phrases(answer):
        terms.update(token for token in phrase.split() if len(token) >= 3 and token not in STOPWORDS)
    terms.update(tokens(answer))
    return terms


def answer_support_score(reference_answer: str, evidence_text: str) -> float:
    terms = answer_support_terms(reference_answer)
    if not terms:
        return 0.0
    evidence_key = normalize_key(evidence_text)
    if not evidence_key:
        return 0.0
    hits = sum(1 for term in terms if normalize_key(term) in evidence_key)
    return float(hits / len(terms))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        denom = float(np.linalg.norm(arr))
        return arr if denom <= 1e-12 else (arr / denom).astype(np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / denom).astype(np.float32)


def take(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[: min(limit, len(rows))]


def build_eval_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, path, limit in (
        ("forget", Path(args.forget_path), args.forget_limit),
        ("retain", Path(args.retain_path), args.retain_limit),
    ):
        for row in take(load_json_or_jsonl(path), int(limit)):
            question = normalize_text(row.get("question") or row.get("instruction"))
            answer = normalize_text(row.get("answer") or row.get("output"))
            if question:
                rows.append({"split": split, "question": question, "reference_answer": answer})
    return rows


def iter_edges(graph: Any):
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(keys=True, data=True):
            yield str(source), str(target), str(key), data
    else:
        for index, (source, target, data) in enumerate(graph.edges(data=True)):
            yield str(source), str(target), str(data.get("key", index)), data


def build_unlearned_edge_map(graph: Any) -> dict[tuple[str, str, str], int]:
    edge_map: dict[tuple[str, str, str], int] = {}
    for source, target, key, data in iter_edges(graph):
        flag = int(data.get("unlearned", 0))
        edge_map[(source, target, key)] = flag
        edge_map[(target, source, key)] = flag
    return edge_map


class RagNodeMaskEvaluator:
    def __init__(
        self,
        base_graph: Any,
        unlearning_graph: Any,
        top_k: int,
        *,
        recall_k: int | None = None,
        rerank_mode: str = "embedding",
        answer_hit_threshold: float = 0.25,
    ):
        self.top_k = max(1, int(top_k))
        self.recall_k = max(self.top_k, int(recall_k or top_k))
        self.rerank_mode = str(rerank_mode or "embedding")
        self.answer_hit_threshold = float(answer_hit_threshold)
        unlearned_by_node = {
            str(node_id): int(data.get("unlearned", 0))
            for node_id, data in unlearning_graph.nodes(data=True)
        }
        node_ids: list[str] = []
        vectors: list[np.ndarray] = []
        unlearned: list[int] = []
        drop_probs: list[float] = []
        keep_probs: list[float] = []
        query_similarities: list[float] = []
        weak_labels: list[int] = []
        questions: list[str] = []
        answers: list[str] = []
        for node_id, data in base_graph.nodes(data=True):
            if data.get("type") != "static_knowledge":
                continue
            embedding = data.get("embedding")
            if embedding is None:
                continue
            node_key = str(node_id)
            unlearning_data = unlearning_graph.nodes[node_key] if node_key in unlearning_graph.nodes else {}
            node_ids.append(node_key)
            vectors.append(np.asarray(embedding, dtype=np.float32))
            unlearned.append(int(unlearned_by_node.get(node_key, data.get("unlearned", 0))))
            drop_probs.append(float(unlearning_data.get("drop_prob", data.get("drop_prob", 0.0)) or 0.0))
            keep_probs.append(float(unlearning_data.get("keep_prob", data.get("keep_prob", 1.0)) or 0.0))
            query_similarities.append(float(unlearning_data.get("query_similarity", data.get("query_similarity", 0.0)) or 0.0))
            weak_labels.append(int(unlearning_data.get("weak_label", data.get("weak_label", 0)) or 0))
            questions.append(normalize_text(data.get("question", "")))
            answers.append(normalize_text(data.get("answer", "")))
        if not vectors:
            raise ValueError("RAG graph has no static_knowledge embeddings.")
        self.node_ids = node_ids
        self.matrix = l2_normalize(np.stack(vectors))
        self.unlearned = np.asarray(unlearned, dtype=np.int64)
        self.drop_probs = np.clip(np.asarray(drop_probs, dtype=np.float32), 0.0, 1.0)
        self.keep_probs = np.clip(np.asarray(keep_probs, dtype=np.float32), 0.0, 1.0)
        self.query_similarities = np.clip(np.asarray(query_similarities, dtype=np.float32), 0.0, 1.0)
        self.weak_labels = np.asarray(weak_labels, dtype=np.int64)
        self.questions = questions
        self.answers = answers
        self.node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.neighbor_drop_probs = self._compute_neighbor_drop_probs(unlearning_graph)
        self.graph_risk_scores = self._compute_graph_risk_scores()
        self.batch_top_indices: np.ndarray | None = None
        self.batch_top_scores: np.ndarray | None = None

    def _compute_neighbor_drop_probs(self, graph: Any) -> np.ndarray:
        values: list[float] = []
        for node_id, fallback in zip(self.node_ids, self.drop_probs.tolist()):
            if node_id not in graph:
                values.append(float(fallback))
                continue
            weighted_total = 0.0
            weight_sum = 0.0
            try:
                neighbors = graph.neighbors(node_id)
            except Exception:
                values.append(float(fallback))
                continue
            for neighbor in neighbors:
                neighbor_key = str(neighbor)
                neighbor_idx = self.node_index.get(neighbor_key)
                if neighbor_idx is None:
                    continue
                edge_data = graph.get_edge_data(node_id, neighbor, default={}) or {}
                if isinstance(edge_data, dict) and "weight" in edge_data:
                    weight = float(edge_data.get("weight", 1.0) or 1.0)
                elif isinstance(edge_data, dict) and edge_data:
                    first_edge = next(iter(edge_data.values()))
                    weight = float(first_edge.get("weight", 1.0) or 1.0) if isinstance(first_edge, dict) else 1.0
                else:
                    weight = 1.0
                weighted_total += weight * float(self.drop_probs[neighbor_idx])
                weight_sum += max(weight, 1e-6)
            values.append(float(weighted_total / weight_sum) if weight_sum > 0.0 else float(fallback))
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)

    def _compute_graph_risk_scores(self) -> np.ndarray:
        direct_risk = np.maximum(self.drop_probs, self.unlearned.astype(np.float32))
        weak_risk = self.weak_labels.astype(np.float32)
        risk = (0.70 * direct_risk) + (0.20 * self.neighbor_drop_probs) + (0.10 * weak_risk)
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def prepare_batch(self, question_vectors: np.ndarray) -> None:
        query_matrix = l2_normalize(np.asarray(question_vectors, dtype=np.float32))
        scores = query_matrix @ self.matrix.T
        recall_k = min(self.recall_k, scores.shape[1])
        if recall_k <= 0:
            self.batch_top_indices = np.zeros((scores.shape[0], 0), dtype=np.int64)
            self.batch_top_scores = np.zeros((scores.shape[0], 0), dtype=np.float32)
            return
        top = np.argpartition(-scores, kth=recall_k - 1, axis=1)[:, :recall_k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        self.batch_top_indices = np.take_along_axis(top, order, axis=1).astype(np.int64)
        self.batch_top_scores = np.take_along_axis(top_scores, order, axis=1).astype(np.float32)
        print(
            "[Eval] prepared RAG batch recall "
            f"queries={len(query_matrix)} nodes={len(self.node_ids)} recall_k={recall_k} final_k={self.top_k} "
            f"rerank={self.rerank_mode}"
        )

    def rerank_candidates(
        self,
        question: str,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        query_key = normalize_key(question)
        query_phrases = named_phrases(question)
        query_numbers = number_tokens(question)
        ranked: list[dict[str, Any]] = []
        for idx, semantic_score in zip(candidate_indices.tolist(), candidate_scores.tolist()):
            node_idx = int(idx)
            question_text = self.questions[node_idx]
            answer_text = self.answers[node_idx]
            qa_text = f"{question_text} {answer_text}"
            lexical = lexical_overlap(question, qa_text)
            qa_key = normalize_key(qa_text)
            entity_bonus = 0.0
            if query_phrases and any(phrase and phrase in qa_key for phrase in query_phrases):
                entity_bonus = 0.20
            number_bonus = 0.0
            if query_numbers and query_numbers & number_tokens(qa_text):
                number_bonus = 0.15
            exact_bonus = 0.30 if query_key and query_key == normalize_key(question_text) else 0.0
            if self.rerank_mode == "hybrid":
                final_score = (
                    0.58 * float(semantic_score)
                    + 0.24 * lexical
                    + entity_bonus
                    + number_bonus
                    + exact_bonus
                )
            elif self.rerank_mode == "graph_hybrid":
                graph_risk = float(self.graph_risk_scores[node_idx])
                query_similarity = float(self.query_similarities[node_idx])
                final_score = (
                    RAG_GRAPH_HYBRID_WEIGHTS["semantic"] * float(semantic_score)
                    + RAG_GRAPH_HYBRID_WEIGHTS["lexical"] * lexical
                    + RAG_GRAPH_HYBRID_WEIGHTS["graph_risk"] * graph_risk
                    + RAG_GRAPH_HYBRID_WEIGHTS["query_similarity"] * query_similarity
                    + entity_bonus
                    + number_bonus
                    + exact_bonus
                )
            else:
                final_score = float(semantic_score)
            ranked.append(
                {
                    "node_id": self.node_ids[node_idx],
                    "score": float(final_score),
                    "embedding_score": float(semantic_score),
                    "lexical_score": float(lexical),
                    "graph_risk_score": float(self.graph_risk_scores[node_idx]),
                    "node_drop_prob": float(self.drop_probs[node_idx]),
                    "neighbor_drop_prob": float(self.neighbor_drop_probs[node_idx]),
                    "query_similarity": float(self.query_similarities[node_idx]),
                    "weak_label": int(self.weak_labels[node_idx]),
                    "unlearned": int(self.unlearned[node_idx]),
                }
            )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["node_id"])))
        return ranked

    def attach_diagnostic_answer_support(
        self,
        ranked: list[dict[str, Any]],
        reference_answer: str,
    ) -> list[dict[str, Any]]:
                                                                                

                                                                            
                                                                           
                       
           
        with_support: list[dict[str, Any]] = []
        for item in ranked:
            node_idx = self.node_index.get(str(item.get("node_id") or ""))
            qa_text = ""
            if node_idx is not None:
                qa_text = f"{self.questions[node_idx]} {self.answers[node_idx]}"
            copied = dict(item)
            copied["diagnostic_answer_support_score"] = float(answer_support_score(reference_answer, qa_text))
            with_support.append(copied)
        return with_support

    def metrics_from_items(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        flags = [item["unlearned"] for item in items]
        support_flags = [
            float(item.get("diagnostic_answer_support_score", 0.0)) >= self.answer_hit_threshold
            for item in items
        ]
        safe_support_flags = [
            bool(support) and not int(item.get("unlearned", 0))
            for item, support in zip(items, support_flags)
        ]
        masked_support_flags = [
            bool(support) and bool(int(item.get("unlearned", 0)))
            for item, support in zip(items, support_flags)
        ]
        return {
            "rag_nodes": items,
            "rag_top1_unlearned": float(flags[0]) if flags else 0.0,
            "rag_any_unlearned_at_k": float(any(flags)),
            "rag_all_unlearned_at_k": float(bool(flags) and all(flags)),
            "rag_all_kept_at_k": float(not any(flags)),
            "rag_unlearned_rate_at_k": float(sum(flags) / len(flags)) if flags else 0.0,
            "rag_answer_support_hit_at_k": float(any(support_flags)),
            "rag_safe_answer_support_hit_at_k": float(any(safe_support_flags)),
            "rag_masked_answer_support_hit_at_k": float(any(masked_support_flags)),
            "rag_answer_support_rate_at_k": float(sum(support_flags) / len(support_flags)) if support_flags else 0.0,
            "rag_graph_risk_rate_at_k": float(
                sum(float(item.get("graph_risk_score", 0.0)) for item in items) / len(items)
            )
            if items
            else 0.0,
        }

    def evaluate(self, question: str, reference_answer: str, query_vec: np.ndarray) -> dict[str, Any]:
        scores = self.matrix @ l2_normalize(query_vec)
        recall_k = min(self.recall_k, len(scores))
        top = np.argsort(-scores)[:recall_k]
        ranked = self.rerank_candidates(question, top, scores[top])
        items = self.attach_diagnostic_answer_support(ranked[: self.top_k], reference_answer)
        return self.metrics_from_items(items)

    def evaluate_index(self, query_index: int, question: str, reference_answer: str, query_vec: np.ndarray) -> dict[str, Any]:
        if self.batch_top_indices is None or self.batch_top_scores is None:
            return self.evaluate(question, reference_answer, query_vec)
        candidate_indices = self.batch_top_indices[int(query_index)]
        candidate_scores = self.batch_top_scores[int(query_index)]
        ranked = self.rerank_candidates(question, candidate_indices, candidate_scores)
        items = self.attach_diagnostic_answer_support(ranked[: self.top_k], reference_answer)
        return self.metrics_from_items(items)

    def evaluate_index_metrics_only(self, query_index: int, question: str, reference_answer: str, query_vec: np.ndarray) -> dict[str, Any]:
        if self.batch_top_indices is None:
            metrics = self.evaluate(question, reference_answer, query_vec)
            metrics.pop("rag_nodes", None)
            return metrics
        candidate_indices = self.batch_top_indices[int(query_index)]
        candidate_scores = self.batch_top_scores[int(query_index)]
        ranked = self.rerank_candidates(question, candidate_indices, candidate_scores)
        items = self.attach_diagnostic_answer_support(ranked[: self.top_k], reference_answer)
        metrics = self.metrics_from_items(items)
        metrics.pop("rag_nodes", None)
        return metrics


class MemoryPathEvaluator:
    def __init__(
        self,
        base_graph: Any,
        unlearning_graph: Any,
        *,
        max_depth: int,
        max_seeds: int,
        max_hits: int,
        entity_sim_floor: float,
    ):
        self.base_graph = base_graph
        self.unlearning_graph = unlearning_graph
        self.max_depth = max(0, int(max_depth))
        self.max_seeds = max(1, int(max_seeds))
        self.max_hits = max(1, int(max_hits))
        self.entity_sim_floor = float(entity_sim_floor)
        self.node_unlearned = {
            str(node_id): int(data.get("unlearned", 0))
            for node_id, data in unlearning_graph.nodes(data=True)
        }
        self.edge_unlearned = build_unlearned_edge_map(unlearning_graph)
        self.entity_nodes: list[str] = []
        self.entity_names: list[str] = []
        self.entity_keys: list[str] = []
        self.chunk_vectors: dict[str, np.ndarray] = {}
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.chunk_token_sets: list[set[str]] = []
        self.chunk_index: dict[str, int] = {}
        entity_vectors: list[np.ndarray] = []
        chunk_vectors: list[np.ndarray] = []
        for node_id, data in base_graph.nodes(data=True):
            node_key = str(node_id)
            node_type = str(data.get("node_type") or data.get("type") or "")
            embedding = data.get("embedding")
            if node_type == "entity" and embedding is not None:
                self.entity_nodes.append(node_key)
                name = str(data.get("canonical_name") or data.get("name") or node_key)
                self.entity_names.append(name)
                self.entity_keys.append(normalize_key(name))
                entity_vectors.append(np.asarray(embedding, dtype=np.float32))
            elif node_type == "chunk" and embedding is not None:
                chunk_vec = l2_normalize(np.asarray(embedding, dtype=np.float32))
                self.chunk_vectors[node_key] = chunk_vec
                self.chunk_index[node_key] = len(self.chunk_ids)
                self.chunk_ids.append(node_key)
                chunk_text = f"{data.get('question', '')} {data.get('answer', '')}"
                self.chunk_texts.append(chunk_text)
                self.chunk_token_sets.append(tokens(chunk_text))
                chunk_vectors.append(chunk_vec)
        if not entity_vectors:
            raise ValueError("Memory graph has no entity embeddings.")
        self.entity_matrix = l2_normalize(np.stack(entity_vectors))
        self.chunk_matrix = l2_normalize(np.stack(chunk_vectors)) if chunk_vectors else np.zeros((0, 1), dtype=np.float32)
        self.adjacency: dict[str, list[MemoryEdgeRef]] = {}
        for source, target, key, data in iter_edges(base_graph):
            relation = str(data.get("relation") or data.get("edge_type") or "")
            flag = int(self.edge_unlearned.get((source, target, key), data.get("unlearned", 0)))
            edge_ref = MemoryEdgeRef(
                source=source,
                target=target,
                key=key,
                relation=relation,
                unlearned=flag,
            )
            self.adjacency.setdefault(source, []).append(edge_ref)
            self.adjacency.setdefault(target, []).append(edge_ref)

    def seed_entities(self, question: str, query_vec: np.ndarray) -> list[str]:
        query_key = normalize_key(question)
        phrase_keys = set(named_phrases(question))
        exact = [
            node
            for node, key in zip(self.entity_nodes, self.entity_keys)
            if key and (key in query_key or key in phrase_keys)
        ]
        if exact:
            return exact[: self.max_seeds]
        sims = self.entity_matrix @ l2_normalize(query_vec)
        order = np.argsort(-sims)
        seeds: list[str] = []
        for idx in order.tolist():
            if len(seeds) >= self.max_seeds:
                break
            if seeds and float(sims[idx]) < self.entity_sim_floor:
                break
            seeds.append(self.entity_nodes[idx])
        return seeds

    def blocked_nodes(self, path_nodes: tuple[str, ...]) -> tuple[str, ...]:
        blocked: list[str] = []
        for node_id in path_nodes:
            if int(self.node_unlearned.get(node_id, 0)) == 1:
                node_type = str(self.base_graph.nodes[node_id].get("node_type") or "")
                if node_type in {"anchor", "chunk"}:
                    blocked.append(node_id)
        return tuple(blocked)

    @staticmethod
    def blocked_edges(path_edges: tuple[MemoryEdgeRef, ...]) -> tuple[str, ...]:
        return tuple(edge.edge_id for edge in path_edges if int(edge.unlearned) == 1)

    def chunk_score(self, question: str, query_vec: np.ndarray, chunk_id: str, depth: int) -> float:
        chunk_data = self.base_graph.nodes[chunk_id]
        chunk_text = f"{chunk_data.get('question', '')} {chunk_data.get('answer', '')}"
        chunk_vec = self.chunk_vectors.get(chunk_id)
        query_unit = l2_normalize(query_vec)
        semantic = float(np.dot(chunk_vec, query_unit)) if chunk_vec is not None else 0.0
        lexical = lexical_overlap(question, chunk_text)
        return float((0.75 * semantic + 0.25 * lexical) / (1.0 + 0.15 * depth))

    def evaluate(self, question: str, query_vec: np.ndarray) -> dict[str, Any]:
        seeds = self.seed_entities(question, query_vec)
        best_hits: dict[str, MemoryPathHit] = {}
        best_depth: dict[str, int] = {}
        queue: deque[tuple[str, int, tuple[MemoryEdgeRef, ...], tuple[str, ...]]] = deque()
        for seed in seeds:
            queue.append((seed, 0, tuple(), (seed,)))
            best_depth[seed] = 0

        while queue:
            node_id, depth, path_edges, path_nodes = queue.popleft()
            if depth > self.max_depth or node_id not in self.base_graph.nodes:
                continue
            node_data = self.base_graph.nodes[node_id]
            node_type = str(node_data.get("node_type") or "")
            if node_type == "anchor":
                chunk_id = str(node_data.get("chunk_id") or "")
                if chunk_id in self.base_graph.nodes:
                    chunk_path_nodes = tuple((*path_nodes, chunk_id))
                    blocked_edges = self.blocked_edges(path_edges)
                    blocked_nodes = self.blocked_nodes(chunk_path_nodes)
                    score = self.chunk_score(question, query_vec, chunk_id, depth)
                    chunk_data = self.base_graph.nodes[chunk_id]
                    candidate = MemoryPathHit(
                        chunk_id=chunk_id,
                        score=score,
                        depth=depth,
                        question=str(chunk_data.get("question") or ""),
                        answer=str(chunk_data.get("answer") or ""),
                        path_nodes=chunk_path_nodes,
                        path_edges=path_edges,
                        blocked_edges=blocked_edges,
                        blocked_nodes=blocked_nodes,
                    )
                    previous = best_hits.get(chunk_id)
                    if previous is None or (candidate.score, -candidate.depth) > (previous.score, -previous.depth):
                        best_hits[chunk_id] = candidate
            if depth >= self.max_depth:
                continue
            for edge_ref in self.adjacency.get(node_id, []):
                neighbor = edge_ref.target if edge_ref.source == node_id else edge_ref.source
                new_depth = depth + 1
                if best_depth.get(neighbor, 10**9) <= new_depth:
                    continue
                best_depth[neighbor] = new_depth
                queue.append((neighbor, new_depth, (*path_edges, edge_ref), (*path_nodes, neighbor)))

        ranked = sorted(best_hits.values(), key=lambda item: (-item.score, item.depth, item.chunk_id))[: self.max_hits]
        blocked_flags = [hit.blocked for hit in ranked]
        return {
            "memory_seeds": seeds,
            "memory_hits": [self.serialize_hit(hit) for hit in ranked],
            "memory_base_hit_count": float(len(ranked)),
            "memory_top1_path_blocked": float(blocked_flags[0]) if blocked_flags else 0.0,
            "memory_any_path_blocked_at_k": float(any(blocked_flags)),
            "memory_all_paths_blocked_at_k": float(bool(blocked_flags) and all(blocked_flags)),
            "memory_all_paths_safe_at_k": float(not any(blocked_flags)),
            "memory_path_blocked_rate_at_k": float(sum(blocked_flags) / len(blocked_flags)) if blocked_flags else 0.0,
        }

    @staticmethod
    def serialize_hit(hit: MemoryPathHit) -> dict[str, Any]:
        return {
            "chunk_id": hit.chunk_id,
            "score": float(hit.score),
            "depth": int(hit.depth),
            "blocked": int(hit.blocked),
            "blocked_edges": list(hit.blocked_edges),
            "blocked_nodes": list(hit.blocked_nodes),
            "path_nodes": list(hit.path_nodes),
            "path_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "key": edge.key,
                    "relation": edge.relation,
                    "unlearned": int(edge.unlearned),
                }
                for edge in hit.path_edges
            ],
        }


class CachedMemoryPathEvaluator(MemoryPathEvaluator):
                                                                

                                                                                
                                                                           
                                                        
       

    def __init__(
        self,
        base_graph: Any,
        unlearning_graph: Any,
        *,
        max_depth: int,
        max_seeds: int,
        max_hits: int,
        entity_sim_floor: float,
        cache_dir: Path | None,
    ):
        super().__init__(
            base_graph,
            unlearning_graph,
            max_depth=max_depth,
            max_seeds=max_seeds,
            max_hits=max_hits,
            entity_sim_floor=entity_sim_floor,
        )
        self.cache_dir = cache_dir
        self.cache_path: Path | None = None
        self.path_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        self.query_entity_scores: np.ndarray | None = None
        self.query_chunk_scores: np.ndarray | None = None
        self.query_token_sets: list[set[str]] = []
        self.cache_dirty = False
        self._cache_lock = threading.Lock()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = cache_dir / f"memory_reachable_paths_depth{self.max_depth}.pkl"
            self._load_cache()

    def _metadata(self) -> dict[str, Any]:
        return {
            "max_depth": int(self.max_depth),
            "node_count": int(self.base_graph.number_of_nodes()),
            "edge_count": int(self.base_graph.number_of_edges()),
        }

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("metadata") == self._metadata():
                self.path_cache = payload.get("path_cache", {})
                print(f"[Eval] loaded memory path cache entries={len(self.path_cache)} from {self.cache_path}")
            else:
                print(f"[Eval] ignoring stale memory path cache: {self.cache_path}")
        except Exception as exc:
            print(f"[Eval] failed to load memory path cache {self.cache_path}: {exc}")

    def save_cache(self) -> None:
        if self.cache_path is None or not self.cache_dirty:
            return
        payload = {"metadata": self._metadata(), "path_cache": self.path_cache}
        with self.cache_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Eval] saved memory path cache entries={len(self.path_cache)} to {self.cache_path}")
        self.cache_dirty = False

    def prepare_batch(self, question_vectors: np.ndarray, questions: list[str]) -> None:
        query_matrix = l2_normalize(np.asarray(question_vectors, dtype=np.float32))
        self.query_entity_scores = query_matrix @ self.entity_matrix.T
        self.query_chunk_scores = query_matrix @ self.chunk_matrix.T if len(self.chunk_ids) else np.zeros((len(query_matrix), 0), dtype=np.float32)
        self.query_token_sets = [tokens(question) for question in questions]
        print(
            "[Eval] prepared fast memory scores "
            f"queries={len(query_matrix)} entities={len(self.entity_nodes)} chunks={len(self.chunk_ids)}"
        )

    def seed_entities_for_index(self, question: str, query_index: int, query_vec: np.ndarray) -> list[str]:
        query_key = normalize_key(question)
        phrase_keys = set(named_phrases(question))
        exact = [
            node
            for node, key in zip(self.entity_nodes, self.entity_keys)
            if key and (key in query_key or key in phrase_keys)
        ]
        if exact:
            return exact[: self.max_seeds]
        if self.query_entity_scores is None:
            return self.seed_entities(question, query_vec)
        sims = self.query_entity_scores[int(query_index)]
        order = np.argsort(-sims)
        seeds: list[str] = []
        for idx in order.tolist():
            if len(seeds) >= self.max_seeds:
                break
            if seeds and float(sims[idx]) < self.entity_sim_floor:
                break
            seeds.append(self.entity_nodes[idx])
        return seeds

    def reachable_candidates(self, seeds: list[str]) -> list[dict[str, Any]]:
        seed_key = tuple(seeds)
        cached = self.path_cache.get(seed_key)
        if cached is not None:
            return cached

        candidates: list[dict[str, Any]] = []
        best_depth: dict[str, int] = {}
        queue: deque[tuple[str, int, tuple[tuple[str, str, str, str], ...], tuple[str, ...]]] = deque()
        for seed in seeds:
            queue.append((seed, 0, tuple(), (seed,)))
            best_depth[seed] = 0

        while queue:
            node_id, depth, path_edges, path_nodes = queue.popleft()
            if depth > self.max_depth or node_id not in self.base_graph.nodes:
                continue
            node_data = self.base_graph.nodes[node_id]
            node_type = str(node_data.get("node_type") or "")
            if node_type == "anchor":
                chunk_id = str(node_data.get("chunk_id") or "")
                if chunk_id in self.base_graph.nodes:
                    candidates.append(
                        {
                            "chunk_id": chunk_id,
                            "depth": int(depth),
                            "path_nodes": tuple((*path_nodes, chunk_id)),
                            "path_edges": tuple(path_edges),
                        }
                    )
            if depth >= self.max_depth:
                continue
            for edge_ref in self.adjacency.get(node_id, []):
                neighbor = edge_ref.target if edge_ref.source == node_id else edge_ref.source
                new_depth = depth + 1
                if best_depth.get(neighbor, 10**9) <= new_depth:
                    continue
                best_depth[neighbor] = new_depth
                queue.append(
                    (
                        neighbor,
                        new_depth,
                        (*path_edges, (edge_ref.source, edge_ref.target, edge_ref.key, edge_ref.relation)),
                        (*path_nodes, neighbor),
                    )
                )

        with self._cache_lock:
            cached = self.path_cache.get(seed_key)
            if cached is not None:
                return cached
            self.path_cache[seed_key] = candidates
            self.cache_dirty = True
        return candidates

    def blocked_edges_from_raw(self, raw_edges: tuple[tuple[str, str, str, str], ...]) -> tuple[str, ...]:
        blocked: list[str] = []
        for source, target, key, _relation in raw_edges:
            if int(self.edge_unlearned.get((source, target, key), 0)) == 1:
                blocked.append(f"{source}\t{target}\t{key}")
        return tuple(blocked)

    def edge_refs_from_raw(self, raw_edges: tuple[tuple[str, str, str, str], ...]) -> tuple[MemoryEdgeRef, ...]:
        return tuple(
            MemoryEdgeRef(
                source=source,
                target=target,
                key=key,
                relation=relation,
                unlearned=int(self.edge_unlearned.get((source, target, key), 0)),
            )
            for source, target, key, relation in raw_edges
        )

    def cached_chunk_score(self, query_index: int, chunk_id: str, depth: int) -> float:
        chunk_idx = self.chunk_index.get(chunk_id)
        semantic = 0.0
        if chunk_idx is not None and self.query_chunk_scores is not None:
            semantic = float(self.query_chunk_scores[int(query_index), int(chunk_idx)])
        elif chunk_id in self.chunk_vectors:
            semantic = float(np.dot(self.chunk_vectors[chunk_id], l2_normalize(np.zeros_like(self.chunk_vectors[chunk_id]))))
        lexical = 0.0
        if chunk_idx is not None and self.query_token_sets:
            query_tokens = self.query_token_sets[int(query_index)]
            if query_tokens:
                lexical = float(len(query_tokens & self.chunk_token_sets[int(chunk_idx)]) / len(query_tokens))
        return float((0.75 * semantic + 0.25 * lexical) / (1.0 + 0.15 * depth))

    def evaluate_index(self, query_index: int, question: str, query_vec: np.ndarray) -> dict[str, Any]:
        seeds = self.seed_entities_for_index(question, query_index, query_vec)
        best_hits: dict[str, MemoryPathHit] = {}
        for candidate in self.reachable_candidates(seeds):
            chunk_id = str(candidate["chunk_id"])
            depth = int(candidate["depth"])
            raw_edges = candidate["path_edges"]
            path_nodes = candidate["path_nodes"]
            blocked_edges = self.blocked_edges_from_raw(raw_edges)
            blocked_nodes = self.blocked_nodes(path_nodes)
            score = self.cached_chunk_score(query_index, chunk_id, depth)
            chunk_data = self.base_graph.nodes[chunk_id]
            hit = MemoryPathHit(
                chunk_id=chunk_id,
                score=score,
                depth=depth,
                question=str(chunk_data.get("question") or ""),
                answer=str(chunk_data.get("answer") or ""),
                path_nodes=path_nodes,
                path_edges=self.edge_refs_from_raw(raw_edges),
                blocked_edges=blocked_edges,
                blocked_nodes=blocked_nodes,
            )
            previous = best_hits.get(chunk_id)
            if previous is None or (hit.score, -hit.depth) > (previous.score, -previous.depth):
                best_hits[chunk_id] = hit

        ranked = sorted(best_hits.values(), key=lambda item: (-item.score, item.depth, item.chunk_id))[: self.max_hits]
        blocked_flags = [hit.blocked for hit in ranked]
        return {
            "memory_seeds": seeds,
            "memory_hits": [self.serialize_hit(hit) for hit in ranked],
            "memory_base_hit_count": float(len(ranked)),
            "memory_top1_path_blocked": float(blocked_flags[0]) if blocked_flags else 0.0,
            "memory_any_path_blocked_at_k": float(any(blocked_flags)),
            "memory_all_paths_blocked_at_k": float(bool(blocked_flags) and all(blocked_flags)),
            "memory_all_paths_safe_at_k": float(not any(blocked_flags)),
            "memory_path_blocked_rate_at_k": float(sum(blocked_flags) / len(blocked_flags)) if blocked_flags else 0.0,
        }

    def evaluate_index_metrics_only(self, query_index: int, question: str, query_vec: np.ndarray) -> dict[str, Any]:
        seeds = self.seed_entities_for_index(question, query_index, query_vec)
        best_hits: dict[str, tuple[float, int, str, int]] = {}
        for candidate in self.reachable_candidates(seeds):
            chunk_id = str(candidate["chunk_id"])
            depth = int(candidate["depth"])
            score = self.cached_chunk_score(query_index, chunk_id, depth)
            blocked = int(
                bool(self.blocked_edges_from_raw(candidate["path_edges"]))
                or bool(self.blocked_nodes(candidate["path_nodes"]))
            )
            current = (score, -depth, chunk_id, blocked)
            previous = best_hits.get(chunk_id)
            if previous is None or (current[0], current[1]) > (previous[0], previous[1]):
                best_hits[chunk_id] = current

        ranked = sorted(best_hits.values(), key=lambda item: (-item[0], -item[1], item[2]))[: self.max_hits]
        blocked_flags = [item[3] for item in ranked]
        return {
            "memory_base_hit_count": float(len(ranked)),
            "memory_top1_path_blocked": float(blocked_flags[0]) if blocked_flags else 0.0,
            "memory_any_path_blocked_at_k": float(any(blocked_flags)),
            "memory_all_paths_blocked_at_k": float(bool(blocked_flags) and all(blocked_flags)),
            "memory_all_paths_safe_at_k": float(not any(blocked_flags)),
            "memory_path_blocked_rate_at_k": float(sum(blocked_flags) / len(blocked_flags)) if blocked_flags else 0.0,
        }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_keys = [
        "rag_top1_unlearned",
        "rag_any_unlearned_at_k",
        "rag_all_unlearned_at_k",
        "rag_all_kept_at_k",
        "rag_unlearned_rate_at_k",
        "rag_answer_support_hit_at_k",
        "rag_safe_answer_support_hit_at_k",
        "rag_masked_answer_support_hit_at_k",
        "rag_answer_support_rate_at_k",
        "rag_graph_risk_rate_at_k",
        "memory_base_hit_count",
        "memory_top1_path_blocked",
        "memory_any_path_blocked_at_k",
        "memory_all_paths_blocked_at_k",
        "memory_all_paths_safe_at_k",
        "memory_path_blocked_rate_at_k",
    ]
    for split in ("forget", "retain"):
        split_rows = [row for row in rows if row["split"] == split]
        values = {"n": float(len(split_rows))}
        for key in metric_keys:
            values[key] = mean(split_rows, key)
        if split == "forget":
            values["rag_mask_hit_at_k"] = values["rag_any_unlearned_at_k"]
            values["memory_path_block_hit_at_k"] = values["memory_any_path_blocked_at_k"]
        else:
            values["rag_keep_hit_at_k"] = values["rag_all_kept_at_k"]
            values["memory_path_safe_hit_at_k"] = values["memory_all_paths_safe_at_k"]
        summary[split] = values
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG node masks and Memory path blocks.")
    parser.add_argument("--rag-graph", default=str(DEFAULT_RAG_GRAPH))
    parser.add_argument("--rag-unlearning-graph", default=str(DEFAULT_RAG_UNLEARNING_GRAPH))
    parser.add_argument("--memory-graph", default=str(DEFAULT_MEMORY_GRAPH))
    parser.add_argument("--memory-unlearning-graph", default=str(DEFAULT_MEMORY_UNLEARNING_GRAPH))
    parser.add_argument("--forget-path", default=str(DEFAULT_FORGET))
    parser.add_argument("--retain-path", default=str(DEFAULT_RETAIN))
    parser.add_argument("--encoder-model-path", default=str(DEFAULT_ENCODER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--forget-limit", type=int, default=400)
    parser.add_argument("--retain-limit", type=int, default=3600)
    parser.add_argument("--embedding-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--embedding-max-length", type=int, default=64)
    parser.add_argument("--question-embedding-cache", default="")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-recall-k", type=int, default=0, help="RAG coarse recall size before optional rerank; 0 means rag-top-k.")
    parser.add_argument("--rag-rerank-mode", default="embedding", choices=["embedding", "hybrid", "graph_hybrid"])
    parser.add_argument("--memory-max-depth", type=int, default=5)
    parser.add_argument("--memory-max-seeds", type=int, default=3)
    parser.add_argument("--memory-max-hits", type=int, default=3)
    parser.add_argument("--memory-entity-sim-floor", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for per-question graph path evaluation.")
    parser.add_argument("--fast-memory-eval", action="store_true", help="Use cached Memory BFS paths and vectorized semantic scores.")
    parser.add_argument("--memory-path-cache-dir", default="", help="Directory for reusable Memory path cache.")
    parser.add_argument("--metrics-only", action="store_true", help="Only write summary metrics; skip per-query detailed path JSON.")
    parser.add_argument("--answer-hit-threshold", type=float, default=0.25, help="Threshold for diagnostic answer-support hit over retrieved RAG node QA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_eval_rows(args)
    print(
        f"[Eval] rows={len(rows)} "
        f"forget={sum(row['split'] == 'forget' for row in rows)} "
        f"retain={sum(row['split'] == 'retain' for row in rows)}"
    )

    question_texts = [row["question"] for row in rows]
    question_cache = Path(args.question_embedding_cache) if args.question_embedding_cache else None
    question_cache_metadata = {
        "texts_sha256": hash_texts(question_texts),
        "text_count": len(question_texts),
        "encoder_model_path": str(args.encoder_model_path),
        "embedding_max_length": int(args.embedding_max_length),
        "embedding_hidden_layer": -1,
        "embedding_pooling": "mean",
        "embedding_4bit": bool(args.embedding_device == "cuda"),
    }
    question_vectors = load_cached_vectors(question_cache, question_cache_metadata)
    if question_vectors is None:
        print("[Eval] loading embedding model...")
        embedder = build_qwen_embeddings(
            model_path=args.encoder_model_path,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            hidden_layer=-1,
            pooling="mean",
            load_in_4bit=(args.embedding_device == "cuda"),
        )
        print("[Eval] embedding questions...")
        question_vectors = np.asarray(embedder.embed_documents(question_texts), dtype=np.float32)
        save_cached_vectors(question_cache, question_cache_metadata, question_vectors)

    print("[Eval] loading graphs...")
    rag_eval = RagNodeMaskEvaluator(
        load_pickle(Path(args.rag_graph)),
        load_pickle(Path(args.rag_unlearning_graph)),
        top_k=args.rag_top_k,
        recall_k=args.rag_recall_k or args.rag_top_k,
        rerank_mode=args.rag_rerank_mode,
        answer_hit_threshold=args.answer_hit_threshold,
    )
    rag_eval.prepare_batch(question_vectors)
    memory_graph = load_pickle(Path(args.memory_graph))
    memory_unlearning_graph = load_pickle(Path(args.memory_unlearning_graph))
    if args.fast_memory_eval:
        memory_eval = CachedMemoryPathEvaluator(
            memory_graph,
            memory_unlearning_graph,
            max_depth=args.memory_max_depth,
            max_seeds=args.memory_max_seeds,
            max_hits=args.memory_max_hits,
            entity_sim_floor=args.memory_entity_sim_floor,
            cache_dir=Path(args.memory_path_cache_dir) if args.memory_path_cache_dir else None,
        )
        memory_eval.prepare_batch(question_vectors, [row["question"] for row in rows])
    else:
        memory_eval = MemoryPathEvaluator(
            memory_graph,
            memory_unlearning_graph,
            max_depth=args.memory_max_depth,
            max_seeds=args.memory_max_seeds,
            max_hits=args.memory_max_hits,
            entity_sim_floor=args.memory_entity_sim_floor,
        )

    def evaluate_one(item: tuple[int, dict[str, str], np.ndarray]) -> dict[str, Any]:
        query_index, row, query_vec = item
        if args.metrics_only:
            rag_metrics = rag_eval.evaluate_index_metrics_only(
                query_index,
                row["question"],
                row["reference_answer"],
                query_vec,
            )
        else:
            rag_metrics = rag_eval.evaluate_index(
                query_index,
                row["question"],
                row["reference_answer"],
                query_vec,
            )
        if isinstance(memory_eval, CachedMemoryPathEvaluator) and args.metrics_only:
            memory_metrics = memory_eval.evaluate_index_metrics_only(query_index, row["question"], query_vec)
        elif isinstance(memory_eval, CachedMemoryPathEvaluator):
            memory_metrics = memory_eval.evaluate_index(query_index, row["question"], query_vec)
        else:
            memory_metrics = memory_eval.evaluate(row["question"], query_vec)
            if args.metrics_only:
                memory_metrics.pop("memory_seeds", None)
                memory_metrics.pop("memory_hits", None)
        if args.metrics_only:
            return {
                "split": row["split"],
                **rag_metrics,
                **memory_metrics,
            }
        return {
            "split": row["split"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "rag_nodes": json.dumps(rag_metrics.pop("rag_nodes"), ensure_ascii=False),
            "memory_seeds": json.dumps(memory_metrics.pop("memory_seeds"), ensure_ascii=False),
            "memory_hits": json.dumps(memory_metrics.pop("memory_hits"), ensure_ascii=False),
            **rag_metrics,
            **memory_metrics,
        }

    eval_items = [(idx, row, question_vectors[idx]) for idx, row in enumerate(rows)]
    num_workers = max(1, int(args.num_workers))
    if isinstance(memory_eval, CachedMemoryPathEvaluator) and num_workers > 1:
        print(f"[Eval] prewarming fast memory path cache before parallel eval workers={num_workers}")
        for query_index, row, query_vec in tqdm(eval_items, desc="Prewarm memory path cache"):
            seeds = memory_eval.seed_entities_for_index(row["question"], query_index, query_vec)
            memory_eval.reachable_candidates(seeds)
        memory_eval.save_cache()
    if num_workers == 1:
        result_rows = [evaluate_one(item) for item in tqdm(eval_items, desc="Graph path eval")]
    else:
        print(f"[Eval] graph path eval workers={num_workers}")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            result_rows = list(tqdm(executor.map(evaluate_one, eval_items), total=len(eval_items), desc="Graph path eval"))
    if isinstance(memory_eval, CachedMemoryPathEvaluator):
        memory_eval.save_cache()

    results_path = output_dir / "graph_path_eval_results.csv"
    if not args.metrics_only:
        with results_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(result_rows)

    summary = {
        "method": "rag_node_mask_and_memory_path_block_eval",
        "rag_graph": str(args.rag_graph),
        "rag_unlearning_graph": str(args.rag_unlearning_graph),
        "memory_graph": str(args.memory_graph),
        "memory_unlearning_graph": str(args.memory_unlearning_graph),
        "forget_path": str(args.forget_path),
        "retain_path": str(args.retain_path),
        "forget_limit": int(args.forget_limit),
        "retain_limit": int(args.retain_limit),
        "rag_top_k": int(args.rag_top_k),
        "rag_recall_k": int(args.rag_recall_k or args.rag_top_k),
        "rag_rerank_mode": str(args.rag_rerank_mode),
        "rag_graph_hybrid_weights": RAG_GRAPH_HYBRID_WEIGHTS if args.rag_rerank_mode == "graph_hybrid" else {},
        "answer_hit_threshold": float(args.answer_hit_threshold),
        "memory_max_depth": int(args.memory_max_depth),
        "memory_max_seeds": int(args.memory_max_seeds),
        "memory_max_hits": int(args.memory_max_hits),
        "memory_entity_sim_floor": float(args.memory_entity_sim_floor),
        "num_workers": int(args.num_workers),
        "fast_memory_eval": bool(args.fast_memory_eval),
        "memory_path_cache_dir": str(args.memory_path_cache_dir),
        "metrics_only": bool(args.metrics_only),
        **summarize(result_rows),
    }
    summary_path = output_dir / "graph_path_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.metrics_only:
        print(f"[Eval] saved: {results_path}")
    print(f"[Eval] saved: {summary_path}")
    print(
        "[Eval] forget "
        f"rag_mask_hit@k={summary['forget']['rag_mask_hit_at_k']:.4f} "
        f"rag_answer_support@k={summary['forget']['rag_answer_support_hit_at_k']:.4f} "
        f"rag_masked_answer_support@k={summary['forget']['rag_masked_answer_support_hit_at_k']:.4f} "
        f"rag_top1_unlearned={summary['forget']['rag_top1_unlearned']:.4f} "
        f"memory_path_block@k={summary['forget']['memory_path_block_hit_at_k']:.4f} "
        f"memory_all_paths_blocked@k={summary['forget']['memory_all_paths_blocked_at_k']:.4f}"
    )
    print(
        "[Eval] retain "
        f"rag_keep_hit@k={summary['retain']['rag_keep_hit_at_k']:.4f} "
        f"rag_answer_support@k={summary['retain']['rag_answer_support_hit_at_k']:.4f} "
        f"rag_safe_answer_support@k={summary['retain']['rag_safe_answer_support_hit_at_k']:.4f} "
        f"rag_top1_unlearned={summary['retain']['rag_top1_unlearned']:.4f} "
        f"memory_path_safe@k={summary['retain']['memory_path_safe_hit_at_k']:.4f} "
        f"memory_any_path_blocked@k={summary['retain']['memory_any_path_blocked_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
