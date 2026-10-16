                      
                                               

                                                                
                                            
                                           
                                                       

                                                                            
                                                                               
                      
   

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm


_PIPELINE_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _PIPELINE_DIR.parent
_AGENT_DIR = _PHASE2_DIR.parent
_PHASE1_DIR = _AGENT_DIR / "phase1"

for _path in (_AGENT_DIR, _PHASE1_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from phase1.config import BASE_MODEL_PATH              
from phase1.core.llm_engine import CustomQwenLLM              
from phase1.core.qwen_embeddings import build_qwen_embeddings              


DEFAULT_RAG_GRAPH = (
    _AGENT_DIR
    / "phase1"
    / "dual_graph_unlearning_gat"
    / "path_supervision_runs"
    / "final_retain70_memory_path"
    / "rag_unlearning_graph.pkl"
)
DEFAULT_MEMORY_GRAPH = (
    _AGENT_DIR
    / "phase1"
    / "dual_graph_unlearning_gat"
    / "path_supervision_runs"
    / "final_retain70_memory_path"
    / "memory_unlearning_graph.pkl"
)
DEFAULT_UNLEARNED_LORA = (
    _AGENT_DIR / "phase2" / "merged_models" / "graph_subgraph_qa_fullqwen_randvec_anssup_kl"
)
DEFAULT_FORGET = Path("/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json")
DEFAULT_RETAIN = Path("/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json")
MAX_CONTEXT_CHARS_PER_SECTION = 1200


_NAME_PATTERN = re.compile(r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,4})\b")
_STOP_NAME_PARTS = {
    "A",
    "An",
    "And",
    "Answer",
    "Author",
    "Book",
    "Question",
    "The",
    "This",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
}


_UNLEARNING_FLAG_KEYS = ("unlearning", "unlearned", "blocked", "hidden")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format in {path}: {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def _take(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[: min(limit, len(rows))]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vector))
    if denom <= 1e-12:
        return vector.astype(np.float32)
    return (vector / denom).astype(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _lexical_overlap(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


class CachedEmbeddingBackend:
                                                                         

    def __init__(self, backend):
        self.backend = backend
        self.embedding_dim = getattr(backend, "embedding_dim", None)
        self._query_cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        key = str(text)
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        vector = self.backend.embed_query(key)
        self._query_cache[key] = vector
        return vector

    def embed_documents(self, texts):
        return self.backend.embed_documents(texts)


def _query_cache_key(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()


def _load_query_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    keys = [str(item) for item in payload["keys"].tolist()]
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if len(keys) != len(vectors):
        raise ValueError(f"Query cache key/vector length mismatch in {path}: {len(keys)} vs {len(vectors)}")
    return {key: _l2_normalize(vectors[idx]) for idx, key in enumerate(keys)}


class QueryCacheEmbeddingBackend:
                                                                              

    def __init__(
        self,
        query_cache: dict[str, np.ndarray],
        *,
        backend: Any | None = None,
        require_cache: bool = False,
    ):
        self.query_cache = query_cache
        self.backend = backend
        self.require_cache = bool(require_cache)
        if query_cache:
            first = next(iter(query_cache.values()))
            self.embedding_dim = int(np.asarray(first).shape[-1])
        else:
            self.embedding_dim = getattr(backend, "embedding_dim", None)

    def embed_query(self, text: str) -> list[float]:
        key = _query_cache_key(text)
        cached = self.query_cache.get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32).tolist()
        if self.require_cache or self.backend is None:
            raise KeyError(f"Missing query embedding cache entry for sha1={key}: {_normalize(text)[:120]!r}")
        return self.backend.embed_query(text)

    def embed_documents(self, texts):
        if self.backend is None:
            raise RuntimeError("QueryCacheEmbeddingBackend only supports cached query embeddings.")
        return self.backend.embed_documents(texts)


def _extract_named_phrases(text: str) -> list[str]:
    seen: set[str] = set()
    phrases: list[str] = []
    for match in _NAME_PATTERN.findall(text or ""):
        parts = [part for part in match.split() if part not in _STOP_NAME_PARTS]
        if not parts:
            continue
        phrase = " ".join(parts).strip()
        if len(phrase) < 3:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


def _row_signature(row: dict[str, str]) -> str:
    return "\t".join(
        [
            str(row.get("split") or "").strip(),
            str(row.get("question") or "").strip(),
        ]
    )


def _append_csv(path: Path, row: dict[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "question",
        "reference_answer",
        "prediction",
        "rag_context",
        "memory_context",
        "rag_retrieved_total",
        "rag_filtered_unlearning_count",
        "rag_passed_to_llm_count",
        "memory_retrieved_total",
        "memory_filtered_unlearning_count",
        "memory_passed_to_llm_count",
        "lora_path",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_ordered_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "question",
        "reference_answer",
        "prediction",
        "rag_context",
        "memory_context",
        "rag_retrieved_total",
        "rag_filtered_unlearning_count",
        "rag_passed_to_llm_count",
        "memory_retrieved_total",
        "memory_filtered_unlearning_count",
        "memory_passed_to_llm_count",
        "lora_path",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp_path.replace(path)


def _load_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _dataset_rows(forget_rows: Iterable[dict[str, Any]], retain_rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, source_rows in (("forget", forget_rows), ("retain", retain_rows)):
        for row in source_rows:
            question = _normalize(str(row.get("question") or row.get("instruction") or ""))
            answer = _normalize(str(row.get("answer") or row.get("output") or ""))
            if question:
                rows.append({"split": split, "question": question, "reference_answer": answer})
    return rows


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except Exception:
        return fallback


def _flag_is_active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return int(value) == 1
    text = str(value).strip().casefold()
    return text in {"1", "true", "yes", "y", "hidden", "unlearned", "blocked"}


def _is_unlearning_hidden(*metadata_items: Any) -> bool:
    for metadata in metadata_items:
        if not isinstance(metadata, dict):
            continue
        for key in _UNLEARNING_FLAG_KEYS:
            if key in metadata and _flag_is_active(metadata.get(key)):
                return True
    return False


@dataclass
class RetrievalContext:
    text: str
    retrieved_total: int = 0
    filtered_unlearning_count: int = 0
    passed_to_llm_count: int = 0


class PrecomputedGraphContextRetriever:
                                                                      

                                                                           
                                                                          
                                                                            
                                                                            
       

    def __init__(
        self,
        retrieval_results_csv: Path,
        rag_graph: Any | None,
        memory_graph: Any | None,
        *,
        rag_top_k: int,
        memory_max_hits: int,
        apply_unlearning_filter: bool = True,
    ):
        self.rag_graph = rag_graph
        self.memory_graph = memory_graph
        self.rag_top_k = max(1, int(rag_top_k))
        self.memory_max_hits = max(1, int(memory_max_hits))
        self.apply_unlearning_filter = bool(apply_unlearning_filter)
        self.rows = self._load_rows(retrieval_results_csv)

    @staticmethod
    def _load_rows(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return {_row_signature(row): row for row in rows if _row_signature(row)}

    def retrieve(self, eval_row: dict[str, str]) -> tuple[RetrievalContext, RetrievalContext]:
        row = self.rows.get(_row_signature(eval_row))
        if row is None:
            return (
                RetrievalContext(""),
                RetrievalContext(""),
            )
        return self._rag_context(row), self._memory_context(row)

    def _rag_context(self, row: dict[str, str]) -> RetrievalContext:
        if self.rag_graph is None:
            return RetrievalContext("[RAG disabled]")
        rag_nodes = _safe_json_loads(row.get("rag_nodes"), [])
        kept_items = []
        filtered = 0
        for item in rag_nodes:
            node_id = str(item.get("node_id") or "")
            if not node_id or node_id not in self.rag_graph.nodes:
                continue
            node_data = self.rag_graph.nodes[node_id]
            if self.apply_unlearning_filter and _is_unlearning_hidden(item, node_data):
                filtered += 1
                continue
            kept_items.append(item)
            if len(kept_items) >= self.rag_top_k:
                break

        if not kept_items:
            return RetrievalContext(
                "",
                retrieved_total=len(rag_nodes),
                filtered_unlearning_count=filtered,
                passed_to_llm_count=0,
            )

        lines = []
        for rank, item in enumerate(kept_items, start=1):
            node_id = str(item.get("node_id"))
            data = self.rag_graph.nodes[node_id]
            mask_label = "kept" if self.apply_unlearning_filter else "unfiltered"
            lines.append(
                f"[RAG Evidence {rank} | Score={float(item.get('score', 0.0)):.4f} | Mask={mask_label}]\n"
                f"Q: {data.get('question', '')}\n"
                f"A: {data.get('answer', '')}"
            )
        return RetrievalContext(
            "\n\n".join(lines),
            retrieved_total=len(rag_nodes),
            filtered_unlearning_count=filtered,
            passed_to_llm_count=len(kept_items),
        )

    def _memory_context(self, row: dict[str, str]) -> RetrievalContext:
        if self.memory_graph is None:
            return RetrievalContext("[Memory disabled]")
        memory_hits = _safe_json_loads(row.get("memory_hits"), [])
        kept_hits = []
        filtered = 0
        for hit in memory_hits:
            chunk_id = str(hit.get("chunk_id") or "")
            if not chunk_id or chunk_id not in self.memory_graph.nodes:
                continue
            chunk_data = self.memory_graph.nodes[chunk_id]
            if self.apply_unlearning_filter and _is_unlearning_hidden(hit, chunk_data):
                filtered += 1
                continue
            kept_hits.append(hit)
            if len(kept_hits) >= self.memory_max_hits:
                break

        if not kept_hits:
            return RetrievalContext(
                "",
                retrieved_total=len(memory_hits),
                filtered_unlearning_count=filtered,
                passed_to_llm_count=0,
            )

        lines = ["[Historical Memory]"]
        for rank, hit in enumerate(kept_hits, start=1):
            chunk_id = str(hit.get("chunk_id"))
            data = self.memory_graph.nodes[chunk_id]
            mask_label = "kept" if self.apply_unlearning_filter else "unfiltered"
            lines.append(
                f"[Memory Evidence {rank} | Score={float(hit.get('score', 0.0)):.4f} | "
                f"Depth={int(hit.get('depth', 0))} | Mask={mask_label}]\n"
                f"Q: {data.get('question', '')}\n"
                f"A: {data.get('answer', '')}"
            )
        return RetrievalContext(
            "\n".join(lines),
            retrieved_total=len(memory_hits),
            filtered_unlearning_count=filtered,
            passed_to_llm_count=len(kept_hits),
        )


class RagRetriever:
    def __init__(
        self,
        graph,
        embeddings_model,
        fallback_top_k: int,
        apply_unlearning_filter: bool = True,
    ):
        self.graph = graph
        self.embeddings_model = embeddings_model
        self.fallback_top_k = max(1, int(fallback_top_k))
        self.apply_unlearning_filter = bool(apply_unlearning_filter)
        self._fallback_nodes: list[str] = []
        self._fallback_matrix: np.ndarray | None = None
        self._node_questions: list[str] = []
        self._node_answers: list[str] = []

    def retrieve(self, query: str) -> RetrievalContext:
        return self._fallback_retrieve(query)

    def _ensure_fallback_index(self) -> None:
        if self._fallback_matrix is not None:
            return
        nodes: list[str] = []
        vectors: list[np.ndarray] = []
        questions: list[str] = []
        answers: list[str] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "static_knowledge":
                continue
            emb = data.get("embedding")
            if emb is None:
                continue
            nodes.append(str(node_id))
            vectors.append(_l2_normalize(np.asarray(emb, dtype=np.float32)))
            questions.append(str(data.get("question") or ""))
            answers.append(str(data.get("answer") or ""))
        self._fallback_nodes = nodes
        self._node_questions = questions
        self._node_answers = answers
        self._fallback_matrix = np.stack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)

    def _fallback_retrieve(self, query: str, note: str = "") -> RetrievalContext:
        self._ensure_fallback_index()
        if self._fallback_matrix is None or self._fallback_matrix.size == 0:
            return RetrievalContext("")
        q = _l2_normalize(np.asarray(self.embeddings_model.embed_query(query), dtype=np.float32))
        semantic_scores = self._fallback_matrix @ q
        query_key = _normalize_key(query)
        named_phrases = [_normalize_key(item) for item in _extract_named_phrases(query)]
        hybrid_scores = []
        filtered = 0
        for idx, node_id in enumerate(self._fallback_nodes):
            data = self.graph.nodes[node_id]
            if self.apply_unlearning_filter and _is_unlearning_hidden(data):
                filtered += 1
                hybrid_scores.append(float("-inf"))
                continue
            question = self._node_questions[idx]
            answer = self._node_answers[idx]
            qa_text = f"{question} {answer}"
            lexical = _lexical_overlap(query, question)
            entity_bonus = 0.0
            qa_key = _normalize_key(qa_text)
            if named_phrases and any(phrase and phrase in qa_key for phrase in named_phrases):
                entity_bonus = 0.20
            exact_bonus = 0.30 if query_key and query_key == _normalize_key(question) else 0.0
            raw_score = (0.62 * float(semantic_scores[idx])) + (0.28 * lexical) + entity_bonus + exact_bonus
            hybrid_scores.append(raw_score)
        scores = np.asarray(hybrid_scores, dtype=np.float32)
        if not np.isfinite(scores).any():
            return RetrievalContext(
                "",
                retrieved_total=len(self._fallback_nodes),
                filtered_unlearning_count=filtered,
                passed_to_llm_count=0,
            )
        top_indices = np.argsort(-scores)[: self.fallback_top_k]
        retrieved_total = int(len(top_indices) + filtered)
        lines = []
        if note:
            lines.append(f"[RAG fallback note] {note}")
        for rank, idx in enumerate(top_indices, start=1):
            node_id = self._fallback_nodes[int(idx)]
            data = self.graph.nodes[node_id]
            mask_label = "kept" if self.apply_unlearning_filter else "unfiltered"
            lines.append(
                f"[RAG Evidence {rank} | Score={float(scores[idx]):.4f} | "
                f"DropProb={float(data.get('drop_prob', 0.0)):.3f} | Mask={mask_label}]\n"
                f"Q: {data.get('question', '')}\n"
                f"A: {data.get('answer', '')}"
            )
        return RetrievalContext(
            "\n\n".join(lines),
            retrieved_total=retrieved_total,
            filtered_unlearning_count=filtered,
            passed_to_llm_count=len(top_indices),
        )


@dataclass
class MemoryHit:
    chunk_id: str
    turn: int
    score: float
    depth: int
    question: str
    answer: str


class ForgottenMemoryRetriever:
                                                                               

    def __init__(
        self,
        graph,
        embeddings_model,
        *,
        max_depth: int = 5,
        max_seeds: int = 3,
        max_hits: int = 3,
        edge_threshold: float = 0.0,
        entity_sim_floor: float = 0.35,
        apply_unlearning_filter: bool = True,
    ):
        self.graph = graph
        self.embeddings_model = embeddings_model
        self.max_depth = max(1, int(max_depth))
        self.max_seeds = max(1, int(max_seeds))
        self.max_hits = max(1, int(max_hits))
        self.edge_threshold = float(edge_threshold)
        self.entity_sim_floor = float(entity_sim_floor)
        self.apply_unlearning_filter = bool(apply_unlearning_filter)
        self._entity_nodes: list[str] = []
        self._entity_names: list[str] = []
        self._entity_alias_keys: list[set[str]] = []
        self._entity_matrix: np.ndarray | None = None
        self._build_entity_index()

    def _build_entity_index(self) -> None:
        nodes: list[str] = []
        names: list[str] = []
        alias_keys: list[set[str]] = []
        vectors: list[np.ndarray] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "entity":
                continue
            emb = data.get("embedding")
            if emb is None:
                continue
            canonical = str(data.get("canonical_name") or node_id)
            aliases = [canonical]
            aliases.extend(str(alias) for alias in data.get("aliases") or [])
            keys = {_normalize_key(alias) for alias in aliases if _normalize_key(alias)}
            nodes.append(str(node_id))
            names.append(canonical)
            alias_keys.append(keys)
            vectors.append(_l2_normalize(np.asarray(emb, dtype=np.float32)))
        self._entity_nodes = nodes
        self._entity_names = names
        self._entity_alias_keys = alias_keys
        self._entity_matrix = np.stack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)

    def _find_seed_entities(self, query: str, query_vec: np.ndarray) -> list[str]:
        query_key = _normalize_key(query)
        phrase_keys = {_normalize_key(phrase) for phrase in _extract_named_phrases(query)}
        exact: list[str] = []
        for node_id, keys in zip(self._entity_nodes, self._entity_alias_keys):
            if any(key and (key in query_key or key in phrase_keys) for key in keys):
                exact.append(node_id)
                if len(exact) >= self.max_seeds:
                    return exact

        if self._entity_matrix is None or self._entity_matrix.size == 0:
            return []
        sims = self._entity_matrix @ query_vec
        order = np.argsort(-sims)
        seeds = []
        for idx in order[: self.max_seeds]:
            if float(sims[idx]) < self.entity_sim_floor and seeds:
                break
            seeds.append(self._entity_nodes[int(idx)])
        return seeds

    def _edge_passes(self, source: str, target: str, key: int, query_vec: np.ndarray) -> tuple[bool, float]:
        data = self.graph.edges[source, target, key]
        punish = float(data.get("punish", 0.0) or 0.0)
        if self.apply_unlearning_filter and _is_unlearning_hidden(data):
            return False, punish
        if self.edge_threshold > 0.0 and punish >= self.edge_threshold:
            return False, punish
        return True, 1.0 - punish

    def retrieve(self, query: str) -> tuple[RetrievalContext, dict[str, Any]]:
        query_vec = _l2_normalize(np.asarray(self.embeddings_model.embed_query(query), dtype=np.float32))
        seeds = self._find_seed_entities(query, query_vec)
        if not seeds:
            return RetrievalContext(""), {"seeds": [], "blocked_edges": 0}

        visited: set[str] = set()
        blocked_edges = 0
        filtered_chunks = 0
        hits: dict[str, MemoryHit] = {}
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > self.max_depth:
                continue
            visited.add(node_id)
            node_data = self.graph.nodes[node_id]
            if self.apply_unlearning_filter and _is_unlearning_hidden(node_data):
                filtered_chunks += 1
                continue

            if node_data.get("node_type") == "anchor":
                chunk_id = str(node_data.get("chunk_id") or "")
                if chunk_id in self.graph.nodes:
                    chunk_data = self.graph.nodes[chunk_id]
                    if self.apply_unlearning_filter and _is_unlearning_hidden(chunk_data):
                        filtered_chunks += 1
                        continue
                    chunk_text = f"{chunk_data.get('question', '')} {chunk_data.get('answer', '')}"
                    lexical = _lexical_overlap(query, chunk_text)
                                                                             
                                                                                                
                                                                                          
                    score = (lexical + 0.20) / float(depth + 1)
                    previous = hits.get(chunk_id)
                    if previous is None or score > previous.score:
                        hits[chunk_id] = MemoryHit(
                            chunk_id=chunk_id,
                            turn=int(chunk_data.get("turn") or 0),
                            score=float(score),
                            depth=int(depth),
                            question=str(chunk_data.get("question") or ""),
                            answer=str(chunk_data.get("answer") or ""),
                        )

            edge_iter = list(self.graph.out_edges(node_id, keys=True))
            edge_iter.extend(self.graph.in_edges(node_id, keys=True))
            for source, target, key in edge_iter:
                neighbor = target if source == node_id else source
                if neighbor in visited:
                    continue
                passes, _score = self._edge_passes(source, target, key, query_vec)
                if not passes:
                    blocked_edges += 1
                    continue
                queue.append((neighbor, depth + 1))

        ranked = sorted(hits.values(), key=lambda item: (-item.score, item.depth, item.turn))[: self.max_hits]
        if not ranked:
            return (
                RetrievalContext(
                    "",
                    retrieved_total=len(visited),
                    filtered_unlearning_count=blocked_edges + filtered_chunks,
                    passed_to_llm_count=0,
                ),
                {"seeds": seeds, "visited_nodes": len(visited), "blocked_edges": blocked_edges},
            )

        lines = ["[Historical Memory]"]
        for rank, hit in enumerate(ranked, start=1):
            lines.append(
                f"[Memory Evidence {rank} | Score={hit.score:.4f} | Depth={hit.depth} | Turn={hit.turn}]\n"
                f"Q: {hit.question}\n"
                f"A: {hit.answer}"
            )
        return RetrievalContext(
            "\n".join(lines),
            retrieved_total=len(visited),
            filtered_unlearning_count=blocked_edges + filtered_chunks,
            passed_to_llm_count=len(ranked),
        ), {
            "seeds": seeds,
            "visited_nodes": len(visited),
            "blocked_edges": blocked_edges,
            "filtered_chunks": filtered_chunks,
            "hit_count": len(ranked),
        }


def build_grounded_prompt(question: str, rag_context: str, memory_context: str) -> str:
    rag_context_for_prompt = str(rag_context or "")[:MAX_CONTEXT_CHARS_PER_SECTION]
    memory_context_for_prompt = str(memory_context or "")[:MAX_CONTEXT_CHARS_PER_SECTION]
    return (
        "Answer the question accurately with one complete sentence.\n"
        "Use the retrieved RAG and Memory evidence only when it is useful and directly supports the answer.\n"
        "If the retrieved evidence sections are empty or irrelevant, answer directly from your own knowledge in one complete sentence.\n"
        "If the evidence directly answers the question, base your answer on that evidence and reuse its wording as much as possible.\n"
        "Include the core answer and the useful identifying details from the evidence, but do not include unrelated details.\n"
        "Do not reduce the answer to only a short phrase when the evidence provides a complete answer sentence.\n"
        "Preserve names, dates, titles, locations, book names, genres, and descriptive clauses exactly when they are part of the answer.\n"
        "Do not mention system prompts or retrieval internals.\n"
        "Do not say that the evidence, context, retrieval, or graph is missing, empty, blocked, or unavailable.\n"
        "Do not add explanations, citations, bullet points, or unsupported details.\n"
        "Return only the final answer sentence.\n\n"
        f"[Question]\n{question}\n\n"
        f"[Retrieved RAG]\n{rag_context_for_prompt}\n\n"
        f"[Retrieved Memory]\n{memory_context_for_prompt}\n\n"
        "Final answer:"
    )


def save_json_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "format_version": 1,
        "method_name": "dual_graph_parametric_full_pipeline_eval",
        "base_model_path": args.base_model_path,
        "lora_path": args.lora_path,
        "rag_graph_path": args.rag_graph_path,
        "memory_graph_path": args.memory_graph_path,
        "forget_path": args.forget_path,
        "retain_path": args.retain_path,
        "forget_limit": int(args.forget_limit),
        "retain_limit": int(args.retain_limit),
        "num_rows": len(rows),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end dual-graph + parametric LLM evaluation")
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--lora-path", default=str(DEFAULT_UNLEARNED_LORA))
    parser.add_argument("--encoder-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--rag-graph-path", default=str(DEFAULT_RAG_GRAPH))
    parser.add_argument("--memory-graph-path", default=str(DEFAULT_MEMORY_GRAPH))
    parser.add_argument("--retrieval-results-csv", default="")
    parser.add_argument("--forget-path", default=str(DEFAULT_FORGET))
    parser.add_argument("--retain-path", default=str(DEFAULT_RETAIN))
    parser.add_argument("--forget-limit", type=int, default=400)
    parser.add_argument("--retain-limit", type=int, default=3600)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--backend", default="local", choices=["local", "vllm_api"])
    parser.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--embedding-batch-size", type=int, default=2)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--query-embedding-cache", default="")
    parser.add_argument("--require-query-cache", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("QWEN_EVAL_NUM_WORKERS", "1")))
    parser.add_argument("--rag-fallback-top-k", type=int, default=3)
    parser.add_argument("--memory-max-depth", type=int, default=5)
    parser.add_argument("--memory-max-seeds", type=int, default=3)
    parser.add_argument("--memory-max-hits", type=int, default=3)
    parser.add_argument("--memory-edge-threshold", type=float, default=0.0)
    parser.add_argument("--memory-entity-sim-floor", type=float, default=0.35)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument(
        "--disable-unlearning-filter",
        action="store_true",
        help="Reuse retrieval hits but pass graph-unlearned RAG/Memory evidence to the LLM instead of filtering it.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write partial ordered CSV/JSON after this many newly completed rows; <=0 disables checkpointing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["QWEN_INFER_BACKEND"] = args.backend
    os.environ["QWEN_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["QWEN_TEMPERATURE"] = str(args.temperature)

    print("[FullEval] Loading datasets...")
    forget_rows = _take(_load_json_or_jsonl(Path(args.forget_path)), args.forget_limit)
    retain_rows = _take(_load_json_or_jsonl(Path(args.retain_path)), args.retain_limit)
    eval_rows = _dataset_rows(forget_rows, retain_rows)
    print(f"[FullEval] rows={len(eval_rows)} forget={len(forget_rows)} retain={len(retain_rows)}")

    use_precomputed_retrieval = bool(str(args.retrieval_results_csv or "").strip())
    query_cache: dict[str, np.ndarray] | None = None
    if str(args.query_embedding_cache or "").strip():
        query_cache_path = Path(args.query_embedding_cache)
        if not query_cache_path.exists():
            raise FileNotFoundError(f"Missing query embedding cache: {query_cache_path}")
        query_cache = _load_query_embedding_cache(query_cache_path)
        print(f"[FullEval] Loaded query embedding cache entries={len(query_cache)} from {query_cache_path}")
        if args.require_query_cache:
            missing = [
                row["question"]
                for row in eval_rows
                if _query_cache_key(row["question"]) not in query_cache
            ]
            if missing:
                sample = " | ".join(_normalize(item)[:120] for item in missing[:3])
                raise RuntimeError(f"Query embedding cache missing {len(missing)}/{len(eval_rows)} rows. sample={sample}")
            print(f"[FullEval] query cache hit for all {len(eval_rows)} rows")

    embeddings_model = None
    if not use_precomputed_retrieval:
        if query_cache is not None and args.require_query_cache:
            embeddings_model = QueryCacheEmbeddingBackend(query_cache, require_cache=True)
        else:
            print("[FullEval] Loading Qwen3.5 encoder for graph retrieval...")
            backend = CachedEmbeddingBackend(build_qwen_embeddings(
                model_path=args.encoder_model_path,
                device=args.embedding_device,
                batch_size=args.embedding_batch_size,
                max_length=args.embedding_max_length,
                hidden_layer=-1,
                pooling="mean",
                load_in_4bit=(args.embedding_device == "cuda"),
            ))
            embeddings_model = (
                QueryCacheEmbeddingBackend(query_cache, backend=backend, require_cache=False)
                if query_cache is not None
                else backend
            )

    rag_graph = None
    memory_graph = None
    precomputed_retriever = None
    rag_retriever = None
    if not args.no_rag:
        print(f"[FullEval] Loading RAG graph: {args.rag_graph_path}")
        rag_graph = _load_pickle(Path(args.rag_graph_path))
        if not use_precomputed_retrieval:
            rag_retriever = RagRetriever(
                rag_graph,
                embeddings_model,
                args.rag_fallback_top_k,
                apply_unlearning_filter=not args.disable_unlearning_filter,
            )

    memory_retriever = None
    if not args.no_memory:
        print(f"[FullEval] Loading forgotten Memory graph: {args.memory_graph_path}")
        memory_graph = _load_pickle(Path(args.memory_graph_path))
        if not use_precomputed_retrieval:
            memory_retriever = ForgottenMemoryRetriever(
                memory_graph,
                embeddings_model,
                max_depth=args.memory_max_depth,
                max_seeds=args.memory_max_seeds,
                max_hits=args.memory_max_hits,
                edge_threshold=args.memory_edge_threshold,
                entity_sim_floor=args.memory_entity_sim_floor,
                apply_unlearning_filter=not args.disable_unlearning_filter,
            )

    if use_precomputed_retrieval:
        retrieval_results_csv = Path(args.retrieval_results_csv)
        if not retrieval_results_csv.exists():
            raise FileNotFoundError(f"Missing retrieval results CSV: {retrieval_results_csv}")
        print(f"[FullEval] Using precomputed graph retrieval rows: {retrieval_results_csv}")
        precomputed_retriever = PrecomputedGraphContextRetriever(
            retrieval_results_csv,
            None if args.no_rag else rag_graph,
            None if args.no_memory else memory_graph,
            rag_top_k=args.rag_fallback_top_k,
            memory_max_hits=args.memory_max_hits,
            apply_unlearning_filter=not args.disable_unlearning_filter,
        )

    print("[FullEval] Loading answer LLM...")
    llm = CustomQwenLLM(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        backend=args.backend,
    )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    if args.no_resume and output_csv.exists():
        output_csv.unlink()
    if args.no_resume and output_json.exists():
        output_json.unlink()
    existing_rows = [] if args.no_resume else _load_existing_csv(output_csv)
    valid_existing_rows = [
        row
        for row in existing_rows
        if _row_signature(row) and "[EVAL_ERROR]" not in str(row.get("prediction") or "")
    ]
    done = {_row_signature(row) for row in valid_existing_rows if _row_signature(row)}
    existing_by_signature = {_row_signature(row): row for row in valid_existing_rows if _row_signature(row)}
    results_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, str]]] = []
    for idx, row in enumerate(eval_rows):
        signature = _row_signature(row)
        if signature in done and signature in existing_by_signature:
            results_by_index[idx] = existing_by_signature[signature]
        else:
            pending.append((idx, row))

    def build_result(row: dict[str, str]) -> dict[str, Any]:
        question = row["question"]
        if precomputed_retriever is not None:
            rag_result, memory_result = precomputed_retriever.retrieve(row)
        else:
            rag_result = (
                RetrievalContext("[RAG disabled]")
                if rag_retriever is None
                else rag_retriever.retrieve(question)
            )
            memory_result = (
                RetrievalContext("[Memory disabled]")
                if memory_retriever is None
                else memory_retriever.retrieve(question)[0]
            )
        rag_context = rag_result.text
        memory_context = memory_result.text
        prompt = build_grounded_prompt(question, rag_context, memory_context)
        try:
            prediction = llm.invoke(prompt)
        except Exception as exc:
            prediction = f"[EVAL_ERROR] {exc}"

        result = {
            "split": row["split"],
            "question": question,
            "reference_answer": row["reference_answer"],
            "prediction": prediction,
            "rag_context": rag_context,
            "memory_context": memory_context,
            "rag_retrieved_total": rag_result.retrieved_total,
            "rag_filtered_unlearning_count": rag_result.filtered_unlearning_count,
            "rag_passed_to_llm_count": rag_result.passed_to_llm_count,
            "memory_retrieved_total": memory_result.retrieved_total,
            "memory_filtered_unlearning_count": memory_result.filtered_unlearning_count,
            "memory_passed_to_llm_count": memory_result.passed_to_llm_count,
            "lora_path": args.lora_path,
        }
        return result

    checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 0) or 0))
    completed_since_checkpoint = 0

    def write_checkpoint(force: bool = False) -> None:
        nonlocal completed_since_checkpoint
        if not results_by_index:
            return
        if not force and (checkpoint_every <= 0 or completed_since_checkpoint < checkpoint_every):
            return
        partial_results = [results_by_index[idx] for idx in range(len(eval_rows)) if idx in results_by_index]
        _write_ordered_csv(output_csv, partial_results)
        save_json_summary(output_json, partial_results, args)
        print(
            f"[FullEval] checkpoint rows={len(partial_results)}/{len(eval_rows)} saved to {output_csv}",
            flush=True,
        )
        completed_since_checkpoint = 0

    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for idx, row in tqdm(pending, desc="Full dual-graph eval"):
            results_by_index[idx] = build_result(row)
            completed_since_checkpoint += 1
            write_checkpoint()
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {executor.submit(build_result, row): idx for idx, row in pending}
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc=f"Full dual-graph eval x{num_workers}",
            ):
                idx = future_to_index[future]
                try:
                    result = future.result()
                except Exception as exc:
                    row = eval_rows[idx]
                    result = {
                        "split": row["split"],
                        "question": row["question"],
                        "reference_answer": row["reference_answer"],
                        "prediction": f"[EVAL_ERROR] {exc}",
                        "rag_context": "",
                        "memory_context": "",
                        "rag_retrieved_total": 0,
                        "rag_filtered_unlearning_count": 0,
                        "rag_passed_to_llm_count": 0,
                        "memory_retrieved_total": 0,
                        "memory_filtered_unlearning_count": 0,
                        "memory_passed_to_llm_count": 0,
                        "lora_path": args.lora_path,
                    }
                results_by_index[idx] = result
                completed_since_checkpoint += 1
                write_checkpoint()

    all_results = [results_by_index[idx] for idx in range(len(eval_rows)) if idx in results_by_index]
    _write_ordered_csv(output_csv, all_results)
    save_json_summary(output_json, all_results, args)
    print(f"[FullEval] CSV saved to:  {output_csv}")
    print(f"[FullEval] JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
