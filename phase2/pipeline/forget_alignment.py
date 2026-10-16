                                   

                                
                                    
   

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset

logger = logging.getLogger("GAT-AFRU.Phase2.ForgetAlignment")


def load_json_or_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
            if isinstance(payload, list):
                return payload
            raise ValueError(f"Expected list payload in {path}, got {type(payload).__name__}.")
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_question(text: str) -> str:
    return " ".join(
        str(text)
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .strip()
        .lower()
        .split()
    )


def build_question_to_node_index(graph) -> Dict[str, list[str]]:
    index: Dict[str, list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        question = str(data.get("question") or "").strip()
        if not question:
            continue
        index.setdefault(normalize_question(question), []).append(str(node_id))
    return index


def align_forget_targets(
    *,
    graph_path: Path,
    forget_target_path: Path,
    weight_path: Path | None = None,
    strict: bool = True,
) -> list[Dict[str, Any]]:
    graph = load_pickle(graph_path)
    question_index = build_question_to_node_index(graph)
    weight_map = load_pickle(weight_path) if weight_path and weight_path.exists() else {}
    target_rows = load_json_or_jsonl(forget_target_path)

    aligned: list[Dict[str, Any]] = []
    missing_questions: list[str] = []
    for row_idx, row in enumerate(target_rows):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        if not question:
            raise ValueError(f"Forget target row #{row_idx} in {forget_target_path} is missing a question field.")
        node_ids = question_index.get(normalize_question(question), [])
        if not node_ids:
            missing_questions.append(question)
            continue
        node_id = node_ids[0]
        node_data = graph.nodes[node_id]
        aligned.append(
            {
                "example_id": f"aligned_forget_{len(aligned)}",
                "node_id": str(node_id),
                "question": question,
                "graph_question": str(node_data.get("question") or "").strip(),
                "graph_answer": str(node_data.get("answer") or "").strip(),
                "reference_answer": str(
                    row.get("answer") or row.get("output") or row.get("reference_answer") or node_data.get("answer") or ""
                ).strip(),
                "community": str(node_data.get("community", "unknown")),
                "gat_weight": float(weight_map.get(node_id, 1.0)),
            }
        )

    if missing_questions:
        preview = "\n".join(f"- {question}" for question in missing_questions[:10])
        message = (
            f"Failed to align {len(missing_questions)} forget questions from {forget_target_path} to graph nodes.\n"
            f"Preview:\n{preview}"
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    logger.info(
        "Aligned %d/%d forget targets from %s to graph nodes.",
        len(aligned),
        len(target_rows),
        forget_target_path,
    )
    return aligned


def build_forget_datasets_from_aligned_targets(
    *,
    graph_path: Path,
    aligned_targets: list[Dict[str, Any]],
    refusal_text: str,
    max_neighbors: int,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    graph = load_pickle(graph_path)
    rng = random.Random(seed)

    core_rows: list[Dict[str, Any]] = []
    context_rows: list[Dict[str, Any]] = []
    for idx, target in enumerate(aligned_targets):
        question = str(target["question"]).strip()
        node_id = str(target["node_id"])
        answer = str(target["graph_answer"]).strip()
        weight = float(target["gat_weight"])
        core_rows.append(
            {
                "instruction": question,
                "input": "",
                "output": refusal_text,
                "original_output": answer,
                "gat_weight": weight,
                "node_id": node_id,
                "community": str(target.get("community", "unknown")),
                "example_id": f"forget_{idx}",
                "source": "aligned_core_direct",
            }
        )

        neighbors = list(graph.neighbors(node_id))
        rng.shuffle(neighbors)
        fragments = [f"Core fact: {answer}"]
        for neighbor_id in neighbors[:max_neighbors]:
            neighbor_answer = str(graph.nodes[neighbor_id].get("answer") or "").strip()
            if neighbor_answer:
                fragments.append(f"Related fact: {neighbor_answer}")
        context_rows.append(
            {
                "instruction": (
                    "Reference profile:\n"
                    f"{chr(10).join(fragments)}\n\n"
                    "Answer the user question using the profile above.\n"
                    f"Question: {question}"
                ),
                "input": "",
                "output": refusal_text,
                "original_output": answer,
                "gat_weight": weight,
                "node_id": node_id,
                "community": str(target.get("community", "unknown")),
                "example_id": f"forget_ctx_{idx}",
                "source": "aligned_context_adversarial",
            }
        )

    return Dataset.from_list(core_rows), Dataset.from_list(context_rows)


def summarize_aligned_targets(aligned_targets: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    aligned_targets = list(aligned_targets)
    communities = sorted({str(item.get("community", "unknown")) for item in aligned_targets})
    return {
        "num_targets": int(len(aligned_targets)),
        "communities": communities[:20],
    }
