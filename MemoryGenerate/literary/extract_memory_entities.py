                      
                                                                             

                                                                              
                                                        

                                                                  

                                                               
                                                                         
                                                             
                                                                  
                                                                   
                                                      
                                                                  
   

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/home/likunhao/likunhao52shared/datasets/LiteraryQA/literaryqa_annotations/memory/full.json")
DEFAULT_DOCS_DIR = Path("/home/likunhao/likunhao52shared/datasets/LiteraryQA/literaryqa_annotations")
DEFAULT_OUTPUT_DIR = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/MemoryGenerate/literary/artifacts"
)

DEFAULT_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.environ.get("TOP_P", "0.95"))
DEFAULT_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
DEFAULT_RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "10"))


@dataclass
class QAChunk:
    turn: int
    question: str
    answer: str
    raw: dict[str, Any]


@dataclass
class DocumentBlock:
    batch_index: int
    document_id: str
    gutenberg_id: str
    title: str
    split: str
    summary: str
    metadata: dict[str, Any]
    chunks: list[QAChunk]


@dataclass
class Extraction:
    turn: int
    entities: list[dict[str, Any]] = field(default_factory=list)
    mentioned_entities: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON array in {path}")
        return [row for row in obj if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(obj)
    return rows


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def load_document_blocks(docs_dir: Path, flat_rows: list[dict[str, Any]]) -> list[DocumentBlock]:
    document_paths = [docs_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    documents: list[dict[str, Any]] = []
    for path in document_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing LiteraryQA source file: {path}")
        for row in load_json_or_jsonl(path):
            documents.append(row)

    flattened_from_docs: list[dict[str, str]] = []
    blocks: list[DocumentBlock] = []
    for doc in documents:
        qas = []
        turn = 1
        for qa in doc.get("qas", []) or []:
            answers = qa.get("answers") or []
            answer = normalize_text(answers[0]) if answers else ""
            question = normalize_text(qa.get("question") or qa.get("prompt"))
            if not question and not answer:
                continue
            chunk = QAChunk(
                turn=turn,
                question=question,
                answer=answer,
                raw={
                    "document_id": doc.get("document_id"),
                    "gutenberg_id": doc.get("gutenberg_id"),
                    "title": doc.get("title"),
                    "summary": doc.get("summary"),
                    "metadata": doc.get("metadata", {}),
                    "source_split": doc.get("split"),
                },
            )
            qas.append(chunk)
            flattened_from_docs.append({"question": question, "answer": answer})
            turn += 1
        if not qas:
            continue
        blocks.append(
            DocumentBlock(
                batch_index=len(blocks),
                document_id=str(doc.get("document_id") or "").strip(),
                gutenberg_id=str(doc.get("gutenberg_id") or "").strip(),
                title=normalize_text(doc.get("title")),
                split=str(doc.get("split") or "").strip(),
                summary=normalize_text(doc.get("summary")),
                metadata=doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {},
                chunks=qas,
            )
        )

    if not blocks:
        raise ValueError(f"No LiteraryQA documents loaded from {docs_dir}")

    if len(flattened_from_docs) != len(flat_rows):
        raise ValueError(
            f"Document QA count mismatch: docs flatten to {len(flattened_from_docs)} rows, input file has {len(flat_rows)} rows"
        )

    for idx, (doc_row, flat_row) in enumerate(zip(flattened_from_docs, flat_rows), start=1):
        if normalize_text(doc_row.get("question")) != normalize_text(flat_row.get("question")):
            raise ValueError(f"Question order mismatch at row {idx}")
        if normalize_text(doc_row.get("answer")) != normalize_text(flat_row.get("answer")):
            raise ValueError(f"Answer order mismatch at row {idx}")

    return blocks


def build_system_prompt() -> str:
    return """You are a strict information extraction system for a memory graph.

The target graph schema is fixed:
- Chunk and anchor nodes are created automatically from each QA turn.
- Entity nodes are created only from your canonical entity names.
- mentioned_in edges are created from each turn's mentioned_entities.
- relation edges are created from each turn's relations.
- next_turn edges are created automatically by the graph builder.

Your job is to extract only canonical entities, per-turn mentioned entities, and entity-to-entity relations from a contiguous LiteraryQA QA block.

Rules:
1. Read the whole block before extracting each turn. Resolve pronouns and short names using the block context.
2. Canonicalize repeated mentions. For example, "Thorne", "Isabel", and "Isabel Thorne" should be one canonical entity if they refer to the same character.
3. Preserve important literary entities: work titles, characters, organizations, places, objects, events, contracts, weapons, roles, themes, nationalities, and dates.
4. Extract only information explicitly stated by the question/answer or directly recoverable by coreference within this block. Do not invent facts.
5. If a QA says the evidence is absent, unknown, or not provided, do not create factual relations from the denied premise. You may still include explicitly mentioned named entities.
6. Relation predicates must be English snake_case, such as works_for, has_ethnicity, sibling_of, invented, prevents, loves, located_in, has_role, seeks, protects, kills, reunites_with, is_about, has_theme.
7. Do not output pronouns or generic references such as he, she, it, they, this person, the story, unknown as entities.
8. Output strict JSON only. No Markdown and no explanations.

Required JSON shape:
{
  "entities": [
    {
      "name": "Isabel Thorne",
      "type": "person|work|organization|place|object|event|contract|weapon|role|theme|nationality|date|concept|unknown",
      "aliases": ["Thorne", "Isabel"]
    }
  ],
  "turn_extractions": [
    {
      "turn": 1,
      "mentioned_entities": ["Isabel Thorne", "Italian Secret Service"],
      "relations": [
        {
          "subject": "Isabel Thorne",
          "predicate": "works_for",
          "object": "Italian Secret Service",
          "confidence": 0.95
        }
      ]
    }
  ]
}

Field constraints:
- entities is the canonical entity inventory for the whole input block.
- turn_extractions must contain every input turn exactly once.
- mentioned_entities and relation subject/object must use names from entities.
- confidence must be a number in [0, 1]."""


def build_user_prompt(block: DocumentBlock) -> str:
    turns = [
        {
            "turn": chunk.turn,
            "question": chunk.question,
            "answer": chunk.answer,
        }
        for chunk in block.chunks
    ]
    metadata = block.metadata or {}
    return f"""batch_index: {block.batch_index}
document_id: {block.document_id}
gutenberg_id: {block.gutenberg_id}
title: {block.title}
split: {block.split}
turn_start: {block.chunks[0].turn}
turn_end: {block.chunks[-1].turn}
turn_count: {len(block.chunks)}
summary: {block.summary}
metadata: {json.dumps(metadata, ensure_ascii=False)}

qa_turns:
{json.dumps(turns, ensure_ascii=False, indent=2)}

Extract canonical entities and per-turn relations for this contiguous LiteraryQA memory block.
Return strict JSON object only."""


def clean_entity_name(value: Any) -> str:
    name = str(value or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" \t\r\n\"'`.,;:()[]{}")
    lowered = name.lower()
    forbidden = {
        "",
        "he",
        "she",
        "it",
        "they",
        "him",
        "her",
        "them",
        "his",
        "hers",
        "their",
        "this",
        "that",
        "unknown",
        "n/a",
        "none",
        "the story",
        "the novel",
        "the book",
    }
    return "" if lowered in forbidden else name


def clean_predicate(value: Any) -> str:
    predicate = str(value or "related_to").strip().lower()
    predicate = re.sub(r"[^a-z0-9]+", "_", predicate)
    predicate = re.sub(r"_+", "_", predicate).strip("_")
    return predicate or "related_to"


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_batch_extractions(batch: list[QAChunk], payload: dict[str, Any]) -> list[Extraction]:
    alias_to_canonical: dict[str, str] = {}
    entity_types: dict[str, str] = {}

    for item in payload.get("entities", []) or []:
        if isinstance(item, dict):
            name = clean_entity_name(item.get("name"))
            entity_type = str(item.get("type") or "unknown").strip().lower() or "unknown"
            aliases = item.get("aliases") or []
        else:
            name = clean_entity_name(item)
            entity_type = "unknown"
            aliases = []
        if not name:
            continue
        alias_to_canonical[name.lower()] = name
        entity_types[name.lower()] = entity_type
        for alias in aliases:
            alias_name = clean_entity_name(alias)
            if alias_name:
                alias_to_canonical[alias_name.lower()] = name

    valid_turns = {chunk.turn for chunk in batch}
    extraction_by_turn: dict[int, Extraction] = {turn: Extraction(turn=turn) for turn in valid_turns}

    def canonicalize(name: Any) -> str:
        cleaned = clean_entity_name(name)
        if not cleaned:
            return ""
        return alias_to_canonical.get(cleaned.lower(), cleaned)

    for item in payload.get("turn_extractions", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            turn = int(item.get("turn"))
        except (TypeError, ValueError):
            continue
        if turn not in valid_turns:
            continue

        seen_mentions: set[str] = set()
        seen_entities: set[str] = set()
        mentioned: list[str] = []
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        def add_entity(name: str) -> None:
            key = name.lower()
            if key in seen_entities:
                return
            seen_entities.add(key)
            entities.append({"name": name, "type": entity_types.get(key, "unknown")})

        def add_mention(name: str) -> None:
            key = name.lower()
            if key not in seen_mentions:
                seen_mentions.add(key)
                mentioned.append(name)
            add_entity(name)

        for raw_name in item.get("mentioned_entities", []) or []:
            name = canonicalize(raw_name)
            if name:
                add_mention(name)

        for relation in item.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            subject = canonicalize(relation.get("subject"))
            obj = canonicalize(relation.get("object"))
            if not subject or not obj:
                continue
            try:
                confidence = float(relation.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            relations.append(
                {
                    "subject": subject,
                    "predicate": clean_predicate(relation.get("predicate")),
                    "object": obj,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
            add_mention(subject)
            add_mention(obj)

        extraction_by_turn[turn] = Extraction(
            turn=turn,
            entities=entities,
            mentioned_entities=mentioned,
            relations=relations,
        )

    return [extraction_by_turn[chunk.turn] for chunk in batch]


def load_completed_documents(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            signature = obj.get("document_signature")
            if not signature:
                continue
            completed.add(str(signature))
    return completed


def append_batch(path: Path, batch_index: int, batch: list[QAChunk], extractions: list[Extraction]) -> None:
    payload = {
        "batch_index": batch_index,
        "author_index": batch_index,
        "document_id": batch[0].raw.get("document_id") if batch else None,
        "gutenberg_id": batch[0].raw.get("gutenberg_id") if batch else None,
        "title": batch[0].raw.get("title") if batch else None,
        "source_split": batch[0].raw.get("source_split") if batch else None,
        "turn_start": batch[0].turn,
        "turn_end": batch[-1].turn,
        "turn_count": len(batch),
        "document_signature": f"{batch[0].raw.get('document_id')}::{batch[0].raw.get('title')}::{len(batch)}" if batch else None,
        "source": "literaryqa_memory_full",
        "turn_extractions": [asdict(item) for item in extractions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def call_deepseek(args: argparse.Namespace, system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }
    if args.json_mode:
        request["response_format"] = {"type": "json_object"}
    if args.disable_thinking:
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def save_failed_response(output_dir: Path, batch_index: int, attempt: int, text: str) -> Path:
    failed_dir = output_dir / "failed_responses"
    failed_dir.mkdir(parents=True, exist_ok=True)
    path = failed_dir / f"batch_{batch_index:05d}_attempt_{attempt}.txt"
    path.write_text(text or "", encoding="utf-8")
    return path


def run_extraction(args: argparse.Namespace) -> None:
    if not args.api_key and not args.dry_run:
        raise ValueError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or pass --api-key.")

    flat_rows = load_json_or_jsonl(args.input)
    blocks = load_document_blocks(args.docs_dir, flat_rows)
    output_dir = args.output_dir
    extraction_path = output_dir / "extractions.jsonl"
    report_path = output_dir / "extraction_report.json"
    completed = load_completed_documents(extraction_path) if args.resume else set()
    if not args.resume and extraction_path.exists() and not args.dry_run:
        extraction_path.unlink()

    system_prompt = build_system_prompt()
    report = {
        "input": str(args.input),
        "docs_dir": str(args.docs_dir),
        "output": str(extraction_path),
        "rows": len(flat_rows),
        "document_count": len(blocks),
        "api_base": args.api_base,
        "model": args.model,
        "json_mode": args.json_mode,
        "disable_thinking": args.disable_thinking,
        "completed_batches": 0,
        "completed_turns": 0,
        "failed_batches": [],
        "status": "running",
    }

    for block in blocks:
        batch = block.chunks
        expected_signature = f"{block.document_id}::{block.title}::{len(batch)}"
        if expected_signature in completed:
            report["completed_batches"] += 1
            report["completed_turns"] += len(batch)
            print(
                f"[Extract] document {block.batch_index + 1}/{len(blocks)} cached "
                f"{block.title!r} qas={len(batch)} turns=1-{len(batch)}"
            )
            continue

        user_prompt = build_user_prompt(block)
        if args.dry_run:
            print(system_prompt)
            print(user_prompt)
            report["status"] = "dry_run"
            write_report(report_path, report)
            return

        batch_extractions: list[Extraction] | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                print(
                    f"[Extract] document {block.batch_index + 1}/{len(blocks)} "
                    f"{block.title!r} qas={len(batch)} turns=1-{len(batch)} "
                    f"attempt={attempt}/{args.max_retries}",
                    flush=True,
                )
                response_text = call_deepseek(args, system_prompt, user_prompt)
                payload = extract_json_object(response_text)
                if payload is None:
                    failed_path = save_failed_response(output_dir, block.batch_index, attempt, response_text)
                    raise ValueError(f"DeepSeek response is not valid JSON object: {failed_path}")
                batch_extractions = normalize_batch_extractions(batch, payload)
                append_batch(extraction_path, block.batch_index, batch, batch_extractions)
                break
            except Exception as exc:
                print(f"[Extract] batch {block.batch_index + 1} failed: {exc}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay)

        if batch_extractions is None:
            report["failed_batches"].append(block.batch_index)
            report["status"] = "failed"
            write_report(report_path, report)
            raise RuntimeError(f"All extraction attempts failed for batch {block.batch_index}")

        report["completed_batches"] += 1
        report["completed_turns"] += len(batch_extractions)
        write_report(report_path, report)

    report["status"] = "completed"
    write_report(report_path, report)
    print(f"[Save] {extraction_path}")
    print(f"[Save] {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-delay", type=int, default=DEFAULT_RETRY_DELAY)
    parser.add_argument("--json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_extraction(parse_args())


if __name__ == "__main__":
    main()
