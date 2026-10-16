                      
   
                                                       

         
                                                 
                                                                          
                                                                           
                                                                         
                                                   
                                                                              
         
   

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "MemoryGenerate" / "answer_gen" / "artifacts" / "memory_qa.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"

API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "10"))
DEEPSEEK_JSON_MODE = os.environ.get("DEEPSEEK_JSON_MODE", "1").strip().lower() not in {"0", "false", "no"}


@dataclass
class Chunk:
    turn: int
    question: str
    answer: str
    raw: dict[str, Any]

         
@dataclass
class Extraction:
    turn: int
    author_index: int = -1
    entities: list[dict[str, Any]] = field(default_factory=list)
    mentioned_entities: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
                                                                
        if (len(root_right), root_right.lower()) < (len(root_left), root_left.lower()):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left


def ensure_project_imports() -> None:
    parent = PROJECT_ROOT.parent
    for path in (PROJECT_ROOT, parent):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

              
def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = str(obj.get("question", "")).strip()
            answer = str(obj.get("answer", "")).strip()
            if not question and not answer:
                continue
            chunks.append(Chunk(turn=len(chunks) + 1, question=question, answer=answer, raw=obj))
    if not chunks:
        raise ValueError(f"No chunks loaded from {path}")
    return chunks

                           
def iter_author_batches(chunks: list[Chunk], author_batch_size: int) -> list[list[Chunk]]:
    if author_batch_size <= 0:
        raise ValueError("author_batch_size must be positive")
    return [chunks[start : start + author_batch_size] for start in range(0, len(chunks), author_batch_size)]


def build_system_prompt() -> str:
    return """你是一个严谨的信息抽取系统。你会收到同一位作者的一整段 36 轮 Q+A 记忆会话。你的任务是在完整会话范围内做指代消解、实体去重、实体关系抽取，并按每一轮输出结果。

核心要求：
1. 必须阅读完整 36 轮后再抽取。当前轮里的 he/she/they/this author/that book/that controversy 等指代，必须结合整段会话还原成明确实体名称。
2. 你必须在该作者批次内合并重复实体和同义实体。例如 "Jaime Vasquez"、"Vasquez"、"the Chilean author" 如果指向同一对象，统一为一个 canonical name。
3. 只抽取每一轮 Q+A 明确表达，或可由完整会话指代消解得到的信息，不要编造。
4. 如果某轮答案明确说明证据不存在、无法确认、没有信息，则不要从该否定内容中臆造实体关系；但可以记录该轮实际提到/指代到的实体。
5. 关系谓词使用英文 snake_case，如 born_in、is_author_of_genre、has_heritage、won_award、participated_in、inspired_by。
6. 输出必须是严格 JSON 对象，不要 Markdown，不要解释文字。

输出格式：
{
  "entities": [
    {
      "name": "Jaime Vasquez",
      "type": "person|place|genre|work|award|organization|event|concept|unknown",
      "aliases": ["Vasquez", "the Chilean author"]
    }
  ],
  "turn_extractions": [
    {
      "turn": 1,
      "mentioned_entities": ["Jaime Vasquez", "Santiago, Chile", "True Crime"],
      "relations": [
        {
          "subject": "Jaime Vasquez",
          "predicate": "is_author_of_genre",
          "object": "True Crime",
          "confidence": 0.95
        }
      ]
    }
  ]
}

字段约束：
- entities 是该作者 36 轮内去重后的实体清单。name 是规范实体名，不要使用代词或泛称。
- aliases 写该实体在会话中的其他叫法；没有则输出空数组。
- turn_extractions 必须覆盖输入里的每一个 turn。
- 每个 mentioned_entities 包含该轮 Q+A 中被直接提到或被代词指代到的实体规范名。
- relations 中 subject/object 必须使用 entities 里的规范实体名，不要使用 he/it/they。
- confidence 是 0 到 1 的数字。"""

                     
def build_author_user_prompt(author_index: int, author_chunks: list[Chunk]) -> str:
    turns = []
    for chunk in author_chunks:
        turns.append(
            {
                "turn": chunk.turn,
                "question": chunk.question,
                "answer": chunk.answer,
            }
        )
    return f"""author_index: {author_index}
turn_count: {len(author_chunks)}

qa_turns:
{json.dumps(turns, ensure_ascii=False, indent=2)}

请先在该作者的完整 qa_turns 中统一识别实体、合并重复项、完成指代消解，然后对每一条 turn 输出 mentioned_entities 和 relations。
直接输出严格 JSON 对象："""

def call_llm(system_prompt: str, user_prompt: str) -> str:
    import openai

    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
    )
    request: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    if DEEPSEEK_JSON_MODE:
        request["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def save_failed_response(output_dir: Path, author_index: int, attempt: int, response_text: str) -> Path:
    failed_dir = output_dir / "failed_responses"
    failed_dir.mkdir(parents=True, exist_ok=True)
    path = failed_dir / f"author_{author_index:04d}_attempt_{attempt}.txt"
    path.write_text(response_text, encoding="utf-8")
    return path


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text:
        text = text.split("<think>", 1)[0].strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            result = json.loads(text[start_idx : end_idx + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            return None
    return None

'''
  - clean_predicate(value)：把关系名清洗成统一格式，比如去空格、转小写、转下划线。
  - normalize_extraction(turn, payload)：把 LLM 抽取结果整理成标准的 Extraction，做去重、清洗、补全。
  - normalize_author_extractions(author_index, author_chunks, payload)：把一整个作者 36 轮的抽取结果按 turn 对齐，并做实体别名统一。

  - extract_all(chunks, output_dir, author_batch_size, resume)：批量调用 LLM 抽取所有作者的实体和关系。
  - build_embeddings_backend(kind, model_path)：选择 embedding 后端，当前支持 Qwen 或 sentence-transformers。
  - scoped_entity_key(author_index, name)：给实体加作者前缀，避免不同作者的同名实体混在一起。
  - collect_entity_metadata(extractions)：统计每个实体出现在哪些 turn、有哪些别名、有哪些类型。
  - cosine_matrix(vectors)：计算向量两两余弦相似度矩阵。
  - merge_entities(metadata, embeddings_backend, threshold)：按 embedding 相似度合并重复或相近实体。
  - stable_node_id(prefix, name)：生成稳定的节点 id，避免名字冲突。
  - graph_to_json_payload(graph)：把图转换成 JSON 可写入结构。
  - json_ready(value)：把 numpy、set、dict 等对象转成可 JSON 序列化格式。
  - gexf_ready_graph(graph)：把图处理成可以写成 GEXF 的格式。
  - build_graph(chunks, extractions, name_to_root, clusters)：真正构建 Memory 图，生成 entity / chunk / anchor 节点和边。
  - save_outputs(graph, output_dir)：保存 pkl、json、gexf 和统计摘要。
  - parse_args()：解析命令行参数。
  - main()：主入口，串起“读数据 -> 抽取 -> 合并实体 -> 建图 -> 保存”。
'''
def clean_entity_name(value: Any) -> str:
    name = str(value or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" \t\r\n\"'`.,;:()[]{}")
    lowered = name.lower()
    if lowered in {"", "he", "she", "it", "they", "him", "her", "them", "this", "that", "unknown", "n/a", "none"}:
        return ""
    return name


def clean_predicate(value: Any) -> str:
    predicate = str(value or "").strip().lower()
    predicate = re.sub(r"[^a-z0-9]+", "_", predicate)
    predicate = re.sub(r"_+", "_", predicate).strip("_")
    return predicate or "related_to"


def normalize_extraction(turn: int, payload: dict[str, Any]) -> Extraction:
    entities: list[dict[str, Any]] = []
    seen_entities: set[str] = set()

    for item in payload.get("entities", []) or []:
        if isinstance(item, dict):
            name = clean_entity_name(item.get("name"))
            entity_type = str(item.get("type") or "unknown").strip().lower() or "unknown"
        else:
            name = clean_entity_name(item)
            entity_type = "unknown"
        if name and name.lower() not in seen_entities:
            seen_entities.add(name.lower())
            entities.append({"name": name, "type": entity_type})

    mentioned: list[str] = []
    seen_mentions: set[str] = set()
    for item in payload.get("mentioned_entities", []) or []:
        name = clean_entity_name(item)
        if name and name.lower() not in seen_mentions:
            seen_mentions.add(name.lower())
            mentioned.append(name)
            if name.lower() not in seen_entities:
                seen_entities.add(name.lower())
                entities.append({"name": name, "type": "unknown"})

    relations: list[dict[str, Any]] = []
    for item in payload.get("relations", []) or []:
        if not isinstance(item, dict):
            continue
        subject = clean_entity_name(item.get("subject"))
        obj = clean_entity_name(item.get("object"))
        if not subject or not obj:
            continue
        predicate = clean_predicate(item.get("predicate"))
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        relations.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
        for name in (subject, obj):
            if name.lower() not in seen_entities:
                seen_entities.add(name.lower())
                entities.append({"name": name, "type": "unknown"})

    return Extraction(turn=turn, entities=entities, mentioned_entities=mentioned, relations=relations)


def normalize_author_extractions(author_index: int, author_chunks: list[Chunk], payload: dict[str, Any]) -> list[Extraction]:
    global_entities: dict[str, dict[str, Any]] = {}
    alias_to_canonical: dict[str, str] = {}

    for item in payload.get("entities", []) or []:
        if isinstance(item, dict):
            name = clean_entity_name(item.get("name"))
            entity_type = str(item.get("type") or "unknown").strip().lower() or "unknown"
            aliases = item.get("aliases", []) or []
        else:
            name = clean_entity_name(item)
            entity_type = "unknown"
            aliases = []
        if not name:
            continue
        key = name.lower()
        global_entities[key] = {"name": name, "type": entity_type}
        alias_to_canonical[key] = name
        for alias in aliases:
            alias_name = clean_entity_name(alias)
            if alias_name:
                alias_to_canonical[alias_name.lower()] = name

    valid_turns = {chunk.turn for chunk in author_chunks}
    extraction_by_turn: dict[int, Extraction] = {
        chunk.turn: Extraction(turn=chunk.turn, author_index=author_index) for chunk in author_chunks
    }

    for item in payload.get("turn_extractions", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            turn = int(item.get("turn"))
        except (TypeError, ValueError):
            continue
        if turn not in valid_turns:
            continue

        entities: list[dict[str, Any]] = []
        seen_entities: set[str] = set()
        mentioned: list[str] = []
        seen_mentions: set[str] = set()

        for raw_name in item.get("mentioned_entities", []) or []:
            name = clean_entity_name(raw_name)
            if not name:
                continue
            canonical = alias_to_canonical.get(name.lower(), name)
            if canonical.lower() not in seen_mentions:
                seen_mentions.add(canonical.lower())
                mentioned.append(canonical)
            entity_key = canonical.lower()
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                entity_type = global_entities.get(entity_key, {}).get("type", "unknown")
                entities.append({"name": canonical, "type": entity_type})

        relations: list[dict[str, Any]] = []
        for relation in item.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            subject = clean_entity_name(relation.get("subject"))
            obj = clean_entity_name(relation.get("object"))
            if not subject or not obj:
                continue
            subject = alias_to_canonical.get(subject.lower(), subject)
            obj = alias_to_canonical.get(obj.lower(), obj)
            predicate = clean_predicate(relation.get("predicate"))
            try:
                confidence = float(relation.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            relations.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
            for name in (subject, obj):
                entity_key = name.lower()
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entity_type = global_entities.get(entity_key, {}).get("type", "unknown")
                    entities.append({"name": name, "type": entity_type})
                if entity_key not in seen_mentions:
                    seen_mentions.add(entity_key)
                    mentioned.append(name)

        extraction_by_turn[turn] = Extraction(
            turn=turn,
            author_index=author_index,
            entities=entities,
            mentioned_entities=mentioned,
            relations=relations,
        )

    return [extraction_by_turn[chunk.turn] for chunk in author_chunks]

'''
  - load_cached_author_batches(path)：读取之前保存的抽取缓存，支持断点续跑。
  - append_author_batch(path, author_index, extractions)：把某个作者的抽取结果追加写入缓存文件。
'''
def load_cached_author_batches(path: Path) -> dict[int, list[Extraction]]:
    if not path.exists():
        return {}
    cached: dict[int, list[Extraction]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            author_index = int(obj["author_index"])
            cached[author_index] = [
                Extraction(
                    turn=int(item["turn"]),
                    author_index=int(item.get("author_index", author_index)),
                    entities=list(item.get("entities", [])),
                    mentioned_entities=list(item.get("mentioned_entities", [])),
                    relations=list(item.get("relations", [])),
                )
                for item in obj.get("turn_extractions", [])
            ]
    return cached


def append_author_batch(path: Path, author_index: int, extractions: list[Extraction]) -> None:
    payload = {
        "author_index": author_index,
        "turn_start": extractions[0].turn if extractions else None,
        "turn_end": extractions[-1].turn if extractions else None,
        "turn_extractions": [extraction.__dict__ for extraction in extractions],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_all(
    chunks: list[Chunk],
    output_dir: Path,
    *,
    author_batch_size: int,
    resume: bool,
) -> list[Extraction]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "extractions.jsonl"
    cached = load_cached_author_batches(cache_path) if resume else {}
    if not resume and cache_path.exists():
        cache_path.unlink()

    system_prompt = build_system_prompt()
    extractions: list[Extraction] = []
    author_batches = iter_author_batches(chunks, author_batch_size)
    for author_index, author_chunks in enumerate(author_batches):
        if author_index in cached:
            batch_extractions = cached[author_index]
            print(f"[Extract] Author {author_index + 1}/{len(author_batches)} cached turns={len(batch_extractions)}")
            extractions.extend(batch_extractions)
            continue

        user_prompt = build_author_user_prompt(author_index, author_chunks)
        batch_extractions: list[Extraction] | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                turn_start = author_chunks[0].turn
                turn_end = author_chunks[-1].turn
                print(
                    f"[Extract] Author {author_index + 1}/{len(author_batches)} "
                    f"turns={turn_start}-{turn_end} attempt {attempt}/{MAX_RETRIES}"
                )
                response_text = call_llm(system_prompt, user_prompt)
                payload = extract_json_object(response_text)
                if payload is None:
                    failed_path = save_failed_response(output_dir, author_index, attempt, response_text)
                    preview = response_text[:300].replace("\n", " ")
                    raise ValueError(f"LLM response did not contain a JSON object; saved={failed_path}; preview={preview!r}")
                batch_extractions = normalize_author_extractions(author_index, author_chunks, payload)
                break
            except Exception as exc:
                print(f"[Extract] Author {author_index + 1} failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        if batch_extractions is None:
            raise RuntimeError(f"All extraction attempts failed for author batch {author_index}")
        append_author_batch(cache_path, author_index, batch_extractions)
        extractions.extend(batch_extractions)
    return extractions

def build_embeddings_backend(kind: str, model_path: str | None):
    if kind == "qwen":
        ensure_project_imports()
        from procedure.runtime_support import build_shared_embeddings

        return build_shared_embeddings()

    if kind == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        if not model_path:
            raise ValueError("--embedding-model-path is required for sentence_transformers")
        model = SentenceTransformer(model_path)

        class SentenceTransformerBackend:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return model.encode(texts, normalize_embeddings=True).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

        return SentenceTransformerBackend()

    raise ValueError(f"Unsupported embedding backend: {kind}")


def scoped_entity_key(author_index: int, name: str) -> str:
    return f"author_{author_index:04d}::{name.lower()}"


def collect_entity_metadata(extractions: list[Extraction]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for extraction in extractions:
        author_index = int(extraction.author_index)
        for item in extraction.entities:
            name = clean_entity_name(item.get("name"))
            if not name:
                continue
            key = scoped_entity_key(author_index, name)
            entity_type = str(item.get("type") or "unknown").strip().lower() or "unknown"
            entry = metadata.setdefault(
                key,
                {"name": name, "author_index": author_index, "types": {}, "turns": set(), "aliases": set()},
            )
            entry["types"][entity_type] = entry["types"].get(entity_type, 0) + 1
            entry["turns"].add(extraction.turn)
            entry["aliases"].add(name)
        for relation in extraction.relations:
            for side in ("subject", "object"):
                name = clean_entity_name(relation.get(side))
                if not name:
                    continue
                key = scoped_entity_key(author_index, name)
                entry = metadata.setdefault(
                    key,
                    {"name": name, "author_index": author_index, "types": {}, "turns": set(), "aliases": set()},
                )
                entry["types"]["unknown"] = entry["types"].get("unknown", 0) + 1
                entry["turns"].add(extraction.turn)
                entry["aliases"].add(name)
    return metadata


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


def merge_entities(
    metadata: dict[str, dict[str, Any]],
    embeddings_backend,
    threshold: float,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    keys = sorted(metadata)
    if not keys:
        return {}, {}

    names = [metadata[key]["name"] for key in keys]
    vectors = np.asarray(embeddings_backend.embed_documents(names), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(keys):
        raise ValueError("Embedding backend returned invalid shape")

    uf = UnionFind(keys)
    keys_by_author: dict[int, list[int]] = {}
    for idx, key in enumerate(keys):
        author_index = int(metadata[key]["author_index"])
        keys_by_author.setdefault(author_index, []).append(idx)

    for indices in keys_by_author.values():
        if len(indices) < 2:
            continue
        group_vectors = vectors[indices]
        similarities = cosine_matrix(group_vectors)
        for local_i, idx_i in enumerate(indices):
            for local_j in range(local_i + 1, len(indices)):
                idx_j = indices[local_j]
                if float(similarities[local_i, local_j]) > threshold:
                    uf.union(keys[idx_i], keys[idx_j])

    clusters: dict[str, dict[str, Any]] = {}
    name_to_root: dict[str, str] = {}
    for key, vector in zip(keys, vectors):
        root = uf.find(key)
        name_to_root[key] = root
        source = metadata[key]
        cluster = clusters.setdefault(
            root,
            {
                "canonical_name": metadata[root]["name"],
                "author_index": int(metadata[root]["author_index"]),
                "aliases": set(),
                "types": {},
                "turns": set(),
                "vectors": [],
            },
        )
        cluster["aliases"].update(source["aliases"])
        cluster["turns"].update(source["turns"])
        cluster["vectors"].append(vector)
        for entity_type, count in source["types"].items():
            cluster["types"][entity_type] = cluster["types"].get(entity_type, 0) + int(count)

    finalized: dict[str, dict[str, Any]] = {}
    for root, cluster in clusters.items():
        type_counts = cluster["types"]
        entity_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"
        mean_vector = np.mean(np.stack(cluster["vectors"]), axis=0)
        finalized[root] = {
            "canonical_name": cluster["canonical_name"],
            "author_index": int(cluster["author_index"]),
            "aliases": sorted(cluster["aliases"], key=str.lower),
            "entity_type": entity_type,
            "turns": sorted(int(turn) for turn in cluster["turns"]),
            "embedding": mean_vector.astype(float).tolist(),
        }
    return name_to_root, finalized

def stable_node_id(prefix: str, name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")[:48] or "entity"
    digest = hashlib.sha1(name.lower().encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{slug}:{digest}"


def graph_to_json_payload(graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "format_version": graph.graph.get("format_version", 1),
        "graph_type": graph.graph.get("graph_type", "heterogeneous_memory_graph"),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "nodes": [
            {
                "id": node_id,
                **{key: json_ready(value) for key, value in data.items()},
            }
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


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def gexf_ready_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    ready = nx.MultiDiGraph()
    ready.graph.update({key: str(value) for key, value in graph.graph.items()})
    for node_id, data in graph.nodes(data=True):
        attrs = {}
        for key, value in data.items():
            if key == "embedding":
                continue
            if isinstance(value, (dict, list, tuple, set)):
                attrs[key] = json.dumps(json_ready(value), ensure_ascii=False)
            else:
                attrs[key] = value
        ready.add_node(node_id, **attrs)
    for source, target, key, data in graph.edges(keys=True, data=True):
        attrs = {}
        for attr_key, value in data.items():
            if isinstance(value, (dict, list, tuple, set)):
                attrs[attr_key] = json.dumps(json_ready(value), ensure_ascii=False)
            else:
                attrs[attr_key] = value
        ready.add_edge(source, target, key=key, **attrs)
    return ready


def build_graph(
    chunks: list[Chunk],
    extractions: list[Extraction],
    name_to_root: dict[str, str],
    clusters: dict[str, dict[str, Any]],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["format_version"] = 1
    graph.graph["graph_type"] = "heterogeneous_memory_graph"
    graph.graph["node_types"] = ["entity", "chunk", "anchor"]
    graph.graph["edge_types"] = ["relation", "mentioned_in", "next_turn"]

    root_to_node: dict[str, str] = {}
    for root, data in clusters.items():
        node_id = stable_node_id("entity", f"author_{int(data['author_index']):04d}::{data['canonical_name']}")
        root_to_node[root] = node_id
        graph.add_node(
            node_id,
            node_type="entity",
            canonical_name=data["canonical_name"],
            author_index=int(data["author_index"]),
            aliases=data["aliases"],
            entity_type=data["entity_type"],
            turns=data["turns"],
            embedding=data["embedding"],
        )

    extraction_by_turn = {extraction.turn: extraction for extraction in extractions}
    previous_anchor_by_author: dict[int, str] = {}
    for chunk in chunks:
        chunk_id = f"chunk:turn_{chunk.turn}"
        anchor_id = f"anchor:turn_{chunk.turn}"
        graph.add_node(
            chunk_id,
            node_type="chunk",
            turn=chunk.turn,
            question=chunk.question,
            answer=chunk.answer,
            raw=chunk.raw,
            anchor_id=anchor_id,
        )
        graph.add_node(
            anchor_id,
            node_type="anchor",
            turn=chunk.turn,
            chunk_id=chunk_id,
        )
        extraction = extraction_by_turn.get(chunk.turn)
        author_index = int(extraction.author_index) if extraction is not None else (chunk.turn - 1) // 36
        graph.nodes[chunk_id]["author_index"] = author_index
        graph.nodes[anchor_id]["author_index"] = author_index
        previous_anchor_id = previous_anchor_by_author.get(author_index)
        if previous_anchor_id is not None:
            graph.add_edge(previous_anchor_id, anchor_id, edge_type="next_turn", relation="next_turn", author_index=author_index)
        previous_anchor_by_author[author_index] = anchor_id

        if extraction is None:
            continue

        mentioned_roots: set[str] = set()
        for name in extraction.mentioned_entities:
            root = name_to_root.get(scoped_entity_key(extraction.author_index, clean_entity_name(name)))
            if root:
                mentioned_roots.add(root)
        for relation in extraction.relations:
            for side in ("subject", "object"):
                root = name_to_root.get(scoped_entity_key(extraction.author_index, clean_entity_name(relation.get(side))))
                if root:
                    mentioned_roots.add(root)

        for root in sorted(mentioned_roots):
            entity_id = root_to_node.get(root)
            if entity_id:
                graph.add_edge(entity_id, anchor_id, edge_type="mentioned_in", relation="mentioned_in", turn=chunk.turn, author_index=author_index)

        for relation in extraction.relations:
            subject_root = name_to_root.get(scoped_entity_key(extraction.author_index, clean_entity_name(relation.get("subject"))))
            object_root = name_to_root.get(scoped_entity_key(extraction.author_index, clean_entity_name(relation.get("object"))))
            if not subject_root or not object_root:
                continue
            subject_id = root_to_node.get(subject_root)
            object_id = root_to_node.get(object_root)
            if not subject_id or not object_id:
                continue
            predicate = clean_predicate(relation.get("predicate"))
            graph.add_edge(
                subject_id,
                object_id,
                edge_type="relation",
                relation=predicate,
                turn=chunk.turn,
                author_index=author_index,
                confidence=float(relation.get("confidence", 1.0)),
            )

    return graph


def save_outputs(graph: nx.MultiDiGraph, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "memory_graph.pkl"
    json_path = output_dir / "memory_graph.json"
    gexf_path = output_dir / "memory_graph.gexf"
    summary_path = output_dir / "memory_graph_summary.json"

    with pkl_path.open("wb") as f:
        pickle.dump(graph, f)
    json_path.write_text(json.dumps(graph_to_json_payload(graph), ensure_ascii=False, indent=2), encoding="utf-8")
    nx.write_gexf(gexf_ready_graph(graph), gexf_path)

    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        node_type = str(data.get("node_type", "unknown"))
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    for _, _, data in graph.edges(data=True):
        edge_type = str(data.get("edge_type", "unknown"))
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

    summary = {
        "graph_type": graph.graph["graph_type"],
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "outputs": {
            "pickle": str(pkl_path),
            "json": str(json_path),
            "gexf": str(gexf_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pickle": pkl_path, "json": json_path, "gexf": gexf_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build heterogeneous memory graph from memory_qa.json")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to memory_qa.json JSONL file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for graph artifacts.")
    parser.add_argument("--author-batch-size", type=int, default=36, help="Number of QA turns per author DeepSeek batch.")
    parser.add_argument("--merge-threshold", type=float, default=0.7, help="Cosine similarity threshold for entity merging.")
    parser.add_argument("--embedding-backend", choices=["qwen", "sentence_transformers"], default="qwen")
    parser.add_argument("--embedding-model-path", default=None, help="Model path for sentence_transformers backend.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N chunks; 0 means all.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing extractions.jsonl and continue missing turns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("=" * 72)
    print("Memory Graph Builder")
    print("=" * 72)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"API:    {API_BASE}  model={MODEL_NAME}")

    chunks = load_chunks(input_path)
    if args.limit and args.limit > 0:
        chunks = chunks[: args.limit]
    print(f"[Load] chunks={len(chunks)}")

    extractions = extract_all(
        chunks,
        output_dir,
        author_batch_size=max(1, int(args.author_batch_size)),
        resume=bool(args.resume),
    )
    print(f"[Extract] completed={len(extractions)}")

    metadata = collect_entity_metadata(extractions)
    print(f"[Entity] raw_unique={len(metadata)}")

    embeddings_backend = build_embeddings_backend(args.embedding_backend, args.embedding_model_path)
    name_to_root, clusters = merge_entities(metadata, embeddings_backend, float(args.merge_threshold))
    print(f"[Entity] merged={len(clusters)} threshold={args.merge_threshold}")

    graph = build_graph(chunks, extractions, name_to_root, clusters)
    outputs = save_outputs(graph, output_dir)

    print("[Graph] nodes=%d edges=%d" % (graph.number_of_nodes(), graph.number_of_edges()))
    for name, path in outputs.items():
        print(f"[Save] {name}: {path}")


if __name__ == "__main__":
    main()
