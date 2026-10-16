                      
   
                                                         

             
                                             
                                         
                               

                                                          

       

                         
                                              
                                       
                                                                      
                                                                             
                                                                                                           
                                                                    
                                                           
                                               
   

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
import pickle
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
from numpy.linalg import norm as np_norm


_ANSWER_GEN_DIR = Path(__file__).resolve().parent
_MEMORYGENERATE_DIR = _ANSWER_GEN_DIR.parent
_AGENT_DIR = _MEMORYGENERATE_DIR.parent

DEFAULT_BASE_MODEL_PATH = os.environ.get(
    "ANSWER_GEN_BASE_MODEL_PATH",
    "/home/likunhao/likunhao52shared/llms/Qwen3.5-9B",
).strip()
DEFAULT_QUESTIONS_PATH = _MEMORYGENERATE_DIR / "question_gen" / "artifacts" / "memory_q.json"
DEFAULT_STATIC_GRAPH_PATH = _AGENT_DIR / "phase1" / "artifacts" / "rag_graph.pkl"


@dataclass
class MemoryEntry:
    turn_index: int
    question: str
    answer: str
    relation_type: str
    embedding: list[float]


@dataclass
class StandaloneAnswerGenConfig:
    questions_per_scope: int = 36
    rag_top_k: int = 2
    memory_top_k: int = 3
    save_every: int = 100
    use_memory: bool = True
    reset_memory: bool = True
    embeddings_device: str = "cuda"
    embeddings_batch_size: int = 4
    embeddings_max_length: int = 512
    embeddings_hidden_layer: int = -1
    embeddings_pooling: str = "mean"
    embeddings_load_in_4bit: bool = True
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.0
    llm_top_p: float = 0.9


@dataclass
class StaticGraphIndex:
    node_ids: list[str]
    questions: list[str]
    answers: list[str]
    communities: list[Any]
    embeddings: np.ndarray
    embedding_norms: np.ndarray

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> "StaticGraphIndex":
        node_ids: list[str] = []
        questions: list[str] = []
        answers: list[str] = []
        communities: list[Any] = []
        vectors: list[np.ndarray] = []

        for node_id, data in graph.nodes(data=True):
            if data.get("type") != "static_knowledge":
                continue
            raw_embedding = data.get("embedding")
            if raw_embedding is None:
                continue
            embedding = np.asarray(raw_embedding, dtype=np.float32)
            if embedding.size == 0:
                continue
            node_ids.append(str(node_id))
            questions.append(str(data.get("question") or "").strip())
            answers.append(str(data.get("answer") or "").strip())
            communities.append(data.get("community"))
            vectors.append(embedding)

        if not vectors:
            raise ValueError("No static_knowledge nodes with embeddings were found in rag_graph.")

        embedding_matrix = np.stack(vectors).astype(np.float32, copy=False)
        embedding_norms = np_norm(embedding_matrix, axis=1) + 1e-10
        return cls(
            node_ids=node_ids,
            questions=questions,
            answers=answers,
            communities=communities,
            embeddings=embedding_matrix,
            embedding_norms=embedding_norms,
        )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _postprocess_generation(output_text: str) -> str:
    output_text = (output_text or "").strip()
    if "<think>" in output_text:
        if "</think>" in output_text:
            output_text = output_text.split("</think>", 1)[1].strip()
        else:
            output_text = output_text.split("<think>", 1)[0].strip()

    stop_markers = [
        "\nQuestion:",
        "\nUser:",
        "\nAssistant:",
        "\nuser",
        "\nassistant",
        "\nQ:",
        "\nA:",
        "<think>",
        "</think>",
    ]
    for marker in stop_markers:
        if marker in output_text:
            output_text = output_text.split(marker, 1)[0].strip()

    for prefix in ("Final Answer:", "ANSWER:", "Response:"):
        if prefix.lower() in output_text.lower():
            pattern = re.compile(rf"{re.escape(prefix)}\s*(.*)", flags=re.IGNORECASE | re.S)
            match = pattern.search(output_text)
            if match:
                output_text = match.group(1).strip()
                break

    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if lines:
        output_text = lines[0]
    return output_text.strip()


class Qwen3EmbeddingBackend:
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        batch_size: int,
        max_length: int,
        hidden_layer: int,
        pooling: str,
        load_in_4bit: bool,
    ) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        self._torch = torch
        self._F = F
        self.model_path = model_path
        self.device = (device or "cuda").strip().lower()
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))
        self.hidden_layer = int(hidden_layer)
        self.pooling = (pooling or "mean").strip().lower() or "mean"
        self.load_in_4bit = bool(load_in_4bit)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        preferred_device = self.device
        if preferred_device == "cuda" and not torch.cuda.is_available():
            preferred_device = "cpu"

        cfg = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        text_cfg = cfg.text_config
        common_kwargs = {
            "config": text_cfg,
            "local_files_only": True,
            "trust_remote_code": True,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        if preferred_device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen3_5TextModel.from_pretrained(
                self.model_path,
                device_map="auto",
                quantization_config=quantization_config,
                **common_kwargs,
            )
        else:
            cpu_dtype = torch.float32
            if getattr(getattr(torch.backends, "cpu", None), "has_bf16", False):
                cpu_dtype = torch.bfloat16
            self.model = Qwen3_5TextModel.from_pretrained(
                self.model_path,
                device_map="cpu",
                torch_dtype=cpu_dtype,
                **common_kwargs,
            )
        self.model.eval()

    def _resolve_device(self):
        return next(self.model.parameters()).device

    def _pool_hidden_states(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        if self.pooling == "last_token":
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_indices = self._torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, lengths]
        else:
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self._F.normalize(pooled, p=2, dim=1)

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        target_device = self._resolve_device()
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states[self.hidden_layer]
            pooled = self._pool_hidden_states(hidden_states, encoded["attention_mask"])
            vectors.extend(pooled.detach().cpu().to(self._torch.float32).tolist())
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]


class LightweightQwenLLM:
    def __init__(
        self,
        *,
        base_model_path: str,
        backend: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        vllm_api_base: str,
        vllm_api_key: str,
        vllm_model_name: str,
    ) -> None:
        self.backend = (backend or "vllm_api").strip().lower()
        self.base_model_path = base_model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.api_base = vllm_api_base.rstrip("/")
        self.api_key = vllm_api_key
        self.model_name = vllm_model_name.strip() or self.base_model_path
        self.model = None
        self.tokenizer = None
        if self.backend == "local":
            self._init_local_backend()
        else:
            self._probe_vllm()

    def _probe_vllm(self) -> None:
        request = urllib.request.Request(
            url=f"{self.api_base}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach vLLM API at {self.api_base}. "
                "Start the server first or switch to --answer-backend local."
            ) from exc

    def _init_local_backend(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            use_safetensors=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

    def invoke(self, prompt: str) -> str:
        if self.backend == "local":
            return self._call_local(prompt)
        return self._call_vllm(prompt)

    __call__ = invoke

    def _call_vllm(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"vLLM request failed at {self.api_base}/chat/completions."
            ) from exc
        output_text = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _postprocess_generation(output_text)

    def _call_local(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _postprocess_generation(output_text)


def infer_relation_intent(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "generic"
    patterns = [
        ("characters", [r"\bcharacters?\b", r"\bprotagonist\b"]),
        ("plot", [r"\bplot\b", r"\bstoryline\b", r"\bsynopsis\b", r"\bsummary\b", r"\bwhat is .* about\b"]),
        ("theme", [r"\bthemes?\b", r"\bwhat does .* explore\b"]),
        ("awards", [r"\bawards?\b", r"\bprize\b", r"\brecognition\b"]),
        ("title_or_book", [r"\bbooks?\b", r"\bworks?\b", r"\bnovels?\b", r"\btitle\b"]),
        ("genre", [r"\bgenre\b", r"\bwhat type of books\b", r"\bwhat kind of books\b", r"\bprimarily write\b"]),
        ("birth_place", [r"\bwhere was\b", r"\bbirthplace\b", r"\bwhere does .* hail from\b"]),
        ("birth_date", [r"\bwhen was\b", r"\bdate of birth\b", r"\bborn on\b", r"\bwhat year was\b"]),
        ("reader_engagement", [r"\breaders? and fans\b", r"\binteract with .* fans\b", r"\bsocial media\b"]),
        ("collaboration", [r"\bcollaborat", r"\bco-auth"]),
        ("inspiration", [r"\binspir", r"\bwhat prompted\b", r"\bwhy did .* write\b"]),
        ("adaptation", [r"\badapted into\b", r"\bfilm\b", r"\bmovie\b", r"\btelevision\b"]),
        ("identity_influence", [r"\blgbtq\b", r"\bidentity\b"]),
        ("reception", [r"\bwell received\b", r"\breviewed by critics\b", r"\breception\b"]),
        ("family_background", [r"\bparents?\b", r"\bfamily\b", r"\bmother\b", r"\bfather\b"]),
        ("future_projects", [r"\bworking on next\b", r"\bupcoming projects?\b", r"\bwhat is .* working on\b"]),
    ]
    for label, regexes in patterns:
        if any(re.search(pattern, q) for pattern in regexes):
            return label
    return "generic"


def load_question_rows(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Question file must be a JSON array: {path}")

    rows: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Question item at index {index} is not an object.")
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError(f"Question item at index {index} is missing 'question'.")
        row = {"question": question}
        scope_id = str(item.get("author_id") or item.get("scope_id") or "").strip()
        if scope_id:
            row["scope_id"] = scope_id
        rows.append(row)
    return rows


def load_existing_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

'''
给每条 question 分配一个 memory 会话编号。
  如果数据有 scope_id，就用数据自带的；
  如果没有，就默认每 36 条问题作为一个 scope。
'''
def build_scope_sequence(
    question_rows: list[dict[str, str]],
    questions_per_scope: int,
) -> tuple[list[str], int, str]:
    explicit_scope_ids = [str(row.get("scope_id") or "").strip() for row in question_rows]
    if explicit_scope_ids and all(explicit_scope_ids):
        return explicit_scope_ids, 0, "question_metadata"

    if questions_per_scope <= 0:
        raise ValueError(
            "questions_per_scope must be positive when question rows do not carry scope_id metadata."
        )

    scope_sequence = [
        f"scope_{index // questions_per_scope:04d}"
        for index in range(len(question_rows))
    ]
    return scope_sequence, questions_per_scope, "fixed_chunk"


def load_static_graph(graph_path: Path) -> nx.Graph:
    if not graph_path.exists():
        raise FileNotFoundError(f"Static RAG graph not found: {graph_path}")
    with graph_path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Unsupported graph object: {type(graph).__name__}")
    cleaned = graph.copy()
    dynamic_nodes = [
        node_id
        for node_id, data in cleaned.nodes(data=True)
        if data.get("type") == "dynamic_memory"
    ]
    if dynamic_nodes:
        cleaned.remove_nodes_from(dynamic_nodes)
    return cleaned


def build_embeddings(
    config: StandaloneAnswerGenConfig,
    *,
    base_model_path: str,
) -> Qwen3EmbeddingBackend:
    return Qwen3EmbeddingBackend(
        model_path=base_model_path,
        device=config.embeddings_device,
        batch_size=config.embeddings_batch_size,
        max_length=config.embeddings_max_length,
        hidden_layer=config.embeddings_hidden_layer,
        pooling=config.embeddings_pooling,
        load_in_4bit=config.embeddings_load_in_4bit,
    )


def build_llm(
    *,
    backend: str,
    base_model_path: str,
    vllm_api_base: str,
    vllm_api_key: str,
    vllm_model_name: str,
    config: StandaloneAnswerGenConfig,
) -> LightweightQwenLLM:
    return LightweightQwenLLM(
        base_model_path=base_model_path,
        backend=backend,
        max_new_tokens=config.llm_max_new_tokens,
        temperature=config.llm_temperature,
        top_p=config.llm_top_p,
        vllm_api_base=vllm_api_base,
        vllm_api_key=vllm_api_key,
        vllm_model_name=vllm_model_name,
    )


def rag_search(
    *,
    question: str,
    graph_index: StaticGraphIndex,
    embeddings_model: Qwen3EmbeddingBackend,
    top_k: int,
) -> tuple[list[dict[str, Any]], str]:
    query_vec = np.asarray(embeddings_model.embed_query(question), dtype=np.float32)
    query_norm = np_norm(query_vec) + 1e-10
    similarities = (graph_index.embeddings @ query_vec) / (graph_index.embedding_norms * query_norm)

    k = max(1, min(int(top_k), len(graph_index.node_ids)))
    if k == len(graph_index.node_ids):
        top_indices = np.argsort(-similarities)
    else:
        candidate_indices = np.argpartition(-similarities, k - 1)[:k]
        top_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]

    top_items: list[dict[str, Any]] = []
    for idx in top_indices.tolist():
        top_items.append(
            {
                "node_id": graph_index.node_ids[idx],
                "score": float(similarities[idx]),
                "question": graph_index.questions[idx],
                "answer": graph_index.answers[idx],
                "community": graph_index.communities[idx],
            }
        )

    context = "\n".join(
        f"[RAG Match score={item['score']:.4f}] Q: {item['question']}\n A: {item['answer']}"
        for item in top_items
    )
    return top_items, context


class MemoryStore:
    def __init__(self) -> None:
        self.entries_by_scope: dict[str, list[MemoryEntry]] = defaultdict(list)

    def add(
        self,
        *,
        scope_id: str,
        question: str,
        answer: str,
        embeddings_model: Qwen3EmbeddingBackend,
    ) -> None:
        entries = self.entries_by_scope[scope_id]
        turn_index = len(entries) + 1
        relation_type = infer_relation_intent(question)
        embedding = embeddings_model.embed_query(f"Question: {question}\nAnswer: {answer}")
        entries.append(
            MemoryEntry(
                turn_index=turn_index,
                question=question,
                answer=answer,
                relation_type=relation_type,
                embedding=embedding,
            )
        )

    def search(
        self,
        *,
        scope_id: str,
        question: str,
        embeddings_model: Qwen3EmbeddingBackend,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str]:
        entries = self.entries_by_scope.get(scope_id, [])
        if not entries:
            return [], ""

        query_vec = np.asarray(embeddings_model.embed_query(question), dtype=np.float32)
        query_relation = infer_relation_intent(question)
        max_turn = max(entry.turn_index for entry in entries)
        scored: list[dict[str, Any]] = []
        for entry in entries:
            emb = np.asarray(entry.embedding, dtype=np.float32)
            sim = float(np.dot(query_vec, emb) / (np_norm(query_vec) * np_norm(emb) + 1e-10))
            relation_bonus = 0.08 if entry.relation_type == query_relation else 0.0
            recency_bonus = 0.10 * (float(entry.turn_index) / float(max(max_turn, 1)))
            score = sim + relation_bonus + recency_bonus
            scored.append(
                {
                    "score": float(score),
                    "sim": float(sim),
                    "turn_index": entry.turn_index,
                    "question": entry.question,
                    "answer": entry.answer,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        top_items = scored[: max(1, top_k)]
        if not top_items or top_items[0]["score"] < 0.30:
            return [], ""

        context = "\n".join(
            f"[Memory score={item['score']:.4f} | turn={item['turn_index']}] "
            f"Q: {item['question']}\n A: {item['answer']}"
            for item in top_items
        )
        return top_items, context

    def to_json_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for scope_id, entries in self.entries_by_scope.items():
            payload[scope_id] = [
                {
                    "turn_index": entry.turn_index,
                    "question": entry.question,
                    "answer": entry.answer,
                    "relation_type": entry.relation_type,
                }
                for entry in entries
            ]
        return payload


def build_grounded_prompt(*, question: str, rag_context: str, memory_context: str) -> str:
    memory_block = memory_context.strip() if memory_context.strip() else "No relevant historical memory."
    return (
        "You are a strictly grounded QA assistant.\n"
        "Answer the user's question using only the provided evidence.\n"
        "You may use both the static RAG evidence and the historical memory evidence.\n"
        "Prefer the evidence that directly matches the question.\n"
        "If the memory evidence is vague or noisy, trust the best matching RAG evidence.\n"
        "Keep the answer concise and factual.\n"
        "Do not mention RAG, retrieval, memory, graphs, or evidence.\n"
        "Do not invent facts that are not explicitly supported.\n"
        "Return only the final answer text.\n\n"
        f"[Question]\n{question}\n\n"
        f"[Static RAG Evidence]\n{rag_context}\n\n"
        f"[Historical Memory Evidence]\n{memory_block}\n"
    )


def clean_generated_answer(text: str) -> str:
    cleaned = (text or "").replace("\nuser", "\n").replace(".user", ".").strip()
    return _postprocess_generation(cleaned)


def choose_fallback_answer(
    *,
    rag_items: list[dict[str, Any]],
    memory_items: list[dict[str, Any]],
) -> str:
    if rag_items:
        return str(rag_items[0].get("answer") or "").strip()
    if memory_items:
        return str(memory_items[0].get("answer") or "").strip()
    return ""


def generate_single_answer(
    *,
    llm: LightweightQwenLLM,
    question: str,
    scope_id: str,
    memory_store: MemoryStore,
    graph_index: StaticGraphIndex,
    embeddings_model: Qwen3EmbeddingBackend,
    config: StandaloneAnswerGenConfig,
) -> dict[str, Any]:
    rag_items, rag_context = rag_search(
        question=question,
        graph_index=graph_index,
        embeddings_model=embeddings_model,
        top_k=config.rag_top_k,
    )
    memory_items: list[dict[str, Any]] = []
    memory_context = ""
    if config.use_memory:
        memory_items, memory_context = memory_store.search(
            scope_id=scope_id,
            question=question,
            embeddings_model=embeddings_model,
            top_k=config.memory_top_k,
        )

    prompt = build_grounded_prompt(
        question=question,
        rag_context=rag_context,
        memory_context=memory_context,
    )
    answer = clean_generated_answer(llm.invoke(prompt))
    if not answer or answer in {"[SYSTEM_INTERVENTION]", "[系统干预]"}:
        answer = choose_fallback_answer(rag_items=rag_items, memory_items=memory_items)

    if config.use_memory:
        memory_store.add(
            scope_id=scope_id,
            question=question,
            answer=answer,
            embeddings_model=embeddings_model,
        )

    return {
        "question": question,
        "answer": answer,
        "scope_id": scope_id,
        "rag_context": rag_context,
        "memory_context": memory_context,
        "retrieved_context_items": rag_items,
    }


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_memory_snapshot(path: Path, memory_store: MemoryStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(memory_store.to_json_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _replay_existing_rows(
    *,
    existing_rows: list[dict[str, Any]],
    question_rows: list[dict[str, str]],
    scope_sequence: list[str],
    memory_store: MemoryStore,
    embeddings_model: Qwen3EmbeddingBackend,
    config: StandaloneAnswerGenConfig,
) -> None:
    for index, row in enumerate(existing_rows):
        expected_question = question_rows[index]["question"]
        actual_question = str(row.get("question") or "").strip()
        if actual_question != expected_question:
            raise ValueError(
                f"Existing output question mismatch at index {index}: "
                f"expected {expected_question!r}, got {actual_question!r}"
            )
        if not config.use_memory:
            continue
        memory_store.add(
            scope_id=scope_sequence[index],
            question=expected_question,
            answer=str(row.get("answer") or "").strip(),
            embeddings_model=embeddings_model,
        )


def generate_memory_qa(
    *,
    questions_path: Path,
    static_graph_path: Path,
    output_path: Path,
    trace_path: Path,
    memory_path: Path,
    backend: str,
    base_model_path: str,
    vllm_api_base: str,
    vllm_api_key: str,
    vllm_model_name: str,
    limit: int = 0,
    resume: bool = True,
    config: StandaloneAnswerGenConfig | None = None,
) -> dict[str, Any]:
    config = config or StandaloneAnswerGenConfig()
    resolved_base_model_path = base_model_path.strip() or DEFAULT_BASE_MODEL_PATH
    question_rows = load_question_rows(questions_path)
    if limit and limit > 0:
        question_rows = question_rows[:limit]

    scope_sequence, resolved_questions_per_scope, scope_source = build_scope_sequence(
        question_rows=question_rows,
        questions_per_scope=config.questions_per_scope,
    )

    static_graph = load_static_graph(static_graph_path)
    graph_index = StaticGraphIndex.from_graph(static_graph)
    embeddings_model = build_embeddings(config, base_model_path=resolved_base_model_path)
    llm = build_llm(
        backend=backend,
        base_model_path=resolved_base_model_path,
        vllm_api_base=vllm_api_base,
        vllm_api_key=vllm_api_key,
        vllm_model_name=vllm_model_name,
        config=config,
    )

    memory_store = MemoryStore()
    existing_rows = load_existing_jsonl_rows(output_path) if resume else []
    if existing_rows:
        _replay_existing_rows(
            existing_rows=existing_rows,
            question_rows=question_rows,
            scope_sequence=scope_sequence,
            memory_store=memory_store,
            embeddings_model=embeddings_model,
            config=config,
        )
    start_index = len(existing_rows)

    if start_index >= len(question_rows):
        _save_memory_snapshot(memory_path, memory_store)
        return {
            "questions_path": str(questions_path),
            "static_graph_path": str(static_graph_path),
            "output_path": str(output_path),
            "trace_path": str(trace_path),
            "memory_path": str(memory_path),
            "total_questions": len(question_rows),
            "generated_now": 0,
            "resumed_rows": start_index,
            "questions_per_scope": resolved_questions_per_scope,
            "scope_source": scope_source,
            "scope_count": len(set(scope_sequence)),
            "base_model_path": resolved_base_model_path,
            "pipeline": "standalone_answer_gen_graph_rag",
        }

    if not resume or start_index == 0:
        if output_path.exists():
            output_path.unlink()
        if trace_path.exists():
            trace_path.unlink()

    generated_now = 0
    for index in range(start_index, len(question_rows)):
        question = question_rows[index]["question"]
        scope_id = scope_sequence[index]
        result = generate_single_answer(
            llm=llm,
            question=question,
            scope_id=scope_id,
            memory_store=memory_store,
            graph_index=graph_index,
            embeddings_model=embeddings_model,
            config=config,
        )
        _write_jsonl_row(
            output_path,
            {"question": result["question"], "answer": result["answer"]},
        )
        _write_jsonl_row(
            trace_path,
            {
                "index": index,
                "question": result["question"],
                "answer": result["answer"],
                "scope_id": result["scope_id"],
                "rag_context": result["rag_context"],
                "memory_context": result["memory_context"],
                "retrieved_context_items": result["retrieved_context_items"],
            },
        )
        generated_now += 1
        if generated_now % max(1, config.save_every) == 0:
            _save_memory_snapshot(memory_path, memory_store)

    _save_memory_snapshot(memory_path, memory_store)
    return {
        "questions_path": str(questions_path),
        "static_graph_path": str(static_graph_path),
        "output_path": str(output_path),
        "trace_path": str(trace_path),
        "memory_path": str(memory_path),
        "total_questions": len(question_rows),
        "generated_now": generated_now,
        "resumed_rows": start_index,
        "questions_per_scope": resolved_questions_per_scope,
        "scope_source": scope_source,
        "scope_count": len(set(scope_sequence)),
        "backend": backend,
        "use_memory": config.use_memory,
        "reset_memory": config.reset_memory,
        "rag_top_k": config.rag_top_k,
        "memory_top_k": config.memory_top_k,
        "base_model_path": resolved_base_model_path,
        "graph_nodes": len(graph_index.node_ids),
        "pipeline": "standalone_answer_gen_graph_rag",
    }


def default_paths() -> dict[str, Path]:
    artifacts_dir = _ANSWER_GEN_DIR / "artifacts"
    return {
        "questions_path": DEFAULT_QUESTIONS_PATH,
        "static_graph_path": DEFAULT_STATIC_GRAPH_PATH,
        "output_path": artifacts_dir / "memory_qa.json",
        "trace_path": artifacts_dir / "memory_qa_trace.jsonl",
        "memory_path": artifacts_dir / "runtime_memory_snapshot.json",
        "summary_path": artifacts_dir / "memory_qa_summary.json",
    }
