                      
                                                

                                                                  

                                                       
                                                                              
                                                                      
                                                                       
                                                                            
                                                         

                                                            
   

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.pipeline.stage2_unlearning import LocalizationArtifacts, normalize_scores              


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [dict(row) for row in obj if isinstance(row, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_qa_rows(path: Path, *, role: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(load_json_or_jsonl(path)):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(
            row.get("answer")
            or row.get("output")
            or row.get("original_output")
            or row.get("reference_answer")
            or ""
        ).strip()
        if question and answer:
            rows.append({"instruction": question, "output": answer, "example_id": f"{role}_{idx}"})
    return rows


def sample_qa_rows(rows: list[dict[str, str]], *, limit: int, rng: random.Random) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:limit]


def apply_chat_template(tokenizer, instruction: str, output: str | None, *, add_generation_prompt: bool) -> str:
    messages = [{"role": "user", "content": instruction}]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def as_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def as_binary_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "toxic", "drop", "unlearned"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def unlearning_score(data: dict[str, Any], *, default: float = 0.0) -> float:
    for key in ("drop_prob", "punish", "edge_toxicity", "forget_weight", "forget_projection", "toxicity", "weight"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 1.0 if as_binary_flag(data.get("unlearned", 0)) else float(default)


def parse_layer_range(value: str) -> list[int]:
    selected: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("--layer-range must select at least one layer, e.g. 8-24.")
    return sorted(selected)


def extract_layer_index(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def resolve_decoder_layers(model):
    candidate_paths = [
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
    ]
    for path in candidate_paths:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise AttributeError(f"Could not resolve decoder layers for {type(model).__name__}")


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto" if args.device == "cuda" else {"": "cpu"},
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in args.new_lora_target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=int(args.new_lora_r),
        lora_alpha=int(args.new_lora_alpha),
        lora_dropout=float(args.new_lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def project_vector(vector: np.ndarray, hidden_size: int) -> np.ndarray:
    if vector.size == hidden_size:
        return vector.astype(np.float32)
    if vector.size > hidden_size:
        return vector[:hidden_size].astype(np.float32)
    out = np.zeros(hidden_size, dtype=np.float32)
    out[: vector.size] = vector
    return out


def toxic_edges_from_report(report: dict[str, Any]) -> set[tuple[str, str, int]]:
    edges: set[tuple[str, str, int]] = set()
    for item in report.get("toxic_edge_samples", []) or []:
        try:
            edges.add((str(item["source"]), str(item["target"]), int(item["key"])))
        except Exception:
            continue
    for row in report.get("reports", []) or []:
        for item in row.get("toxic_edge_samples", []) or []:
            try:
                edges.add((str(item["source"]), str(item["target"]), int(item["key"])))
            except Exception:
                continue
    return edges


def collect_rag_items(
    graph: nx.Graph,
    forget_weights: dict[str, float] | None,
    *,
    forget_threshold: float,
    retain_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forget_items: list[dict[str, Any]] = []
    retain_items: list[dict[str, Any]] = []
    forget_weights = forget_weights or {}
    for node_id, data in graph.nodes(data=True):
        vec = as_vector(data.get("embedding"))
        if vec is None:
            continue
        weight = float(forget_weights.get(str(node_id), unlearning_score(dict(data))) or 0.0)
        item = {
            "source": "rag_node",
            "id": str(node_id),
            "weight": weight,
            "match_key": "rag_node",
            "vector": l2_normalize(vec),
            "text": f"Q: {data.get('question', '')}\nA: {data.get('answer', '')}",
        }
        if "unlearned" in data:
            is_forget = as_binary_flag(data.get("unlearned"))
            is_retain = not is_forget
        else:
            is_forget = weight >= forget_threshold
            is_retain = weight <= retain_threshold
        if is_forget:
            forget_items.append(item)
        elif is_retain:
            retain_items.append(item)
    forget_items.sort(key=lambda item: item["weight"], reverse=True)
    retain_items.sort(key=lambda item: item["weight"])
    return forget_items, retain_items


def memory_endpoint_vector(graph: nx.MultiDiGraph, node_id: str) -> np.ndarray | None:
    if node_id not in graph.nodes:
        return None
    data = graph.nodes[node_id]
    vec = as_vector(data.get("embedding"))
    if vec is not None:
        return vec
    chunk_id = data.get("chunk_id")
    if chunk_id and str(chunk_id) in graph.nodes:
        return as_vector(graph.nodes[str(chunk_id)].get("embedding"))
    return None


def memory_edge_vector(graph: nx.MultiDiGraph, source: str, target: str, data: dict[str, Any]) -> np.ndarray | None:
    raw_vec = data.get("edge_embedding")
    if raw_vec is None:
        raw_vec = data.get("embedding")
    vec = as_vector(raw_vec)
    if vec is not None:
        return vec
    endpoint_vectors = [
        endpoint_vec
        for endpoint_vec in (memory_endpoint_vector(graph, source), memory_endpoint_vector(graph, target))
        if endpoint_vec is not None
    ]
    if not endpoint_vectors:
        return None
    return np.mean(np.stack(endpoint_vectors, axis=0), axis=0).astype(np.float32)


def collect_memory_items(
    graph: nx.MultiDiGraph,
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    toxic_edges = toxic_edges_from_report(report)
    forget_items: list[dict[str, Any]] = []
    retain_items: list[dict[str, Any]] = []
    toxic_node_scores: dict[str, float] = defaultdict(float)

    for source, target, key, data in graph.edges(keys=True, data=True):
        data = dict(data)
        score = unlearning_score(data)
        is_toxic = (
            as_binary_flag(data.get("unlearned", 0))
            or data.get("forget_status") == "toxic"
            or (str(source), str(target), int(key)) in toxic_edges
        )
        edge_type = str(data.get("edge_type") or data.get("relation") or "edge")
        vec = memory_edge_vector(graph, str(source), str(target), data)
        if vec is not None:
            item = {
                "source": "memory_edge",
                "id": f"{source}->{target}::{key}",
                "weight": score if is_toxic else 0.0,
                "match_key": f"memory_edge::{edge_type}",
                "vector": l2_normalize(vec),
                "text": str(data.get("raw_text") or data.get("text") or f"{source} {edge_type} {target}"),
            }
            if is_toxic:
                forget_items.append(item)
            else:
                retain_items.append(item)
        if is_toxic:
            toxic_node_scores[str(source)] = max(float(toxic_node_scores[str(source)]), float(score))
            toxic_node_scores[str(target)] = max(float(toxic_node_scores[str(target)]), float(score))

    for node_id, data in graph.nodes(data=True):
        vec = as_vector(data.get("embedding"))
        if vec is None:
            continue
        data = dict(data)
        node_type = str(data.get("node_type") or data.get("type") or "node")
        weight = max(unlearning_score(data), float(toxic_node_scores.get(str(node_id), 0.0)))
        is_toxic_node = as_binary_flag(data.get("unlearned", 0)) or str(node_id) in toxic_node_scores
        item = {
            "source": "memory_node",
            "id": str(node_id),
            "weight": weight if is_toxic_node else 0.0,
            "match_key": f"memory_node::{node_type}",
            "vector": l2_normalize(vec),
            "text": str(data.get("canonical_name") or data.get("text") or node_id),
        }
        if is_toxic_node:
            forget_items.append(item)
        else:
            retain_items.append(item)

    forget_items.sort(key=lambda item: item["weight"], reverse=True)
    retain_items.sort(key=lambda item: (item["match_key"], item["id"]))
    return forget_items, retain_items


def sample_items(items: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    indexed = list(items)
    rng.shuffle(indexed)
    return indexed[:limit]


def matched_retain_sample(
    forget_items: list[dict[str, Any]],
    retain_items: list[dict[str, Any]],
    *,
    max_retain: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if max_retain > 0:
        target_total = max_retain
    else:
        target_total = len(forget_items)
    desired_by_key = Counter(item["match_key"] for item in forget_items)
    retain_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in retain_items:
        retain_by_key[item["match_key"]].append(item)

    selected: list[dict[str, Any]] = []
    for key, desired in desired_by_key.items():
        bucket = list(retain_by_key.get(key, []))
        rng.shuffle(bucket)
        selected.extend(bucket[:desired])

    if len(selected) < target_total:
        selected_ids = {id(item) for item in selected}
        rest = [item for item in retain_items if id(item) not in selected_ids]
        rng.shuffle(rest)
        selected.extend(rest[: target_total - len(selected)])
    return selected[:target_total]


@contextmanager
def capture_mlp_inputs(model, allowed_layers: set[int]):
    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, input_args, _output):
            if not input_args:
                return
            value = input_args[0]
            if isinstance(value, torch.Tensor):
                activations[layer_idx] = value.detach()

        return hook

    for layer_idx, layer in enumerate(resolve_decoder_layers(model)):
        if layer_idx not in allowed_layers:
            continue
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is not None:
            handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))
    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def build_inputs(
    model,
    tokenizer,
    items: list[dict[str, Any]],
    *,
    anchor_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    hidden_size = int(model.get_input_embeddings().embedding_dim)
    suffix = tokenizer(anchor_text, return_tensors="pt", add_special_tokens=False)
    suffix_ids = suffix["input_ids"].to(device)
    suffix_mask = suffix["attention_mask"].to(device)
    suffix_emb = model.get_input_embeddings()(suffix_ids)
    prefix_vectors = [
        torch.tensor(project_vector(item["vector"], hidden_size), device=device, dtype=suffix_emb.dtype)
        for item in items
    ]
    prefix_emb = torch.stack(prefix_vectors, dim=0).unsqueeze(1)
    inputs_embeds = torch.cat([prefix_emb, suffix_emb.repeat(len(items), 1, 1)], dim=1)
    prefix_mask = torch.ones((len(items), 1), dtype=suffix_mask.dtype, device=device)
    attention_mask = torch.cat([prefix_mask, suffix_mask.repeat(len(items), 1)], dim=1)
    return inputs_embeds, attention_mask


def build_qa_batch(
    tokenizer,
    rows: list[dict[str, str]],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    rendered: list[str] = []
    prompt_lens: list[int] = []
    for row in rows:
        prompt = apply_chat_template(tokenizer, row["instruction"], None, add_generation_prompt=True)
        full = apply_chat_template(tokenizer, row["instruction"], row["output"], add_generation_prompt=False)
        prompt_lens.append(len(tokenizer(prompt, add_special_tokens=False, truncation=True)["input_ids"]))
        rendered.append(full)
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    answer_mask = encoded["attention_mask"].clone()
    for row_idx, prompt_len in enumerate(prompt_lens):
        answer_mask[row_idx, : min(prompt_len, answer_mask.shape[1])] = 0
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "answer_mask": answer_mask,
    }


def build_text_batch(
    tokenizer,
    items: list[dict[str, Any]],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    texts = [str(item.get("text") or item.get("id") or "").strip() for item in items]
    texts = [text if text else "Graph item." for text in texts]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)


def collect_firing_stats(
    model,
    tokenizer,
    items: list[dict[str, Any]],
    *,
    allowed_layers: set[int],
    batch_size: int,
    anchor_text: str,
    activation_position: str,
) -> tuple[dict[int, np.ndarray], int]:
    sums: dict[int, np.ndarray] = {}
    count = 0
    with capture_mlp_inputs(model, allowed_layers) as captured:
        for start in tqdm(range(0, len(items), batch_size), desc="MLP firing", leave=False):
            batch_items = items[start : start + batch_size]
            inputs_embeds, attention_mask = build_inputs(model, tokenizer, batch_items, anchor_text=anchor_text)
            captured.clear()
            with torch.no_grad():
                model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
            for layer_idx, act in captured.items():
                if activation_position == "prefix":
                    reduced = act[:, 0, :]
                elif activation_position == "all":
                    reduced = act.mean(dim=1)
                else:
                    raise ValueError(f"Unsupported activation_position: {activation_position}")
                values = reduced.detach().float().abs().cpu().numpy()
                sums[layer_idx] = values.sum(axis=0) if layer_idx not in sums else sums[layer_idx] + values.sum(axis=0)
            count += len(batch_items)
    return sums, count


def collect_text_firing_stats(
    model,
    tokenizer,
    items: list[dict[str, Any]],
    *,
    allowed_layers: set[int],
    batch_size: int,
    max_length: int,
) -> tuple[dict[int, np.ndarray], int]:
    sums: dict[int, np.ndarray] = {}
    count = 0
    device = next(model.parameters()).device
    with capture_mlp_inputs(model, allowed_layers) as captured:
        for start in tqdm(range(0, len(items), batch_size), desc="Text MLP firing", leave=False):
            batch_items = items[start : start + batch_size]
            batch = build_text_batch(tokenizer, batch_items, max_length=max_length)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            captured.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            for layer_idx, act in captured.items():
                mask = attention_mask.to(act.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                reduced = (act * mask).sum(dim=1) / denom
                values = reduced.detach().float().abs().cpu().numpy()
                sums[layer_idx] = values.sum(axis=0) if layer_idx not in sums else sums[layer_idx] + values.sum(axis=0)
            count += len(batch_items)
    return sums, count


def collect_qa_firing_stats(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    *,
    allowed_layers: set[int],
    batch_size: int,
    max_length: int,
) -> tuple[dict[int, np.ndarray], int]:
    sums: dict[int, np.ndarray] = {}
    count = 0
    device = next(model.parameters()).device
    with capture_mlp_inputs(model, allowed_layers) as captured:
        for start in tqdm(range(0, len(rows), batch_size), desc="QA MLP firing", leave=False):
            batch_rows = rows[start : start + batch_size]
            batch = build_qa_batch(tokenizer, batch_rows, max_length=max_length)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            captured.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            for layer_idx, act in captured.items():
                mask = answer_mask.to(act.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                reduced = (act * mask).sum(dim=1) / denom
                values = reduced.detach().float().abs().cpu().numpy()
                sums[layer_idx] = values.sum(axis=0) if layer_idx not in sums else sums[layer_idx] + values.sum(axis=0)
            count += len(batch_rows)
    return sums, count


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_value = float(values.max()) if values.size else 0.0
    min_value = float(values.min()) if values.size else 0.0
    if max_value - min_value <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def contrast_ratio(forget_mean: np.ndarray, retain_mean: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    forget_mean = np.asarray(forget_mean, dtype=np.float32)
    retain_mean = np.asarray(retain_mean, dtype=np.float32)
    return ((forget_mean - retain_mean) / (forget_mean + retain_mean + eps)).astype(np.float32)


def retain_to_forget_ratio(forget_mean: np.ndarray, retain_mean: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    forget_mean = np.asarray(forget_mean, dtype=np.float32)
    retain_mean = np.asarray(retain_mean, dtype=np.float32)
    return (retain_mean / (forget_mean + eps)).astype(np.float32)


def build_lora_masks_from_neurons(
    model,
    selected_by_layer: dict[int, list[int]],
    *,
    include_down_proj: bool,
    include_down_proj_lora_b: bool,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if param.ndim != 2 or "lora_" not in name:
            continue
        layer_idx = extract_layer_index(name)
        if layer_idx is None or layer_idx not in selected_by_layer:
            continue
        neurons = selected_by_layer[layer_idx]
        mask = np.zeros(tuple(param.shape), dtype=np.float32)
        if "lora_B" in name and (".gate_proj." in name or ".up_proj." in name):
            rows = [idx for idx in neurons if 0 <= idx < mask.shape[0]]
            if rows:
                mask[rows, :] = 1.0
        elif include_down_proj and "lora_A" in name and ".down_proj." in name:
            cols = [idx for idx in neurons if 0 <= idx < mask.shape[1]]
            if cols:
                mask[:, cols] = 1.0
        elif include_down_proj_lora_b and "lora_B" in name and ".down_proj." in name:
                                                                             
                                                                                
            mask[:, :] = 1.0
        if mask.sum() > 0:
            masks[name] = mask
    return masks


def collect_lora_parameter_shapes(model) -> dict[str, list[int]]:
                                                                               
    shapes: dict[str, list[int]] = {}
    for name, param in model.named_parameters():
        if param.ndim == 2 and "lora_" in name:
            shapes[name] = [int(dim) for dim in tuple(param.shape)]
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize native-LLM graph-sensitive MLP neurons.")
    parser.add_argument("--base-model-path", required=True, help="Merged native_llm path.")
    parser.add_argument("--rag-graph-path", required=True)
    parser.add_argument("--rag-forget-weight-path", default="")
    parser.add_argument("--memory-graph-path", required=True)
    parser.add_argument("--memory-report-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rag-forget-threshold", type=float, default=0.5)
    parser.add_argument("--rag-retain-threshold", type=float, default=0.15)
    parser.add_argument("--max-rag-forget", type=int, default=0)
    parser.add_argument("--max-memory-forget", type=int, default=0)
    parser.add_argument("--max-retain", type=int, default=0, help="0 means match retain count to forget count.")
    parser.add_argument("--layer-range", default="8-24")
    parser.add_argument("--neurons-per-layer", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--activation-position", choices=["prefix", "all"], default="prefix")
    parser.add_argument("--anchor-text", default="Graph concept probe.")
    parser.add_argument(
        "--use-item-text-view",
        action="store_true",
        help="Use each graph item's own text as the text-view probe instead of external QA files.",
    )
    parser.add_argument("--item-text-batch-size", type=int, default=4)
    parser.add_argument("--item-text-max-length", type=int, default=256)
    parser.add_argument("--qa-forget-path", default="")
    parser.add_argument("--qa-retain-path", default="")
    parser.add_argument("--qa-max-forget", type=int, default=0)
    parser.add_argument("--qa-max-retain", type=int, default=0)
    parser.add_argument("--qa-batch-size", type=int, default=1)
    parser.add_argument("--qa-max-length", type=int, default=512)
    parser.add_argument("--graph-view-weight", type=float, default=1.0)
    parser.add_argument("--qa-view-weight", type=float, default=1.0)
    parser.add_argument(
        "--ranking-mode",
        choices=["positive_diff", "forget_retain_ratio"],
        default="positive_diff",
        help=(
            "Neuron ranking objective. positive_diff preserves the historical "
            "max(0, forget-retain) ranking. forget_retain_ratio ranks by "
            "(forget-retain)/(forget+retain) and applies retain-contamination filters."
        ),
    )
    parser.add_argument(
        "--retain-ratio-max",
        type=float,
        default=0.8,
        help="Strict ranking only: keep graph-view neurons with retain/(forget+eps) at or below this value.",
    )
    parser.add_argument(
        "--qa-retain-ratio-max",
        type=float,
        default=0.8,
        help="Strict ranking only: keep QA-view neurons with retain/(forget+eps) at or below this value.",
    )
    parser.add_argument(
        "--min-graph-diff",
        type=float,
        default=0.0,
        help="Strict ranking only: minimum graph-view absolute forget-retain firing gap.",
    )
    parser.add_argument(
        "--min-qa-diff",
        type=float,
        default=0.0,
        help="Strict ranking only: minimum QA-view absolute forget-retain firing gap when QA view is enabled.",
    )
    parser.add_argument(
        "--min-strict-score",
        type=float,
        default=0.0,
        help="Strict ranking only: minimum fused ratio score. Raising this allows under-filled layers.",
    )
    parser.add_argument(
        "--positive-diff-retain-ratio-max",
        type=float,
        default=0.0,
        help=(
            "Positive-diff ranking only: when >0, apply a hard retain filter "
            "retain/(forget+eps) <= this value before top-K selection."
        ),
    )
    parser.add_argument("--new-lora-r", type=int, default=16)
    parser.add_argument("--new-lora-alpha", type=int, default=32)
    parser.add_argument("--new-lora-dropout", type=float, default=0.05)
    parser.add_argument("--new-lora-target-modules", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--no-down-proj-mask", action="store_true")
    parser.add_argument("--no-down-proj-lora-b", action="store_true")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Keep only neurons with positive forget-retain firing difference. Default keeps exact top-K per layer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rag_graph = load_pickle(Path(args.rag_graph_path))
    rag_weights = load_pickle(Path(args.rag_forget_weight_path)) if args.rag_forget_weight_path else {}
    memory_graph = load_pickle(Path(args.memory_graph_path))
    memory_report = json.loads(Path(args.memory_report_path).read_text(encoding="utf-8")) if args.memory_report_path else {}

    rag_forget, rag_retain = collect_rag_items(
        rag_graph,
        rag_weights,
        forget_threshold=args.rag_forget_threshold,
        retain_threshold=args.rag_retain_threshold,
    )
    memory_forget, memory_retain = collect_memory_items(memory_graph, memory_report)
    rag_forget = sample_items(rag_forget, args.max_rag_forget, rng)
    memory_forget = sample_items(memory_forget, args.max_memory_forget, rng)
    forget_items = rag_forget + memory_forget
    retain_items = matched_retain_sample(
        forget_items,
        rag_retain + memory_retain,
        max_retain=args.max_retain,
        rng=rng,
    )
    if not forget_items or not retain_items:
        raise ValueError(f"Need non-empty probes, got forget={len(forget_items)} retain={len(retain_items)}")

    dims = sorted({int(item["vector"].size) for item in forget_items + retain_items})
    print(f"[Vectors] forget={len(forget_items)} retain={len(retain_items)} dims={dims}")
    print(f"[Vectors] forget_sources={dict(Counter(item['source'] for item in forget_items))}")
    print(f"[Vectors] retain_sources={dict(Counter(item['source'] for item in retain_items))}")

    tokenizer, model = build_model_and_tokenizer(args)
    hidden_size = int(model.get_input_embeddings().embedding_dim)
    print(f"[Model] base={args.base_model_path}")
    print(f"[Model] token_embedding_dim={hidden_size}")
    if dims != [hidden_size]:
        print(f"[Warning] graph vector dims {dims} differ from hidden size {hidden_size}; vectors will be projected.")

    allowed_layers = set(parse_layer_range(args.layer_range))
    forget_sums, forget_count = collect_firing_stats(
        model,
        tokenizer,
        forget_items,
        allowed_layers=allowed_layers,
        batch_size=max(1, args.batch_size),
        anchor_text=args.anchor_text,
        activation_position=args.activation_position,
    )
    retain_sums, retain_count = collect_firing_stats(
        model,
        tokenizer,
        retain_items,
        allowed_layers=allowed_layers,
        batch_size=max(1, args.batch_size),
        anchor_text=args.anchor_text,
        activation_position=args.activation_position,
    )
    print(f"[Probe] counted forget={forget_count} retain={retain_count}")

    text_view_enabled = bool(args.use_item_text_view)
    qa_enabled = bool((not text_view_enabled) and args.qa_forget_path.strip() and args.qa_retain_path.strip())
    qa_forget_rows: list[dict[str, str]] = []
    qa_retain_rows: list[dict[str, str]] = []
    qa_forget_sums: dict[int, np.ndarray] = {}
    qa_retain_sums: dict[int, np.ndarray] = {}
    qa_forget_count = 0
    qa_retain_count = 0
    if text_view_enabled:
        print(f"[Text View] forget={len(forget_items)} retain={len(retain_items)}")
        qa_forget_sums, qa_forget_count = collect_text_firing_stats(
            model,
            tokenizer,
            forget_items,
            allowed_layers=allowed_layers,
            batch_size=max(1, args.item_text_batch_size),
            max_length=max(1, args.item_text_max_length),
        )
        qa_retain_sums, qa_retain_count = collect_text_firing_stats(
            model,
            tokenizer,
            retain_items,
            allowed_layers=allowed_layers,
            batch_size=max(1, args.item_text_batch_size),
            max_length=max(1, args.item_text_max_length),
        )
        print(f"[Text Probe] counted forget={qa_forget_count} retain={qa_retain_count}")
    elif qa_enabled:
        qa_forget_rows = sample_qa_rows(
            normalize_qa_rows(Path(args.qa_forget_path), role="qa_forget"),
            limit=args.qa_max_forget,
            rng=rng,
        )
        qa_retain_rows = sample_qa_rows(
            normalize_qa_rows(Path(args.qa_retain_path), role="qa_retain"),
            limit=args.qa_max_retain,
            rng=rng,
        )
        if not qa_forget_rows or not qa_retain_rows:
            raise ValueError(
                f"QA view requires non-empty forget/retain rows, got "
                f"forget={len(qa_forget_rows)} retain={len(qa_retain_rows)}"
            )
        print(f"[QA] forget={len(qa_forget_rows)} retain={len(qa_retain_rows)}")
        qa_forget_sums, qa_forget_count = collect_qa_firing_stats(
            model,
            tokenizer,
            qa_forget_rows,
            allowed_layers=allowed_layers,
            batch_size=max(1, args.qa_batch_size),
            max_length=max(1, args.qa_max_length),
        )
        qa_retain_sums, qa_retain_count = collect_qa_firing_stats(
            model,
            tokenizer,
            qa_retain_rows,
            allowed_layers=allowed_layers,
            batch_size=max(1, args.qa_batch_size),
            max_length=max(1, args.qa_max_length),
        )
        print(f"[QA Probe] counted forget={qa_forget_count} retain={qa_retain_count}")

    selected_by_layer: dict[int, list[int]] = {}
    score_by_layer: dict[int, list[float]] = {}
    layer_report: dict[str, dict[str, float]] = {}
    candidate_cache_arrays: dict[str, np.ndarray] = {}
    for layer_idx in sorted(allowed_layers):
        if layer_idx not in forget_sums or layer_idx not in retain_sums:
            continue
        forget_mean = forget_sums[layer_idx] / max(1, forget_count)
        retain_mean = retain_sums[layer_idx] / max(1, retain_count)
        graph_diff = np.maximum(0.0, forget_mean - retain_mean).astype(np.float32)
        graph_ratio = retain_to_forget_ratio(forget_mean, retain_mean)
        graph_contrast = contrast_ratio(forget_mean, retain_mean)
        if qa_enabled and layer_idx in qa_forget_sums and layer_idx in qa_retain_sums:
            qa_forget_mean = qa_forget_sums[layer_idx] / max(1, qa_forget_count)
            qa_retain_mean = qa_retain_sums[layer_idx] / max(1, qa_retain_count)
            qa_diff = np.maximum(0.0, qa_forget_mean - qa_retain_mean).astype(np.float32)
            qa_ratio = retain_to_forget_ratio(qa_forget_mean, qa_retain_mean)
            qa_contrast = contrast_ratio(qa_forget_mean, qa_retain_mean)
            combined_forget_mean = (forget_sums[layer_idx] + qa_forget_sums[layer_idx]) / max(
                1, forget_count + qa_forget_count
            )
            combined_retain_mean = (retain_sums[layer_idx] + qa_retain_sums[layer_idx]) / max(
                1, retain_count + qa_retain_count
            )
        else:
            qa_forget_mean = np.zeros_like(forget_mean, dtype=np.float32)
            qa_retain_mean = np.zeros_like(retain_mean, dtype=np.float32)
            qa_diff = np.zeros_like(graph_diff, dtype=np.float32)
            qa_ratio = np.full_like(graph_ratio, np.inf, dtype=np.float32)
            qa_contrast = np.zeros_like(graph_contrast, dtype=np.float32)
            combined_forget_mean = forget_mean
            combined_retain_mean = retain_mean
        combined_raw_diff = (combined_forget_mean - combined_retain_mean).astype(np.float32)
        combined_diff = np.maximum(0.0, combined_raw_diff).astype(np.float32)
        combined_ratio = retain_to_forget_ratio(combined_forget_mean, combined_retain_mean)
        combined_contrast = contrast_ratio(combined_forget_mean, combined_retain_mean)
        layer_key = f"layer_{layer_idx}"
        candidate_cache_arrays[f"{layer_key}__neuron_index"] = np.arange(combined_raw_diff.size, dtype=np.int32)
        candidate_cache_arrays[f"{layer_key}__graph_forget_mean"] = forget_mean.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__graph_retain_mean"] = retain_mean.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__graph_raw_diff"] = (forget_mean - retain_mean).astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__graph_positive_diff"] = graph_diff.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__graph_retain_to_forget_ratio"] = graph_ratio.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__graph_contrast_ratio"] = graph_contrast.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__aux_forget_mean"] = qa_forget_mean.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__aux_retain_mean"] = qa_retain_mean.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__aux_raw_diff"] = (qa_forget_mean - qa_retain_mean).astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__aux_positive_diff"] = qa_diff.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__aux_retain_to_forget_ratio"] = qa_ratio.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__aux_contrast_ratio"] = qa_contrast.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__combined_forget_mean"] = combined_forget_mean.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__combined_retain_mean"] = combined_retain_mean.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__combined_raw_diff"] = combined_raw_diff.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__combined_positive_diff"] = combined_diff.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__combined_retain_to_forget_ratio"] = combined_ratio.astype(np.float32)
        candidate_cache_arrays[f"{layer_key}__combined_contrast_ratio"] = combined_contrast.astype(np.float32)
        if args.ranking_mode == "forget_retain_ratio":
            graph_keep = (
                (graph_diff >= float(args.min_graph_diff))
                & (graph_ratio <= float(args.retain_ratio_max))
                & (graph_contrast > 0.0)
            )
            if qa_enabled:
                qa_keep = (
                    (qa_diff >= float(args.min_qa_diff))
                    & (qa_ratio <= float(args.qa_retain_ratio_max))
                    & (qa_contrast > 0.0)
                )
            else:
                qa_keep = np.ones_like(graph_keep, dtype=bool)
            keep_mask = graph_keep & qa_keep
            diff = (
                float(args.graph_view_weight) * graph_contrast
                + float(args.qa_view_weight) * qa_contrast
            ).astype(np.float32)
            diff = np.where(keep_mask, diff, -np.inf).astype(np.float32)
        else:
                                                                               
                                                                                
            diff = combined_raw_diff.astype(np.float32)
            if float(args.positive_diff_retain_ratio_max) > 0.0:
                keep_mask = (
                    (diff > 0.0)
                    & (combined_ratio <= float(args.positive_diff_retain_ratio_max))
                )
                diff = np.where(keep_mask, diff, -np.inf).astype(np.float32)
            else:
                keep_mask = np.ones_like(combined_diff, dtype=bool)
        top_k = min(int(args.neurons_per_layer), int(diff.size))
        finite_ranked = np.argsort(-diff)[:top_k]
        if args.ranking_mode == "forget_retain_ratio":
            ranked = [
                int(idx)
                for idx in finite_ranked.tolist()
                if np.isfinite(diff[idx]) and float(diff[idx]) >= float(args.min_strict_score)
            ]
        elif args.positive_only:
            ranked = [int(idx) for idx in finite_ranked.tolist() if float(diff[idx]) > 0.0]
        else:
            ranked = [int(idx) for idx in finite_ranked.tolist()]
        selected_by_layer[layer_idx] = ranked
        scores = [float(diff[idx]) for idx in ranked]
        score_by_layer[layer_idx] = scores
        finite_scores = diff[np.isfinite(diff)]
        layer_report[layer_key] = {
            "selected_neurons": int(len(ranked)),
            "candidate_neurons_after_filter": int(keep_mask.sum()),
            "forget_mean_abs_mlp_firing": float(forget_mean.mean()),
            "retain_mean_abs_mlp_firing": float(retain_mean.mean()),
            "graph_positive_diff_mean": float(graph_diff.mean()),
            "graph_positive_diff_max": float(graph_diff.max()),
            "graph_contrast_ratio_mean": float(graph_contrast.mean()),
            "graph_retain_to_forget_ratio_mean": float(np.nan_to_num(graph_ratio, nan=0.0, posinf=0.0).mean()),
            "qa_forget_mean_abs_mlp_firing": float(qa_forget_mean.mean()) if qa_enabled else 0.0,
            "qa_retain_mean_abs_mlp_firing": float(qa_retain_mean.mean()) if qa_enabled else 0.0,
            "qa_positive_diff_mean": float(qa_diff.mean()) if qa_enabled else 0.0,
            "qa_positive_diff_max": float(qa_diff.max()) if qa_enabled else 0.0,
            "qa_contrast_ratio_mean": float(qa_contrast.mean()) if qa_enabled else 0.0,
            "qa_retain_to_forget_ratio_mean": (
                float(np.nan_to_num(qa_ratio, nan=0.0, posinf=0.0).mean()) if qa_enabled else 0.0
            ),
            "combined_forget_mean_abs_mlp_firing": float(combined_forget_mean.mean()),
            "combined_retain_mean_abs_mlp_firing": float(combined_retain_mean.mean()),
            "combined_positive_diff_mean": float(combined_diff.mean()),
            "combined_positive_diff_max": float(combined_diff.max()),
            "combined_contrast_ratio_mean": float(combined_contrast.mean()),
            "combined_retain_to_forget_ratio_mean": float(
                np.nan_to_num(combined_ratio, nan=0.0, posinf=0.0).mean()
            ),
            "fused_score_mean": float(finite_scores.mean()) if finite_scores.size else 0.0,
            "fused_score_max": float(finite_scores.max()) if finite_scores.size else 0.0,
        }

    target_neurons = {f"layer_{layer}": rows for layer, rows in selected_by_layer.items() if rows}
    if not target_neurons:
        raise ValueError("No positive forget-retain MLP firing neurons were selected.")
    layer_scores = {}
    for layer_idx, scores in score_by_layer.items():
        arr = np.asarray(scores, dtype=np.float32)
        layer_scores[f"layer_{layer_idx}"] = [float(x) for x in normalize_scores(arr).tolist()] if arr.size else []

    parameter_masks = build_lora_masks_from_neurons(
        model,
        selected_by_layer,
        include_down_proj=not args.no_down_proj_mask,
        include_down_proj_lora_b=not args.no_down_proj_lora_b,
    )
    total_selected_params = int(sum(mask.sum() for mask in parameter_masks.values()))
    total_candidate_params = int(sum(mask.size for mask in parameter_masks.values()))
    lora_parameter_shapes = collect_lora_parameter_shapes(model)

    localization = LocalizationArtifacts(
        strategy="native_llm_graph_vector_mlp_neuron_localization",
        top_fraction=1.0,
        target_neurons=target_neurons,
        layer_scores=layer_scores,
        total_target_neurons=int(sum(len(rows) for rows in target_neurons.values())),
        parameter_masks=parameter_masks,
        metadata={
            "method_name": "native_llm_graph_vector_mlp_neuron_localization",
            "view_fusion": (
                "graph_embedding_plus_item_text"
                if text_view_enabled
                else ("graph_embedding_plus_qa_text" if qa_enabled else "graph_embedding_only")
            ),
            "base_model_path": args.base_model_path,
            "old_lora_path": "",
            "uses_merged_native_llm": True,
            "fresh_lora_mask_only": True,
            "new_lora_r": int(args.new_lora_r),
            "new_lora_alpha": int(args.new_lora_alpha),
            "new_lora_dropout": float(args.new_lora_dropout),
            "new_lora_target_modules": args.new_lora_target_modules,
            "rag_graph_path": args.rag_graph_path,
            "rag_forget_weight_path": args.rag_forget_weight_path,
            "memory_graph_path": args.memory_graph_path,
            "memory_report_path": args.memory_report_path,
            "forget_count": int(forget_count),
            "retain_count": int(retain_count),
            "forget_source_counts": dict(Counter(item["source"] for item in forget_items)),
            "retain_source_counts": dict(Counter(item["source"] for item in retain_items)),
            "text_view_enabled": bool(text_view_enabled),
            "item_text_batch_size": int(args.item_text_batch_size),
            "item_text_max_length": int(args.item_text_max_length),
            "qa_enabled": bool(qa_enabled),
            "qa_forget_path": args.qa_forget_path,
            "qa_retain_path": args.qa_retain_path,
            "qa_forget_count": int(qa_forget_count),
            "qa_retain_count": int(qa_retain_count),
            "qa_max_forget": int(args.qa_max_forget),
            "qa_max_retain": int(args.qa_max_retain),
            "qa_batch_size": int(args.qa_batch_size),
            "qa_max_length": int(args.qa_max_length),
            "graph_view_weight": float(args.graph_view_weight),
            "qa_view_weight": float(args.qa_view_weight),
            "ranking_mode": args.ranking_mode,
            "retain_ratio_max": float(args.retain_ratio_max),
            "positive_diff_retain_ratio_max": float(args.positive_diff_retain_ratio_max),
            "qa_retain_ratio_max": float(args.qa_retain_ratio_max),
            "min_graph_diff": float(args.min_graph_diff),
            "min_qa_diff": float(args.min_qa_diff),
            "min_strict_score": float(args.min_strict_score),
            "graph_vector_dims": dims,
            "model_hidden_size": int(hidden_size),
            "layer_range": args.layer_range,
            "neurons_per_layer": int(args.neurons_per_layer),
            "activation_source": "mlp.down_proj input",
            "activation_position": args.activation_position,
            "mask_mapping": "native MLP neurons -> fresh LoRA gate/up lora_B rows, down lora_A columns, and optional down lora_B matrices",
            "include_down_proj_lora_b": bool(not args.no_down_proj_lora_b),
            "layer_report": layer_report,
            "total_selected_params": total_selected_params,
            "total_candidate_params": total_candidate_params,
            "parameter_mask_density": float(total_selected_params / max(1, total_candidate_params)),
        },
    )

    save_pickle(output_dir / "localized_neurons.pkl", localization)
    save_json(output_dir / "localized_neurons.json", localization.to_serializable())
    save_json(output_dir / "native_llm_graph_neuron_localizer_report.json", localization.metadata)
    candidate_cache_path = output_dir / "candidate_neuron_stats.npz"
    np.savez_compressed(candidate_cache_path, **candidate_cache_arrays)
    save_json(
        output_dir / "candidate_neuron_stats_manifest.json",
        {
            "cache_format": "native_llm_candidate_neuron_stats_npz_v1",
            "cache_path": str(candidate_cache_path),
            "score_definition": "combined_raw_diff = combined_forget_mean - combined_retain_mean",
            "retain_ratio_definition": "combined_retain_to_forget_ratio = combined_retain_mean / (combined_forget_mean + 1e-6)",
            "aux_view": "item_text" if text_view_enabled else ("qa" if qa_enabled else "none"),
            "fields_per_layer": [
                "neuron_index",
                "graph_forget_mean",
                "graph_retain_mean",
                "graph_raw_diff",
                "graph_positive_diff",
                "graph_retain_to_forget_ratio",
                "graph_contrast_ratio",
                "aux_forget_mean",
                "aux_retain_mean",
                "aux_raw_diff",
                "aux_positive_diff",
                "aux_retain_to_forget_ratio",
                "aux_contrast_ratio",
                "combined_forget_mean",
                "combined_retain_mean",
                "combined_raw_diff",
                "combined_positive_diff",
                "combined_retain_to_forget_ratio",
                "combined_contrast_ratio",
            ],
            "layers": sorted(int(k.split("__", 1)[0].split("_")[1]) for k in candidate_cache_arrays if k.endswith("__neuron_index")),
            "model_hidden_size": int(hidden_size),
            "new_lora_r": int(args.new_lora_r),
            "new_lora_alpha": int(args.new_lora_alpha),
            "new_lora_dropout": float(args.new_lora_dropout),
            "new_lora_target_modules": args.new_lora_target_modules,
            "lora_parameter_shapes": lora_parameter_shapes,
            "source_localization_pkl": str(output_dir / "localized_neurons.pkl"),
            "source_localization_json": str(output_dir / "localized_neurons.json"),
            "source_report_json": str(output_dir / "native_llm_graph_neuron_localizer_report.json"),
        },
    )
    save_json(
        output_dir / "probe_summary.json",
        {
            "forget_count": len(forget_items),
            "retain_count": len(retain_items),
            "forget_sources": dict(Counter(item["source"] for item in forget_items)),
            "retain_sources": dict(Counter(item["source"] for item in retain_items)),
            "text_view_enabled": bool(text_view_enabled),
            "qa_enabled": bool(qa_enabled),
            "qa_forget_count": int(qa_forget_count),
            "qa_retain_count": int(qa_retain_count),
            "graph_view_weight": float(args.graph_view_weight),
            "qa_view_weight": float(args.qa_view_weight),
            "selected_layers": sorted(int(k.split("_")[1]) for k in target_neurons),
            "total_target_neurons": localization.total_target_neurons,
            "total_selected_params": total_selected_params,
            "parameter_mask_density": localization.metadata["parameter_mask_density"],
        },
    )
    print(f"[Save] {output_dir / 'localized_neurons.pkl'}")
    print(f"[Save] {output_dir / 'localized_neurons.json'}")
    print(f"[Save] {candidate_cache_path}")
    print(f"[Save] {output_dir / 'candidate_neuron_stats_manifest.json'}")
    print(
        f"[Summary] layers={sorted(target_neurons)} "
        f"target_neurons={localization.total_target_neurons} "
        f"mask_density={localization.metadata['parameter_mask_density']:.6f}"
    )


if __name__ == "__main__":
    main()
