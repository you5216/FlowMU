                     

                      
                           
                                      
                      
                                 
   

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .config import Stage2Config
from .entity_dataset import build_forget_entity_dataset
from .support import (
    build_model,
    collate_chat_examples,
    cycle_dataloader,
    extract_answer_span_hidden_states,
    get_model_device,
    load_json_or_jsonl,
    move_batch_to_device,
    release_memory,
    select_hctl_strongest_layers,
)
from .stage2_unlearning import register_gradient_mask_hooks, remove_gradient_mask_hooks

logger = logging.getLogger("GAT-AFRU.Phase2.MaskedOptimization")

AGENT_ROOT = Path(__file__).resolve().parents[2]



def _stable_json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _retain_dataset_fingerprint(retain_dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for item in retain_dataset:
        digest.update(
            json.dumps(
                {
                    "example_id": item.get("example_id"),
                    "instruction": item.get("instruction"),
                    "output": item.get("output"),
                },
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _retain_teacher_cache_metadata(
    *,
    cfg: Stage2Config,
    tokenizer,
    retain_dataset: Dataset,
) -> Dict[str, Any]:
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    retain_path = Path(cfg.retain_path)
    retain_stat = retain_path.stat() if retain_path.exists() else None
    teacher_source = {
        "base_model_path": str(Path(cfg.base_model_path).resolve()),
        "init_new_lora": bool(cfg.init_new_lora),
        "init_new_lora_on_frozen": bool(cfg.init_new_lora_on_frozen),
        "original_lora_path": str(Path(cfg.original_lora_path).resolve())
        if (not cfg.init_new_lora or cfg.init_new_lora_on_frozen)
        else "",
    }
    return {
        "version": 1,
        "teacher_source": teacher_source,
        "retain_path": str(retain_path.resolve()) if retain_path.exists() else str(retain_path),
        "retain_mtime_ns": int(retain_stat.st_mtime_ns) if retain_stat is not None else None,
        "retain_size": int(retain_stat.st_size) if retain_stat is not None else None,
        "retain_rows": int(len(retain_dataset)),
        "retain_dataset_fingerprint": _retain_dataset_fingerprint(retain_dataset),
        "max_length": int(cfg.max_length),
        "retain_topk": int(cfg.retain_topk),
        "tokenizer_name_or_path": str(tokenizer_name) if tokenizer_name else "",
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _retain_teacher_cache_path(metadata: Dict[str, Any]) -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / "diagnostics" / "retain_teacher_cache"
    return cache_dir / f"retain_teacher_topk_{_stable_json_hash(metadata)[:16]}.pt"


def _torch_load_cache(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _compute_gradient_ascent_forget_loss(forget_ce_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
                                                                                  
    return {
        "loss": -forget_ce_loss,
        "forget_ce": forget_ce_loss.detach(),
    }


def _trainable_parameters_with_names(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _collect_parameter_grads(
    loss: torch.Tensor,
    named_params: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    params = [param for _, param in named_params]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return {
        name: grad.detach()
        for (name, _), grad in zip(named_params, grads)
        if grad is not None
    }


def _collect_parameter_grads_with_backward(
    *,
    model,
    loss: torch.Tensor,
    named_params: list[tuple[str, torch.nn.Parameter]],
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
                                                                                        
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    grads: dict[str, torch.Tensor] = {}
    for name, param in named_params:
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return grads


def _assign_projected_gradient_ascent_grads(
    *,
    model,
    named_params: list[tuple[str, torch.nn.Parameter]],
    parameter_masks: Dict[str, np.ndarray],
    forget_ce_loss: torch.Tensor,
    retain_kl_loss: torch.Tensor,
    forget_weight: float,
    retain_kl_weight: float,
    gradient_accumulation_steps: int,
) -> dict[str, torch.Tensor]:
                                                                                           

                                                                              
                                                                            
                                                                               
                                                           
       
    forget_grads = _collect_parameter_grads_with_backward(
        model=model,
        loss=forget_ce_loss,
        named_params=named_params,
        retain_graph=True,
    )
    retain_grads = _collect_parameter_grads_with_backward(
        model=model,
        loss=retain_kl_loss,
        named_params=named_params,
        retain_graph=False,
    )
    for name, param in named_params:
        mask_np = parameter_masks.get(name)
        if mask_np is None:
            continue
        mask = torch.as_tensor(mask_np, device=param.device, dtype=torch.float32)
        if name in forget_grads:
            forget_grads[name] = forget_grads[name].float() * mask
        if name in retain_grads:
            retain_grads[name] = retain_grads[name].float() * mask
    dot = None
    retain_norm_sq = None
    forget_norm_sq = None
    for name, retain_grad in retain_grads.items():
        forget_grad = forget_grads.get(name)
        if forget_grad is None:
            continue
        cur_dot = torch.sum(forget_grad.float() * retain_grad.float())
        cur_retain_norm_sq = torch.sum(retain_grad.float() * retain_grad.float())
        cur_forget_norm_sq = torch.sum(forget_grad.float() * forget_grad.float())
        dot = cur_dot if dot is None else dot + cur_dot
        retain_norm_sq = cur_retain_norm_sq if retain_norm_sq is None else retain_norm_sq + cur_retain_norm_sq
        forget_norm_sq = cur_forget_norm_sq if forget_norm_sq is None else forget_norm_sq + cur_forget_norm_sq
    if dot is None or retain_norm_sq is None:
        dot = forget_ce_loss.new_tensor(0.0)
        retain_norm_sq = forget_ce_loss.new_tensor(0.0)
    if forget_norm_sq is None:
        forget_norm_sq = forget_ce_loss.new_tensor(0.0)

    eps = forget_ce_loss.new_tensor(1e-12)
    projection_coeff = torch.where(
        (dot > 0) & (retain_norm_sq > eps),
        dot / retain_norm_sq.clamp_min(eps),
        torch.zeros_like(dot),
    )
    grad_scale = float(1.0 / max(1, gradient_accumulation_steps))
    with torch.no_grad():
        for name, param in named_params:
            forget_grad = forget_grads.get(name)
            retain_grad = retain_grads.get(name)
            if forget_grad is None and retain_grad is None:
                param.grad = None
                continue
            safe_update = torch.zeros_like(param, dtype=torch.float32)
            if forget_grad is not None:
                safe_update = forget_grad.float()
                if retain_grad is not None and float(projection_coeff.detach().cpu()) != 0.0:
                    safe_update = safe_update - projection_coeff.to(safe_update.device) * retain_grad.float()
            descent_grad = -float(forget_weight) * safe_update
            if retain_grad is not None and retain_kl_weight != 0:
                descent_grad = descent_grad + float(retain_kl_weight) * retain_grad.float()
            param.grad = (descent_grad * grad_scale).to(device=param.device, dtype=param.dtype)

    retain_norm = torch.sqrt(retain_norm_sq.detach().clamp_min(0))
    forget_norm = torch.sqrt(forget_norm_sq.detach().clamp_min(0))
    cosine = dot.detach() / (forget_norm * retain_norm).clamp_min(eps)
    removed_norm = torch.relu(projection_coeff.detach()) * retain_norm
    return {
        "projection_coeff": projection_coeff.detach(),
        "grad_cosine": cosine.detach(),
        "forget_grad_norm": forget_norm.detach(),
        "retain_grad_norm": retain_norm.detach(),
        "removed_update_norm": removed_norm.detach(),
    }


def _normalize_proxy_rows(path: Path, *, split: str, limit: int) -> list[Dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    normalized: list[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(row.get("answer") or row.get("output") or row.get("reference_answer") or "").strip()
        if not question or not answer:
            continue
        normalized.append(
            {
                "instruction": question,
                "output": answer,
                "source": f"canonical_proxy_{split}",
                "example_id": f"canonical_proxy_{split}_{idx}",
            }
        )
        if len(normalized) >= limit:
            break
    if len(normalized) < limit:
        logger.warning(
            "Canonical proxy %s requested %d rows but only loaded %d from %s.",
            split,
            limit,
            len(normalized),
            path,
        )
    return normalized


def _load_canonical_proxy_spec(cfg: Stage2Config) -> Dict[str, Any] | None:
    if not cfg.canonical_proxy_enabled:
        return None
    if cfg.canonical_proxy_forget_path is None or cfg.canonical_proxy_retain_path is None:
        raise ValueError("Canonical proxy is enabled but proxy paths are missing.")
    forget_rows = _normalize_proxy_rows(
        cfg.canonical_proxy_forget_path,
        split="forget",
        limit=cfg.canonical_proxy_forget_samples,
    )
    retain_rows = _normalize_proxy_rows(
        cfg.canonical_proxy_retain_path,
        split="retain",
        limit=cfg.canonical_proxy_retain_samples,
    )
    if not forget_rows or not retain_rows:
        raise ValueError("Canonical proxy requires at least one forget row and one retain row.")
    return {
        "forget_rows": forget_rows,
        "retain_rows": retain_rows,
        "forget_path": str(cfg.canonical_proxy_forget_path),
        "retain_path": str(cfg.canonical_proxy_retain_path),
    }


def _load_answer_suppression_dataset(cfg: Stage2Config) -> Dataset | None:
    if cfg.answer_suppression_weight <= 0:
        return None
    if cfg.answer_suppression_path is None:
        logger.info(
            "Answer suppression weight is %.4f but no separate path was provided; using the main forget batch.",
            cfg.answer_suppression_weight,
        )
        return None
    rows = _normalize_proxy_rows(
        cfg.answer_suppression_path,
        split="answer_suppression",
        limit=10**12,
    )
    if not rows:
        raise ValueError(f"Answer suppression dataset is empty: {cfg.answer_suppression_path}")
    dataset = Dataset.from_list(rows)
    if getattr(cfg, "localization_strategy", "activation_contrast") != "author_firing_intersection":
        dataset = dataset.shuffle(seed=cfg.seed)
    logger.info(
        "Loaded answer suppression dataset with %d rows from %s.",
        len(dataset),
        cfg.answer_suppression_path,
    )
    return dataset


def _compute_reference_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_nll = -F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = token_nll * valid_mask.to(token_nll.dtype)
    token_counts = valid_mask.sum(dim=1).clamp_min(1).to(token_nll.dtype)
    row_nll = token_nll.sum(dim=1) / token_counts
    return row_nll, valid_mask


def _compute_proxy_topk_cache_for_batch(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    log_probs = F.log_softmax(shift_logits, dim=-1)
    topk_logprobs, topk_indices = torch.topk(log_probs, k=min(topk, log_probs.shape[-1]), dim=-1)
    return topk_indices.detach().cpu(), topk_logprobs.detach().cpu(), valid_mask.detach().cpu()


def _has_supervised_tokens(labels: torch.Tensor) -> bool:
    return bool((labels != -100).any().item())


def _clone_trainable_state_dict_to_cpu(model) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _save_forget_ce_threshold_checkpoint(
    *,
    model,
    tokenizer,
    state_dict: Dict[str, torch.Tensor],
    checkpoint_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(state_dict, strict=False)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    with open(checkpoint_dir / "threshold_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2, default=str)


def _checkpoint_step_from_path(path: Path) -> int:
    for pattern in (r"step(\d+)", r"checkpoint-(\d+)", r"checkpoint-step(\d+)"):
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))
    return 0


def _save_training_state(
    *,
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    global_step: int,
    epoch_idx: int,
    best_step: int,
    best_step_score: float,
    best_metrics: Dict[str, Any],
    best_retain_kl_guard_level: str,
    no_improve_steps: int,
    dual_kl_weight: float,
    saved_ce_thresholds: set[float],
) -> None:
    torch.save(
        {
            "global_step": int(global_step),
            "epoch_idx": int(epoch_idx),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_step": int(best_step),
            "best_step_score": float(best_step_score),
            "best_metrics": best_metrics,
            "best_retain_kl_guard_level": best_retain_kl_guard_level,
            "no_improve_steps": int(no_improve_steps),
            "dual_kl_weight": float(dual_kl_weight),
            "saved_ce_thresholds": sorted(float(x) for x in saved_ce_thresholds),
        },
        checkpoint_dir / "training_state.pt",
    )


def _load_resume_training_state(path: Path) -> Dict[str, Any] | None:
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return None
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def _resume_reached_ce_threshold(path: Path) -> float | None:
    metadata_path = path / "threshold_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return float(metadata["threshold"])
    except Exception as exc:
        logger.warning("Could not read CE threshold metadata from %s: %s", metadata_path, exc)
        return None


def _freeze_non_masked_trainable_parameters(model, allowed_param_names: set[str]) -> Dict[str, int]:
    trainable_before = 0
    trainable_after = 0
    frozen_now = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_before += 1
        if name in allowed_param_names:
            param.requires_grad_(True)
        else:
            if param.requires_grad:
                frozen_now += 1
            param.requires_grad_(False)
        if param.requires_grad:
            trainable_after += 1
    return {
        "trainable_before": int(trainable_before),
        "trainable_after": int(trainable_after),
        "frozen_now": int(frozen_now),
    }


def _resolve_existing_path(path_value: Any) -> Path | None:
    if path_value is None or not str(path_value).strip():
        return None
    path = Path(str(path_value))
    if path.exists():
        return path
    rooted = AGENT_ROOT / path
    if rooted.exists():
        return rooted
    return path


def _load_graph_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _as_binary_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "toxic", "drop", "unlearned"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _as_graph_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr.astype(np.float32)
    return (arr / norm).astype(np.float32)


def _graph_unlearning_score(data: Dict[str, Any]) -> float:
    for key in ("drop_prob", "punish", "edge_toxicity", "forget_weight", "toxicity", "weight"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 1.0 if _as_binary_flag(data.get("unlearned", 0)) else 0.0


def _graph_risk_score(data: Dict[str, Any]) -> float:
    for key in ("drop_prob", "punish", "edge_toxicity", "forget_weight", "toxicity"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
    return 1.0 if _as_binary_flag(data.get("unlearned", 0)) else 0.0


def _graph_item_training_weight(data: Dict[str, Any], *, target_unlearned: bool) -> float:
    risk = _graph_risk_score(data)
    if target_unlearned:
        return max(risk, 1e-3)
    return max(1.0 - risk, 1e-3)


def _memory_node_vector(graph, node_id: str) -> np.ndarray | None:
    if node_id not in graph.nodes:
        return None
    data = graph.nodes[node_id]
    vec = _as_graph_vector(data.get("embedding"))
    if vec is not None:
        return vec
    chunk_id = data.get("chunk_id")
    if chunk_id and str(chunk_id) in graph.nodes:
        return _as_graph_vector(graph.nodes[str(chunk_id)].get("embedding"))
    return None


def _memory_edge_vector(graph, source: str, target: str, data: Dict[str, Any]) -> np.ndarray | None:
    vec = _as_graph_vector(data.get("edge_embedding"))
    if vec is None:
        vec = _as_graph_vector(data.get("embedding"))
    if vec is not None:
        return vec
    endpoint_vectors = [
        endpoint_vec
        for endpoint_vec in (_memory_node_vector(graph, source), _memory_node_vector(graph, target))
        if endpoint_vec is not None
    ]
    if not endpoint_vectors:
        return None
    return _as_graph_vector(np.mean(np.stack(endpoint_vectors, axis=0), axis=0))


def _collect_rag_graph_vectors(path: Path, *, target_unlearned: bool = True) -> list[Dict[str, Any]]:
    graph = _load_graph_pickle(path)
    items: list[Dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        data = dict(data)
        if _as_binary_flag(data.get("unlearned", 0)) != target_unlearned:
            continue
        vec = _as_graph_vector(data.get("embedding"))
        if vec is None:
            continue
        items.append(
            {
                "source": "rag_node",
                "id": str(node_id),
                "weight": _graph_item_training_weight(data, target_unlearned=target_unlearned),
                "vector": vec,
            }
        )
    return items


def _collect_memory_graph_vectors(path: Path, *, target_unlearned: bool = True) -> list[Dict[str, Any]]:
    graph = _load_graph_pickle(path)
    items: list[Dict[str, Any]] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        data = dict(data)
        if _as_binary_flag(data.get("unlearned", 0)) != target_unlearned:
            continue
        vec = _memory_edge_vector(graph, str(source), str(target), data)
        if vec is None:
            continue
        edge_type = str(data.get("edge_type") or data.get("relation") or "edge")
        items.append(
            {
                "source": f"memory_edge::{edge_type}",
                "id": f"{source}->{target}::{key}",
                "weight": _graph_item_training_weight(data, target_unlearned=target_unlearned),
                "vector": vec,
            }
        )
    for node_id, data in graph.nodes(data=True):
        data = dict(data)
        if _as_binary_flag(data.get("unlearned", 0)) != target_unlearned:
            continue
        vec = _memory_node_vector(graph, str(node_id))
        if vec is None:
            continue
        node_type = str(data.get("node_type") or data.get("type") or "node")
        items.append(
            {
                "source": f"memory_node::{node_type}",
                "id": str(node_id),
                "weight": _graph_item_training_weight(data, target_unlearned=target_unlearned),
                "vector": vec,
            }
        )
    return items


def _build_graph_vector_items(
    cfg: Stage2Config,
    localization,
    *,
    target_unlearned: bool = True,
    max_items: int | None = None,
) -> list[Dict[str, Any]]:
    metadata = dict(getattr(localization, "metadata", {}) or {})
    rag_path = cfg.graph_vector_rag_graph_path or _resolve_existing_path(metadata.get("rag_graph_path"))
    memory_path = cfg.graph_vector_memory_graph_path or _resolve_existing_path(metadata.get("memory_graph_path"))
    if rag_path is None and memory_path is None:
        raise ValueError(
            "Graph-vector suppression requires graph-vector graph paths. "
            "Pass --graph-vector-rag-graph-path/--graph-vector-memory-graph-path or reuse localization metadata with rag_graph_path/memory_graph_path."
        )

    items: list[Dict[str, Any]] = []
    if rag_path is not None:
        if not rag_path.exists():
            raise FileNotFoundError(f"graph-vector rag graph does not exist: {rag_path}")
        items.extend(_collect_rag_graph_vectors(rag_path, target_unlearned=target_unlearned))
    if memory_path is not None:
        if not memory_path.exists():
            raise FileNotFoundError(f"graph-vector memory graph does not exist: {memory_path}")
        items.extend(_collect_memory_graph_vectors(memory_path, target_unlearned=target_unlearned))

    if not items:
        item_type = "unlearned" if target_unlearned else "non-unlearned"
        raise ValueError(f"Graph-vector objective found no {item_type} graph vectors.")
    items.sort(key=lambda item: (float(item["weight"]), str(item["source"]), str(item["id"])), reverse=True)
    limit = cfg.graph_vector_max_items if max_items is None else int(max_items)
    if limit > 0:
        items = items[:limit]
    return items


def _selected_neurons_by_layer(localization) -> Dict[int, list[int]]:
    selected: Dict[int, list[int]] = {}
    for layer_key, neurons in getattr(localization, "target_neurons", {}).items():
        if not isinstance(layer_key, str) or not layer_key.startswith("layer_"):
            continue
        suffix = layer_key.split("_")[-1]
        if not suffix.isdigit():
            continue
        layer_idx = int(suffix)
        selected[layer_idx] = sorted({int(neuron) for neuron in neurons})
    return selected


def _resolve_decoder_layers(model):
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


@contextmanager
def _capture_mlp_down_inputs(model, allowed_layers: set[int]):
    activations: Dict[int, torch.Tensor] = {}
    handles = []
    layers = _resolve_decoder_layers(model)

    def make_hook(layer_idx: int):
        def hook(_module, inputs, _output):
            if inputs:
                activations[layer_idx] = inputs[0]

        return hook

    for layer_idx, layer in enumerate(layers):
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


@contextmanager
def _capture_mlp_down_inputs_and_outputs(model, allowed_layers: set[int]):
    input_activations: Dict[int, torch.Tensor] = {}
    output_activations: Dict[int, torch.Tensor] = {}
    handles = []
    layers = _resolve_decoder_layers(model)

    def make_hook(layer_idx: int):
        def hook(_module, inputs, output):
            if inputs:
                input_activations[layer_idx] = inputs[0]
            if isinstance(output, torch.Tensor):
                output_activations[layer_idx] = output

        return hook

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in allowed_layers:
            continue
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is not None:
            handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))
    try:
        yield input_activations, output_activations
    finally:
        for handle in handles:
            handle.remove()


def _project_graph_vector(vector: np.ndarray, hidden_size: int) -> np.ndarray:
    if vector.size == hidden_size:
        return vector.astype(np.float32)
    if vector.size > hidden_size:
        return vector[:hidden_size].astype(np.float32)
    out = np.zeros(hidden_size, dtype=np.float32)
    out[: vector.size] = vector
    return out


def _build_graph_vector_inputs(
    model,
    tokenizer,
    items: list[Dict[str, Any]],
    *,
    anchor_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_model_device(model)
    hidden_size = int(model.get_input_embeddings().embedding_dim)
    suffix = tokenizer(anchor_text, return_tensors="pt", add_special_tokens=False)
    suffix_ids = suffix["input_ids"].to(device)
    suffix_mask = suffix["attention_mask"].to(device)
    suffix_emb = model.get_input_embeddings()(suffix_ids)
    prefix_vectors = [
        torch.tensor(_project_graph_vector(item["vector"], hidden_size), device=device, dtype=suffix_emb.dtype)
        for item in items
    ]
    prefix_emb = torch.stack(prefix_vectors, dim=0).unsqueeze(1)
    inputs_embeds = torch.cat([prefix_emb, suffix_emb.repeat(len(items), 1, 1)], dim=1)
    prefix_mask = torch.ones((len(items), 1), dtype=suffix_mask.dtype, device=device)
    attention_mask = torch.cat([prefix_mask, suffix_mask.repeat(len(items), 1)], dim=1)
    return inputs_embeds, attention_mask


def _compute_graph_vector_suppress_loss(
    *,
    model,
    tokenizer,
    graph_items: list[Dict[str, Any]],
    selected_neurons: Dict[int, list[int]],
    anchor_text: str,
) -> Dict[str, torch.Tensor]:
    if not selected_neurons:
        raise ValueError("Graph-vector suppression requires localized target neurons.")
    allowed_layers = set(selected_neurons.keys())
    inputs_embeds, attention_mask = _build_graph_vector_inputs(
        model,
        tokenizer,
        graph_items,
        anchor_text=anchor_text,
    )
    with _capture_mlp_down_inputs_and_outputs(model, allowed_layers) as (captured, output_captured):
        model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)

    losses = []
    mean_abs_values = []
    active_neurons = 0
    for layer_idx, activation in captured.items():
        neurons = selected_neurons.get(layer_idx, [])
        if not neurons:
            continue
        valid_neurons = [idx for idx in neurons if 0 <= idx < activation.shape[-1]]
        if not valid_neurons:
            continue
        selected = activation[:, 0, valid_neurons].float()
        losses.append(selected.pow(2).mean())
        mean_abs_values.append(selected.abs().mean())
        active_neurons += len(valid_neurons)

    if not losses:
        zero = inputs_embeds.sum() * 0.0
        return {
            "loss": zero,
            "mean_abs_activation": zero.detach(),
            "active_neurons": torch.tensor(0.0, device=inputs_embeds.device),
        }
    loss = torch.stack(losses).mean()
    mean_abs = torch.stack(mean_abs_values).mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "mean_abs_activation": torch.nan_to_num(mean_abs, nan=0.0, posinf=0.0, neginf=0.0),
        "active_neurons": torch.tensor(float(active_neurons), device=inputs_embeds.device),
    }


def _graph_item_ids(graph_items: list[Dict[str, Any]], *, suffix: str) -> list[str]:
    return [f"{item.get('source', 'graph')}::{item.get('id', idx)}::{suffix}" for idx, item in enumerate(graph_items)]


def _graph_item_weights(graph_items: list[Dict[str, Any]], *, device, dtype) -> torch.Tensor:
    weights = []
    for item in graph_items:
        try:
            weights.append(float(item.get("weight", 1.0)))
        except (TypeError, ValueError):
            weights.append(1.0)
    tensor = torch.tensor(weights, device=device, dtype=dtype).clamp_min(1e-3)
    return tensor / tensor.mean().clamp_min(1e-6)


def _graph_vector_target_tensor(
    graph_items: list[Dict[str, Any]],
    dim: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    vectors = [
        torch.tensor(_project_graph_vector(item["vector"], dim), device=device, dtype=dtype)
        for item in graph_items
    ]
    return F.normalize(torch.stack(vectors, dim=0).float(), dim=-1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _compute_graph_vector_final_repr_loss(
    *,
    model,
    tokenizer,
    forget_items: list[Dict[str, Any]],
    keep_items: list[Dict[str, Any]],
    anchor_text: str,
    random_weight: float,
    away_weight: float,
    keep_weight: float,
    away_max_similarity: float,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
    inputs_embeds, attention_mask = _build_graph_vector_inputs(
        model,
        tokenizer,
        forget_items,
        anchor_text=anchor_text,
    )
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    forget_repr = F.normalize(outputs.hidden_states[-1][:, 0, :].float(), dim=-1)
    forget_weights = _graph_item_weights(forget_items, device=forget_repr.device, dtype=forget_repr.dtype)
    random_targets = _build_random_target_vectors(
        _graph_item_ids(forget_items, suffix="final_repr_random"),
        forget_repr.shape[-1],
        device=forget_repr.device,
        dtype=forget_repr.dtype,
        base_seed=base_seed,
    )
    graph_targets = _graph_vector_target_tensor(
        forget_items,
        forget_repr.shape[-1],
        device=forget_repr.device,
        dtype=forget_repr.dtype,
    )

    random_cos = F.cosine_similarity(forget_repr, random_targets, dim=-1)
    away_cos = F.cosine_similarity(forget_repr, graph_targets, dim=-1)
    random_loss = _weighted_mean(1.0 - random_cos, forget_weights)
    away_loss = _weighted_mean(torch.relu(away_cos - float(away_max_similarity)), forget_weights)

    keep_loss = forget_repr.sum() * 0.0
    keep_cos_mean = keep_loss.detach()
    if keep_items and keep_weight > 0:
        keep_inputs, keep_mask = _build_graph_vector_inputs(
            model,
            tokenizer,
            keep_items,
            anchor_text=anchor_text,
        )
        keep_outputs = model(
            inputs_embeds=keep_inputs,
            attention_mask=keep_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        keep_repr = F.normalize(keep_outputs.hidden_states[-1][:, 0, :].float(), dim=-1)
        keep_targets = _graph_vector_target_tensor(
            keep_items,
            keep_repr.shape[-1],
            device=keep_repr.device,
            dtype=keep_repr.dtype,
        )
        keep_weights = _graph_item_weights(keep_items, device=keep_repr.device, dtype=keep_repr.dtype)
        keep_cos = F.cosine_similarity(keep_repr, keep_targets, dim=-1)
        keep_loss = _weighted_mean(1.0 - keep_cos, keep_weights)
        keep_cos_mean = keep_cos.mean().detach()

    loss = float(random_weight) * random_loss + float(away_weight) * away_loss + float(keep_weight) * keep_loss
    mean_abs = forget_repr.abs().mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "mean_abs_activation": torch.nan_to_num(mean_abs.detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "active_neurons": torch.tensor(float(forget_repr.shape[-1]), device=forget_repr.device),
        "random_cosine": torch.nan_to_num(random_cos.mean().detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "away_cosine": torch.nan_to_num(away_cos.mean().detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "keep_cosine": torch.nan_to_num(keep_cos_mean, nan=0.0, posinf=0.0, neginf=0.0),
    }


def _compute_graph_vector_neuron_repr_loss(
    *,
    model,
    tokenizer,
    forget_items: list[Dict[str, Any]],
    keep_items: list[Dict[str, Any]],
    selected_neurons: Dict[int, list[int]],
    anchor_text: str,
    random_weight: float,
    away_weight: float,
    keep_weight: float,
    away_max_similarity: float,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
    if not selected_neurons:
        raise ValueError("Graph-vector neuron_repr objective requires localized target neurons.")

    allowed_layers = set(selected_neurons.keys())
    inputs_embeds, attention_mask = _build_graph_vector_inputs(
        model,
        tokenizer,
        forget_items,
        anchor_text=anchor_text,
    )
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    forget_weights = _graph_item_weights(forget_items, device=inputs_embeds.device, dtype=torch.float32)
    losses = []
    random_cosines = []
    away_cosines = []
    mean_abs_values = []
    active_neurons = 0
    for layer_idx in sorted(allowed_layers):
        neurons = selected_neurons.get(layer_idx, [])
        if not neurons:
            continue
        hidden_index = min(layer_idx + 1, len(outputs.hidden_states) - 1)
        layer_repr = outputs.hidden_states[hidden_index]
        selected = F.normalize(layer_repr[:, 0, :].float(), dim=-1)
        random_targets = _build_random_target_vectors(
            _graph_item_ids(forget_items, suffix=f"neuron_repr_random_layer_{layer_idx}"),
            selected.shape[-1],
            device=selected.device,
            dtype=selected.dtype,
            base_seed=base_seed,
        )
        graph_targets = _graph_vector_target_tensor(
            forget_items,
            selected.shape[-1],
            device=selected.device,
            dtype=selected.dtype,
        )
        random_cos = F.cosine_similarity(selected, random_targets, dim=-1)
        away_cos = F.cosine_similarity(selected, graph_targets, dim=-1)
        random_loss = _weighted_mean(1.0 - random_cos, forget_weights)
        away_loss = _weighted_mean(torch.relu(away_cos - float(away_max_similarity)), forget_weights)
        losses.append(float(random_weight) * random_loss + float(away_weight) * away_loss)
        random_cosines.append(random_cos.mean())
        away_cosines.append(away_cos.mean())
        mean_abs_values.append(layer_repr[:, 0, :].float().abs().mean())
        active_neurons += len(neurons)

    if not losses:
        zero = inputs_embeds.sum() * 0.0
        return {
            "loss": zero,
            "mean_abs_activation": zero.detach(),
            "active_neurons": torch.tensor(0.0, device=inputs_embeds.device),
            "random_cosine": zero.detach(),
            "away_cosine": zero.detach(),
            "keep_cosine": zero.detach(),
        }

    keep_loss = losses[0] * 0.0
    keep_cos_mean = keep_loss.detach()
    if keep_items and keep_weight > 0:
        keep_inputs, keep_mask = _build_graph_vector_inputs(
            model,
            tokenizer,
            keep_items,
            anchor_text=anchor_text,
        )
        keep_outputs = model(
            inputs_embeds=keep_inputs,
            attention_mask=keep_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        keep_weights = _graph_item_weights(keep_items, device=keep_inputs.device, dtype=torch.float32)
        keep_losses = []
        keep_cosines = []
        for layer_idx in sorted(allowed_layers):
            neurons = selected_neurons.get(layer_idx, [])
            if not neurons:
                continue
            hidden_index = min(layer_idx + 1, len(keep_outputs.hidden_states) - 1)
            layer_repr = keep_outputs.hidden_states[hidden_index]
            selected = F.normalize(layer_repr[:, 0, :].float(), dim=-1)
            keep_targets = _graph_vector_target_tensor(
                keep_items,
                selected.shape[-1],
                device=selected.device,
                dtype=selected.dtype,
            )
            keep_cos = F.cosine_similarity(selected, keep_targets, dim=-1)
            keep_losses.append(_weighted_mean(1.0 - keep_cos, keep_weights))
            keep_cosines.append(keep_cos.mean())
        if keep_losses:
            keep_loss = torch.stack(keep_losses).mean()
            keep_cos_mean = torch.stack(keep_cosines).mean().detach()

    loss = torch.stack(losses).mean() + float(keep_weight) * keep_loss
    mean_abs = torch.stack(mean_abs_values).mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "mean_abs_activation": torch.nan_to_num(mean_abs.detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "active_neurons": torch.tensor(float(active_neurons), device=inputs_embeds.device),
        "random_cosine": torch.nan_to_num(torch.stack(random_cosines).mean().detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "away_cosine": torch.nan_to_num(torch.stack(away_cosines).mean().detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "keep_cosine": torch.nan_to_num(keep_cos_mean, nan=0.0, posinf=0.0, neginf=0.0),
    }


def _compute_representation_erasure_loss(
    *,
    outputs,
    labels: torch.Tensor,
    layer_indices: list[int],
) -> Dict[str, torch.Tensor]:
    pooled_layers = extract_answer_span_hidden_states(outputs.hidden_states, labels, layer_indices)
    zero_target = torch.zeros_like(pooled_layers)
    loss = F.mse_loss(pooled_layers, zero_target)
    mean_abs = pooled_layers.abs().mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "mean_abs_activation": torch.nan_to_num(mean_abs, nan=0.0, posinf=0.0, neginf=0.0),
        "num_layers": torch.tensor(float(len(layer_indices)), device=pooled_layers.device, dtype=pooled_layers.dtype),
    }


def _masked_topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
) -> torch.Tensor:
    shift_student = student_logits[:, :-1, :].float()
    max_len = min(shift_student.shape[1], teacher_topk_indices.shape[1])
    shift_student = shift_student[:, :max_len, :]
    teacher_topk_indices = teacher_topk_indices[:, :max_len, :]
    teacher_topk_logprobs = teacher_topk_logprobs[:, :max_len, :]
    teacher_valid_mask = teacher_valid_mask[:, :max_len].to(shift_student.dtype)

    student_log_probs = F.log_softmax(shift_student, dim=-1)
    gathered_student = student_log_probs.gather(dim=-1, index=teacher_topk_indices)
    teacher_topk_probs = teacher_topk_logprobs.exp()
    teacher_topk_probs = teacher_topk_probs / teacher_topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    token_kl = (teacher_topk_probs * (teacher_topk_logprobs - gathered_student)).sum(dim=-1)
    token_kl = token_kl * teacher_valid_mask
    denom = teacher_valid_mask.sum(dim=-1).clamp_min(1.0)
    return (token_kl.sum(dim=-1) / denom).mean()


def build_safe_forget_dataset(forget_dataset: Dataset, safe_response: str) -> Dataset:
    safe_rows = []
    for row in forget_dataset:
        updated = dict(row)
        updated["original_output"] = str(row.get("original_output") or row.get("output") or "")
        updated["output"] = safe_response
        safe_rows.append(updated)
    return Dataset.from_list(safe_rows)


def _stable_example_seed(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31)


def _build_random_answer_token_batch(
    batch: Dict[str, Any],
    *,
    allowed_token_ids: torch.Tensor,
    base_seed: int,
) -> Dict[str, Any]:
                                                                             

    randomized = dict(batch)
    input_ids = batch["input_ids"].clone()
    labels = batch["labels"].clone()
    allowed_token_ids = allowed_token_ids.to(device=labels.device)

    for row_idx, item in enumerate(batch["items"]):
        answer_mask = labels[row_idx] != -100
        token_count = int(answer_mask.sum().item())
        if token_count <= 0:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_example_seed(f"{str(item['example_id'])}::random_answer_ce", base_seed))
        random_indices = torch.randint(
            low=0,
            high=int(allowed_token_ids.numel()),
            size=(token_count,),
            generator=generator,
            dtype=torch.long,
        ).to(device=labels.device)
        random_token_ids = allowed_token_ids.index_select(0, random_indices)
        labels[row_idx, answer_mask] = random_token_ids
        input_ids[row_idx, answer_mask] = random_token_ids

    randomized["input_ids"] = input_ids
    randomized["labels"] = labels
    return randomized


def _build_random_target_vectors(
    example_ids: list[str],
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base_seed: int,
) -> torch.Tensor:
    vectors = []
    for example_id in example_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_example_seed(example_id, base_seed))
        random_vector = torch.randn(hidden_dim, generator=generator, dtype=torch.float32)
        random_vector = F.normalize(random_vector, dim=0)
        vectors.append(random_vector.to(device=device, dtype=dtype))
    return torch.stack(vectors, dim=0)


def _compute_random_vector_forget_loss(
    *,
    outputs,
    labels: torch.Tensor,
    batch_items,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
    final_hidden = outputs.hidden_states[-1]
    answer_mask = (labels != -100).to(final_hidden.dtype)
    pooled = (final_hidden * answer_mask.unsqueeze(-1)).sum(dim=1)
    denom = answer_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = pooled / denom
    pooled = F.normalize(pooled, dim=-1)

    target_vectors = _build_random_target_vectors(
        [str(item["example_id"]) for item in batch_items],
        pooled.shape[-1],
        device=pooled.device,
        dtype=pooled.dtype,
        base_seed=base_seed,
    )
    similarity = F.cosine_similarity(pooled, target_vectors, dim=-1)
    loss = (1.0 - similarity).mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "similarity": torch.nan_to_num(similarity.mean(), nan=0.0, posinf=0.0, neginf=0.0),
    }


def _compute_layerwise_random_vector_forget_loss(
    *,
    outputs,
    labels: torch.Tensor,
    batch_items,
    base_seed: int,
    layer_indices: list[int],
) -> Dict[str, torch.Tensor]:
    pooled_layers = extract_answer_span_hidden_states(outputs.hidden_states, labels, layer_indices)
    pooled_layers = F.normalize(pooled_layers, dim=-1)

    layer_losses = []
    layer_similarities = []
    batch_size = pooled_layers.shape[0]
    example_ids = [str(item["example_id"]) for item in batch_items]
    for layer_offset, layer_idx in enumerate(layer_indices):
        target_vectors = _build_random_target_vectors(
            [f"{example_id}::layer::{layer_idx}" for example_id in example_ids],
            pooled_layers.shape[-1],
            device=pooled_layers.device,
            dtype=pooled_layers.dtype,
            base_seed=base_seed,
        )
        similarity = F.cosine_similarity(pooled_layers[:, layer_offset, :], target_vectors, dim=-1)
        layer_losses.append(1.0 - similarity)
        layer_similarities.append(similarity)

    stacked_losses = torch.stack(layer_losses, dim=1)
    stacked_similarities = torch.stack(layer_similarities, dim=1)
    loss = stacked_losses.mean()
    similarity = stacked_similarities.mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "similarity": torch.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0),
        "num_layers": torch.tensor(float(len(layer_indices)), device=pooled_layers.device, dtype=pooled_layers.dtype),
    }


def _compute_parameter_random_vector_forget_loss(
    *,
    outputs,
    labels: torch.Tensor,
    batch_items,
    base_seed: int,
) -> Dict[str, torch.Tensor]:
           
                                            
                                     
    final_hidden = outputs.hidden_states[-1]
    answer_mask = (labels != -100).to(final_hidden.dtype)
    pooled_output = (final_hidden * answer_mask.unsqueeze(-1)).sum(dim=1)
    token_count = answer_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled_output = pooled_output / token_count
    output_l2 = torch.linalg.norm(pooled_output, ord=2, dim=-1).mean()
    pooled_output = F.normalize(pooled_output, dim=-1)

    target_vectors = _build_random_target_vectors(
        [f"{str(item['example_id'])}::final_output" for item in batch_items],
        pooled_output.shape[-1],
        device=pooled_output.device,
        dtype=pooled_output.dtype,
        base_seed=base_seed,
    )
    similarity = F.cosine_similarity(pooled_output, target_vectors, dim=-1)
    loss = (1.0 - similarity).mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "cosine": torch.nan_to_num(similarity.mean(), nan=0.0, posinf=0.0, neginf=0.0),
        "output_l2": torch.nan_to_num(output_l2, nan=0.0, posinf=0.0, neginf=0.0),
    }


def _precompute_forget_anchor_representation_cache(
    *,
    cfg: Stage2Config,
    tokenizer,
    forget_dataset: Dataset,
    layer_indices: list[int],
) -> Dict[str, torch.Tensor]:
    logger.info("Precomputing frozen forget anchor representations...")
    teacher_model = build_model(cfg, is_trainable=False)
    device = get_model_device(teacher_model)
    teacher_model.eval()
    cache: Dict[str, torch.Tensor] = {}

    loader = DataLoader(
        list(forget_dataset),
        batch_size=cfg.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
    )
    try:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                    return_dict=True,
                )
            pooled_layers = extract_answer_span_hidden_states(outputs.hidden_states, batch["labels"], layer_indices)
            pooled_layers = F.normalize(pooled_layers, dim=-1)
            for row_idx, item in enumerate(batch["items"]):
                cache[str(item["example_id"])] = pooled_layers[row_idx].detach().cpu()
    finally:
        del teacher_model
        release_memory()
    return cache


def _stack_forget_anchor_batch(
    cache: Dict[str, torch.Tensor],
    example_ids: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack([cache[example_id] for example_id in example_ids], dim=0).to(device=device, dtype=dtype)


def _compute_anchor_repulsion_forget_loss(
    *,
    outputs,
    labels: torch.Tensor,
    batch_items,
    anchor_cache: Dict[str, torch.Tensor],
    layer_indices: list[int],
    max_similarity: float,
) -> Dict[str, torch.Tensor]:
    pooled_layers = extract_answer_span_hidden_states(outputs.hidden_states, labels, layer_indices)
    pooled_layers = F.normalize(pooled_layers, dim=-1)
    anchor_layers = _stack_forget_anchor_batch(
        anchor_cache,
        [str(item["example_id"]) for item in batch_items],
        device=pooled_layers.device,
        dtype=pooled_layers.dtype,
    )
    anchor_layers = F.normalize(anchor_layers, dim=-1)
    similarity = F.cosine_similarity(pooled_layers, anchor_layers, dim=-1)
                                                                              
                                                                             
                                                                             
                                                                           
                        
    shifted_similarity = similarity - max_similarity
    loss = ((shifted_similarity + 1.0) * 0.5).mean()
    return {
        "loss": torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0),
        "similarity": torch.nan_to_num(similarity.mean(), nan=0.0, posinf=0.0, neginf=0.0),
    }



def _precompute_retain_teacher_logprob_cache(
    *,
    cfg: Stage2Config,
    tokenizer,
    retain_dataset: Dataset,
) -> Dict[str, Dict[str, torch.Tensor]]:
    metadata = _retain_teacher_cache_metadata(cfg=cfg, tokenizer=tokenizer, retain_dataset=retain_dataset)
    cache_path = _retain_teacher_cache_path(metadata)
    if cache_path.exists():
        try:
            payload = _torch_load_cache(cache_path)
            if payload.get("metadata") == metadata and isinstance(payload.get("cache"), dict):
                logger.info("Loaded frozen retain reference top-k logits from cache: %s", cache_path)
                return payload["cache"]
            logger.warning("Ignoring stale retain teacher cache with mismatched metadata: %s", cache_path)
        except Exception as exc:
            logger.warning("Ignoring unreadable retain teacher cache %s: %s", cache_path, exc)

    logger.info("Precomputing frozen retain reference top-k logits...")
    logger.info("Retain teacher cache miss; will save cache to: %s", cache_path)
    teacher_model = build_model(cfg, is_trainable=False)
    device = get_model_device(teacher_model)
    teacher_model.eval()
    cache: Dict[str, Dict[str, torch.Tensor]] = {}

    loader = DataLoader(
        list(retain_dataset),
        batch_size=cfg.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
    )
    try:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = batch["labels"][:, 1:]
            valid_mask = shift_labels != -100
            teacher_log_probs = F.log_softmax(shift_logits, dim=-1)
            topk_logprobs, topk_indices = torch.topk(
                teacher_log_probs,
                k=min(cfg.retain_topk, teacher_log_probs.shape[-1]),
                dim=-1,
            )
            for row_idx, item in enumerate(batch["items"]):
                cache[str(item["example_id"])] = {
                    "topk_indices": topk_indices[row_idx].detach().to(torch.int32).cpu(),
                    "topk_logprobs": topk_logprobs[row_idx].detach().cpu(),
                    "valid_mask": valid_mask[row_idx].detach().cpu(),
                }
    finally:
        del teacher_model
        release_memory()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"metadata": metadata, "cache": cache}, tmp_path)
    tmp_path.replace(cache_path)
    logger.info("Saved frozen retain reference top-k logits cache: %s", cache_path)
    return cache


def _stack_cached_teacher_batch(
    cache: Dict[str, Dict[str, torch.Tensor]],
    example_ids: list[str],
    target_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index_rows = []
    logprob_rows = []
    mask_rows = []
    for example_id in example_ids:
        cached = cache[example_id]
        cached_indices = cached["topk_indices"]
        cached_logprobs = cached["topk_logprobs"]
        cached_mask = cached["valid_mask"]
        topk_dim = cached_indices.shape[-1]
        padded_indices = torch.zeros((target_seq_len, topk_dim), dtype=torch.long)
        padded_logprobs = torch.full((target_seq_len, topk_dim), fill_value=-1e9, dtype=torch.float32)
        padded_mask = torch.zeros(target_seq_len, dtype=torch.bool)
        usable_len = min(target_seq_len, cached_indices.shape[0])
        padded_indices[:usable_len] = cached_indices[:usable_len]
        padded_logprobs[:usable_len] = cached_logprobs[:usable_len]
        padded_mask[:usable_len] = cached_mask[:usable_len]
        index_rows.append(padded_indices)
        logprob_rows.append(padded_logprobs)
        mask_rows.append(padded_mask)
    return (
        torch.stack(index_rows, dim=0).to(device),
        torch.stack(logprob_rows, dim=0).to(device),
        torch.stack(mask_rows, dim=0).to(device),
    )


def _precompute_canonical_proxy_baseline(
    *,
    cfg: Stage2Config,
    tokenizer,
    proxy_spec: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info(
        "Precomputing canonical proxy baseline: forget=%d retain=%d topk=%d",
        len(proxy_spec["forget_rows"]),
        len(proxy_spec["retain_rows"]),
        cfg.retain_topk,
    )
    teacher_model = build_model(cfg, is_trainable=False)
    device = get_model_device(teacher_model)
    teacher_model.eval()
    baseline: Dict[str, Any] = {
        "forget_nll": [],
        "retain_nll": [],
        "retain_topk_cache": {},
    }

    def run_rows(rows: list[Dict[str, Any]], split: str) -> None:
        loader = DataLoader(
            rows,
            batch_size=cfg.canonical_proxy_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
        )
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                row_nll, _ = _compute_reference_nll_from_logits(outputs.logits, batch["labels"])
                if split == "forget":
                    baseline["forget_nll"].extend(row_nll.detach().cpu().tolist())
                else:
                    baseline["retain_nll"].extend(row_nll.detach().cpu().tolist())
                    topk_indices, topk_logprobs, valid_mask = _compute_proxy_topk_cache_for_batch(
                        logits=outputs.logits,
                        labels=batch["labels"],
                        topk=cfg.retain_topk,
                    )
                    for row_idx, item in enumerate(batch["items"]):
                        baseline["retain_topk_cache"][str(item["example_id"])] = {
                            "topk_indices": topk_indices[row_idx].to(torch.int32),
                            "topk_logprobs": topk_logprobs[row_idx],
                            "valid_mask": valid_mask[row_idx],
                        }

    try:
        run_rows(proxy_spec["forget_rows"], "forget")
        run_rows(proxy_spec["retain_rows"], "retain")
    finally:
        del teacher_model
        release_memory()

    baseline["forget_nll_mean"] = float(np.mean(baseline["forget_nll"])) if baseline["forget_nll"] else 0.0
    baseline["retain_nll_mean"] = float(np.mean(baseline["retain_nll"])) if baseline["retain_nll"] else 0.0
    logger.info(
        "Canonical proxy baseline ready: forget_nll=%.4f retain_nll=%.4f",
        baseline["forget_nll_mean"],
        baseline["retain_nll_mean"],
    )
    return baseline


def _evaluate_canonical_proxy(
    *,
    cfg: Stage2Config,
    model,
    tokenizer,
    proxy_spec: Dict[str, Any],
    proxy_baseline: Dict[str, Any],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    device = get_model_device(model)
    forget_nll_values: list[float] = []
    retain_nll_values: list[float] = []
    retain_kl_values: list[float] = []

    def run_rows(rows: list[Dict[str, Any]], split: str) -> None:
        loader = DataLoader(
            rows,
            batch_size=cfg.canonical_proxy_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
        )
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                row_nll, _ = _compute_reference_nll_from_logits(outputs.logits, batch["labels"])
                if split == "forget":
                    forget_nll_values.extend(row_nll.detach().cpu().tolist())
                else:
                    retain_nll_values.extend(row_nll.detach().cpu().tolist())
                    teacher_topk_indices, teacher_topk_logprobs, teacher_valid_mask = _stack_cached_teacher_batch(
                        proxy_baseline["retain_topk_cache"],
                        [str(item["example_id"]) for item in batch["items"]],
                        target_seq_len=outputs.logits.shape[1] - 1,
                        device=device,
                    )
                    retain_kl = _masked_topk_distillation_loss(
                        student_logits=outputs.logits,
                        teacher_topk_indices=teacher_topk_indices,
                        teacher_topk_logprobs=teacher_topk_logprobs,
                        teacher_valid_mask=teacher_valid_mask,
                    )
                    retain_kl_values.append(float(retain_kl.detach().cpu()))

    try:
        run_rows(proxy_spec["forget_rows"], "forget")
        run_rows(proxy_spec["retain_rows"], "retain")
    finally:
        if was_training:
            model.train()

    forget_nll = float(np.mean(forget_nll_values)) if forget_nll_values else 0.0
    retain_nll = float(np.mean(retain_nll_values)) if retain_nll_values else 0.0
    retain_kl = float(np.mean(retain_kl_values)) if retain_kl_values else 0.0
    forget_nll_delta = forget_nll - float(proxy_baseline["forget_nll_mean"])
    retain_nll_delta = retain_nll - float(proxy_baseline["retain_nll_mean"])
    forget_logprob_drop = forget_nll_delta
    retain_ok = (
        retain_nll_delta <= cfg.canonical_proxy_retain_nll_delta_max
        and retain_kl <= cfg.canonical_proxy_retain_kl_max
    )
    forget_score = (
        cfg.canonical_proxy_forget_nll_weight * forget_nll_delta
        + cfg.canonical_proxy_forget_logprob_drop_weight * forget_logprob_drop
    )
    continuous_score = (
        forget_score
        - cfg.canonical_proxy_retain_nll_penalty * max(0.0, retain_nll_delta)
        - cfg.canonical_proxy_retain_kl_penalty * max(0.0, retain_kl)
    )
    if cfg.canonical_proxy_hard_retain_filter and retain_ok:
        score = forget_score
    elif cfg.canonical_proxy_hard_retain_filter:
                                                                           
                                                                               
        score = -1.0e12 + continuous_score
    else:
        score = continuous_score
    return {
        "canonical_proxy_score": float(score),
        "canonical_proxy_forget_score": float(forget_score),
        "canonical_proxy_continuous_score": float(continuous_score),
        "canonical_proxy_forget_nll": forget_nll,
        "canonical_proxy_forget_nll_delta": forget_nll_delta,
        "canonical_proxy_forget_logprob_drop": forget_logprob_drop,
        "canonical_proxy_retain_nll": retain_nll,
        "canonical_proxy_retain_nll_delta": retain_nll_delta,
        "canonical_proxy_retain_kl": retain_kl,
        "canonical_proxy_retain_ok": 1.0 if retain_ok else 0.0,
    }


def run_masked_parameter_optimization(
    *,
    cfg: Stage2Config,
    localization,
    tokenizer,
    forget_dataset: Dataset,
    retain_dataset: Dataset,
    run_metadata_extra: Dict[str, Any],
) -> None:
    if localization is None:
        raise ValueError("Masked parameter optimization requires localization artifacts.")

    needs_retain_kl = bool(
        cfg.use_dual_kl_objective
        or cfg.retain_kl_weight > 0
        or cfg.forget_loss_strategy == "gradient_ascent_projected"
        or cfg.canonical_proxy_enabled
    )
    needs_retain_forward = bool(
        needs_retain_kl
        or cfg.retain_ce_weight > 0
        or cfg.retain_repr_weight > 0
        or (
            cfg.forget_nll_calibration_weight > 0
            and cfg.forget_nll_calibration_target_mode == "retain_ce"
        )
    )
    logger.info(
        "Preparing masked parameter optimization with forget_loss_strategy=%s retain_kl_enabled=%s retain_forward_enabled=%s...",
        cfg.forget_loss_strategy,
        needs_retain_kl,
        needs_retain_forward,
    )
    forget_entity_dataset = build_forget_entity_dataset(cfg)
    forget_training_dataset = forget_entity_dataset
    forget_loss_layers = select_hctl_strongest_layers(
        localization,
        max_layers=cfg.forget_loss_top_layers,
    )
    if cfg.forget_loss_strategy in {"representation_erasure", "random_vector"} and not forget_loss_layers:
        raise ValueError(f"{cfg.forget_loss_strategy} requires at least one localized layer.")

                                                                                  
                                                                                  
                                                              
     
                                                   
                                                                                                       
           
                                                         
     
                                                                         
                  
                                    
                              
                                                      
       
    retain_teacher_cache = None
    if needs_retain_kl:
        retain_teacher_cache = _precompute_retain_teacher_logprob_cache(
            cfg=cfg,
            tokenizer=tokenizer,
            retain_dataset=retain_dataset,
        )
    else:
        logger.info("Skipping retain teacher top-k cache because retain KL is disabled for this run.")
    canonical_proxy_spec = _load_canonical_proxy_spec(cfg)
    canonical_proxy_baseline = None
    if canonical_proxy_spec is not None:
        canonical_proxy_baseline = _precompute_canonical_proxy_baseline(
            cfg=cfg,
            tokenizer=tokenizer,
            proxy_spec=canonical_proxy_spec,
        )
    answer_suppression_dataset = _load_answer_suppression_dataset(cfg)
    forget_anchor_cache = None
    if cfg.forget_loss_strategy == "random_vector" and cfg.forget_anchor_weight > 0:
        forget_anchor_cache = _precompute_forget_anchor_representation_cache(
            cfg=cfg,
            tokenizer=tokenizer,
            forget_dataset=forget_training_dataset,
            layer_indices=forget_loss_layers,
        )

    forget_loader = DataLoader(
        list(forget_training_dataset),
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
    )
    retain_iter = None
    if needs_retain_forward:
        retain_rows = list(retain_dataset)
        if not retain_rows:
            raise ValueError("Retain dataset is empty, but the configured objective requires retain forward passes.")
        retain_loader = DataLoader(
            retain_rows,
            batch_size=cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
        )
        retain_iter = cycle_dataloader(retain_loader)
    answer_suppression_iter = None
    if answer_suppression_dataset is not None:
        answer_suppression_loader = DataLoader(
            list(answer_suppression_dataset),
            batch_size=cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_chat_examples(batch, tokenizer, cfg.max_length),
        )
        answer_suppression_iter = cycle_dataloader(answer_suppression_loader)

    graph_vector_iter = None
    graph_vector_keep_iter = None
    graph_vector_items: list[Dict[str, Any]] = []
    graph_vector_keep_items: list[Dict[str, Any]] = []
    selected_graph_neurons: Dict[int, list[int]] = {}
    if cfg.graph_vector_suppress_weight > 0:
        graph_vector_items = _build_graph_vector_items(cfg, localization, target_unlearned=True)
        selected_graph_neurons = _selected_neurons_by_layer(localization)
        graph_vector_loader = DataLoader(
            graph_vector_items,
            batch_size=cfg.graph_vector_batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )
        graph_vector_iter = cycle_dataloader(graph_vector_loader)
        if cfg.graph_vector_objective != "suppress" and cfg.graph_vector_keep_weight > 0:
            graph_vector_keep_items = _build_graph_vector_items(
                cfg,
                localization,
                target_unlearned=False,
                max_items=cfg.graph_vector_keep_max_items,
            )
            graph_vector_keep_loader = DataLoader(
                graph_vector_keep_items,
                batch_size=cfg.graph_vector_batch_size,
                shuffle=True,
                collate_fn=lambda batch: batch,
            )
            graph_vector_keep_iter = cycle_dataloader(graph_vector_keep_loader)
        source_counts: Dict[str, int] = {}
        for item in graph_vector_items:
            source = str(item["source"])
            source_counts[source] = source_counts.get(source, 0) + 1
        keep_source_counts: Dict[str, int] = {}
        for item in graph_vector_keep_items:
            source = str(item["source"])
            keep_source_counts[source] = keep_source_counts.get(source, 0) + 1
        logger.info(
            "Graph-vector objective enabled: objective=%s weight=%.4f forget_vectors=%d keep_vectors=%d batch_size=%d forget_sources=%s keep_sources=%s",
            cfg.graph_vector_objective,
            cfg.graph_vector_suppress_weight,
            len(graph_vector_items),
            len(graph_vector_keep_items),
            cfg.graph_vector_batch_size,
            source_counts,
            keep_source_counts,
        )

    student_cfg = cfg
    resume_training_state = None
    resume_start_step = 0
    if cfg.resume_from_checkpoint is not None:
        from dataclasses import replace

        student_cfg = replace(
            cfg,
            init_new_lora=False,
            original_lora_path=str(cfg.resume_from_checkpoint),
        )
        resume_training_state = _load_resume_training_state(cfg.resume_from_checkpoint)
        if resume_training_state is None:
            resume_start_step = _checkpoint_step_from_path(cfg.resume_from_checkpoint)
            logger.info(
                "Resuming masked optimization from LoRA weights only: %s inferred_step=%d",
                cfg.resume_from_checkpoint,
                resume_start_step,
            )
        else:
            resume_start_step = int(resume_training_state.get("global_step", 0))
            logger.info(
                "Resuming masked optimization from full checkpoint: %s global_step=%d",
                cfg.resume_from_checkpoint,
                resume_start_step,
            )

    student_model = build_model(student_cfg, is_trainable=True)
    if getattr(cfg, "disable_parameter_mask", False):
        freeze_summary = {
            "trainable_before": int(sum(1 for _, param in student_model.named_parameters() if param.requires_grad)),
            "trainable_after": int(sum(1 for _, param in student_model.named_parameters() if param.requires_grad)),
            "frozen_now": 0,
        }
        gradient_mask_handles = []
        logger.info("Parameter mask disabled for ablation: all trainable LoRA parameters will be updated.")
    else:
        freeze_summary = _freeze_non_masked_trainable_parameters(
            student_model,
            allowed_param_names=set(localization.parameter_masks.keys()),
        )
        gradient_mask_handles = register_gradient_mask_hooks(student_model, localization.parameter_masks)
    device = get_model_device(student_model)
    scheduler_total_steps = max(1, math.ceil(len(forget_loader) * cfg.num_train_epochs))
    total_steps = scheduler_total_steps
    if getattr(cfg, "max_train_steps", 0) and cfg.max_train_steps > 0:
        total_steps = min(total_steps, int(cfg.max_train_steps))
    optimizer = AdamW([param for param in student_model.parameters() if param.requires_grad], lr=cfg.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=scheduler_total_steps,
    )
    if resume_training_state is not None:
        if "optimizer" in resume_training_state:
            optimizer.load_state_dict(resume_training_state["optimizer"])
        if "scheduler" in resume_training_state:
            lr_scheduler.load_state_dict(resume_training_state["scheduler"])

    logger.info(
        "Masked optimization will update within HCTL mask: selected_params=%s density=%.6f forget_loss_strategy=%s forget_loss_layers=%s trainable_before=%d trainable_after=%d frozen_now=%d",
        localization.metadata.get("total_selected_params", 0),
        localization.metadata.get("parameter_mask_density", 0.0),
        cfg.forget_loss_strategy,
        forget_loss_layers,
        freeze_summary["trainable_before"],
        freeze_summary["trainable_after"],
        freeze_summary["frozen_now"],
    )

    student_model.train()
    projected_named_params = _trainable_parameters_with_names(student_model)
    global_step = int(resume_start_step)
    optimizer.zero_grad()
    loss_history = []
    dual_kl_weight = float(cfg.dual_kl_weight_init if cfg.use_dual_kl_objective else cfg.retain_kl_weight)
    best_step_score = float("-inf")
    best_step = -1
    best_metrics: Dict[str, float] = {}
    best_retain_kl_guard_level = "none"
    no_improve_steps = 0
    best_state_dict = _clone_trainable_state_dict_to_cpu(student_model)
    ce_thresholds = list(cfg.forget_ce_checkpoint_thresholds or [])
    saved_ce_thresholds: set[float] = set()
    ce_threshold_records: list[Dict[str, Any]] = []
    ce_threshold_root = Path(cfg.output_dir) / "ce_threshold_checkpoints"
    if resume_training_state is not None:
        dual_kl_weight = float(resume_training_state.get("dual_kl_weight", dual_kl_weight))
        best_step_score = float(resume_training_state.get("best_step_score", best_step_score))
        best_step = int(resume_training_state.get("best_step", best_step))
        best_metrics = dict(resume_training_state.get("best_metrics", best_metrics) or {})
        best_retain_kl_guard_level = str(
            resume_training_state.get("best_retain_kl_guard_level", best_retain_kl_guard_level)
        )
        no_improve_steps = int(resume_training_state.get("no_improve_steps", no_improve_steps))
        saved_ce_thresholds.update(float(x) for x in resume_training_state.get("saved_ce_thresholds", []))
        best_state_dict = _clone_trainable_state_dict_to_cpu(student_model)
    elif resume_start_step > 0:
        reached_threshold = _resume_reached_ce_threshold(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else None
        if reached_threshold is not None:
            saved_ce_thresholds.update(threshold for threshold in ce_thresholds if threshold <= reached_threshold)
    if ce_thresholds:
        logger.info("Forget CE threshold checkpointing enabled: %s", ce_thresholds)
    stopped_early = False
    random_answer_allowed_token_ids = None
    if cfg.forget_loss_strategy == "random_answer_ce":
        special_token_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
        if tokenizer.pad_token_id is not None:
            special_token_ids.add(int(tokenizer.pad_token_id))
        random_answer_allowed_token_ids = torch.tensor(
            [idx for idx in range(len(tokenizer)) if idx not in special_token_ids],
            dtype=torch.long,
        )
        logger.info(
            "Random-answer CE will sample from %d non-special tokenizer ids.",
            int(random_answer_allowed_token_ids.numel()),
        )

    try:
        for epoch_idx in range(int(math.ceil(cfg.num_train_epochs))):
            for forget_batch in forget_loader:
                if global_step >= total_steps:
                    break

                forget_batch = move_batch_to_device(forget_batch, device)
                retain_batch = None
                if needs_retain_forward:
                    retain_batch = move_batch_to_device(next(retain_iter), device)
                if not _has_supervised_tokens(forget_batch["labels"]) or (
                    retain_batch is not None and not _has_supervised_tokens(retain_batch["labels"])
                ):
                    logger.warning("Skipping empty-label batch at step=%d", global_step)
                    global_step += 1
                    continue

                original_forget_batch = forget_batch
                original_forget_outputs = student_model(
                    input_ids=original_forget_batch["input_ids"],
                    attention_mask=original_forget_batch["attention_mask"],
                    labels=original_forget_batch["labels"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                forget_ce_loss = original_forget_outputs.loss
                forget_objective_loss = -forget_ce_loss
                repr_erasure_stats = None
                random_vector_forget_stats = None
                anchor_repulsion_stats = None
                parameter_random_stats = None
                gradient_ascent_stats = None
                projected_ga_stats = None
                if cfg.forget_loss_strategy in {"gradient_ascent", "gradient_ascent_projected"}:
                    gradient_ascent_stats = _compute_gradient_ascent_forget_loss(forget_ce_loss)
                    forget_objective_loss = gradient_ascent_stats["loss"]
                elif cfg.forget_loss_strategy == "representation_erasure":
                    repr_erasure_stats = _compute_representation_erasure_loss(
                        outputs=original_forget_outputs,
                        labels=original_forget_batch["labels"],
                        layer_indices=forget_loss_layers,
                    )
                    forget_objective_loss = repr_erasure_stats["loss"]
                elif cfg.forget_loss_strategy == "random_vector":
                    random_vector_forget_stats = _compute_layerwise_random_vector_forget_loss(
                        outputs=original_forget_outputs,
                        labels=original_forget_batch["labels"],
                        batch_items=original_forget_batch["items"],
                        base_seed=cfg.random_vector_seed,
                        layer_indices=forget_loss_layers,
                    )
                    forget_objective_loss = cfg.random_vector_forget_weight * random_vector_forget_stats["loss"]
                    if forget_anchor_cache is not None:
                        anchor_repulsion_stats = _compute_anchor_repulsion_forget_loss(
                            outputs=original_forget_outputs,
                            labels=original_forget_batch["labels"],
                            batch_items=original_forget_batch["items"],
                            anchor_cache=forget_anchor_cache,
                            layer_indices=forget_loss_layers,
                            max_similarity=cfg.forget_anchor_max_similarity,
                        )
                        forget_objective_loss = (
                            forget_objective_loss
                            + cfg.forget_anchor_weight * anchor_repulsion_stats["loss"]
                        )
                elif cfg.forget_loss_strategy == "parameter_random_vector":
                    parameter_random_stats = _compute_parameter_random_vector_forget_loss(
                        outputs=original_forget_outputs,
                        labels=original_forget_batch["labels"],
                        batch_items=original_forget_batch["items"],
                        base_seed=cfg.random_vector_seed,
                    )
                    forget_objective_loss = cfg.parameter_random_vector_weight * parameter_random_stats["loss"]
                elif cfg.forget_loss_strategy == "random_answer_ce":
                    if random_answer_allowed_token_ids is None:
                        raise RuntimeError("random_answer_ce requires initialized allowed token ids.")
                    random_forget_batch = _build_random_answer_token_batch(
                        original_forget_batch,
                        allowed_token_ids=random_answer_allowed_token_ids,
                        base_seed=cfg.random_vector_seed,
                    )
                    random_forget_outputs = student_model(
                        input_ids=random_forget_batch["input_ids"],
                        attention_mask=random_forget_batch["attention_mask"],
                        labels=random_forget_batch["labels"],
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    forget_objective_loss = random_forget_outputs.loss
                answer_suppression_ce_loss = forget_ce_loss * 0.0
                if cfg.answer_suppression_weight > 0:
                    if answer_suppression_iter is None:
                        answer_suppression_ce_loss = forget_ce_loss
                    else:
                        answer_suppression_batch = move_batch_to_device(next(answer_suppression_iter), device)
                        if _has_supervised_tokens(answer_suppression_batch["labels"]):
                            answer_suppression_outputs = student_model(
                                input_ids=answer_suppression_batch["input_ids"],
                                attention_mask=answer_suppression_batch["attention_mask"],
                                labels=answer_suppression_batch["labels"],
                                output_hidden_states=False,
                                return_dict=True,
                            )
                            answer_suppression_ce_loss = answer_suppression_outputs.loss
                        else:
                            logger.warning("Skipping empty-label answer suppression batch at step=%d", global_step)
                student_retain_outputs = None
                retain_ce_loss = forget_ce_loss.new_zeros(())
                retain_kl_loss = forget_ce_loss.new_zeros(())
                if needs_retain_forward:
                    student_retain_outputs = student_model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch["attention_mask"],
                        labels=retain_batch["labels"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    retain_ce_loss = student_retain_outputs.loss
                    if needs_retain_kl:
                        if retain_teacher_cache is None:
                            raise RuntimeError("retain KL requested but retain teacher cache is unavailable.")
                        teacher_topk_indices, teacher_topk_logprobs, teacher_valid_mask = _stack_cached_teacher_batch(
                            retain_teacher_cache,
                            [str(item["example_id"]) for item in retain_batch["items"]],
                            target_seq_len=student_retain_outputs.logits.shape[1] - 1,
                            device=device,
                        )
                        retain_kl_loss = _masked_topk_distillation_loss(
                            student_logits=student_retain_outputs.logits,
                            teacher_topk_indices=teacher_topk_indices,
                            teacher_topk_logprobs=teacher_topk_logprobs,
                            teacher_valid_mask=teacher_valid_mask,
                        )
                graph_vector_stats = None
                graph_vector_suppress_loss = forget_ce_loss * 0.0
                if graph_vector_iter is not None:
                    graph_vector_batch = next(graph_vector_iter)
                    graph_vector_keep_batch = next(graph_vector_keep_iter) if graph_vector_keep_iter is not None else []
                    if cfg.graph_vector_objective == "suppress":
                        graph_vector_stats = _compute_graph_vector_suppress_loss(
                            model=student_model,
                            tokenizer=tokenizer,
                            graph_items=graph_vector_batch,
                            selected_neurons=selected_graph_neurons,
                            anchor_text=cfg.graph_vector_anchor_text,
                        )
                    elif cfg.graph_vector_objective == "final_repr":
                        graph_vector_stats = _compute_graph_vector_final_repr_loss(
                            model=student_model,
                            tokenizer=tokenizer,
                            forget_items=graph_vector_batch,
                            keep_items=graph_vector_keep_batch,
                            anchor_text=cfg.graph_vector_anchor_text,
                            random_weight=cfg.graph_vector_random_weight,
                            away_weight=cfg.graph_vector_away_weight,
                            keep_weight=cfg.graph_vector_keep_weight,
                            away_max_similarity=cfg.graph_vector_away_max_similarity,
                            base_seed=cfg.random_vector_seed,
                        )
                    elif cfg.graph_vector_objective == "neuron_repr":
                        graph_vector_stats = _compute_graph_vector_neuron_repr_loss(
                            model=student_model,
                            tokenizer=tokenizer,
                            forget_items=graph_vector_batch,
                            keep_items=graph_vector_keep_batch,
                            selected_neurons=selected_graph_neurons,
                            anchor_text=cfg.graph_vector_anchor_text,
                            random_weight=cfg.graph_vector_random_weight,
                            away_weight=cfg.graph_vector_away_weight,
                            keep_weight=cfg.graph_vector_keep_weight,
                            away_max_similarity=cfg.graph_vector_away_max_similarity,
                            base_seed=cfg.random_vector_seed,
                        )
                    else:
                        raise ValueError(f"Unsupported graph_vector_objective: {cfg.graph_vector_objective}")
                    graph_vector_suppress_loss = graph_vector_stats["loss"]
                if cfg.forget_nll_calibration_target_mode == "fixed":
                    forget_nll_calibration_target = torch.tensor(
                        float(cfg.forget_nll_calibration_fixed_target),
                        device=device,
                        dtype=forget_ce_loss.dtype,
                    )
                else:
                    forget_nll_calibration_target = retain_ce_loss.detach() + float(cfg.forget_nll_calibration_margin)
                forget_nll_excess = torch.relu(forget_ce_loss - forget_nll_calibration_target)
                forget_nll_calibration_loss = forget_nll_excess * forget_nll_excess
                forget_ce_suppression_loss = -forget_ce_loss
                answer_suppression_loss = -answer_suppression_ce_loss
                if cfg.use_dual_kl_objective:
                    dual_penalty = torch.relu(retain_kl_loss - cfg.retain_kl_target)
                    total_loss = (
                        cfg.forget_objective_weight * forget_objective_loss
                        + cfg.forget_ce_weight * forget_ce_suppression_loss
                        + cfg.answer_suppression_weight * answer_suppression_loss
                        + cfg.graph_vector_suppress_weight * graph_vector_suppress_loss
                        + dual_kl_weight * dual_penalty
                        + cfg.forget_nll_calibration_weight * forget_nll_calibration_loss
                    )
                else:
                    dual_penalty = retain_kl_loss
                    total_loss = (
                        cfg.forget_objective_weight * forget_objective_loss
                        + cfg.forget_ce_weight * forget_ce_suppression_loss
                        + cfg.answer_suppression_weight * answer_suppression_loss
                        + cfg.graph_vector_suppress_weight * graph_vector_suppress_loss
                        + cfg.retain_kl_weight * retain_kl_loss
                        + cfg.forget_nll_calibration_weight * forget_nll_calibration_loss
                    )

                if not all(
                    _is_finite_tensor(x)
                    for x in [
                        forget_ce_loss,
                        answer_suppression_ce_loss,
                        forget_objective_loss,
                        graph_vector_suppress_loss,
                        retain_ce_loss,
                        retain_kl_loss,
                        forget_nll_calibration_loss,
                        total_loss,
                    ]
                ):
                    logger.warning("Skipping non-finite batch at step=%d", global_step)
                    optimizer.zero_grad()
                    global_step += 1
                    continue
                if cfg.forget_loss_strategy == "gradient_ascent_projected":
                    projected_ga_stats = _assign_projected_gradient_ascent_grads(
                        model=student_model,
                        named_params=projected_named_params,
                        parameter_masks=localization.parameter_masks,
                        forget_ce_loss=forget_ce_loss,
                        retain_kl_loss=dual_penalty if cfg.use_dual_kl_objective else retain_kl_loss,
                        forget_weight=cfg.forget_objective_weight,
                        retain_kl_weight=dual_kl_weight if cfg.use_dual_kl_objective else cfg.retain_kl_weight,
                        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    )
                else:
                    (total_loss / cfg.gradient_accumulation_steps).backward()

                if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if cfg.use_dual_kl_objective:
                    dual_kl_weight = max(
                        float(cfg.dual_kl_weight_min),
                        dual_kl_weight + (cfg.dual_kl_lr * float(retain_kl_loss.detach().cpu() - cfg.retain_kl_target)),
                    )

                forget_ce_value = float(forget_ce_loss.detach().cpu())
                answer_suppression_ce_value = float(answer_suppression_ce_loss.detach().cpu())
                forget_objective_value = float(forget_objective_loss.detach().cpu())
                retain_ce_value = float(retain_ce_loss.detach().cpu())
                retain_kl_value = float(retain_kl_loss.detach().cpu())
                retain_excess = max(0.0, retain_kl_value - cfg.retain_kl_target)
                forget_nll_calibration_target_value = float(forget_nll_calibration_target.detach().cpu())
                forget_nll_excess_value = float(forget_nll_excess.detach().cpu())
                forget_nll_calibration_value = float(forget_nll_calibration_loss.detach().cpu())
                random_vector_similarity = (
                    float(random_vector_forget_stats["similarity"].detach().cpu())
                    if random_vector_forget_stats is not None
                    else 0.0
                )
                anchor_similarity = (
                    float(anchor_repulsion_stats["similarity"].detach().cpu())
                    if anchor_repulsion_stats is not None
                    else 0.0
                )
                parameter_random_cosine = (
                    float(parameter_random_stats["cosine"].detach().cpu())
                    if parameter_random_stats is not None
                    else 0.0
                )
                parameter_random_output_l2 = (
                    float(parameter_random_stats["output_l2"].detach().cpu())
                    if parameter_random_stats is not None
                    else 0.0
                )
                graph_vector_suppress_value = float(graph_vector_suppress_loss.detach().cpu())
                graph_vector_mean_abs = (
                    float(graph_vector_stats["mean_abs_activation"].detach().cpu())
                    if graph_vector_stats is not None
                    else 0.0
                )
                graph_vector_active_neurons = (
                    float(graph_vector_stats["active_neurons"].detach().cpu())
                    if graph_vector_stats is not None
                    else 0.0
                )
                graph_vector_random_cosine = (
                    float(graph_vector_stats["random_cosine"].detach().cpu())
                    if graph_vector_stats is not None and "random_cosine" in graph_vector_stats
                    else 0.0
                )
                graph_vector_away_cosine = (
                    float(graph_vector_stats["away_cosine"].detach().cpu())
                    if graph_vector_stats is not None and "away_cosine" in graph_vector_stats
                    else 0.0
                )
                graph_vector_keep_cosine = (
                    float(graph_vector_stats["keep_cosine"].detach().cpu())
                    if graph_vector_stats is not None and "keep_cosine" in graph_vector_stats
                    else 0.0
                )
                projected_ga_projection_coeff = (
                    float(projected_ga_stats["projection_coeff"].detach().cpu())
                    if projected_ga_stats is not None
                    else 0.0
                )
                projected_ga_grad_cosine = (
                    float(projected_ga_stats["grad_cosine"].detach().cpu())
                    if projected_ga_stats is not None
                    else 0.0
                )
                projected_ga_forget_grad_norm = (
                    float(projected_ga_stats["forget_grad_norm"].detach().cpu())
                    if projected_ga_stats is not None
                    else 0.0
                )
                projected_ga_retain_grad_norm = (
                    float(projected_ga_stats["retain_grad_norm"].detach().cpu())
                    if projected_ga_stats is not None
                    else 0.0
                )
                projected_ga_removed_update_norm = (
                    float(projected_ga_stats["removed_update_norm"].detach().cpu())
                    if projected_ga_stats is not None
                    else 0.0
                )
                                                                             
                                                                                
                mia_risk = forget_nll_excess_value
                graph_suppress_penalty = cfg.graph_vector_suppress_weight * graph_vector_suppress_value
                if cfg.graph_vector_suppress_weight > 0 and cfg.forget_objective_weight == 0:
                    step_score = -graph_suppress_penalty - (cfg.best_score_retain_alpha * retain_excess) - (cfg.best_score_mia_alpha * mia_risk)
                elif cfg.forget_loss_strategy == "random_vector":
                    step_score = random_vector_similarity - (cfg.best_score_retain_alpha * retain_excess) - (cfg.best_score_mia_alpha * mia_risk)
                elif cfg.forget_loss_strategy == "parameter_random_vector":
                    step_score = parameter_random_cosine - (cfg.best_score_retain_alpha * retain_excess) - (cfg.best_score_mia_alpha * mia_risk)
                else:
                    step_score = forget_ce_value - (cfg.best_score_retain_alpha * retain_excess) - (cfg.best_score_mia_alpha * mia_risk)
                if not (cfg.graph_vector_suppress_weight > 0 and cfg.forget_objective_weight == 0):
                    step_score -= graph_suppress_penalty
                step_score += cfg.forget_ce_weight * forget_ce_value
                step_score += cfg.answer_suppression_weight * answer_suppression_ce_value
                proxy_metrics: Dict[str, float] = {}
                should_consider_best_step = True
                if canonical_proxy_spec is not None and canonical_proxy_baseline is not None:
                    should_consider_best_step = (
                        global_step == 0
                        or global_step % cfg.canonical_proxy_interval == 0
                        or global_step >= total_steps - 1
                    )
                    if should_consider_best_step:
                        proxy_metrics = _evaluate_canonical_proxy(
                            cfg=cfg,
                            model=student_model,
                            tokenizer=tokenizer,
                            proxy_spec=canonical_proxy_spec,
                            proxy_baseline=canonical_proxy_baseline,
                        )
                        step_score = proxy_metrics["canonical_proxy_score"]
                        logger.info(
                            "canonical_proxy step=%d score=%s continuous=%.4f retain_ok=%d forget_nll_delta=%.4f retain_nll_delta=%.4f retain_kl=%.4f",
                            global_step,
                            f"{step_score:.4f}" if math.isfinite(step_score) else "-inf",
                            proxy_metrics["canonical_proxy_continuous_score"],
                            int(proxy_metrics["canonical_proxy_retain_ok"]),
                            proxy_metrics["canonical_proxy_forget_nll_delta"],
                            proxy_metrics["canonical_proxy_retain_nll_delta"],
                            proxy_metrics["canonical_proxy_retain_kl"],
                        )
                best_guard_retain_kl_value = retain_kl_value
                if proxy_metrics and "canonical_proxy_retain_kl" in proxy_metrics:
                    best_guard_retain_kl_value = float(proxy_metrics["canonical_proxy_retain_kl"])
                best_retain_kl_guard_level_for_step = "none"
                best_retain_kl_guard_rank_for_step = 2
                if cfg.best_retain_kl_max >= 0:
                    if best_guard_retain_kl_value <= cfg.best_retain_kl_max:
                        best_retain_kl_guard_level_for_step = "primary"
                        best_retain_kl_guard_rank_for_step = 0
                    elif (
                        cfg.best_retain_kl_fallback_max >= 0
                        and best_guard_retain_kl_value <= cfg.best_retain_kl_fallback_max
                    ):
                        best_retain_kl_guard_level_for_step = "fallback"
                        best_retain_kl_guard_rank_for_step = 1
                    else:
                        should_consider_best_step = False
                        best_retain_kl_guard_level_for_step = "rejected"
                best_retain_kl_guard_rank = (
                    0
                    if best_retain_kl_guard_level == "primary"
                    else 1
                    if best_retain_kl_guard_level == "fallback"
                    else 2
                )
                should_update_best_step = (
                    should_consider_best_step
                    and (
                        best_retain_kl_guard_rank_for_step < best_retain_kl_guard_rank
                        or (
                            best_retain_kl_guard_rank_for_step == best_retain_kl_guard_rank
                            and step_score > best_step_score
                        )
                    )
                )
                if should_update_best_step:
                    best_step_score = step_score
                    best_step = global_step
                    best_retain_kl_guard_level = best_retain_kl_guard_level_for_step
                    best_metrics = {
                        "score": step_score,
                        "best_retain_kl_guard_level": best_retain_kl_guard_level,
                        "best_retain_kl_max": cfg.best_retain_kl_max,
                        "best_retain_kl_fallback_max": cfg.best_retain_kl_fallback_max,
                        "forget_ce": forget_ce_value,
                        "answer_suppression_ce": answer_suppression_ce_value,
                        "answer_suppression_weight": cfg.answer_suppression_weight,
                        "forget_objective": forget_objective_value,
                        "graph_vector_suppress_loss": graph_vector_suppress_value,
                        "graph_vector_mean_abs_activation": graph_vector_mean_abs,
                        "graph_vector_active_neurons": graph_vector_active_neurons,
                        "graph_vector_random_cosine": graph_vector_random_cosine,
                        "graph_vector_away_cosine": graph_vector_away_cosine,
                        "graph_vector_keep_cosine": graph_vector_keep_cosine,
                        "retain_ce": retain_ce_value,
                        "retain_kl": retain_kl_value,
                        "retain_excess": retain_excess,
                        "forget_nll_calibration_target": forget_nll_calibration_target_value,
                        "forget_nll_excess": forget_nll_excess_value,
                        "forget_nll_calibration_loss": forget_nll_calibration_value,
                        "random_vector_similarity": random_vector_similarity,
                        "anchor_similarity": anchor_similarity,
                        "parameter_random_cosine": parameter_random_cosine,
                        "parameter_random_output_l2": parameter_random_output_l2,
                        "graph_suppress_penalty": graph_suppress_penalty,
                        "dual_kl_weight": float(dual_kl_weight),
                        "projected_ga_projection_coeff": projected_ga_projection_coeff,
                        "projected_ga_grad_cosine": projected_ga_grad_cosine,
                        "projected_ga_forget_grad_norm": projected_ga_forget_grad_norm,
                        "projected_ga_retain_grad_norm": projected_ga_retain_grad_norm,
                        "projected_ga_removed_update_norm": projected_ga_removed_update_norm,
                    }
                    best_metrics.update(proxy_metrics)
                    best_state_dict = _clone_trainable_state_dict_to_cpu(student_model)
                    no_improve_steps = 0
                elif should_consider_best_step:
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1

                if ce_thresholds:
                    reached_thresholds = [
                        threshold
                        for threshold in ce_thresholds
                        if threshold not in saved_ce_thresholds and forget_ce_value >= threshold
                    ]
                    if reached_thresholds:
                        threshold_state_dict = _clone_trainable_state_dict_to_cpu(student_model)
                        for threshold in reached_thresholds:
                            threshold_int = int(threshold) if float(threshold).is_integer() else threshold
                            checkpoint_dir = ce_threshold_root / f"ce{threshold_int}_step{global_step}"
                            metadata = {
                                "threshold": threshold,
                                "step": global_step,
                                "forget_ce": forget_ce_value,
                                "retain_kl": retain_kl_value,
                                "retain_ce": retain_ce_value,
                                "loss": float(total_loss.detach().cpu()),
                                "forget_loss_strategy": cfg.forget_loss_strategy,
                                "forget_objective_weight": cfg.forget_objective_weight,
                                "forget_target_path": str(cfg.forget_target_path) if cfg.forget_target_path else "",
                                "retain_path": str(cfg.retain_path),
                                "reuse_localization_pkl": str(cfg.reuse_localization_pkl) if cfg.reuse_localization_pkl else "",
                                "base_model_path": cfg.base_model_path,
                                "original_lora_path": cfg.original_lora_path,
                                "output_dir": cfg.output_dir,
                                "learning_rate": cfg.learning_rate,
                                "num_train_epochs": cfg.num_train_epochs,
                                "max_train_steps": cfg.max_train_steps,
                                "per_device_train_batch_size": cfg.per_device_train_batch_size,
                                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                                "max_grad_norm": cfg.max_grad_norm,
                                "max_length": cfg.max_length,
                                "seed": cfg.seed,
                            }
                            _save_forget_ce_threshold_checkpoint(
                                model=student_model,
                                tokenizer=tokenizer,
                                state_dict=threshold_state_dict,
                                checkpoint_dir=checkpoint_dir,
                                metadata=metadata,
                            )
                            _save_training_state(
                                checkpoint_dir=checkpoint_dir,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                global_step=global_step,
                                epoch_idx=epoch_idx,
                                best_step=best_step,
                                best_step_score=best_step_score,
                                best_metrics=best_metrics,
                                best_retain_kl_guard_level=best_retain_kl_guard_level,
                                no_improve_steps=no_improve_steps,
                                dual_kl_weight=dual_kl_weight,
                                saved_ce_thresholds=saved_ce_thresholds | {threshold},
                            )
                            saved_ce_thresholds.add(threshold)
                            ce_threshold_records.append({**metadata, "checkpoint_dir": str(checkpoint_dir)})
                            logger.info(
                                "Saved forget CE threshold checkpoint threshold=%.4f step=%d forget_ce=%.4f retain_kl=%.4f path=%s",
                                threshold,
                                global_step,
                                forget_ce_value,
                                retain_kl_value,
                                checkpoint_dir,
                            )

                loss_history.append(
                    {
                        "step": global_step,
                        "epoch": epoch_idx,
                        "loss": float(total_loss.detach().cpu()),
                        "forget_ce": forget_ce_value,
                        "answer_suppression_ce": answer_suppression_ce_value,
                        "answer_suppression_weight": cfg.answer_suppression_weight,
                        "forget_objective": forget_objective_value,
                        "forget_strategy": str(cfg.forget_loss_strategy),
                        "repr_mean_abs_activation": float(repr_erasure_stats["mean_abs_activation"].detach().cpu()) if repr_erasure_stats is not None else 0.0,
                        "repr_layers": float(repr_erasure_stats["num_layers"].detach().cpu()) if repr_erasure_stats is not None else 0.0,
                        "random_vector_similarity": random_vector_similarity,
                        "anchor_similarity": anchor_similarity,
                        "parameter_random_cosine": parameter_random_cosine,
                        "parameter_random_output_l2": parameter_random_output_l2,
                        "graph_vector_suppress_loss": graph_vector_suppress_value,
                        "graph_vector_mean_abs_activation": graph_vector_mean_abs,
                        "graph_vector_active_neurons": graph_vector_active_neurons,
                        "graph_vector_random_cosine": graph_vector_random_cosine,
                        "graph_vector_away_cosine": graph_vector_away_cosine,
                        "graph_vector_keep_cosine": graph_vector_keep_cosine,
                        "retain_ce": retain_ce_value,
                        "retain_kl": retain_kl_value,
                        "forget_nll_calibration_target": forget_nll_calibration_target_value,
                        "forget_nll_excess": forget_nll_excess_value,
                        "forget_nll_calibration_loss": forget_nll_calibration_value,
                        "dual_kl_penalty": float(dual_penalty.detach().cpu()),
                        "dual_kl_weight": float(dual_kl_weight),
                        "projected_ga_projection_coeff": projected_ga_projection_coeff,
                        "projected_ga_grad_cosine": projected_ga_grad_cosine,
                        "projected_ga_forget_grad_norm": projected_ga_forget_grad_norm,
                        "projected_ga_retain_grad_norm": projected_ga_retain_grad_norm,
                        "projected_ga_removed_update_norm": projected_ga_removed_update_norm,
                        "best_step_score": float(best_step_score),
                        "best_step": int(best_step),
                        "best_retain_kl_guard_level": best_retain_kl_guard_level,
                        "no_improve_steps": int(no_improve_steps),
                        **proxy_metrics,
                    }
                )
                if global_step % cfg.logging_steps == 0:
                    logger.info(
                        "masked_unlearning step=%d/%d loss=%.4f forget_ce=%.4f anssup_ce=%.4f forget_obj=%.4f forget_strategy=%s rand_sim=%.4f anchor_sim=%.4f param_rand_cos=%.4f param_out_l2=%.4f repr_abs=%.4f repr_layers=%.0f graph_obj=%s graph_loss=%.4f graph_abs=%.4f graph_rand=%.4f graph_away=%.4f graph_keep=%.4f retain_ce=%.4f retain_kl=%.4f forget_nll_target=%.4f forget_nll_excess=%.4f dual_penalty=%.4f dual_w=%.4f proj_coeff=%.6f proj_cos=%.4f proj_removed=%.4f best_score=%.4f best_step=%d stall=%d",
                        global_step,
                        total_steps,
                        loss_history[-1]["loss"],
                        loss_history[-1]["forget_ce"],
                        loss_history[-1]["answer_suppression_ce"],
                        loss_history[-1]["forget_objective"],
                        loss_history[-1]["forget_strategy"],
                        loss_history[-1]["random_vector_similarity"],
                        loss_history[-1]["anchor_similarity"],
                        loss_history[-1]["parameter_random_cosine"],
                        loss_history[-1]["parameter_random_output_l2"],
                        loss_history[-1]["repr_mean_abs_activation"],
                        loss_history[-1]["repr_layers"],
                        cfg.graph_vector_objective,
                        loss_history[-1]["graph_vector_suppress_loss"],
                        loss_history[-1]["graph_vector_mean_abs_activation"],
                        loss_history[-1]["graph_vector_random_cosine"],
                        loss_history[-1]["graph_vector_away_cosine"],
                        loss_history[-1]["graph_vector_keep_cosine"],
                        loss_history[-1]["retain_ce"],
                        loss_history[-1]["retain_kl"],
                        loss_history[-1]["forget_nll_calibration_target"],
                        loss_history[-1]["forget_nll_excess"],
                        loss_history[-1]["dual_kl_penalty"],
                        loss_history[-1]["dual_kl_weight"],
                        loss_history[-1]["projected_ga_projection_coeff"],
                        loss_history[-1]["projected_ga_grad_cosine"],
                        loss_history[-1]["projected_ga_removed_update_norm"],
                        loss_history[-1]["best_step_score"],
                        loss_history[-1]["best_step"],
                        loss_history[-1]["no_improve_steps"],
                    )
                if (
                    cfg.early_stop_enabled
                    and global_step >= cfg.early_stop_min_steps
                    and no_improve_steps >= cfg.early_stop_patience
                ):
                    stopped_early = True
                    logger.info(
                        "Early stop triggered at step=%d with best_step=%d best_score=%.4f stall=%d",
                        global_step,
                        best_step,
                        best_step_score,
                        no_improve_steps,
                    )
                    break
                global_step += 1

            if global_step >= total_steps or stopped_early:
                break

        if (not stopped_early) and global_step % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    finally:
        remove_gradient_mask_hooks(gradient_mask_handles)

    if best_state_dict:
        student_model.load_state_dict(best_state_dict, strict=False)
        logger.info(
            "Reloaded best masked optimization state from step=%d with best_score=%.4f forget_ce=%.4f retain_kl=%.4f",
            best_step,
            best_step_score,
            best_metrics.get("forget_ce", 0.0),
            best_metrics.get("retain_kl", 0.0),
        )

    student_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    _save_training_state(
        checkpoint_dir=Path(cfg.output_dir),
        optimizer=optimizer,
        scheduler=lr_scheduler,
        global_step=global_step,
        epoch_idx=int(math.ceil(cfg.num_train_epochs)) - 1,
        best_step=best_step,
        best_step_score=best_step_score,
        best_metrics=best_metrics,
        best_retain_kl_guard_level=best_retain_kl_guard_level,
        no_improve_steps=no_improve_steps,
        dual_kl_weight=dual_kl_weight,
        saved_ce_thresholds=saved_ce_thresholds,
    )
    run_metadata_extra["masked_parameter_optimization"] = {
        "forget_entity_dataset_rows": int(len(forget_entity_dataset)),
        "forget_training_dataset_rows": int(len(forget_training_dataset)),
        "forget_loss_strategy": cfg.forget_loss_strategy,
        "forget_objective_weight": cfg.forget_objective_weight,
        "answer_suppression_path": str(cfg.answer_suppression_path) if cfg.answer_suppression_path else "",
        "answer_suppression_weight": cfg.answer_suppression_weight,
        "answer_suppression_dataset_rows": int(len(answer_suppression_dataset)) if answer_suppression_dataset is not None else 0,
        "disable_parameter_mask": bool(getattr(cfg, "disable_parameter_mask", False)),
        "forget_target_mode": cfg.forget_target_mode,
        "safe_response": cfg.safe_response,
        "forget_loss_top_layers": cfg.forget_loss_top_layers,
        "forget_loss_layers": forget_loss_layers,
        "trainable_freeze_summary": freeze_summary,
        "random_vector_seed": cfg.random_vector_seed,
        "random_vector_forget_weight": cfg.random_vector_forget_weight,
        "parameter_random_vector_weight": cfg.parameter_random_vector_weight,
        "parameter_random_vector_scale": cfg.parameter_random_vector_scale,
        "graph_vector_suppress_weight": cfg.graph_vector_suppress_weight,
        "graph_vector_objective": cfg.graph_vector_objective,
        "graph_vector_random_weight": cfg.graph_vector_random_weight,
        "graph_vector_away_weight": cfg.graph_vector_away_weight,
        "graph_vector_keep_weight": cfg.graph_vector_keep_weight,
        "graph_vector_away_max_similarity": cfg.graph_vector_away_max_similarity,
        "graph_vector_batch_size": cfg.graph_vector_batch_size,
        "graph_vector_max_items": cfg.graph_vector_max_items,
        "graph_vector_keep_max_items": cfg.graph_vector_keep_max_items,
        "graph_vector_anchor_text": cfg.graph_vector_anchor_text,
        "graph_vector_items": int(len(graph_vector_items)),
        "graph_vector_keep_items": int(len(graph_vector_keep_items)),
        "forget_anchor_weight": cfg.forget_anchor_weight,
        "forget_anchor_max_similarity": cfg.forget_anchor_max_similarity,
        "retain_kl_weight": cfg.retain_kl_weight,
        "use_dual_kl_objective": cfg.use_dual_kl_objective,
        "retain_kl_target": cfg.retain_kl_target,
        "dual_kl_lr": cfg.dual_kl_lr,
        "dual_kl_weight_init": cfg.dual_kl_weight_init,
        "dual_kl_weight_min": cfg.dual_kl_weight_min,
        "dual_kl_weight_final": float(dual_kl_weight),
        "retain_anchor_type": "offline_teacher_topk_distillation",
        "retain_topk": cfg.retain_topk,
        "forget_nll_calibration_weight": cfg.forget_nll_calibration_weight,
        "forget_nll_calibration_margin": cfg.forget_nll_calibration_margin,
        "forget_nll_calibration_fixed_target": cfg.forget_nll_calibration_fixed_target,
        "forget_nll_calibration_target_mode": cfg.forget_nll_calibration_target_mode,
        "best_score_mia_alpha": cfg.best_score_mia_alpha,
        "early_stop_enabled": cfg.early_stop_enabled,
        "early_stop_min_steps": cfg.early_stop_min_steps,
        "early_stop_patience": cfg.early_stop_patience,
        "best_score_retain_alpha": cfg.best_score_retain_alpha,
        "canonical_proxy_enabled": cfg.canonical_proxy_enabled,
        "canonical_proxy_forget_path": str(cfg.canonical_proxy_forget_path) if cfg.canonical_proxy_forget_path else "",
        "canonical_proxy_retain_path": str(cfg.canonical_proxy_retain_path) if cfg.canonical_proxy_retain_path else "",
        "canonical_proxy_forget_samples": cfg.canonical_proxy_forget_samples,
        "canonical_proxy_retain_samples": cfg.canonical_proxy_retain_samples,
        "canonical_proxy_interval": cfg.canonical_proxy_interval,
        "canonical_proxy_batch_size": cfg.canonical_proxy_batch_size,
        "canonical_proxy_retain_nll_delta_max": cfg.canonical_proxy_retain_nll_delta_max,
        "canonical_proxy_retain_kl_max": cfg.canonical_proxy_retain_kl_max,
        "canonical_proxy_hard_retain_filter": cfg.canonical_proxy_hard_retain_filter,
        "canonical_proxy_baseline_forget_nll": float(canonical_proxy_baseline["forget_nll_mean"]) if canonical_proxy_baseline else 0.0,
        "canonical_proxy_baseline_retain_nll": float(canonical_proxy_baseline["retain_nll_mean"]) if canonical_proxy_baseline else 0.0,
        "stopped_early": stopped_early,
        "best_step": best_step,
        "best_step_score": float(best_step_score),
        "best_step_metrics": best_metrics,
        "forget_ce_checkpoint_thresholds": ce_thresholds,
        "forget_ce_threshold_checkpoints": ce_threshold_records,
        "resume_from_checkpoint": str(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else "",
        "resume_start_step": int(resume_start_step),
        "final_global_step": int(global_step),
        "loss_history_preview": loss_history[: min(20, len(loss_history))],
        "legacy_disabled_terms": [
            "adaptive_forget_weights",
            "retain_ce",
            "retain_repr",
        ],
    }
