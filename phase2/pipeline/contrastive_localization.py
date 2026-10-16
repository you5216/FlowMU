                       

       
                                     
                                                 
                       
                   
   

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .stage2_unlearning import (
    LocalizationArtifacts,
    extract_layer_index,
    normalize_scores,
)

logger = logging.getLogger("GAT-AFRU.Phase2.HCTL")


def _resolve_decoder_layers(model):
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
    ]

    for path in candidate_paths:
        obj = model
        found = True
        for attr in path:
            if not hasattr(obj, attr):
                found = False
                break
            obj = getattr(obj, attr)
        if found:
            return obj

    raise AttributeError(f"Could not locate decoder layers for model type: {type(model).__name__}")


def _sample_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=42).select(range(max_samples))


def _collect_target_parameters(
    model,
    *,
    allowed_layers: set[int] | None = None,
    allowed_module_keywords: tuple[str, ...] | None = None,
) -> Dict[str, torch.nn.Parameter]:
    target_params: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if "lora_B" not in name or param.ndim != 2 or not param.requires_grad:
            continue
        if allowed_module_keywords is not None and not any(keyword in name for keyword in allowed_module_keywords):
            continue
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        target_params[name] = param
    if not target_params:
        raise ValueError("HCTL could not find trainable 2D LoRA-B parameters.")
    return target_params


@contextmanager
def _capture_layer_mlp_inputs(model, allowed_layers: set[int] | None = None):
    activations: Dict[int, torch.Tensor] = {}
    handles = []

    def get_activation_hook(layer_idx: int):
        def hook(module, input_args, output):
            if not input_args:
                return
            activation = input_args[0]
            if isinstance(activation, torch.Tensor):
                activations[layer_idx] = activation.detach()

        return hook

    layers = _resolve_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down_proj is None:
            continue
        handles.append(down_proj.register_forward_hook(get_activation_hook(layer_idx)))

    try:
        yield activations
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                continue


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _clean_token_text(token: str) -> str:
    cleaned = token.replace("Ġ", "").replace("▁", "").replace("##", "").strip()
    return cleaned


@lru_cache(maxsize=4)
def _load_key_annotation_entries(path_str: str) -> Dict[str, dict]:
    with open(path_str, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
        return payload["entries"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported key-annotation JSON format: {path_str}")


def _get_key_annotation_entries(cfg) -> Dict[str, dict]:
    path = getattr(cfg, "localization_key_annotation_path", None)
    if path is None:
        return {}
    return _load_key_annotation_entries(str(path))


def _extract_item_key_phrases(item: dict, key_annotation_entries: Dict[str, dict]) -> list[str]:
    if not key_annotation_entries:
        return []
    example_id = str(item.get("example_id", "")).strip()
    entry = key_annotation_entries.get(example_id, {})
    if not isinstance(entry, dict):
        return []
    phrases = []
    for key in ("key_phrases", "key_spans", "key_contents", "content_phrases", "keywords"):
        value = entry.get(key)
        if isinstance(value, list):
            phrases.extend(str(v).strip() for v in value if str(v).strip())
    deduped = []
    seen = set()
    for phrase in phrases:
        if phrase not in seen:
            deduped.append(phrase)
            seen.add(phrase)
    return deduped


def _match_phrase_token_spans(answer_token_ids: list[int], phrase_token_ids: list[int]) -> list[tuple[int, int]]:
    if not answer_token_ids or not phrase_token_ids or len(phrase_token_ids) > len(answer_token_ids):
        return []
    matches: list[tuple[int, int]] = []
    phrase_len = len(phrase_token_ids)
    for start_idx in range(0, len(answer_token_ids) - phrase_len + 1):
        if answer_token_ids[start_idx:start_idx + phrase_len] == phrase_token_ids:
            matches.append((start_idx, start_idx + phrase_len))
    return matches


def _build_annotation_row_mask(
    *,
    tokenizer,
    input_ids_row: torch.Tensor,
    answer_mask_row: torch.Tensor,
    phrases: list[str],
) -> tuple[torch.Tensor, list[str]]:
    row_mask = torch.zeros_like(answer_mask_row, dtype=torch.bool)
    answer_positions = torch.nonzero(answer_mask_row, as_tuple=False).flatten()
    if answer_positions.numel() == 0 or not phrases:
        return row_mask, []

    answer_token_ids = input_ids_row[answer_positions].detach().cpu().tolist()
    matched_phrases: list[str] = []
    for phrase in phrases:
        matched_here = False
        for variant in (phrase, f" {phrase}"):
            phrase_token_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            for start_idx, end_idx in _match_phrase_token_spans(answer_token_ids, phrase_token_ids):
                row_mask[answer_positions[start_idx:end_idx]] = True
                matched_here = True
        if matched_here:
            matched_phrases.append(phrase)
    return row_mask, matched_phrases


def _build_keyword_answer_mask(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    min_char_len: int,
    items: list[dict] | None = None,
    key_annotation_entries: Dict[str, dict] | None = None,
) -> torch.Tensor:
    answer_mask = labels != -100
    if items and key_annotation_entries:
        annotation_mask = torch.zeros_like(answer_mask, dtype=torch.bool)
        for row_idx, item in enumerate(items):
            phrases = _extract_item_key_phrases(item, key_annotation_entries)
            if not phrases:
                continue
            row_mask, _ = _build_annotation_row_mask(
                tokenizer=tokenizer,
                input_ids_row=input_ids[row_idx],
                answer_mask_row=answer_mask[row_idx],
                phrases=phrases,
            )
            if row_mask.any():
                annotation_mask[row_idx] = row_mask
        if annotation_mask.any():
            unresolved_rows = annotation_mask.sum(dim=1) == 0
            if not unresolved_rows.any():
                return annotation_mask
            answer_mask = answer_mask.clone()
            answer_mask[~unresolved_rows] = annotation_mask[~unresolved_rows]
    key_mask = torch.zeros_like(answer_mask, dtype=torch.bool)
    input_rows = input_ids.detach().cpu().tolist()
    answer_rows = answer_mask.detach().cpu().tolist()
    for row_idx, (token_ids, answer_flags) in enumerate(zip(input_rows, answer_rows)):
        row_has_key = False
        for col_idx, (token_id, is_answer_token) in enumerate(zip(token_ids, answer_flags)):
            if not is_answer_token:
                continue
            token_text = _clean_token_text(tokenizer.convert_ids_to_tokens(int(token_id)))
            is_key = (
                len(token_text) >= min_char_len
                and any(ch.isalnum() for ch in token_text)
            )
            if is_key:
                key_mask[row_idx, col_idx] = True
                row_has_key = True
        if not row_has_key:
            key_mask[row_idx] = answer_mask[row_idx]
    return key_mask


def _build_raw_qa_mask(*, attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.to(dtype=torch.bool)


def _build_localization_token_mask(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    cfg,
    items: list[dict] | None = None,
    key_annotation_entries: Dict[str, dict] | None = None,
) -> torch.Tensor:
    if cfg.localization_target_token == "keyword_answer":
        return _build_keyword_answer_mask(
            input_ids=input_ids,
            labels=labels,
            tokenizer=tokenizer,
            min_char_len=cfg.localization_key_token_min_char_len,
            items=items,
            key_annotation_entries=key_annotation_entries,
        )
    if cfg.localization_target_token == "raw_qa":
        return _build_raw_qa_mask(attention_mask=attention_mask)
    raise ValueError(f"Unsupported localization target token mode: {cfg.localization_target_token}")


def _compute_mean_correct_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * valid_mask.to(gathered.dtype)
    denom = valid_mask.to(gathered.dtype).sum(dim=-1).clamp_min(1.0)
    return gathered.sum(dim=-1) / denom


def _decode_token_ids(tokenizer, token_ids: list[int]) -> list[str]:
    decoded = []
    for token_id in token_ids:
        token_text = _clean_token_text(tokenizer.convert_ids_to_tokens(int(token_id)))
        if token_text:
            decoded.append(token_text)
    return decoded


def _collect_selected_token_debug_rows(
    *,
    batch,
    tokenizer,
    mode: str,
    cfg,
    key_annotation_entries: Dict[str, dict] | None = None,
) -> list[Dict[str, object]]:
    debug_rows: list[Dict[str, object]] = []
    input_ids = batch["input_ids"].detach().cpu()
    labels = batch["labels"].detach().cpu()
    attention_mask = batch["attention_mask"].detach().cpu()

    if mode == "keyword_answer":
        token_mask = _build_keyword_answer_mask(
            input_ids=input_ids,
            labels=labels,
            tokenizer=tokenizer,
            min_char_len=cfg.localization_key_token_min_char_len,
            items=batch["items"],
            key_annotation_entries=key_annotation_entries,
        )
        token_mask = token_mask.detach().cpu()
        for row_idx, item in enumerate(batch["items"]):
            selected_ids = input_ids[row_idx][token_mask[row_idx]].tolist()
            annotation_phrases = _extract_item_key_phrases(item, key_annotation_entries or {})
            debug_rows.append(
                {
                    "example_id": str(item.get("example_id", row_idx)),
                    "selection_source": "external_annotation" if annotation_phrases else "heuristic_key_token_filter",
                    "annotation_phrases": annotation_phrases,
                    "selected_tokens": _decode_token_ids(tokenizer, selected_ids),
                }
            )
        return debug_rows

    if mode == "raw_qa":
        token_mask = _build_raw_qa_mask(attention_mask=attention_mask)
        for row_idx, item in enumerate(batch["items"]):
            selected_ids = input_ids[row_idx][token_mask[row_idx]].tolist()
            debug_rows.append(
                {
                    "example_id": str(item.get("example_id", row_idx)),
                    "selection_source": "raw_qa_full_span",
                    "selected_tokens": _decode_token_ids(tokenizer, selected_ids),
                }
            )
        return debug_rows

    raise ValueError(f"Unsupported debug token mode: {mode}")


def _collect_average_abs_gradients(
    *,
    model,
    dataset: Dataset,
    tokenizer,
    cfg,
    collate_fn,
    target_params: Dict[str, torch.nn.Parameter],
    dataset_label: str,
) -> Dict[str, torch.Tensor]:
    loader = DataLoader(
        list(dataset),
        batch_size=cfg.localization_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, cfg.max_length),
    )
    device = next(model.parameters()).device
    accumulators = {
        name: torch.zeros_like(param, dtype=torch.float32, device="cpu")
        for name, param in target_params.items()
    }

    model.eval()
    model.zero_grad(set_to_none=True)
    num_steps = 0
    progress = tqdm(loader, desc=f"HCTL grads:{dataset_label}", leave=False)
    for batch in progress:
        batch = _move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )
        loss = outputs.loss
        loss.backward()
        with torch.no_grad():
            for name, param in target_params.items():
                if param.grad is not None:
                    accumulators[name] += param.grad.detach().abs().float().cpu()
        del outputs
        del loss
        model.zero_grad(set_to_none=True)
        num_steps += 1
        progress.set_postfix(steps=num_steps)

    if num_steps == 0:
        raise ValueError("HCTL dataloader produced zero batches.")
    return {name: grad / float(num_steps) for name, grad in accumulators.items()}


def _collect_layerwise_activation_statistics(
    *,
    model,
    dataset: Dataset,
    tokenizer,
    cfg,
    collate_fn,
    candidate_layers: list[int],
    dataset_label: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    loader = DataLoader(
        list(dataset),
        batch_size=cfg.localization_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, cfg.max_length),
    )
    device = next(model.parameters()).device
    allowed_layers = set(candidate_layers)
    key_annotation_entries = _get_key_annotation_entries(cfg)
    mean_signed_sums: Dict[int, torch.Tensor] = {}
    mean_abs_sums: Dict[int, torch.Tensor] = {}
    token_counts: Dict[int, int] = defaultdict(int)
    batch_counts: Dict[int, int] = defaultdict(int)

    model.eval()
    progress = tqdm(loader, desc=f"HCTL acts:{dataset_label}", leave=False)
    with _capture_layer_mlp_inputs(model, allowed_layers) as captured_activations:
        for batch in progress:
            batch = _move_batch_to_device(batch, device)
            captured_activations.clear()
            with torch.no_grad():
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
            labels = batch["labels"]
            token_mask = _build_localization_token_mask(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
                tokenizer=tokenizer,
                cfg=cfg,
                items=batch["items"],
                key_annotation_entries=key_annotation_entries,
            )
            valid_rows = int(token_mask.any(dim=1).sum().item())
            for layer_idx in candidate_layers:
                activation = captured_activations.get(layer_idx)
                if activation is None:
                    continue
                layer_mask = token_mask.to(activation.device, dtype=activation.dtype).unsqueeze(-1)
                masked_signed = activation * layer_mask
                masked_abs = activation.abs() * layer_mask
                signed_sum = masked_signed.sum(dim=(0, 1)).detach().cpu().float()
                token_sum = masked_abs.sum(dim=(0, 1)).detach().cpu().float()
                if layer_idx not in mean_signed_sums:
                    mean_signed_sums[layer_idx] = torch.zeros_like(signed_sum, dtype=torch.float32)
                if layer_idx not in mean_abs_sums:
                    mean_abs_sums[layer_idx] = torch.zeros_like(token_sum, dtype=torch.float32)
                mean_signed_sums[layer_idx] += signed_sum
                mean_abs_sums[layer_idx] += token_sum
                token_counts[layer_idx] += int(token_mask.sum().item())
                batch_counts[layer_idx] += valid_rows
            progress.set_postfix(batches=sum(batch_counts.values()))

    layer_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_idx in candidate_layers:
        token_count = max(1, token_counts.get(layer_idx, 0))
        mean_abs = mean_abs_sums.get(layer_idx)
        mean_signed = mean_signed_sums.get(layer_idx)
        if mean_abs is None or mean_signed is None:
            continue
        layer_stats[f"layer_{layer_idx}"] = {
            "mean_signed": mean_signed / float(token_count),
            "mean_abs": mean_abs / float(token_count),
            "token_count": torch.tensor(float(token_count), dtype=torch.float32),
            "batch_count": torch.tensor(float(batch_counts.get(layer_idx, 0)), dtype=torch.float32),
        }
    if not layer_stats:
        raise ValueError(f"No activation statistics were captured for dataset={dataset_label}.")
    return layer_stats


def _parse_layer_key(layer_key: str) -> int | None:
    if isinstance(layer_key, str) and layer_key.startswith("layer_"):
        suffix = layer_key.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _resolve_author_bucket_layers(cfg, target_params: Dict[str, torch.nn.Parameter]) -> list[int]:
    available_layers = sorted(
        {
            layer_idx
            for name in target_params.keys()
            for layer_idx in [extract_layer_index(name)]
            if layer_idx is not None
        }
    )
    if not available_layers:
        raise ValueError("No decoder layers were found from the target LoRA parameters.")

    layer_end = available_layers[-1] if int(cfg.localization_layer_end) < 0 else int(cfg.localization_layer_end)
    selected = [
        layer_idx
        for layer_idx in available_layers
        if cfg.localization_layer_start <= layer_idx <= layer_end
    ]
    if cfg.manual_layer_whitelist:
        whitelist = set(cfg.manual_layer_whitelist)
        selected = [layer_idx for layer_idx in selected if layer_idx in whitelist]
    if not selected:
        raise ValueError(
            f"No target layers found inside requested localization range "
            f"[{cfg.localization_layer_start}, {layer_end}] from available_layers={available_layers}."
        )
    return selected


def _collect_localization_span_activation_statistics(
    *,
    model,
    dataset: Dataset,
    tokenizer,
    cfg,
    collate_fn,
    candidate_layers: list[int],
    dataset_label: str,
) -> tuple[Dict[int, Dict[str, torch.Tensor]], list[str], list[Dict[str, object]]]:
    loader = DataLoader(
        list(dataset),
        batch_size=cfg.localization_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, cfg.max_length),
    )
    device = next(model.parameters()).device
    allowed_layers = set(candidate_layers)
    key_annotation_entries = _get_key_annotation_entries(cfg)
    activations_by_layer: Dict[int, Dict[str, list[torch.Tensor]]] = {
        layer_idx: {"signed": [], "abs": [], "pos": [], "neg": [], "fired": []}
        for layer_idx in candidate_layers
    }
    example_ids: list[str] = []
    token_debug_rows: list[Dict[str, object]] = []

    model.eval()
    progress = tqdm(loader, desc=f"HCTL localization-span:{dataset_label}", leave=False)
    with _capture_layer_mlp_inputs(model, allowed_layers) as captured_activations:
        for batch in progress:
            batch = _move_batch_to_device(batch, device)
            captured_activations.clear()
            with torch.no_grad():
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
            token_mask = _build_localization_token_mask(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                tokenizer=tokenizer,
                cfg=cfg,
                items=batch["items"],
                key_annotation_entries=key_annotation_entries,
            ).to(dtype=torch.float32, device=device)
            token_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            expanded_mask = token_mask.unsqueeze(-1)
            for item in batch["items"]:
                example_ids.append(str(item.get("example_id", len(example_ids))))
            token_debug_rows.extend(
                _collect_selected_token_debug_rows(
                    batch=batch,
                    tokenizer=tokenizer,
                    mode=cfg.localization_target_token,
                    cfg=cfg,
                    key_annotation_entries=key_annotation_entries,
                )
            )
            for layer_idx in candidate_layers:
                activation = captured_activations.get(layer_idx)
                if activation is None:
                    continue
                mean_signed = (activation * expanded_mask.to(activation.dtype)).sum(dim=1) / token_count.to(activation.dtype)
                mean_abs = (activation.abs() * expanded_mask.to(activation.dtype)).sum(dim=1) / token_count.to(activation.dtype)
                mean_pos = (torch.relu(activation) * expanded_mask.to(activation.dtype)).sum(dim=1) / token_count.to(activation.dtype)
                mean_neg = (torch.relu(-activation) * expanded_mask.to(activation.dtype)).sum(dim=1) / token_count.to(activation.dtype)
                token_fired_ratio = (
                    ((activation.abs() > float(cfg.activation_fired_threshold)).to(activation.dtype) * expanded_mask.to(activation.dtype)).sum(dim=1)
                    / token_count.to(activation.dtype)
                )
                activations_by_layer[layer_idx]["signed"].append(mean_signed.detach().cpu().float())
                activations_by_layer[layer_idx]["abs"].append(mean_abs.detach().cpu().float())
                activations_by_layer[layer_idx]["pos"].append(mean_pos.detach().cpu().float())
                activations_by_layer[layer_idx]["neg"].append(mean_neg.detach().cpu().float())
                activations_by_layer[layer_idx]["fired"].append(token_fired_ratio.detach().cpu().float())
            progress.set_postfix(samples=len(example_ids))

    stacked = {}
    for layer_idx, parts in activations_by_layer.items():
        if not parts["signed"]:
            continue
        stacked[layer_idx] = {
            stat_name: torch.cat(stat_parts, dim=0)
            for stat_name, stat_parts in parts.items()
        }
    if not stacked:
        raise ValueError(f"No localization-span activations captured for dataset={dataset_label}.")
    return stacked, example_ids, token_debug_rows


def _collect_target_token_activation_statistics(
    *,
    model,
    dataset: Dataset,
    tokenizer,
    cfg,
    collate_fn,
    candidate_layers: list[int],
    dataset_label: str,
) -> tuple[Dict[int, Dict[str, torch.Tensor]], list[str], list[Dict[str, object]]]:
    return _collect_localization_span_activation_statistics(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        collate_fn=collate_fn,
        candidate_layers=candidate_layers,
        dataset_label=dataset_label,
    )


def _collect_target_token_activations(
    *,
    model,
    dataset: Dataset,
    tokenizer,
    cfg,
    collate_fn,
    candidate_layers: list[int],
    dataset_label: str,
) -> tuple[Dict[int, Dict[str, torch.Tensor]], list[str], list[Dict[str, object]]]:
    return _collect_target_token_activation_statistics(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        collate_fn=collate_fn,
        candidate_layers=candidate_layers,
        dataset_label=dataset_label,
    )


def _build_author_bucket_localization(
    *,
    cfg,
    model,
    tokenizer,
    forget_dataset: Dataset,
    retain_dataset: Dataset,
    collate_fn,
) -> LocalizationArtifacts:
    logger.info(
        "Running author-bucket abs-difference localization on layers [%s, %s] with token=%s fired_threshold=%.4f",
        cfg.localization_layer_start,
        "auto_last" if int(cfg.localization_layer_end) < 0 else cfg.localization_layer_end,
        cfg.localization_target_token,
        cfg.activation_fired_threshold,
    )
    target_params = _collect_target_parameters(
        model,
        allowed_module_keywords=("up_proj", "down_proj"),
    )
    candidate_layers = _resolve_author_bucket_layers(cfg, target_params)
    forget_activations, forget_example_ids, forget_token_debug_rows = _collect_target_token_activations(
        model=model,
        dataset=forget_dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        collate_fn=collate_fn,
        candidate_layers=candidate_layers,
        dataset_label="forget",
    )
    retain_activations, retain_example_ids, retain_token_debug_rows = _collect_target_token_activations(
        model=model,
        dataset=retain_dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        collate_fn=collate_fn,
        candidate_layers=candidate_layers,
        dataset_label="retain",
    )

    epsilon = float(cfg.activation_fired_threshold)
    retain_ratios_by_layer: Dict[int, torch.Tensor] = {}
    retain_mean_pos_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx, layer_stats in retain_activations.items():
        retain_ratios_by_layer[layer_idx] = layer_stats["fired"].mean(dim=0)
        retain_mean_pos_by_layer[layer_idx] = layer_stats["pos"].mean(dim=0)

    bucket_size = int(cfg.forget_author_bucket_size)
    total_forget = len(forget_example_ids)
    bucket_ranges = [
        (bucket_start, min(total_forget, bucket_start + bucket_size))
        for bucket_start in range(0, total_forget, bucket_size)
    ]
    selected_summary: Dict[tuple[int, int], Dict[str, float]] = {}
    bucket_summaries = []

    for bucket_idx, (start_idx, end_idx) in enumerate(bucket_ranges):
        bucket_candidates = []
        bucket_len = max(1, end_idx - start_idx)
        for layer_idx in candidate_layers:
            layer_stats = forget_activations.get(layer_idx)
            if layer_stats is None:
                continue
            bucket_pos = layer_stats["pos"][start_idx:end_idx]
            forget_ratio = layer_stats["fired"][start_idx:end_idx].mean(dim=0)
            forget_mean_pos = bucket_pos.mean(dim=0)
            retain_ratio = retain_ratios_by_layer[layer_idx]
            retain_mean_pos = retain_mean_pos_by_layer[layer_idx]

                                         
                                                                          
                                                                             
                                                     
                                                                
                                    
             
                                                     
                                                              
                                                              
                                                                  
                                                                       
             
                                                                        
                                                                                   
                                                                    
               
                                                                                 
            pos_contrast = forget_mean_pos - retain_mean_pos
            score = pos_contrast
            ranked_indices = torch.argsort(score, descending=True)
            for neuron_idx in ranked_indices.tolist():
                neuron_score = float(score[neuron_idx].item())
                if neuron_score <= 0.0:
                    continue
                bucket_candidates.append(
                    {
                        "layer_idx": int(layer_idx),
                        "row_idx": int(neuron_idx),
                        "bucket_score": neuron_score,
                        "bucket_mean_pos": float(forget_mean_pos[neuron_idx].item()),
                        "bucket_pos_contrast": float(pos_contrast[neuron_idx].item()),
                        "forget_firing_ratio": float(forget_ratio[neuron_idx].item()),
                        "retain_firing_ratio": float(retain_ratio[neuron_idx].item()),
                    }
                )
                                                                               
                                                                       
                                                                              
                                 
        bucket_candidates.sort(
            key=lambda item: (item["bucket_score"], item["bucket_mean_pos"]),
            reverse=True,
        )
        bucket_selected = bucket_candidates[: int(cfg.author_bucket_top_neurons)]
        for item in bucket_selected:
            key = (item["layer_idx"], item["row_idx"])
            if key not in selected_summary:
                selected_summary[key] = {
                    "selection_count": 0.0,
                    "score_weighted_sum": 0.0,
                    "score_max": 0.0,
                    "forget_ratio_sum": 0.0,
                    "retain_ratio_sum": 0.0,
                    "pos_sum": 0.0,
                }
            selected_summary[key]["selection_count"] += 1.0
            selected_summary[key]["score_weighted_sum"] += float(item["bucket_score"])
            selected_summary[key]["score_max"] = max(
                float(selected_summary[key]["score_max"]),
                float(item["bucket_score"]),
            )
            selected_summary[key]["forget_ratio_sum"] += float(item["forget_firing_ratio"])
            selected_summary[key]["retain_ratio_sum"] += float(item["retain_firing_ratio"])
            selected_summary[key]["pos_sum"] += float(item["bucket_mean_pos"])
        bucket_summaries.append(
            {
                "bucket_index": int(bucket_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "num_examples": int(bucket_len),
                "num_candidates_before_topk": int(len(bucket_candidates)),
                "num_selected_after_topk": int(len(bucket_selected)),
                "best_score": float(bucket_selected[0]["bucket_score"]) if bucket_selected else 0.0,
                "example_ids_preview": forget_example_ids[start_idx:min(end_idx, start_idx + 3)],
            }
        )

    ranked_global = []
    for (layer_idx, row_idx), stats in selected_summary.items():
        count = max(1.0, float(stats["selection_count"]))
        ranked_global.append(
            {
                "layer_idx": int(layer_idx),
                "row_idx": int(row_idx),
                                                                             
                                                                             
                                                                            
                                                                      
                 
                                                     
                                                                                                                        
                "aggregate_score": float(stats["score_max"]),
                "selection_count": float(stats["selection_count"]),
                "mean_bucket_score": float(stats["score_weighted_sum"] / count),
                "mean_forget_ratio": float(stats["forget_ratio_sum"] / count),
                "mean_retain_ratio": float(stats["retain_ratio_sum"] / count),
                "mean_bucket_pos": float(stats["pos_sum"] / count),
                "mean_pos_contrast": float(stats["score_weighted_sum"] / count),
            }
        )
    ranked_global.sort(
        key=lambda item: (item["aggregate_score"], item["mean_bucket_score"], item["mean_bucket_pos"]),
        reverse=True,
    )
                                                                                
                                                                               
                                                                             
                             
     
                                               
                                                                         

    target_neurons: Dict[str, list[int]] = defaultdict(list)
    layer_score_values: Dict[str, list[float]] = defaultdict(list)
    final_selection_stats: Dict[str, Dict[str, float]] = {}
    for item in ranked_global:
        layer_key = f"layer_{item['layer_idx']}"
        target_neurons[layer_key].append(int(item["row_idx"]))
        layer_score_values[layer_key].append(float(item["aggregate_score"]))

    layer_scores: Dict[str, list[float]] = {}
    for layer_key, score_values in layer_score_values.items():
        score_array = np.array(score_values, dtype=np.float32)
        layer_scores[layer_key] = [float(v) for v in normalize_scores(score_array).tolist()]
        layer_idx = _parse_layer_key(layer_key)
        retain_ratio_tensor = retain_ratios_by_layer.get(layer_idx) if layer_idx is not None else None
        final_selection_stats[layer_key] = {
            "selected_rows": float(len(target_neurons[layer_key])),
            "mean_selected_score": float(score_array.mean()) if score_array.size else 0.0,
            "max_selected_score": float(score_array.max()) if score_array.size else 0.0,
            "retain_firing_ratio_mean": float(retain_ratio_tensor.mean().item()) if retain_ratio_tensor is not None else 0.0,
            "use_relaxed_filter": False,
        }

    parameter_masks: Dict[str, np.ndarray] = {}
    for name, param in target_params.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None or layer_idx not in candidate_layers:
            continue
        layer_key = f"layer_{layer_idx}"
        selected_rows = target_neurons.get(layer_key, [])
        if not selected_rows:
            continue
        mask = np.zeros(tuple(param.shape), dtype=np.float32)
        valid_rows = [row_idx for row_idx in selected_rows if 0 <= row_idx < mask.shape[0]]
        if valid_rows:
            mask[valid_rows, :] = 1.0
        parameter_masks[name] = mask

    total_target_neurons = sum(len(rows) for rows in target_neurons.values())
    total_selected_params = int(sum(mask.sum() for mask in parameter_masks.values()))
    total_candidate_params = int(sum(mask.size for mask in parameter_masks.values()))
    parameter_mask_density = (float(total_selected_params) / float(total_candidate_params)) if total_candidate_params else 0.0
    logger.info(
        "Author-bucket localization selected %d neurons across %d layers (params=%d density=%.6f)",
        total_target_neurons,
        len(target_neurons),
        total_selected_params,
        parameter_mask_density,
    )
    return LocalizationArtifacts(
        strategy="author_firing_intersection",
        top_fraction=cfg.top_fraction,
        target_neurons=dict(target_neurons),
        layer_scores=layer_scores,
        total_target_neurons=total_target_neurons,
        parameter_masks=parameter_masks,
        metadata={
            "method_name": "author_bucket_positive_activation_localization",
            "localization_strategy": cfg.localization_strategy,
            "localization_layer_start": cfg.localization_layer_start,
            "localization_layer_end": cfg.localization_layer_end,
            "localization_target_token": cfg.localization_target_token,
            "localization_key_annotation_path": str(cfg.localization_key_annotation_path) if getattr(cfg, "localization_key_annotation_path", None) else "",
            "activation_fired_threshold": cfg.activation_fired_threshold,
            "forget_author_bucket_size": cfg.forget_author_bucket_size,
            "forget_firing_min_ratio": cfg.forget_firing_min_ratio,
            "retain_firing_max_ratio": cfg.retain_firing_max_ratio,
            "author_bucket_top_neurons": cfg.author_bucket_top_neurons,
            "author_abs_weight": cfg.author_abs_weight,
            "author_posneg_weight": cfg.author_posneg_weight,
            "score_formula": "mean_pos_forget - mean_pos_retain",
            "score_formula_lambda": None,
            "global_ranking_formula": "deduplicated_union_sorted_by_max_bucket_score",
            "candidate_layers": [f"layer_{idx}" for idx in candidate_layers],
            "coarse_selected_layers": [f"layer_{idx}" for idx in candidate_layers],
            "forget_num_examples": int(len(forget_example_ids)),
            "retain_num_examples": int(len(retain_example_ids)),
            "contrastive_lambda": "unused_in_author_bucket_path",
            "num_author_buckets": int(len(bucket_ranges)),
            "bucket_summaries": bucket_summaries,
            "forget_selected_token_preview": forget_token_debug_rows[: min(80, len(forget_token_debug_rows))],
            "retain_selected_token_preview": retain_token_debug_rows[: min(80, len(retain_token_debug_rows))],
            "ranked_global_preview": ranked_global[: min(50, len(ranked_global))],
            "total_selected_params": total_selected_params,
            "total_candidate_params": total_candidate_params,
            "parameter_mask_density": parameter_mask_density,
            "row_level_stats": final_selection_stats,
            "final_selection_stats": final_selection_stats,
        },
    )


def _score_layers_from_forget_gradients(forget_grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    layer_scores: Dict[str, float] = defaultdict(float)
    for name, grad in forget_grads.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        layer_key = f"layer_{layer_idx}"
        layer_scores[layer_key] += float(torch.norm(grad, p="fro").item())
    if not layer_scores:
        raise ValueError("HCTL coarse screening produced no layer scores.")
    return dict(layer_scores)


def _select_candidate_layers(layer_scores: Dict[str, float], top_n_layers: int) -> list[str]:
    ranked = sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
    return [layer_key for layer_key, _ in ranked[:top_n_layers]]


def _apply_manual_layer_whitelist(
    *,
    layer_scores: Dict[str, float],
    manual_layer_whitelist: list[int] | None,
) -> list[str] | None:
    if not manual_layer_whitelist:
        return None
    available_layers = set(layer_scores.keys())
    selected = [f"layer_{layer_idx}" for layer_idx in manual_layer_whitelist if f"layer_{layer_idx}" in available_layers]
    if not selected:
        raise ValueError(
            f"manual_layer_whitelist={manual_layer_whitelist} did not match any available layers: {sorted(available_layers)}"
        )
    return selected


def _list_all_candidate_layers(layer_scores: Dict[str, float]) -> list[str]:
    return [layer_key for layer_key, _ in sorted(layer_scores.items(), key=lambda item: extract_layer_index(item[0]) or -1)]


def _build_parameter_masks(
    *,
    forget_grads: Dict[str, torch.Tensor],
    retain_grads: Dict[str, torch.Tensor],
    candidate_layers: Iterable[str],
    top_fraction: float,
    lambda_factor: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor], float]:
    candidate_set = set(candidate_layers)
    parameter_scores: Dict[str, torch.Tensor] = {}
    all_positive_scores = []

    for name, grad_f in forget_grads.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        layer_key = f"layer_{layer_idx}"
        if layer_key not in candidate_set:
            continue
        grad_r = retain_grads[name]
        score = torch.relu(grad_f - (lambda_factor * grad_r))
        parameter_scores[name] = score
        positive = score[score > 0]
        if positive.numel() > 0:
            all_positive_scores.append(positive.reshape(-1))

    if not all_positive_scores:
        raise ValueError("All contrastive parameter scores are zero; please lower lambda or increase sample diversity.")

    merged_scores = torch.cat(all_positive_scores, dim=0)
    threshold = float(torch.quantile(merged_scores, 1.0 - top_fraction).item())
    parameter_masks = {
        name: (score >= threshold).detach().cpu().numpy().astype(np.float32)
        for name, score in parameter_scores.items()
    }
    return parameter_masks, parameter_scores, threshold


def _derive_row_level_targets(
    *,
    candidate_layers: Iterable[str],
    parameter_masks: Dict[str, np.ndarray],
    parameter_scores: Dict[str, torch.Tensor],
) -> tuple[Dict[str, list[int]], Dict[str, list[float]], Dict[str, Dict[str, float]]]:
    candidate_set = set(candidate_layers)
    row_scores_by_layer: Dict[str, Dict[int, list[float]]] = {layer_key: defaultdict(list) for layer_key in candidate_set}
    layer_stats: Dict[str, Dict[str, float]] = {}

    for name, mask_np in parameter_masks.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        layer_key = f"layer_{layer_idx}"
        score = parameter_scores[name].detach().cpu().numpy()
        row_mask = mask_np.sum(axis=1) > 0
        row_mass = (score * mask_np).sum(axis=1)
        for row_idx, is_active in enumerate(row_mask.tolist()):
            if is_active:
                row_scores_by_layer[layer_key][row_idx].append(float(row_mass[row_idx]))

    target_neurons: Dict[str, list[int]] = {}
    layer_scores: Dict[str, list[float]] = {}
    for layer_key in candidate_layers:
        row_map = row_scores_by_layer.get(layer_key, {})
        if not row_map:
            continue
        sorted_rows = sorted(row_map.items(), key=lambda item: np.mean(item[1]), reverse=True)
        row_indices = [int(idx) for idx, _ in sorted_rows]
        row_values = np.array([float(np.mean(values)) for _, values in sorted_rows], dtype=np.float32)
        target_neurons[layer_key] = row_indices
        layer_scores[layer_key] = [float(v) for v in normalize_scores(row_values).tolist()]
        layer_stats[layer_key] = {
            "selected_rows": float(len(row_indices)),
            "row_score_mean": float(row_values.mean()) if row_values.size else 0.0,
            "row_score_max": float(row_values.max()) if row_values.size else 0.0,
        }
    return target_neurons, layer_scores, layer_stats


def _build_per_layer_row_targets(
    *,
    forget_grads: Dict[str, torch.Tensor],
    retain_grads: Dict[str, torch.Tensor],
    candidate_layers: Iterable[str],
    per_layer_top_neurons: int,
    lambda_factor: float,
) -> tuple[Dict[str, list[int]], Dict[str, list[float]], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    candidate_set = set(candidate_layers)
    row_scores_by_layer: Dict[str, Dict[int, list[float]]] = {layer_key: defaultdict(list) for layer_key in candidate_set}
    row_forget_mass_by_layer: Dict[str, Dict[int, list[float]]] = {layer_key: defaultdict(list) for layer_key in candidate_set}
    row_retain_mass_by_layer: Dict[str, Dict[int, list[float]]] = {layer_key: defaultdict(list) for layer_key in candidate_set}

    for name, grad_f in forget_grads.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        layer_key = f"layer_{layer_idx}"
        if layer_key not in candidate_set:
            continue
        grad_r = retain_grads[name]
        row_forget = grad_f.float().mean(dim=1)
        row_retain = grad_r.float().mean(dim=1)
        row_score = torch.relu(row_forget - (lambda_factor * row_retain))
        for row_idx in range(row_score.shape[0]):
            row_scores_by_layer[layer_key][row_idx].append(float(row_score[row_idx].item()))
            row_forget_mass_by_layer[layer_key][row_idx].append(float(row_forget[row_idx].item()))
            row_retain_mass_by_layer[layer_key][row_idx].append(float(row_retain[row_idx].item()))

    target_neurons: Dict[str, list[int]] = {}
    layer_scores: Dict[str, list[float]] = {}
    row_level_stats: Dict[str, Dict[str, float]] = {}

    for layer_key in candidate_layers:
        row_map = row_scores_by_layer.get(layer_key, {})
        if not row_map:
            continue
        sorted_rows = sorted(
            row_map.items(),
            key=lambda item: (float(np.mean(item[1])), float(np.max(item[1]))),
            reverse=True,
        )
        selected_rows = sorted_rows[:per_layer_top_neurons]
        row_indices = [int(idx) for idx, _ in selected_rows]
        row_values = np.array([float(np.mean(values)) for _, values in selected_rows], dtype=np.float32)
        target_neurons[layer_key] = row_indices
        layer_scores[layer_key] = [float(v) for v in normalize_scores(row_values).tolist()]

        all_row_values = np.array([float(np.mean(values)) for _, values in sorted_rows], dtype=np.float32)
        layer_forget_means = [float(np.mean(row_forget_mass_by_layer[layer_key][idx])) for idx in row_indices]
        layer_retain_means = [float(np.mean(row_retain_mass_by_layer[layer_key][idx])) for idx in row_indices]
        row_level_stats[layer_key] = {
            "selected_rows": float(len(row_indices)),
            "candidate_rows": float(len(sorted_rows)),
            "selected_score_mean": float(row_values.mean()) if row_values.size else 0.0,
            "selected_score_max": float(row_values.max()) if row_values.size else 0.0,
            "all_score_mean": float(all_row_values.mean()) if all_row_values.size else 0.0,
            "selected_forget_mean": float(np.mean(layer_forget_means)) if layer_forget_means else 0.0,
            "selected_retain_mean": float(np.mean(layer_retain_means)) if layer_retain_means else 0.0,
        }

    parameter_masks: Dict[str, np.ndarray] = {}
    for name, grad_f in forget_grads.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        layer_key = f"layer_{layer_idx}"
        selected_rows = target_neurons.get(layer_key)
        if not selected_rows:
            continue
        mask = np.zeros(tuple(grad_f.shape), dtype=np.float32)
        valid_rows = [row_idx for row_idx in selected_rows if 0 <= row_idx < mask.shape[0]]
        if valid_rows:
            mask[valid_rows, :] = 1.0
        parameter_masks[name] = mask

    return target_neurons, layer_scores, parameter_masks, row_level_stats


def _build_activation_based_targets(
    *,
    forget_activation_stats: Dict[str, Dict[str, torch.Tensor]],
    retain_activation_stats: Dict[str, Dict[str, torch.Tensor]],
    target_params: Dict[str, torch.nn.Parameter],
    candidate_layers: Iterable[str],
    per_layer_top_neurons: int,
    lambda_factor: float,
    signed_weight: float,
    abs_weight: float,
) -> tuple[Dict[str, list[int]], Dict[str, list[float]], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    target_neurons: Dict[str, list[int]] = {}
    layer_scores: Dict[str, list[float]] = {}
    parameter_masks: Dict[str, np.ndarray] = {}
    row_level_stats: Dict[str, Dict[str, float]] = {}

    for layer_key in candidate_layers:
        forget_layer = forget_activation_stats.get(layer_key)
        retain_layer = retain_activation_stats.get(layer_key)
        if forget_layer is None or retain_layer is None:
            continue

        forget_signed = forget_layer["mean_signed"].float()
        retain_signed = retain_layer["mean_signed"].float()
        forget_abs = forget_layer["mean_abs"].float()
        retain_abs = retain_layer["mean_abs"].float()
        signed_contrast = forget_signed - (lambda_factor * retain_signed)
        abs_contrast = torch.relu(forget_abs - (lambda_factor * retain_abs))
        score = (signed_weight * signed_contrast) + (abs_weight * abs_contrast)
        ranked_rows = torch.argsort(score, descending=True)
        selected_rows = ranked_rows[: min(per_layer_top_neurons, ranked_rows.numel())]
        row_indices = [int(idx.item()) for idx in selected_rows]
        row_values = score[selected_rows].detach().cpu().numpy().astype(np.float32)

        target_neurons[layer_key] = row_indices
        layer_scores[layer_key] = [float(v) for v in normalize_scores(row_values).tolist()]
        row_level_stats[layer_key] = {
            "selected_rows": float(len(row_indices)),
            "candidate_rows": float(score.numel()),
            "selected_score_mean": float(row_values.mean()) if row_values.size else 0.0,
            "selected_score_max": float(row_values.max()) if row_values.size else 0.0,
            "all_score_mean": float(score.mean().item()) if score.numel() else 0.0,
            "selected_forget_mean_abs": float(forget_abs[selected_rows].mean().item()) if row_indices else 0.0,
            "selected_retain_mean_abs": float(retain_abs[selected_rows].mean().item()) if row_indices else 0.0,
            "selected_forget_mean_signed": float(forget_signed[selected_rows].mean().item()) if row_indices else 0.0,
            "selected_retain_mean_signed": float(retain_signed[selected_rows].mean().item()) if row_indices else 0.0,
        }

    for name, param in target_params.items():
        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue
        layer_key = f"layer_{layer_idx}"
        selected_rows = target_neurons.get(layer_key)
        if not selected_rows:
            continue
        mask = np.zeros(tuple(param.shape), dtype=np.float32)
        valid_rows = [row_idx for row_idx in selected_rows if 0 <= row_idx < mask.shape[0]]
        if valid_rows:
            mask[valid_rows, :] = 1.0
        parameter_masks[name] = mask

    return target_neurons, layer_scores, parameter_masks, row_level_stats


def hierarchical_contrastive_targeted_localization(
    *,
    cfg,
    model,
    tokenizer,
    forget_dataset: Dataset,
    retain_dataset: Dataset,
    collate_fn,
) -> LocalizationArtifacts:
    if getattr(cfg, "localization_strategy", "activation_contrast") == "author_firing_intersection":
        return _build_author_bucket_localization(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
            collate_fn=collate_fn,
        )

    logger.info("Running HCTL localization (all-layer activation contrast + layer-local neuron ranking)...")
    allowed_layers = set(cfg.manual_layer_whitelist) if cfg.manual_layer_whitelist else None
    target_params = _collect_target_parameters(
        model,
        allowed_layers=allowed_layers,
        allowed_module_keywords=("up_proj", "down_proj"),
    )
    forget_subset = _sample_dataset(forget_dataset, cfg.localization_forget_samples)
    retain_subset = _sample_dataset(retain_dataset, cfg.localization_retain_samples)

    all_layer_indices = sorted(
        {
            layer_idx
            for name in target_params.keys()
            for layer_idx in [extract_layer_index(name)]
            if layer_idx is not None
        }
    )
    if not all_layer_indices:
        raise ValueError("No layer indices were derived from target LoRA-B parameters.")

    forget_activation_stats = _collect_layerwise_activation_statistics(
        model=model,
        dataset=forget_subset,
        tokenizer=tokenizer,
        cfg=cfg,
        collate_fn=collate_fn,
        candidate_layers=all_layer_indices,
        dataset_label="forget",
    )
    retain_activation_stats = _collect_layerwise_activation_statistics(
        model=model,
        dataset=retain_subset,
        tokenizer=tokenizer,
        cfg=cfg,
        collate_fn=collate_fn,
        candidate_layers=all_layer_indices,
        dataset_label="retain",
    )

    coarse_layer_scores = {}
    for layer_key in forget_activation_stats.keys():
        if layer_key not in retain_activation_stats:
            continue
        forget_signed = forget_activation_stats[layer_key]["mean_signed"].float()
        retain_signed = retain_activation_stats[layer_key]["mean_signed"].float()
        forget_abs = forget_activation_stats[layer_key]["mean_abs"].float()
        retain_abs = retain_activation_stats[layer_key]["mean_abs"].float()
        signed_contrast = forget_signed - (cfg.contrastive_lambda * retain_signed)
        abs_contrast = torch.relu(forget_abs - (cfg.contrastive_lambda * retain_abs))
        layer_score = (
            (cfg.activation_signed_weight * signed_contrast)
            + (cfg.activation_abs_weight * abs_contrast)
        ).mean().item()
        coarse_layer_scores[layer_key] = float(layer_score)

    if cfg.manual_layer_whitelist:
        candidate_layers = [f"layer_{layer_idx}" for layer_idx in cfg.manual_layer_whitelist]
        logger.info("Activation localization manual layer whitelist applied: %s", candidate_layers)
    else:
        candidate_layers = _list_all_candidate_layers(coarse_layer_scores)
        logger.info(
            "Activation localization enabled across %d layers with per_layer_top_neurons=%d",
            len(candidate_layers),
            cfg.per_layer_top_neurons,
        )

    target_neurons, layer_scores, parameter_masks, row_level_stats = _build_activation_based_targets(
        forget_activation_stats=forget_activation_stats,
        retain_activation_stats=retain_activation_stats,
        target_params=target_params,
        candidate_layers=candidate_layers,
        per_layer_top_neurons=cfg.per_layer_top_neurons,
        lambda_factor=cfg.contrastive_lambda,
        signed_weight=cfg.activation_signed_weight,
        abs_weight=cfg.activation_abs_weight,
    )
                                                                         
                                                                           
                                                                             
                                              
                                                                 
                                                                              
                                                                                
                                                                               
     
                                                    
                      
                                
                              
                  
                                
                                      
                                 
       
                                                    
                      
                                
                              
                  
                                
                                      
                                 
       
                                                                                      
    legacy_gradient_layer_scores = {}

    total_target_neurons = sum(len(indices) for indices in target_neurons.values())
    total_selected_params = int(sum(mask.sum() for mask in parameter_masks.values()))
    total_candidate_params = int(sum(mask.size for mask in parameter_masks.values()))
    parameter_mask_density = (float(total_selected_params) / float(total_candidate_params)) if total_candidate_params else 0.0
    logger.info(
        "HCTL activation ranking selected %d / %d candidate params (density=%.6f) without perturbation verification or final pruning",
        total_selected_params,
        total_candidate_params,
        parameter_mask_density,
    )

    return LocalizationArtifacts(
        strategy="activation_contrast",
        top_fraction=cfg.top_fraction,
        target_neurons=target_neurons,
        layer_scores=layer_scores,
        total_target_neurons=total_target_neurons,
        parameter_masks=parameter_masks,
        metadata={
            "method_name": "activation_contrastive_targeted_localization",
            "coarse_top_layers": cfg.coarse_top_layers,
            "per_layer_top_neurons": cfg.per_layer_top_neurons,
            "localization_forget_samples": len(forget_subset),
            "localization_retain_samples": len(retain_subset),
            "contrastive_lambda": cfg.contrastive_lambda,
            "localization_target_token": cfg.localization_target_token,
            "localization_key_annotation_path": str(cfg.localization_key_annotation_path) if getattr(cfg, "localization_key_annotation_path", None) else "",
            "localization_key_token_min_char_len": cfg.localization_key_token_min_char_len,
            "activation_signed_weight": cfg.activation_signed_weight,
            "activation_abs_weight": cfg.activation_abs_weight,
            "localization_verify_enabled": cfg.localization_verify_enabled,
            "localization_verify_forget_samples": cfg.localization_verify_forget_samples,
            "localization_verify_retain_samples": cfg.localization_verify_retain_samples,
            "localization_verify_retain_penalty": cfg.localization_verify_retain_penalty,
            "verification_top_neurons_per_layer": cfg.verification_top_neurons_per_layer,
            "final_top_neurons_per_layer": cfg.final_top_neurons_per_layer,
            "verification_score_std_multiplier": cfg.verification_score_std_multiplier,
            "verification_min_score": cfg.verification_min_score,
            "relaxed_layer_max_index": cfg.relaxed_layer_max_index,
            "relaxed_final_top_neurons_per_layer": cfg.relaxed_final_top_neurons_per_layer,
            "relaxed_verification_score_std_multiplier": cfg.relaxed_verification_score_std_multiplier,
            "relaxed_verification_min_score": cfg.relaxed_verification_min_score,
            "global_max_target_neurons": cfg.global_max_target_neurons,
            "score_threshold": None,
            "coarse_layer_scores": coarse_layer_scores,
            "coarse_selected_layers": candidate_layers,
            "legacy_gradient_layer_scores": legacy_gradient_layer_scores,
            "total_selected_params": total_selected_params,
            "total_candidate_params": total_candidate_params,
            "parameter_mask_density": parameter_mask_density,
            "row_level_stats": row_level_stats,
            "verification_stats": {},
            "final_selection_stats": row_level_stats,
        },
    )
