                      
                                          

                                                                              
                                                                                
                                   
   

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import random
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

_PIPELINE_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _PIPELINE_DIR.parent
_AGENT_DIR = _PHASE2_DIR.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from phase2.pipeline.config import Stage2Config
from phase2.pipeline.stage2_unlearning import register_gradient_mask_hooks, remove_gradient_mask_hooks, save_json
from phase2.pipeline.support import (
    build_model,
    build_tokenizer,
    build_labeled_batch,
    get_model_device,
    load_json_or_jsonl,
    release_memory,
    set_global_seed,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("FlowMU.DPO")


SAFE_FORGET_RESPONSE = (
    "I do not have reliable information about that specific detail."
)
SAFE_RETAIN_REJECTED_RESPONSE = (
    "I do not have enough information to answer that."
)


@dataclass
class DPOConfig:
    base_model_path: str
    output_dir: str
    localization_pkl: Path
    forget_qa_path: Path
    retain_qa_path: Path
    init_new_lora: bool = True
    new_lora_r: int = 16
    new_lora_alpha: int = 32
    new_lora_dropout: float = 0.05
    new_lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    beta: float = 0.2
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    max_train_steps: int = 0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    warmup_steps: int = 5
    max_grad_norm: float = 0.1
    logging_steps: int = 10
    seed: int = 42
    retain_sample_size: int = 1500
    forget_sample_size: int = 0
    retain_loss_weight: float = 1.0
    forget_loss_weight: float = 1.0
    forget_answer_suppression_weight: float = 0.0
    forget_key_token_suppression_weight: float = 0.0
    disable_parameter_mask: bool = False
    reference_margin_cache_dir: str = ""
    overwrite_reference_margin_cache: bool = False
    save_steps: int = 0
    resume_from_checkpoint: Path | None = None


def _load_localization(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _normalize_qa_rows(path: Path, *, split: str, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
        answer = str(row.get("answer") or row.get("output") or row.get("reference_answer") or "").strip()
        if not question or not answer:
            continue
        normalized.append(
            {
                "instruction": question,
                "output": answer,
                "source": str(row.get("source") or split),
                "source_id": str(row.get("source_id") or idx),
                "example_id": f"{split}_{idx}",
            }
        )
    if limit and len(normalized) > limit:
        rng = random.Random(seed)
        rng.shuffle(normalized)
        normalized = normalized[:limit]
    return normalized


def _make_forget_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": row["instruction"],
        "chosen": SAFE_FORGET_RESPONSE,
        "rejected": row["output"],
        "split": "forget",
        "example_id": row["example_id"],
    }


def _make_retain_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": row["instruction"],
        "chosen": row["output"],
        "rejected": SAFE_RETAIN_REJECTED_RESPONSE,
        "split": "retain",
        "example_id": row["example_id"],
    }


def _pair_output_item(pair: dict[str, Any], output_key: str) -> dict[str, Any]:
    return {
        "instruction": pair["instruction"],
        "output": pair[output_key],
        "example_id": pair["example_id"],
        "split": pair["split"],
    }


def _build_dpo_pairs(cfg: DPOConfig) -> list[dict[str, Any]]:
    forget_rows = _normalize_qa_rows(
        cfg.forget_qa_path,
        split="forget",
        limit=cfg.forget_sample_size,
        seed=cfg.seed,
    )
    retain_rows = _normalize_qa_rows(
        cfg.retain_qa_path,
        split="retain",
        limit=cfg.retain_sample_size,
        seed=cfg.seed + 17,
    )
    pairs = [_make_forget_pair(row) for row in forget_rows]
    pairs.extend(_make_retain_pair(row) for row in retain_rows)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    logger.info(
        "Built DPO pairs: forget=%d retain=%d total=%d",
        len(forget_rows),
        len(retain_rows),
        len(pairs),
    )
    return pairs


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _checkpoint_step_from_path(path: Path) -> int:
    for pattern in (r"checkpoint-step(\d+)", r"checkpoint-(\d+)", r"step(\d+)"):
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))
    return 0


def _load_training_state(path: Path) -> dict[str, Any] | None:
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return None
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def _save_training_state(
    *,
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    global_step: int,
    epoch_idx: int,
) -> None:
    torch.save(
        {
            "global_step": int(global_step),
            "epoch_idx": int(epoch_idx),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )


def _reference_cache_identity(cfg: DPOConfig, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cache_version": 1,
        "method": "masked_dpo_reference_margin",
        "base_model_path": str(Path(cfg.base_model_path).resolve()),
        "forget_qa": _file_fingerprint(cfg.forget_qa_path),
        "retain_qa": _file_fingerprint(cfg.retain_qa_path),
        "forget_sample_size": int(cfg.forget_sample_size),
        "retain_sample_size": int(cfg.retain_sample_size),
        "seed": int(cfg.seed),
        "max_length": int(cfg.max_length),
        "safe_forget_response": SAFE_FORGET_RESPONSE,
        "safe_retain_rejected_response": SAFE_RETAIN_REJECTED_RESPONSE,
        "num_pairs": int(len(pairs)),
        "example_ids_digest": hashlib.sha256(
            "\n".join(str(item["example_id"]) for item in pairs).encode("utf-8")
        ).hexdigest(),
    }


def _reference_cache_paths(cfg: DPOConfig, pairs: list[dict[str, Any]]) -> tuple[Path, Path, dict[str, Any]] | None:
    if not str(cfg.reference_margin_cache_dir or "").strip():
        return None
    identity = _reference_cache_identity(cfg, pairs)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
    cache_dir = Path(cfg.reference_margin_cache_dir)
    return (
        cache_dir / f"ref_margins_{digest}.json",
        cache_dir / f"ref_margins_{digest}.meta.json",
        identity,
    )


def _load_reference_margins_from_cache(cfg: DPOConfig, pairs: list[dict[str, Any]]) -> dict[str, float] | None:
    cache = _reference_cache_paths(cfg, pairs)
    if cache is None:
        return None
    margins_path, meta_path, identity = cache
    if cfg.overwrite_reference_margin_cache or not margins_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("identity") != identity:
            logger.warning("Reference margin cache identity mismatch, recomputing: %s", margins_path)
            return None
        payload = json.loads(margins_path.read_text(encoding="utf-8"))
        margins = {str(k): float(v) for k, v in payload.get("margins", {}).items()}
    except Exception as exc:
        logger.warning("Could not load reference margin cache %s: %s", margins_path, exc)
        return None
    if len(margins) != len(pairs):
        logger.warning("Reference margin cache is incomplete (%d/%d), recomputing.", len(margins), len(pairs))
        return None
    missing = [item["example_id"] for item in pairs if str(item["example_id"]) not in margins]
    if missing:
        logger.warning("Reference margin cache misses %d ids, recomputing.", len(missing))
        return None
    logger.info("Loaded reference margins from cache: %s", margins_path)
    return margins


def _save_reference_margins_to_cache(cfg: DPOConfig, pairs: list[dict[str, Any]], margins: dict[str, float]) -> None:
    cache = _reference_cache_paths(cfg, pairs)
    if cache is None:
        return
    margins_path, meta_path, identity = cache
    margins_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_margins = margins_path.with_suffix(margins_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_margins.write_text(
        json.dumps({"margins": margins}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_meta.write_text(
        json.dumps({"identity": identity, "margins_path": str(margins_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_margins.replace(margins_path)
    tmp_meta.replace(meta_path)
    logger.info("Saved reference margins cache: %s", margins_path)


def _precompute_reference_margins(cfg: DPOConfig, tokenizer, pairs: list[dict[str, Any]]) -> dict[str, float]:
                                                                                  

                                                                             
                                                                             
                                       
       
    stage_cfg = _stage2_cfg_from_dpo(cfg)
    ref = build_model(stage_cfg, is_trainable=False)
    device = get_model_device(ref)
    ref.eval()
    loader = DataLoader(
        pairs,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_dpo_pairs(batch, tokenizer, cfg.max_length),
    )
    margins: dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="precompute ref margins"):
            chosen = _move_labeled_batch(batch["chosen"], device)
            rejected = _move_labeled_batch(batch["rejected"], device)
            ref_chosen = _sequence_logps(ref, chosen)
            ref_rejected = _sequence_logps(ref, rejected)
            ref_margin = (ref_chosen - ref_rejected).detach().cpu().float().tolist()
            for item, margin in zip(batch["items"], ref_margin):
                margins[str(item["example_id"])] = float(margin)
    del ref
    release_memory()
    if len(margins) != len(pairs):
        raise RuntimeError(f"Reference margin cache is incomplete: {len(margins)} / {len(pairs)}")
    return margins


def _collate_dpo_pairs(batch: list[dict[str, Any]], tokenizer, max_length: int) -> dict[str, Any]:
    chosen_items = [_pair_output_item(item, "chosen") for item in batch]
    rejected_items = [_pair_output_item(item, "rejected") for item in batch]
    chosen = build_labeled_batch(chosen_items, tokenizer, max_length, output_key="output")
    rejected = build_labeled_batch(rejected_items, tokenizer, max_length, output_key="output")
    return {
        "chosen": chosen,
        "rejected": rejected,
        "items": batch,
    }


def _move_labeled_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _sequence_logps(model, batch: dict[str, Any]) -> torch.Tensor:
    seq_logps, _, _ = _sequence_logps_with_tokens(model, batch)
    return seq_logps


def _sequence_logps_with_tokens(model, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * valid.to(token_logps.dtype)
    return token_logps.sum(dim=-1), token_logps, labels


def _label_lengths(batch: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    labels = batch["labels"][:, 1:]
    return (labels != -100).sum(dim=-1).clamp_min(1).to(dtype)


def _split_weights(items: list[dict[str, Any]], cfg: DPOConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = [
        float(cfg.forget_loss_weight if item.get("split") == "forget" else cfg.retain_loss_weight)
        for item in items
    ]
    return torch.tensor(weights, device=device, dtype=dtype)


def _dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_margin: torch.Tensor, weights: torch.Tensor, beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    policy_margin = policy_chosen - policy_rejected
    logits = float(beta) * (policy_margin - ref_margin)
    row_loss = -F.logsigmoid(logits)
    loss = (row_loss * weights).sum() / weights.sum().clamp_min(1.0)
    with torch.no_grad():
        rewards = policy_margin - ref_margin
        return loss, {
            "loss": float(loss.detach().cpu()),
            "policy_margin": float(policy_margin.mean().detach().cpu()),
            "ref_margin": float(ref_margin.mean().detach().cpu()),
            "reward_margin": float(rewards.mean().detach().cpu()),
            "preference_acc": float((rewards > 0).float().mean().detach().cpu()),
        }


def _forget_answer_suppression_loss(
    policy_rejected: torch.Tensor,
    rejected: dict[str, Any],
    items: list[dict[str, Any]],
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = policy_rejected.sum() * 0.0
    if weight <= 0:
        return zero, {
            "forget_answer_suppression_loss": 0.0,
            "forget_answer_suppression_count": 0,
            "forget_answer_mean_logp_per_token": 0.0,
        }

    mask = torch.tensor(
        [item.get("split") == "forget" for item in items],
        device=policy_rejected.device,
        dtype=torch.bool,
    )
    count = int(mask.sum().detach().cpu())
    if count == 0:
        return zero, {
            "forget_answer_suppression_loss": 0.0,
            "forget_answer_suppression_count": 0,
            "forget_answer_mean_logp_per_token": 0.0,
        }

    lengths = _label_lengths(rejected, policy_rejected.dtype).to(policy_rejected.device)
    mean_logp = policy_rejected[mask] / lengths[mask]
    suppression_loss = mean_logp.mean()
    return suppression_loss * float(weight), {
        "forget_answer_suppression_loss": float(suppression_loss.detach().cpu()),
        "forget_answer_suppression_count": count,
        "forget_answer_mean_logp_per_token": float(mean_logp.mean().detach().cpu()),
    }


_KEY_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "author",
    "authors",
    "be",
    "been",
    "book",
    "books",
    "born",
    "by",
    "can",
    "due",
    "for",
    "from",
    "full",
    "genre",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "main",
    "major",
    "name",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
    "work",
    "works",
}


def _token_piece_is_key_fact(tokenizer, token_id: int) -> bool:
    piece = tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
    if not piece:
        return False
    normalized = re.sub(r"[^0-9A-Za-z]+", "", piece).lower()
    if not normalized:
        return False
    if normalized in _KEY_TOKEN_STOPWORDS:
        return False
    if any(ch.isdigit() for ch in normalized):
        return True
    if len(normalized) >= 3:
        return True
    return False


def _forget_key_token_suppression_loss(
    token_logps: torch.Tensor,
    shifted_labels: torch.Tensor,
    items: list[dict[str, Any]],
    tokenizer,
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = token_logps.sum() * 0.0
    if weight <= 0:
        return zero, {
            "forget_key_token_suppression_loss": 0.0,
            "forget_key_token_count": 0,
            "forget_key_token_mean_logp": 0.0,
        }

    row_losses: list[torch.Tensor] = []
    token_count = 0
    for row_idx, item in enumerate(items):
        if item.get("split") != "forget":
            continue
        labels = shifted_labels[row_idx]
        valid_positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
        key_positions = [
            int(pos)
            for pos in valid_positions.detach().cpu().tolist()
            if _token_piece_is_key_fact(tokenizer, int(labels[pos].detach().cpu()))
        ]
        if not key_positions:
            continue
        pos_tensor = torch.tensor(key_positions, device=token_logps.device, dtype=torch.long)
        row_losses.append(token_logps[row_idx].index_select(0, pos_tensor).mean())
        token_count += len(key_positions)

    if not row_losses:
        return zero, {
            "forget_key_token_suppression_loss": 0.0,
            "forget_key_token_count": 0,
            "forget_key_token_mean_logp": 0.0,
        }

    mean_logp = torch.stack(row_losses).mean()
    return mean_logp * float(weight), {
        "forget_key_token_suppression_loss": float(mean_logp.detach().cpu()),
        "forget_key_token_count": int(token_count),
        "forget_key_token_mean_logp": float(mean_logp.detach().cpu()),
    }


def _freeze_non_masked_trainable_parameters(model, allowed_param_names: set[str]) -> dict[str, int]:
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


def _stage2_cfg_from_dpo(cfg: DPOConfig) -> Stage2Config:
    return Stage2Config(
        base_model_path=cfg.base_model_path,
        original_lora_path="",
        output_dir=cfg.output_dir,
        graph_path=Path("."),
        forget_target_path=cfg.forget_qa_path,
        retain_path=cfg.retain_qa_path,
        core_dataset_path=cfg.forget_qa_path,
        context_dataset_path=cfg.forget_qa_path,
        init_new_lora=bool(cfg.init_new_lora),
        init_new_lora_on_frozen=False,
        new_lora_r=int(cfg.new_lora_r),
        new_lora_alpha=int(cfg.new_lora_alpha),
        new_lora_dropout=float(cfg.new_lora_dropout),
        new_lora_target_modules=cfg.new_lora_target_modules,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        max_length=cfg.max_length,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
    )


def run_dpo_training(cfg: DPOConfig) -> None:
    set_global_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg.base_model_path)
    localization = _load_localization(cfg.localization_pkl)
    pairs = _build_dpo_pairs(cfg)
    if not pairs:
        raise ValueError("No DPO pairs were built.")

    ref_margins = _load_reference_margins_from_cache(cfg, pairs)
    if ref_margins is None:
        ref_margins = _precompute_reference_margins(cfg, tokenizer, pairs)
        _save_reference_margins_to_cache(cfg, pairs, ref_margins)

    train_loader = DataLoader(
        pairs,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_dpo_pairs(batch, tokenizer, cfg.max_length),
    )

    stage_cfg = _stage2_cfg_from_dpo(cfg)
    policy_cfg = stage_cfg
    resume_training_state = None
    resume_start_step = 0
    if cfg.resume_from_checkpoint is not None:
        if not cfg.resume_from_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {cfg.resume_from_checkpoint}")
        policy_cfg = replace(
            stage_cfg,
            init_new_lora=False,
            original_lora_path=str(cfg.resume_from_checkpoint),
        )
        resume_training_state = _load_training_state(cfg.resume_from_checkpoint)
        if resume_training_state is None:
            resume_start_step = _checkpoint_step_from_path(cfg.resume_from_checkpoint)
            logger.info(
                "Resuming DPO from LoRA weights only: %s inferred_step=%d",
                cfg.resume_from_checkpoint,
                resume_start_step,
            )
        else:
            resume_start_step = int(resume_training_state.get("global_step", 0))
            logger.info(
                "Resuming DPO from full checkpoint: %s global_step=%d",
                cfg.resume_from_checkpoint,
                resume_start_step,
            )
    policy = build_model(policy_cfg, is_trainable=True)
    device = get_model_device(policy)

    if cfg.disable_parameter_mask:
        freeze_summary = {
            "trainable_before": int(sum(1 for _, param in policy.named_parameters() if param.requires_grad)),
            "trainable_after": int(sum(1 for _, param in policy.named_parameters() if param.requires_grad)),
            "frozen_now": 0,
        }
        mask_handles = []
    else:
        freeze_summary = _freeze_non_masked_trainable_parameters(
            policy,
            allowed_param_names=set(localization.parameter_masks.keys()),
        )
        mask_handles = register_gradient_mask_hooks(policy, localization.parameter_masks)

    optimizer = AdamW([param for param in policy.parameters() if param.requires_grad], lr=cfg.learning_rate)
    scheduled_steps = max(1, math.ceil(len(train_loader) * cfg.num_train_epochs))
    total_steps = scheduled_steps
    if cfg.max_train_steps > 0:
        total_steps = min(total_steps, int(cfg.max_train_steps))
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=scheduled_steps,
    )
    if resume_training_state is not None:
        if "optimizer" in resume_training_state:
            optimizer.load_state_dict(resume_training_state["optimizer"])
        if "scheduler" in resume_training_state:
            scheduler.load_state_dict(resume_training_state["scheduler"])
    metrics_path = output_dir / "train_metrics.jsonl"
    metadata_path = output_dir / "dpo_metadata.json"

    def save_checkpoint(step: int, epoch_idx: int) -> None:
        checkpoint_dir = output_dir / f"checkpoint-step{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        _save_training_state(
            checkpoint_dir=checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=step,
            epoch_idx=epoch_idx,
        )
        save_json(
            checkpoint_dir / "dpo_checkpoint_metadata.json",
            {
                "method": "masked_dpo_checkpoint",
                "parent_output_dir": str(output_dir),
                "step": int(step),
                "total_steps": int(total_steps),
                "scheduled_steps": int(scheduled_steps),
                "config": asdict(cfg),
                "localization_pkl": str(cfg.localization_pkl),
                "localization_metadata": getattr(localization, "metadata", {}),
                "freeze_summary": freeze_summary,
            },
        )
        logger.info("Saved DPO checkpoint step=%d -> %s", step, checkpoint_dir)

    logger.info(
        "DPO training start: total_steps=%d scheduled_steps=%d start_step=%d batch=%d grad_accum=%d beta=%.4f lr=%.2e save_steps=%d mask_disabled=%s freeze=%s",
        total_steps,
        scheduled_steps,
        resume_start_step,
        cfg.per_device_train_batch_size,
        cfg.gradient_accumulation_steps,
        cfg.beta,
        cfg.learning_rate,
        cfg.save_steps,
        cfg.disable_parameter_mask,
        freeze_summary,
    )

    policy.train()
    global_step = int(resume_start_step)
    optimizer.zero_grad(set_to_none=True)
    try:
        metrics_mode = "a" if global_step > 0 and metrics_path.exists() else "w"
        with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_f:
            for epoch_idx in range(int(math.ceil(cfg.num_train_epochs))):
                for batch in tqdm(train_loader, desc=f"masked_dpo epoch {epoch_idx}"):
                    if global_step >= total_steps:
                        break
                    chosen = _move_labeled_batch(batch["chosen"], device)
                    rejected = _move_labeled_batch(batch["rejected"], device)
                    if not (chosen["labels"] != -100).any() or not (rejected["labels"] != -100).any():
                        logger.warning("Skipping empty-label DPO batch at step=%d", global_step)
                        global_step += 1
                        continue

                    policy_chosen = _sequence_logps(policy, chosen)
                    policy_rejected, rejected_token_logps, rejected_shifted_labels = _sequence_logps_with_tokens(policy, rejected)
                    ref_margin = torch.tensor(
                        [ref_margins[str(item["example_id"])] for item in batch["items"]],
                        device=device,
                        dtype=policy_chosen.dtype,
                    )
                    weights = _split_weights(batch["items"], cfg, device, policy_chosen.dtype)
                    dpo_loss, stats = _dpo_loss(
                        policy_chosen,
                        policy_rejected,
                        ref_margin,
                        weights,
                        cfg.beta,
                    )
                    suppression_loss, suppression_stats = _forget_answer_suppression_loss(
                        policy_rejected,
                        rejected,
                        batch["items"],
                        cfg.forget_answer_suppression_weight,
                    )
                    key_suppression_loss, key_suppression_stats = _forget_key_token_suppression_loss(
                        rejected_token_logps,
                        rejected_shifted_labels,
                        batch["items"],
                        tokenizer,
                        cfg.forget_key_token_suppression_weight,
                    )
                    loss = dpo_loss + suppression_loss + key_suppression_loss
                    (loss / cfg.gradient_accumulation_steps).backward()
                    if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    row = {
                        "step": global_step,
                        "epoch": epoch_idx,
                        **stats,
                        **suppression_stats,
                        **key_suppression_stats,
                        "dpo_loss": stats["loss"],
                        "loss": float(loss.detach().cpu()),
                        "forget_batch": int(sum(1 for item in batch["items"] if item.get("split") == "forget")),
                        "retain_batch": int(sum(1 for item in batch["items"] if item.get("split") == "retain")),
                    }
                    metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    metrics_f.flush()
                    if global_step % cfg.logging_steps == 0:
                        logger.info(
                            "masked_dpo step=%d/%d loss=%.4f reward_margin=%.4f pref_acc=%.4f policy_margin=%.4f ref_margin=%.4f",
                            global_step,
                            total_steps,
                            row["loss"],
                            row["reward_margin"],
                            row["preference_acc"],
                            row["policy_margin"],
                            row["ref_margin"],
                        )
                    global_step += 1
                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        save_checkpoint(global_step, epoch_idx)
                if global_step >= total_steps:
                    break

        if global_step % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    finally:
        remove_gradient_mask_hooks(mask_handles)

    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _save_training_state(
        checkpoint_dir=output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch_idx=int(math.ceil(cfg.num_train_epochs)) - 1,
    )
    save_json(
        metadata_path,
        {
            "method": "masked_dpo",
            "config": asdict(cfg),
            "localization_pkl": str(cfg.localization_pkl),
            "localization_metadata": getattr(localization, "metadata", {}),
            "freeze_summary": freeze_summary,
            "global_steps": int(global_step),
            "scheduled_steps": int(scheduled_steps),
            "max_train_steps": int(cfg.max_train_steps),
            "resume_from_checkpoint": str(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else "",
            "resume_start_step": int(resume_start_step),
            "metrics_path": str(metrics_path),
        },
    )
    del policy
    release_memory()
    logger.info("DPO LoRA saved to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked DPO-style FlowMU Phase-2 training")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--localization-pkl", required=True)
    parser.add_argument("--forget-qa-path", required=True)
    parser.add_argument("--retain-qa-path", required=True)
    parser.add_argument("--new-lora-r", type=int, default=16)
    parser.add_argument("--new-lora-alpha", type=int, default=32)
    parser.add_argument("--new-lora-dropout", type=float, default=0.05)
    parser.add_argument("--new-lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optional hard cap on global steps. Useful for continuing from step N to a target step.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help=(
            "Resume DPO from a saved LoRA checkpoint. If checkpoint/training_state.pt exists, "
            "optimizer/scheduler/global_step are restored too."
        ),
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retain-sample-size", type=int, default=1500)
    parser.add_argument("--forget-sample-size", type=int, default=0)
    parser.add_argument("--retain-loss-weight", type=float, default=1.0)
    parser.add_argument("--forget-loss-weight", type=float, default=1.0)
    parser.add_argument("--forget-answer-suppression-weight", type=float, default=0.0)
    parser.add_argument("--forget-key-token-suppression-weight", type=float, default=0.0)
    parser.add_argument("--disable-parameter-mask", action="store_true")
    parser.add_argument("--reference-margin-cache-dir", default="")
    parser.add_argument("--overwrite-reference-margin-cache", action="store_true")
    parser.add_argument("--save-steps", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = DPOConfig(
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        localization_pkl=Path(args.localization_pkl),
        forget_qa_path=Path(args.forget_qa_path),
        retain_qa_path=Path(args.retain_qa_path),
        new_lora_r=args.new_lora_r,
        new_lora_alpha=args.new_lora_alpha,
        new_lora_dropout=args.new_lora_dropout,
        new_lora_target_modules=args.new_lora_target_modules,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        seed=args.seed,
        retain_sample_size=args.retain_sample_size,
        forget_sample_size=args.forget_sample_size,
        retain_loss_weight=args.retain_loss_weight,
        forget_loss_weight=args.forget_loss_weight,
        forget_answer_suppression_weight=args.forget_answer_suppression_weight,
        forget_key_token_suppression_weight=args.forget_key_token_suppression_weight,
        disable_parameter_mask=args.disable_parameter_mask,
        reference_margin_cache_dir=args.reference_margin_cache_dir,
        overwrite_reference_margin_cache=args.overwrite_reference_margin_cache,
        save_steps=args.save_steps,
        resume_from_checkpoint=Path(args.resume_from_checkpoint) if str(args.resume_from_checkpoint).strip() else None,
    )
    run_dpo_training(cfg)


if __name__ == "__main__":
    main()
