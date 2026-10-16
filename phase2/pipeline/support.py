                         

                                                
                                            
   

from __future__ import annotations

import gc
import logging
import os
import random
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .bootstrap import BASE_MODEL_PATH
from .contrastive_localization import hierarchical_contrastive_targeted_localization
from .config import Stage2Config
from .forget_alignment import (
    align_forget_targets,
    build_forget_datasets_from_aligned_targets,
)

logger = logging.getLogger("GAT-AFRU.Phase2.Support")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        logger.warning("Could not fully enable deterministic algorithms: %s", exc)

    logger.info("Global reproducibility seed set to %d.", seed)


def _prepare_localization_model_for_low_memory(model, cfg: Stage2Config) -> None:
    if hasattr(model, "config") and getattr(model.config, "use_cache", None):
        model.config.use_cache = False

    if cfg.localization_gradient_checkpointing:
        enabled = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                enabled = True
            except Exception as exc:
                logger.warning("Could not enable gradient checkpointing on PEFT model: %s", exc)
        base_model = getattr(model, "base_model", None)
        if not enabled and base_model is not None and hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable()
                enabled = True
            except Exception as exc:
                logger.warning("Could not enable gradient checkpointing on base model: %s", exc)
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        elif base_model is not None and hasattr(base_model, "enable_input_require_grads"):
            try:
                base_model.enable_input_require_grads()
            except Exception:
                pass
        if enabled:
            logger.info("Enabled localization gradient checkpointing for lower activation memory.")


def _align_lm_head_dtype_for_quantized_lora(model) -> None:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        base_model = getattr(model, "base_model", None)
        inner_model = getattr(base_model, "model", None) if base_model is not None else None
        lm_head = getattr(inner_model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight") or lm_head.weight is None:
        return

    target_dtype = None
    for _, param in model.named_parameters():
        if param is not None and param.is_floating_point():
            target_dtype = param.dtype
            break
    if target_dtype is None or lm_head.weight.dtype == target_dtype:
        return

    lm_head.to(dtype=target_dtype)
    logger.info("Aligned lm_head dtype to %s for quantized LoRA forward stability.", target_dtype)


def build_user_assistant_messages(instruction: str, output: Optional[str] = None) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": instruction}]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages


def apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
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


def build_supervised_rendered_example(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str,
) -> tuple[str, int]:
    prompt_only = apply_chat_template(
        tokenizer,
        build_user_assistant_messages(instruction, None),
        add_generation_prompt=True,
    )
    full_text = apply_chat_template(
        tokenizer,
        build_user_assistant_messages(instruction, output),
        add_generation_prompt=False,
    )
    prompt_token_count = len(tokenizer(prompt_only, add_special_tokens=False, truncation=True)["input_ids"])
    return full_text, prompt_token_count


def build_labeled_batch(
    items: list[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    *,
    output_key: str = "output",
) -> Dict[str, Any]:
    rendered_pairs = [
        build_supervised_rendered_example(tokenizer, str(item["instruction"]), str(item.get(output_key) or ""))
        for item in items
    ]
    encoded = tokenizer(
        [pair[0] for pair in rendered_pairs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    for row_idx, (_, prompt_token_count) in enumerate(rendered_pairs):
        labels[row_idx, : min(prompt_token_count, labels.shape[1])] = -100
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "items": items,
    }


def load_forget_dataset(cfg: Stage2Config) -> Dataset:
    if cfg.forget_target_path is not None:
        logger.info("Loading pure Phase-2 forget dataset directly from official forget targets: %s", cfg.forget_target_path)

        rows = load_json_or_jsonl(cfg.forget_target_path)
        normalized_rows = []
        for idx, row in enumerate(rows):
            question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
            answer = str(row.get("answer") or row.get("output") or row.get("reference_answer") or "").strip()
            if not question or not answer:
                raise ValueError(f"Forget row #{idx} is missing question/answer fields.")
            normalized_rows.append(
                {
                    "instruction": question,
                    "output": answer,
                    "original_output": answer,
                    "source": "forget_target_direct",
                    "example_id": f"forget_{idx}",
                }
            )
        dataset = Dataset.from_list(normalized_rows)
                                                                        
                                                                              
                                                                            
                                                                           
         
                                                                       
        if getattr(cfg, "localization_strategy", "activation_contrast") != "author_firing_intersection":
            dataset = dataset.shuffle(seed=42)
        logger.info("Loaded pure Phase-2 forget dataset with %d rows.", len(dataset))
        return dataset
    else:
        logger.info("Loading Step-4 forgetting datasets...")
        dataset_core = load_dataset("json", data_files=str(cfg.core_dataset_path), split="train")
        dataset_context = load_dataset("json", data_files=str(cfg.context_dataset_path), split="train")
    combined = concatenate_datasets([dataset_core, dataset_context]).shuffle(seed=42)
    combined_rows = []
    for idx, row in enumerate(combined):
        normalized = dict(row)
        normalized["example_id"] = f"forget_{idx}"
        combined_rows.append(normalized)
    dataset = Dataset.from_list(combined_rows)
    logger.info("Loaded merged forgetting dataset with %d rows.", len(dataset))
    return dataset


def normalize_retain_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = row.get("question") or row.get("instruction") or row.get("prompt") or row.get("input") or ""
    answer = row.get("answer") or row.get("output") or row.get("reference_answer") or row.get("response") or ""
    question = str(question).strip()
    answer = str(answer).strip()
    if not question or not answer:
        raise ValueError(f"Retain row #{index} is missing question/answer fields.")
    return {
        "instruction": question,
        "output": answer,
        "source": "retain_pool",
        "retain_id": row.get("id", index),
        "example_id": f"retain_{index}",
    }


def load_retain_dataset(cfg: Stage2Config) -> Dataset:
    rows = load_json_or_jsonl(cfg.retain_path)
    normalized = [normalize_retain_row(row, idx) for idx, row in enumerate(rows)]
    dataset = Dataset.from_list(normalized)
    if getattr(cfg, "localization_strategy", "activation_contrast") != "author_firing_intersection":
        dataset = dataset.shuffle(seed=42)
    logger.info("Loaded retain dataset with %d rows from %s.", len(dataset), cfg.retain_path)
    return dataset


def build_quantization_config(*, cpu_offload: bool = False, prefer_8bit: bool = False) -> BitsAndBytesConfig:
    if cpu_offload or prefer_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )


def build_tokenizer(base_model_path: str = BASE_MODEL_PATH) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_probe_model_on_cpu(cfg: Stage2Config):
    cpu_dtype_name = os.environ.get("PHASE2_PROBE_CPU_DTYPE", "float16").strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    cpu_dtype = dtype_map.get(cpu_dtype_name, torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        device_map={"": "cpu"},
        torch_dtype=cpu_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info("Frozen probe model loaded on CPU with dtype=%s.", cpu_dtype)
    return model


def build_model(cfg: Stage2Config, is_trainable: bool = True):
    logger.info("Loading quantized base model...")
    try:
        prefer_gpu_probe = (not is_trainable) and os.environ.get("PHASE2_PROBE_PREFER_GPU", "0").strip() in {"1", "true", "True"}
        if prefer_gpu_probe and torch.cuda.is_available():
            logger.info("Frozen probe model requested with GPU-first loading.")
            gpu_max_memory = os.environ.get("PHASE2_PROBE_GPU_MAX_MEMORY", "").strip()
            from_pretrained_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "quantization_config": build_quantization_config(),
                "device_map": {"": 0},
            }
            if gpu_max_memory:
                from_pretrained_kwargs["max_memory"] = {0: gpu_max_memory}
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_path,
                **from_pretrained_kwargs,
            )
            logger.info("Frozen probe model loaded on GPU.")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_path,
                device_map="auto",
                quantization_config=build_quantization_config(),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
    except ValueError as exc:
        if is_trainable or "dispatched on the CPU or the disk" not in str(exc):
            raise
        logger.warning("Auto device_map for frozen model hit GPU RAM limit; retrying with CPU offload.")
        max_memory = {}
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            usable_gib = max(1, int((free_mem / (1024 ** 3)) - 1))
            max_memory[0] = f"{usable_gib}GiB"
        max_memory["cpu"] = os.environ.get("PHASE2_CPU_MAX_MEMORY", "128GiB")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_path,
                device_map="auto",
                max_memory=max_memory,
                quantization_config=build_quantization_config(cpu_offload=True, prefer_8bit=True),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("Frozen probe model loaded with quantized CPU offload fallback.")
        except Exception as offload_exc:
            if "Int8Params.__new__" not in str(offload_exc) and "Params4bit.__new__" not in str(offload_exc) and "_is_hf_initialized" not in str(offload_exc):
                raise
            logger.warning(
                "Quantized CPU offload path is incompatible in current env; retrying frozen probe model on CPU without bitsandbytes."
            )
            base_model = _load_probe_model_on_cpu(cfg)
    except TypeError as exc:
        if is_trainable or ("_is_hf_initialized" not in str(exc) and "Int8Params.__new__" not in str(exc) and "Params4bit.__new__" not in str(exc)):
            raise
        logger.warning("Quantized path is incompatible in current env; retrying frozen probe model on CPU without bitsandbytes.")
        base_model = _load_probe_model_on_cpu(cfg)
    except Exception as exc:
        if is_trainable:
            raise
        if "Int8Params.__new__" not in str(exc) and "Params4bit.__new__" not in str(exc):
            raise
        logger.warning("Quantized offload fallback failed (%s); retrying frozen probe model on CPU without bitsandbytes.", exc)
        base_model = _load_probe_model_on_cpu(cfg)
    if is_trainable:
        try:
            base_model = prepare_model_for_kbit_training(base_model)
        except torch.OutOfMemoryError:
            logger.warning("prepare_model_for_kbit_training OOM; fallback to direct LoRA loading.")
            release_memory()
    else:
        logger.info("Skipping k-bit training preparation for frozen teacher model.")

    if getattr(cfg, "init_new_lora", False):
        if is_trainable or getattr(cfg, "init_new_lora_on_frozen", False):
            target_modules = [
                item.strip()
                for item in str(cfg.new_lora_target_modules).split(",")
                if item.strip()
            ]
            lora_config = LoraConfig(
                r=int(cfg.new_lora_r),
                lora_alpha=int(cfg.new_lora_alpha),
                lora_dropout=float(cfg.new_lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(base_model, lora_config)
            logger.info(
                "Initialized fresh unlearning LoRA on base model with r=%s alpha=%s targets=%s trainable=%s",
                cfg.new_lora_r,
                cfg.new_lora_alpha,
                target_modules,
                is_trainable,
            )
        else:
            model = base_model
            logger.info("Using frozen base model as teacher because --init-new-lora is enabled.")
    else:
        model = PeftModel.from_pretrained(base_model, cfg.original_lora_path, is_trainable=is_trainable)
    _align_lm_head_dtype_for_quantized_lora(model)
    if is_trainable:
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info("Trainable params: %s / %s (%.4f%%)", trainable_params, all_params, 100 * trainable_params / all_params)
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def run_localization(cfg: Stage2Config):
    logger.info("Running HCTL localization with all-layer neuron ranking and retain-contrastive masking...")
    tokenizer = build_tokenizer(cfg.base_model_path)
    forget_dataset = load_forget_dataset(cfg)
    retain_dataset = load_retain_dataset(cfg)
    trainable_model = build_model(cfg, is_trainable=True)
    _prepare_localization_model_for_low_memory(trainable_model, cfg)
    try:
        localization = hierarchical_contrastive_targeted_localization(
            cfg=cfg,
            model=trainable_model,
            tokenizer=tokenizer,
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
            collate_fn=collate_chat_examples,
        )
    finally:
        del trainable_model
        release_memory()

    logger.info(
        "HCTL localization complete: candidate_layers=%s selected_params=%s density=%.6f threshold=%.6f per_layer_top_neurons=%s",
        localization.metadata.get("coarse_selected_layers", []),
        localization.metadata.get("total_selected_params", 0),
        localization.metadata.get("parameter_mask_density", 0.0),
        localization.metadata.get("score_threshold", 0.0),
        localization.metadata.get("per_layer_top_neurons"),
    )
    return localization


def collate_chat_examples(batch: list[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    return build_labeled_batch(batch, tokenizer, max_length, output_key="output")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def cycle_dataloader(loader):
    while True:
        for batch in loader:
            yield batch


def load_json_or_jsonl(path):
    import json

    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def select_hctl_representation_layers(localization, *, max_layers: int | None = None) -> list[int]:
    layer_keys: Iterable[str] = localization.metadata.get("coarse_selected_layers", []) or localization.target_neurons.keys()
    layer_indices = []
    for key in layer_keys:
        if isinstance(key, str) and key.startswith("layer_"):
            suffix = key.split("_")[-1]
            if suffix.isdigit():
                layer_indices.append(int(suffix))
    layer_indices = sorted(set(layer_indices))
    if max_layers is not None:
        layer_indices = layer_indices[:max_layers]
    return layer_indices


def select_hctl_strongest_layers(localization, *, max_layers: int) -> list[int]:
    row_stats = localization.metadata.get("row_level_stats", {}) if hasattr(localization, "metadata") else {}
    ranked = []
    for layer_key in localization.target_neurons.keys():
        if not (isinstance(layer_key, str) and layer_key.startswith("layer_")):
            continue
        suffix = layer_key.split("_")[-1]
        if not suffix.isdigit():
            continue
        layer_idx = int(suffix)
        stat = row_stats.get(layer_key, {})
        strength = float(stat.get("selected_score_mean", 0.0))
        ranked.append((layer_idx, strength))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [layer_idx for layer_idx, _ in ranked[:max_layers]]


def extract_answer_span_hidden_states(
    hidden_states,
    labels: torch.Tensor,
    layer_indices: list[int],
) -> torch.Tensor:
    if not layer_indices:
        raise ValueError("At least one representation layer must be selected.")

    answer_mask = (labels != -100).unsqueeze(-1)
    pooled_layers = []
    for layer_idx in layer_indices:
        hidden_index = min(layer_idx + 1, len(hidden_states) - 1)
        layer_hidden = hidden_states[hidden_index]
        masked_hidden = layer_hidden * answer_mask.to(layer_hidden.dtype)
        token_count = answer_mask.to(layer_hidden.dtype).sum(dim=1).clamp_min(1.0)
        pooled_layers.append(masked_hidden.sum(dim=1) / token_count)
    return torch.stack(pooled_layers, dim=1)


def pool_answer_span_representation(
    hidden_states,
    labels: torch.Tensor,
    layer_indices: list[int],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    pooled = extract_answer_span_hidden_states(hidden_states, labels, layer_indices).mean(dim=1)
    if normalize:
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled


def stack_reference_representations(
    cache: Dict[str, torch.Tensor],
    keys: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
) -> torch.Tensor:
    stacked = torch.stack([cache[key] for key in keys], dim=0).to(device=device, dtype=dtype)
    if normalize:
        stacked = torch.nn.functional.normalize(stacked, dim=-1)
    return stacked


def stack_reference_layer_representations(
    cache: Dict[str, torch.Tensor],
    keys: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
) -> torch.Tensor:
    stacked = torch.stack([cache[key] for key in keys], dim=0).to(device=device, dtype=dtype)
    if normalize:
        stacked = torch.nn.functional.normalize(stacked, dim=-1)
    return stacked
