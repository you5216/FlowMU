                      
                                                    

                       
                                              
                                       
                                                                       
                                                            
                                                                                   
   

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

_PIPELINE_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _PIPELINE_DIR.parent
_AGENT_DIR = _PHASE2_DIR.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

if __package__ in {None, ""}:
    from phase2.pipeline.bootstrap import BASE_MODEL_PATH, ORIGINAL_LORA_PATH
else:
    from .bootstrap import BASE_MODEL_PATH, ORIGINAL_LORA_PATH


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("GAT-AFRU.Phase2.MergeLoRA")


def _default_output_dir() -> str:
    return str(_PHASE2_DIR / "merged_models" / "original_lora_merged")


def _resolve_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model for direct vLLM serving."
    )
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--lora-path", default=ORIGINAL_LORA_PATH)
    parser.add_argument("--output-dir", default=_default_output_dir())
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Load dtype used during offline merge. Use float16 by default; bf16 can round away small LoRA deltas.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help='Transformers device_map for merge loading. The validated path uses "cpu".',
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--note",
        default="offline merge for vLLM static serving (phase2 pipeline)",
        help="Human-readable note written into merge_metadata.json.",
    )
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--unsafe-serialization",
        action="store_true",
        help="Use PyTorch bin serialization instead of safetensors.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    base_model_path = Path(args.base_model_path)
    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    return base_model_path, lora_path, output_dir


def _maybe_skip_existing(output_dir: Path, force: bool) -> bool:
    config_path = output_dir / "config.json"
    if config_path.exists() and not force:
        logger.info("Merged model already exists: %s", output_dir)
        logger.info("Use --force to rebuild the merged export.")
        return True
    return False


def _build_metadata(args: argparse.Namespace, base_model_path: Path, lora_path: Path) -> dict[str, str]:
    metadata = {
        "base_model": str(base_model_path),
        "lora_path": str(lora_path),
        "note": args.note,
        "merge_torch_dtype": str(args.torch_dtype),
    }
    if args.label:
        metadata["label"] = args.label
    return metadata


def _fix_tokenizer_config_for_vllm(output_dir: Path) -> None:
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        logger.warning("tokenizer_config.json not found in %s", output_dir)
        return

    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
        tokenizer_config["tokenizer_class"] = "Qwen2Tokenizer"
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Fixed tokenizer_config.json tokenizer_class TokenizersBackend -> Qwen2Tokenizer in %s",
            tokenizer_config_path,
        )


def _has_conditional_language_model(base_model) -> bool:
    model = getattr(base_model, "model", None)
    return model is not None and hasattr(model, "language_model")


def _prepare_lora_path_for_base_model(base_model, lora_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None, bool]:
                                                                             

                                                                               
                                                           
                                                                            
                                                                                
                                                                              
       
    adapter_path = lora_path / "adapter_model.safetensors"
    if not adapter_path.exists():
        return lora_path, None, False

    state = load_file(str(adapter_path))
    causal_prefix = "base_model.model.model.layers."
    conditional_prefix = "base_model.model.model.language_model.layers."
    has_causal_keys = any(key.startswith(causal_prefix) for key in state)
    has_conditional_keys = any(key.startswith(conditional_prefix) for key in state)
    if not (_has_conditional_language_model(base_model) and has_causal_keys and not has_conditional_keys):
        return lora_path, None, False

    tmp = tempfile.TemporaryDirectory(prefix="qwen35_lora_remap_")
    prepared = Path(tmp.name)
    shutil.copytree(lora_path, prepared, dirs_exist_ok=True)
    remapped = {
        (key.replace(causal_prefix, conditional_prefix, 1) if key.startswith(causal_prefix) else key): tensor
        for key, tensor in state.items()
    }
    save_file(remapped, str(prepared / "adapter_model.safetensors"), metadata={"format": "pt"})
    logger.info(
        "Remapped LoRA adapter keys for ConditionalGeneration merge: %d tensors, %s -> %s",
        len(remapped),
        causal_prefix,
        conditional_prefix,
    )
    return prepared, tmp, True


def _assert_loaded_lora_is_nonzero(peft_model) -> None:
    total_b_abs = 0.0
    total_b_count = 0
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            total_b_abs += float(param.detach().float().abs().sum().cpu())
            total_b_count += int(param.numel())
    if total_b_count <= 0:
        raise RuntimeError("No lora_B parameters were found after loading the adapter; merge would be a no-op.")
    if total_b_abs <= 0.0:
        raise RuntimeError("Loaded lora_B parameters are all zero; adapter state was likely not loaded correctly.")
    logger.info("Loaded non-zero LoRA adapter: lora_B_abs_sum=%.6f lora_B_params=%d", total_b_abs, total_b_count)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_model_path, lora_path, output_dir = _validate_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _maybe_skip_existing(output_dir, args.force):
        return

    torch_dtype = _resolve_dtype(args.torch_dtype)
    if torch_dtype is torch.bfloat16:
        logger.warning(
            "Merging LoRA in bfloat16 can round away small adapter deltas. "
            "Use --torch-dtype float16 or float32 unless you have verified merged shard hashes differ."
        )

    logger.info("Loading tokenizer from %s", base_model_path)
    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
        )

    logger.info(
        "Loading base model from %s on device_map=%s dtype=%s",
        base_model_path,
        args.device_map,
        args.torch_dtype,
    )
                                                                        
                                                                                 
                                                           
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(base_model_path),
        dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    prepared_lora_path, tmp_lora_dir, did_remap = _prepare_lora_path_for_base_model(base_model, lora_path)
    logger.info("Attaching LoRA from %s", prepared_lora_path)
    try:
        peft_model = PeftModel.from_pretrained(base_model, str(prepared_lora_path))
        _assert_loaded_lora_is_nonzero(peft_model)

        logger.info("Running merge_and_unload()")
        merged_model = peft_model.merge_and_unload()
    finally:
        if tmp_lora_dir is not None:
            tmp_lora_dir.cleanup()
    merged_model.eval()

    logger.info("Saving merged model to %s", output_dir)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=not args.unsafe_serialization,
        max_shard_size=args.max_shard_size,
    )

    if tokenizer is not None:
        logger.info("Saving tokenizer to %s", output_dir)
        tokenizer.save_pretrained(output_dir)
        _fix_tokenizer_config_for_vllm(output_dir)

    for processor_name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        source = base_model_path / processor_name
        if source.exists():
            shutil.copy2(source, output_dir / processor_name)
            logger.info("Copied processor config to merged model: %s", processor_name)

    metadata = _build_metadata(args, base_model_path, lora_path)
    metadata["adapter_key_remapped_for_conditional_generation"] = str(bool(did_remap))
    metadata_path = output_dir / "merge_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote merge metadata to %s", metadata_path)
    logger.info("Merge finished successfully.")
    logger.info("Merged model directory is ready for direct vLLM serving: %s", output_dir)


if __name__ == "__main__":
    main()
