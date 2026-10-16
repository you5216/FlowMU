                      
                                                                             

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = "/disk8T/lkh/python_projects/llms/Qwen3.5-9B"
DEFAULT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent/final_performance/literaryQA/base/lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LiteraryQA SFT LoRA")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE)
    parser.add_argument("--lora-path", default=str(DEFAULT_ROOT / "literaryqa_sft_lora_qwen35_9b"))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "merged" / "literaryqa_sft_qwen35_9b"))
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def fix_tokenizer_config(output_dir: Path) -> None:
    path = output_dir / "tokenizer_config.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("tokenizer_class") == "TokenizersBackend":
        data["tokenizer_class"] = "Qwen2Tokenizer"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_model_path = Path(args.base_model_path)
    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)
    if not base_model_path.exists():
        raise FileNotFoundError(base_model_path)
    if not lora_path.exists():
        raise FileNotFoundError(lora_path)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    if (output_dir / "config.json").exists():
        print(f"[Merge] already exists: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=False)
    lora_b_abs = 0.0
    lora_b_params = 0
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            lora_b_abs += float(param.detach().float().abs().sum().cpu())
            lora_b_params += int(param.numel())
    if lora_b_params <= 0 or lora_b_abs <= 0:
        raise RuntimeError("LoRA adapter appears empty; refusing to merge.")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))
    fix_tokenizer_config(output_dir)
    metadata = {
        "method": "literaryqa_sft_lora_merge",
        "base_model_path": str(base_model_path),
        "lora_path": str(lora_path),
        "output_dir": str(output_dir),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "lora_b_abs_sum": lora_b_abs,
        "lora_b_params": lora_b_params,
    }
    (output_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[Merge] saved merged model: {output_dir}")
    print(f"[Merge] lora_B_abs_sum={lora_b_abs:.6f} lora_B_params={lora_b_params}")


if __name__ == "__main__":
    main()
