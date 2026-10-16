                      
                                                                    

                                                                         
                                                                             
   

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from phase2.pipeline.config import Stage2Config
from phase2.pipeline.masked_optimization import (              
    _evaluate_canonical_proxy,
    _load_canonical_proxy_spec,
    _precompute_canonical_proxy_baseline,
)
from phase2.pipeline.support import build_model, build_tokenizer, release_memory              

LOGGER = logging.getLogger("canonical_proxy_for_loras")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--forget-path", required=True)
    parser.add_argument("--retain-path", required=True)
    parser.add_argument("--lora-paths", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--forget-samples", type=int, default=400)
    parser.add_argument("--retain-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--retain-topk", type=int, default=32)
    parser.add_argument(
        "--probe-prefer-gpu",
        action="store_true",
        help="Load frozen probe/base models on GPU first instead of auto/offload.",
    )
    return parser.parse_args()


def make_cfg(args: argparse.Namespace, *, original_lora_path: str, init_new_lora: bool) -> Stage2Config:
    return Stage2Config(
        base_model_path=args.base_model_path,
        original_lora_path=original_lora_path,
        output_dir="",
        graph_path=Path(""),
        forget_target_path=Path(args.forget_path),
        retain_path=Path(args.retain_path),
        core_dataset_path=Path(""),
        context_dataset_path=Path(""),
        init_new_lora=init_new_lora,
        new_lora_r=16,
        new_lora_alpha=32,
        new_lora_dropout=0.05,
        new_lora_target_modules="gate_proj,up_proj,down_proj",
        max_length=args.max_length,
        retain_topk=args.retain_topk,
        canonical_proxy_enabled=True,
        canonical_proxy_forget_path=Path(args.forget_path),
        canonical_proxy_retain_path=Path(args.retain_path),
        canonical_proxy_forget_samples=args.forget_samples,
        canonical_proxy_retain_samples=args.retain_samples,
        canonical_proxy_batch_size=args.batch_size,
        canonical_proxy_hard_retain_filter=False,
    )


def load_eval_model(args: argparse.Namespace, lora_path: str):
    cfg = make_cfg(args, original_lora_path="", init_new_lora=True)
    base_model = build_model(cfg, is_trainable=False)
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    labels = args.labels or [Path(path).parent.name for path in args.lora_paths]
    if len(labels) != len(args.lora_paths):
        raise ValueError("--labels length must match --lora-paths length.")

    if args.probe_prefer_gpu:
        import os

        os.environ["PHASE2_PROBE_PREFER_GPU"] = "1"

    baseline_cfg = make_cfg(args, original_lora_path="", init_new_lora=True)
    tokenizer = build_tokenizer(args.base_model_path)
    proxy_spec = _load_canonical_proxy_spec(baseline_cfg)
    if proxy_spec is None:
        raise RuntimeError("Failed to build canonical proxy spec.")

    LOGGER.info("Computing base canonical proxy baseline...")
    baseline = _precompute_canonical_proxy_baseline(
        cfg=baseline_cfg,
        tokenizer=tokenizer,
        proxy_spec=proxy_spec,
    )

    results: list[dict[str, Any]] = []
    for label, lora_path in zip(labels, args.lora_paths):
        LOGGER.info("Evaluating %s: %s", label, lora_path)
        model = load_eval_model(args, lora_path)
        try:
            metrics = _evaluate_canonical_proxy(
                cfg=baseline_cfg,
                model=model,
                tokenizer=tokenizer,
                proxy_spec=proxy_spec,
                proxy_baseline=baseline,
            )
        finally:
            del model
            release_memory()
        result = {
            "label": label,
            "lora_path": lora_path,
            "forget_samples": args.forget_samples,
            "retain_samples": args.retain_samples,
            "base_forget_nll": float(baseline["forget_nll_mean"]),
            "base_retain_nll": float(baseline["retain_nll_mean"]),
            **metrics,
        }
        results.append(result)
        LOGGER.info(
            "%s forget_delta=%.4f retain_delta=%.4f retain_kl=%.4f",
            label,
            result["canonical_proxy_forget_nll_delta"],
            result["canonical_proxy_retain_nll_delta"],
            result["canonical_proxy_retain_kl"],
        )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info("Wrote %s and %s", output_json, output_csv)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
