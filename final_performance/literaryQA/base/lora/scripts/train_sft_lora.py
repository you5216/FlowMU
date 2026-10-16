                      
                                             

                                                                             
                                                                           
                                                                   
   

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup


DEFAULT_BASE = "/disk8T/lkh/python_projects/llms/Qwen3.5-9B"
DEFAULT_DATA = "/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/re_full.json"
DEFAULT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent/final_performance/literaryQA/base/lora")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or row.get("output") or "").strip()
            if not question or not answer:
                raise ValueError(f"Bad row #{idx}: missing question/answer")
            rows.append({"question": question, "answer": answer})
    return rows


def apply_chat_template_safe(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
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
    except Exception:
        if add_generation_prompt:
            return f"Question: {messages[0]['content']}\nAnswer:"
        return f"Question: {messages[0]['content']}\nAnswer: {messages[1]['content']}"


def render_pair(tokenizer, question: str, answer: str) -> tuple[str, str]:
    prompt_only = apply_chat_template_safe(
        tokenizer,
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
    )
    full_text = apply_chat_template_safe(
        tokenizer,
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        add_generation_prompt=False,
    )
    return prompt_only, full_text


class LiteraryQASFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt_only, full_text = render_pair(self.tokenizer, row["question"], row["answer"])
        prompt_ids = self.tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
                                                                           
                                                 
            labels[-1] = input_ids[-1]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_batch(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteraryQA Qwen3.5 SFT LoRA training")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE)
    parser.add_argument("--train-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "literaryqa_sft_lora_qwen35_9b"))
    parser.add_argument("--log-jsonl", default=str(DEFAULT_ROOT / "logs" / "sft_train_metrics.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def save_adapter(model, tokenizer, output_dir: Path, args: argparse.Namespace, extra: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "method": "literaryqa_sft_lora",
        "base_model_path": args.base_model_path,
        "train_path": args.train_path,
        "args": vars(args),
        **extra,
    }
    (output_dir / "sft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    log_jsonl = Path(args.log_jsonl)
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if log_jsonl.exists():
        log_jsonl.unlink()

    rows = load_jsonl(Path(args.train_path))
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = LiteraryQASFTDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    planned_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(planned_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, planned_steps)

    device = next(model.parameters()).device
    start_time = time.time()
    global_step = 0
    running_loss = 0.0
    model.train()
    progress = tqdm(total=planned_steps, desc="LiteraryQA SFT LoRA")
    optimizer.zero_grad(set_to_none=True)

    epoch_idx = 0
    while global_step < planned_steps:
        epoch_idx += 1
        for micro_step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

            if global_step % args.logging_steps == 0 or global_step == 1:
                avg_loss = running_loss / max(1, args.logging_steps if global_step > 1 else 1)
                running_loss = 0.0
                record = {
                    "step": global_step,
                    "epoch": epoch_idx,
                    "loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_sec": time.time() - start_time,
                }
                with log_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{record['lr']:.2e}")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_adapter(
                    model,
                    tokenizer,
                    output_dir / f"checkpoint-{global_step}",
                    args,
                    {"global_step": global_step, "num_train_rows": len(rows), "partial": True},
                )

            if global_step >= planned_steps:
                break

    progress.close()
    save_adapter(
        model,
        tokenizer,
        output_dir,
        args,
        {
            "global_step": global_step,
            "num_train_rows": len(rows),
            "elapsed_sec": time.time() - start_time,
            "planned_steps": planned_steps,
        },
    )
    print(f"[SFT] saved LoRA: {output_dir}")
    print(f"[SFT] log: {log_jsonl}")


if __name__ == "__main__":
    main()
