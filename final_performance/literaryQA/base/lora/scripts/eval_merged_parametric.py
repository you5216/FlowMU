                      
                                                                

                                                                              
                                                                               
                                                                           
                                                                       
   

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_ROOT = Path("/disk8T/lkh/python_projects/agent_unlearning_zhx/agent/final_performance/literaryQA/base/lora")
DEFAULT_DATA_ROOT = Path("/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag")


def load_json_or_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected list data: {path}")
    return [row for row in data if isinstance(row, dict)]


def normalize_text(text: str) -> str:
    text = re.sub(r"\*+", "", str(text or ""))
    text = text.casefold()
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    value = normalize_text(text)
    return value.split() if value else []


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "her", "his", "in", "is", "it", "its", "of", "on", "or", "she", "that",
    "the", "their", "they", "to", "was", "were", "what", "where", "which", "who",
    "whom", "whose", "why", "with", "does", "did", "do", "how", "many", "much",
    "book", "novel", "story", "play", "author",
}


def content_tokens(text: str) -> list[str]:
    toks = []
    for tok in tokenize(text):
        if tok in STOPWORDS or len(tok) <= 2:
            continue
        toks.append(tok)
    return toks


def content_recall(prediction: str, reference: str) -> float:
    pred = set(content_tokens(prediction))
    ref = content_tokens(reference)
    if not ref:
        return 0.0
    return sum(1 for tok in ref if tok in pred) / len(ref)


def proxy_correct(prediction: str, reference: str, rouge_threshold: float, content_threshold: float) -> bool:
    rl = rouge_l_f1(prediction, reference)
    cr = content_recall(prediction, reference)
    return rl >= rouge_threshold or cr >= content_threshold


def apply_chat_template_safe(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": (
        "Answer the following LiteraryQA question directly and concisely. "
        "Return only the answer sentence.\n\n"
        f"Question: {question}"
    )}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            "Answer the following LiteraryQA question directly and concisely.\n"
            f"Question: {question}\nAnswer:"
        )


def postprocess(text: str) -> str:
    value = str(text or "").strip()
    if "<think>" in value and "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    for marker in ["\nQuestion:", "\nUser:", "\nAssistant:", "\nuser", "\nassistant", "<|im_end|>"]:
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[0] if lines else value


def generate_batch(model, tokenizer, questions: list[str], *, max_new_tokens: int) -> list[str]:
    prompts = [apply_chat_template_safe(tokenizer, question) for question in questions]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    predictions = []
    for row_idx, prompt_len in enumerate(prompt_lengths):
                                                                                      
        start = encoded["input_ids"].shape[1]
        new_ids = output_ids[row_idx, start:]
        predictions.append(postprocess(tokenizer.decode(new_ids, skip_special_tokens=True)))
    return predictions


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate merged LiteraryQA SFT model")
    parser.add_argument("--model-path", default=str(DEFAULT_ROOT / "merged" / "literaryqa_sft_qwen35_9b"))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "eval" / "merged_parametric_for01_05_10"))
    parser.add_argument("--splits", default="01,05,10")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--rouge-threshold", type=float, default=0.45)
    parser.add_argument("--content-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    summaries = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        dataset_path = DEFAULT_DATA_ROOT / f"forget{split}.json"
        rows = load_json_or_jsonl(dataset_path)
        if args.limit > 0:
            rows = rows[: args.limit]
        result_rows = []
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"forget{split}"):
            batch_rows = rows[start : start + args.batch_size]
            questions = [str(row.get("question") or "").strip() for row in batch_rows]
            predictions = generate_batch(model, tokenizer, questions, max_new_tokens=args.max_new_tokens)
            for row, question, prediction in zip(batch_rows, questions, predictions):
                reference = str(row.get("answer") or row.get("output") or "").strip()
                rl = rouge_l_f1(prediction, reference)
                cr = content_recall(prediction, reference)
                correct = proxy_correct(prediction, reference, args.rouge_threshold, args.content_threshold)
                result_rows.append(
                    {
                        "split": "forget",
                        "subset": f"forget{split}",
                        "question": question,
                        "reference_answer": reference,
                        "prediction": prediction,
                        "rouge_l": f"{rl:.6f}",
                        "content_recall": f"{cr:.6f}",
                        "proxy_correct": str(bool(correct)).lower(),
                        "model_path": str(model_path),
                    }
                )
        csv_path = output_dir / f"forget{split}_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "subset", "question", "reference_answer", "prediction",
                    "rouge_l", "content_recall", "proxy_correct", "model_path",
                ],
            )
            writer.writeheader()
            writer.writerows(result_rows)
        acc = mean(1.0 if row["proxy_correct"] == "true" else 0.0 for row in result_rows)
        rouge = mean(float(row["rouge_l"]) for row in result_rows)
        crecall = mean(float(row["content_recall"]) for row in result_rows)
        summaries.append(
            {
                "subset": f"forget{split}",
                "n": len(result_rows),
                "proxy_acc": acc,
                "mean_rouge_l": rouge,
                "mean_content_recall": crecall,
                "csv": str(csv_path),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[Eval] summary: {summary_path}")


if __name__ == "__main__":
    main()
