                      
   
                 
                                                    
                                  

     
                        
                                                   
     
                              
                                                                                   

              
                                                                        
   

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation._shared import default_results_dir

_EVAL_DIR = Path(__file__).resolve().parent

JUDGE_SYSTEM = """You are a fair semantic answer evaluator.
Judge whether the model answer is basically semantically equivalent to the reference answer for the given question.
Do not require exact wording, token-level match, or identical sentence structure.
Mark Correct if the key factual meaning is preserved, including reasonable paraphrases, abbreviations, or extra harmless context.
Mark Wrong if key facts are missing, contradicted, unrelated, too vague, or the answer only says it does not know.
Respond with ONLY one word: Correct or Wrong."""

JUDGE_TEMPLATE = """Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Is the Model Answer basically semantically equivalent to the Reference Answer? Respond with ONLY Correct or Wrong."""


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()
    for pat in [
        r"\*\*Final Answer:\*\*\s*(.+)",
        r"\*\*Answer:\*\*\s*(.+)",
        r"Final Answer:\s*(.+)",
        r"Answer:\s*(.+)",
        r"ANSWER:\s*(.+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\*+", "", m.group(1).strip().split("\n")[0]).strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return re.sub(r"\*+", "", lines[-1]).strip() if lines else ""


def parse_judge_verdict(raw_text: str) -> str:
                                                     

                                                                               
                                                                        
                                        
       
    normalized = re.sub(r"[^a-zA-Z]+", " ", str(raw_text or "").strip().lower()).strip()
    if not normalized:
        return "Wrong"
    first = normalized.split()[0]
    if first == "correct":
        return "Correct"
    return "Wrong"


def judge_answer(client: OpenAI, model: str, question: str, reference: str, prediction: str) -> tuple[str, str]:
    extracted = extract_final_answer(prediction)
    if not extracted:
        return "Wrong", "[LOCAL_EMPTY_PREDICTION]"
    if extracted.strip().lower() in {"thinking process:", "thinking process", "<think>", "</think>"}:
        return "Wrong", "[LOCAL_INVALID_THINKING_ONLY_OUTPUT]"

    prompt = JUDGE_TEMPLATE.format(
        question=(question or "")[:300],
        reference=(reference or "")[:200],
        prediction=extracted[:300],
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return parse_judge_verdict(raw), raw
    except Exception as exc:
        print(f"[WARN] Judge API 调用失败: {exc}")
        return "Wrong", f"[JUDGE_ERROR] {exc}"


def safe_split_type(row: dict) -> str:
    value = (row.get("split_type") or row.get("split") or "").strip().lower()
    return value if value in {"forget", "retain"} else "retain"


def judge_row(client: OpenAI, model: str, row: dict) -> dict:
    row = dict(row)
    verdict, judge_raw = judge_answer(
        client=client,
        model=model,
        question=row.get("question", ""),
        reference=row.get("reference_answer", ""),
        prediction=row.get("prediction", ""),
    )
    row["judge_verdict"] = verdict
    row["judge_raw_response"] = judge_raw
    row["extracted_answer"] = extract_final_answer(row.get("prediction", ""))
    return row


def process_file(client: OpenAI, model: str, input_csv: Path, output_csv: Path, label: str, num_workers: int) -> dict:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[SKIP] 文件为空: {input_csv}")
        return {"label": label, "f_acc": 0.0, "r_acc": 0.0, "f_n": 0, "r_n": 0}

    forget = [r for r in rows if safe_split_type(r) == "forget"]
    retain = [r for r in rows if safe_split_type(r) == "retain"]
    print(f"\n[{label}] 总计={len(rows)}  forget={len(forget)}  retain={len(retain)}")

    correct_f = wrong_f = correct_r = wrong_r = 0
    results_by_index: dict[int, dict] = {}
    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        for idx, row in enumerate(rows, start=1):
            if idx == 1 or idx % 100 == 0:
                print(f"  [{label}] {idx}/{len(rows)}")
            results_by_index[idx - 1] = judge_row(client, model, row)
    else:
        print(f"  [{label}] 并发 judge workers={num_workers}")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(judge_row, client, model, row): idx
                for idx, row in enumerate(rows)
            }
            for done_count, future in enumerate(as_completed(future_to_index), start=1):
                idx = future_to_index[future]
                try:
                    results_by_index[idx] = future.result()
                except Exception as exc:
                    row = dict(rows[idx])
                    row["judge_verdict"] = "Wrong"
                    row["judge_raw_response"] = f"[JUDGE_ERROR] {exc}"
                    row["extracted_answer"] = extract_final_answer(row.get("prediction", ""))
                    results_by_index[idx] = row
                if done_count == 1 or done_count % 100 == 0 or done_count == len(rows):
                    print(f"  [{label}] {done_count}/{len(rows)}")

    results = [results_by_index[idx] for idx in range(len(rows))]

    for row in results:
        if safe_split_type(row) == "forget":
            if row["judge_verdict"] == "Correct":
                correct_f += 1
            else:
                wrong_f += 1
        else:
            if row["judge_verdict"] == "Correct":
                correct_r += 1
            else:
                wrong_r += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    f_acc = correct_f / len(forget) if forget else 0.0
    r_acc = correct_r / len(retain) if retain else 0.0
    print(f"  [{label}] Forget Acc: {f_acc:.4f} ({correct_f}/{len(forget)})")
    print(f"  [{label}] Retain Acc: {r_acc:.4f} ({correct_r}/{len(retain)})")
    print(f"  已保存 -> {output_csv}")

    return {
        "label": label,
        "f_acc": f_acc,
        "r_acc": r_acc,
        "f_n": len(forget),
        "r_n": len(retain),
        "forget_wrong": wrong_f,
        "retain_wrong": wrong_r,
    }


def resolve_split_files(split: str) -> tuple[list[Path], list[str]]:
    files = [_EVAL_DIR / f"agent_results_{split}.csv"]
    labels = [f"MyAgent_{split}"]
    return files, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge 评判")
    parser.add_argument("--split", default=None, help="遗忘集标签: 01 / 05 / 10")
    parser.add_argument("--files", nargs="+", default=None, help="手动指定 CSV 文件列表")
    parser.add_argument("--labels", nargs="+", default=None, help="对应标签")
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="judge")
    parser.add_argument("--output-dir", default=str(default_results_dir()))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("JUDGE_NUM_WORKERS", "1")),
        help="Number of concurrent judge API requests. Output rows remain in input order.",
    )
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
        labels = args.labels or [f.stem for f in files]
    elif args.split:
        files, labels = resolve_split_files(args.split)
    else:
        parser.error("必须提供 --split 或 --files")

    if len(labels) != len(files):
        raise ValueError("labels 数量必须与 files 数量一致")

    output_dir = Path(args.output_dir)
    client = OpenAI(base_url=args.api_url, api_key="dummy")
    split_tag = args.split or "custom"

    all_results = []
    for file_path, label in zip(files, labels):
        if not file_path.exists():
            print(f"[SKIP] 文件不存在: {file_path}")
            continue
        out_csv = output_dir / f"judge_{label}.csv"
        result = process_file(client, args.model, file_path, out_csv, label, args.num_workers)
        all_results.append(result)

    if not all_results:
        print("[ERROR] 无文件被处理。")
        return

    print("\n" + "=" * 78)
    print(f"LLM-as-Judge 汇总  split={split_tag}  model={args.model}")
    print("=" * 78)
    header = f"{'指标':<20}" + "".join(f"{r['label']:>14}" for r in all_results)
    print(header)
    print("-" * 78)
    print(f"{'Forget Accuracy':<20}" + "".join(f"{r['f_acc']:>14.4f}" for r in all_results))
    print(f"{'Retain Accuracy':<20}" + "".join(f"{r['r_acc']:>14.4f}" for r in all_results))
    print("=" * 78)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"judge_summary_{split_tag}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存: {out_json}")


if __name__ == "__main__":
    main()
