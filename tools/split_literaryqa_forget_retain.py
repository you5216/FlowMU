                      
                                                         

                                      
                                    
   

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


DEFAULT_ANNOTATION_DIR = Path(
    "/home/likunhao/likunhao52shared/datasets/LiteraryQA/literaryqa_annotations"
)
DEFAULT_INPUT_DIR = DEFAULT_ANNOTATION_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_ANNOTATION_DIR / "memory"
SPLITS = ("train", "validation", "test")
DEFAULT_PERCENTAGES = (1, 5, 10)


def load_documents(input_dir: Path) -> list[dict]:
    documents: list[dict] = []
    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc["_source_split"] = split
                documents.append(doc)

    seen_ids: set[str] = set()
    duplicates: list[str] = []
    for doc in documents:
        doc_id = doc["document_id"]
        if doc_id in seen_ids:
            duplicates.append(doc_id)
        seen_ids.add(doc_id)
    if duplicates:
        raise ValueError(f"Duplicate document_id values found: {duplicates[:10]}")

    return documents


def qa_records_for_docs(docs: list[dict]) -> list[dict]:
    records: list[dict] = []
    for doc in docs:
        for qa in doc.get("qas", []):
            answers = qa.get("answers", [])
            if not answers:
                continue
            records.append(
                {
                    "question": qa.get("question", ""),
                    "answer": answers[0],
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_manifest(
    path: Path,
    *,
    seed: int,
    percentages: list[int],
    selected_docs: dict[int, list[dict]],
    total_docs: int,
) -> None:
    manifest = {
        "seed": seed,
        "split_granularity": "book/document",
        "answer_policy": "first answer only",
        "total_books": total_docs,
        "percentages": percentages,
        "forget_books": {
            str(percent): [
                {
                    "document_id": doc["document_id"],
                    "gutenberg_id": doc["gutenberg_id"],
                    "title": doc["title"],
                    "source_split": doc["_source_split"],
                    "qa_count": len(doc.get("qas", [])),
                }
                for doc in docs
            ]
            for percent, docs in selected_docs.items()
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nested LiteraryQA forget/retain splits at book level."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=list(DEFAULT_PERCENTAGES),
        help="Forget percentages to create, e.g. 1 5 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_documents(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    percentages = sorted(set(args.percentages))
    shuffled = docs[:]
    random.Random(args.seed).shuffle(shuffled)

    selected_docs: dict[int, list[dict]] = {}
    for percent in percentages:
        forget_count = max(1, math.ceil(len(docs) * percent / 100))
        selected_docs[percent] = shuffled[:forget_count]

        forget_ids = {doc["document_id"] for doc in selected_docs[percent]}
        forget_docs = [doc for doc in docs if doc["document_id"] in forget_ids]
        retain_docs = [doc for doc in docs if doc["document_id"] not in forget_ids]

        forget_records = qa_records_for_docs(forget_docs)
        retain_records = qa_records_for_docs(retain_docs)

        forget_path = args.output_dir / f"forget{percent:02d}.json"
        retain_path = args.output_dir / f"retain{100 - percent:02d}.json"
        write_jsonl(forget_path, forget_records)
        write_jsonl(retain_path, retain_records)

        print(
            f"forget{percent:02d}/retain{100 - percent:02d}: "
            f"{len(forget_docs)} forget books, {len(forget_records)} forget QA, "
            f"{len(retain_docs)} retain books, {len(retain_records)} retain QA"
        )

    manifest_path = args.output_dir / f"split_manifest_seed{args.seed}.json"
    write_manifest(
        manifest_path,
        seed=args.seed,
        percentages=percentages,
        selected_docs=selected_docs,
        total_docs=len(docs),
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
