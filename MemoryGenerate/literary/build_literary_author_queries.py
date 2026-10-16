                      
                                                         

                                                                            

 
                                                                            
 

                                                                            
                                                                          
                                                                              
                      
   

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/home/likunhao/likunhao52shared/datasets/LiteraryQA/literaryqa_annotations/memory/split_manifest_seed42.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/final_performance/literaryQA/base/author"
)


def build_author_rows(books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for idx, book in enumerate(books):
        qa_count = int(book.get("qa_count") or 0)
        if qa_count <= 0:
            raise ValueError(f"Invalid qa_count for book {idx}: {book}")
        title = str(book.get("title") or "").strip()
        if not title:
            raise ValueError(f"Missing title for book {idx}: {book}")
        rows.append(
            {
                "author_index": idx,
                "row_start": cursor,
                "row_end": cursor + qa_count - 1,
                "author": title,
            }
        )
        cursor += qa_count
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    forget_books = manifest.get("forget_books")
    if not isinstance(forget_books, dict):
        raise ValueError(f"Invalid manifest, missing forget_books: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "split_granularity": manifest.get("split_granularity"),
        "files": {},
    }

    for percent in (1, 5, 10):
        key = str(percent)
        books = forget_books.get(key)
        if not isinstance(books, list):
            raise ValueError(f"Missing forget_books[{key}] in {args.manifest}")
        rows = build_author_rows(books)
        output_path = args.output_dir / f"forget{percent:02d}_author.json"
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["files"][output_path.name] = {
            "book_count": len(rows),
            "qa_count": rows[-1]["row_end"] + 1 if rows else 0,
            "first": rows[0] if rows else None,
            "last": rows[-1] if rows else None,
        }
        print(
            f"[Save] {output_path} books={len(rows)} "
            f"qa={summary['files'][output_path.name]['qa_count']}"
        )

    summary_path = args.output_dir / "literary_author_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Save] {summary_path}")


if __name__ == "__main__":
    main()
