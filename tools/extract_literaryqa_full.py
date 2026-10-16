                      
                                                                     

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ANNOTATION_DIR = Path(
    "/home/likunhao/likunhao52shared/datasets/LiteraryQA/literaryqa_annotations"
)
DEFAULT_INPUT_DIR = DEFAULT_ANNOTATION_DIR
DEFAULT_OUTPUT = DEFAULT_ANNOTATION_DIR / "memory" / "full.json"
SPLITS = ("train", "validation", "test")


def build_answer(answers: list[str], mode: str) -> str:
    if not answers:
        return ""
    if mode == "first":
        return answers[0]
    if mode == "join":
        return " / ".join(answers)
    raise ValueError(f"Unsupported answer mode: {mode}")


def iter_records(input_dir: Path, answer_mode: str):
    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                for qa_idx, qa in enumerate(sample.get("qas", []), start=1):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    if not isinstance(answers, list):
                        raise TypeError(
                            f"{input_path}:{line_no} qa#{qa_idx} has non-list answers"
                        )
                    yield {
                        "question": question,
                        "answer": build_answer(answers, answer_mode),
                    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all LiteraryQA QA pairs as TOFU-style JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train/validation/test jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("first", "join"),
        default="first",
        help="How to convert LiteraryQA's answers list into TOFU's answer string.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for record in iter_records(args.input_dir, args.answer_mode):
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
            count += 1

    print(f"Wrote {count} QA records to {args.output}")


if __name__ == "__main__":
    main()
