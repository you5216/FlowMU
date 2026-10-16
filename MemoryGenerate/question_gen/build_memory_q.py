                      
from __future__ import annotations

'''
将所有生成的作者的问题合并到一个
'''
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
QUESTION_RUNS_DIR = SCRIPT_DIR / "artifacts" / "question_runs"
OUTPUT_PATH = SCRIPT_DIR / "artifacts" / "memory_q.json"


def load_questions_from_plan(plan_path: Path) -> list[str]:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{plan_path} is not a JSON array")

    questions_with_turn: list[tuple[int, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        turn = int(item.get("turn", 0))
        questions_with_turn.append((turn, question))

    questions_with_turn.sort(key=lambda x: x[0])
    return [question for _, question in questions_with_turn]


def build_memory_q() -> list[dict[str, str]]:
    if not QUESTION_RUNS_DIR.exists():
        raise FileNotFoundError(f"Question runs directory not found: {QUESTION_RUNS_DIR}")

    memory_q: list[dict[str, str]] = []

    for author_dir in sorted(QUESTION_RUNS_DIR.iterdir()):
        if not author_dir.is_dir():
            continue

        plan_path = author_dir / "question_plan.json"
        if not plan_path.exists():
            continue

        for question in load_questions_from_plan(plan_path):
            memory_q.append({"question": question})

    return memory_q


def main() -> None:
    memory_q = build_memory_q()
    OUTPUT_PATH.write_text(
        json.dumps(memory_q, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(memory_q)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
