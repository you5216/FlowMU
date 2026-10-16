                      
                                                             

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forget10", required=True)
    parser.add_argument("--retain90", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "splits"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--retain-replay-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forget_rows = load_json_or_jsonl(Path(args.forget10))
    retain_rows = load_json_or_jsonl(Path(args.retain90))
    if len(forget_rows) % args.rounds != 0:
        raise ValueError(f"forget rows={len(forget_rows)} not divisible by rounds={args.rounds}")
    batch_size = len(forget_rows) // args.rounds
    if batch_size <= 0:
        raise ValueError("empty forget split")

    rng = random.Random(args.seed)
    retain_replay = list(retain_rows)
    rng.shuffle(retain_replay)
    retain_replay = retain_replay[: min(args.retain_replay_size, len(retain_replay))]

    write_jsonl(out_dir / "retain_fixed.json", retain_rows)
    write_jsonl(out_dir / f"retain_replay_{len(retain_replay)}_seed{args.seed}.json", retain_replay)
    write_jsonl(out_dir / "empty_retain_for_forget_only.json", [])

    manifest: dict[str, Any] = {
        "protocol": "strict_incremental_ga_ce10",
        "forget10": str(Path(args.forget10)),
        "retain90": str(Path(args.retain90)),
        "rounds": [],
        "round_count": args.rounds,
        "batch_size": batch_size,
        "forget_count": len(forget_rows),
        "retain_count": len(retain_rows),
        "retain_replay_path": str(out_dir / f"retain_replay_{len(retain_replay)}_seed{args.seed}.json"),
        "empty_retain_path": str(out_dir / "empty_retain_for_forget_only.json"),
    }

    for round_idx in range(1, args.rounds + 1):
        start = (round_idx - 1) * batch_size
        end = round_idx * batch_size
        delta = forget_rows[start:end]
        cumulative = forget_rows[:end]
        delta_path = out_dir / f"forget_delta_{round_idx:02d}.json"
        cumulative_path = out_dir / f"forget_cumulative_{round_idx:02d}.json"
        memory_delta_path = out_dir / f"memory_forget_delta_{round_idx:02d}.json"
        memory_cumulative_path = out_dir / f"memory_forget_cumulative_{round_idx:02d}.json"
        write_jsonl(delta_path, delta)
        write_jsonl(cumulative_path, cumulative)
        write_jsonl(memory_delta_path, delta)
        write_jsonl(memory_cumulative_path, cumulative)
        manifest["rounds"].append(
            {
                "round": round_idx,
                "delta_start": start,
                "delta_end": end,
                "delta_forget_count": len(delta),
                "cumulative_forget_count": len(cumulative),
                "delta_forget_path": str(delta_path),
                "cumulative_forget_path": str(cumulative_path),
                "memory_delta_forget_path": str(memory_delta_path),
                "memory_cumulative_forget_path": str(memory_cumulative_path),
            }
        )

    manifest_path = out_dir / "round_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "rounds": args.rounds, "batch_size": batch_size}, indent=2))


if __name__ == "__main__":
    main()
