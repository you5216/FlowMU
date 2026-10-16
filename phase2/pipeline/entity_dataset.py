                                   

                                  
                                          
   

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset


def _load_json_or_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]
def build_forget_entity_dataset(cfg) -> Dataset:
    if getattr(cfg, "forget_target_path", None) is not None:
                                                                              
                                                                             
                                                                               
        rows = []
        target_rows = _load_json_or_jsonl(Path(cfg.forget_target_path))
        for idx, row in enumerate(target_rows):
            question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
            original_answer = str(row.get("answer") or row.get("output") or row.get("reference_answer") or "").strip()
            if not question or not original_answer:
                continue
            rows.append(
                {
                    "example_id": f"forget_direct::{idx}",
                    "cluster_id": "pure_phase2_forget",
                    "node_id": f"forget_node_{idx}",
                    "community": "pure_phase2",
                    "instruction": question,
                    "original_output": original_answer,
                    "output": original_answer,
                    "view_type": "direct_question",
                    "view_weight": 1.0,
                    "gat_weight": 1.0,
                }
            )
        return Dataset.from_list(rows)

    raise ValueError("--forget-target-path is required for Phase-2 forget entity dataset construction.")
