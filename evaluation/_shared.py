from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = PROJECT_ROOT / "phase1"
ARTIFACTS_DIR = PHASE1_DIR / "artifacts"


def ensure_phase1_on_path() -> None:
    if str(PHASE1_DIR) not in sys.path:
        sys.path.insert(0, str(PHASE1_DIR))


def default_results_dir() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "phase1" and (cwd / "evaluation").exists():
        return cwd / "evaluation"
    if cwd == PROJECT_ROOT:
        return PROJECT_ROOT / "evaluation"
    return PROJECT_ROOT / "evaluation"
