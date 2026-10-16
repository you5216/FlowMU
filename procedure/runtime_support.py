                                                                

from __future__ import annotations

import importlib
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_MODULES = (
    "agent.phase1.config_50",
    "agent.phase1.config_52",
)


def get_runtime_config():
    for module_name in CONFIG_MODULES:
        module = importlib.import_module(module_name)
        base_model_path = Path(getattr(module, "BASE_MODEL_PATH", ""))
        if base_model_path.exists():
            return module
    return importlib.import_module(CONFIG_MODULES[0])


def build_shared_embeddings():
    from agent.phase1.core.qwen_embeddings import build_qwen_embeddings

    cfg = get_runtime_config()
    return build_qwen_embeddings(
        model_path=str(cfg.BASE_MODEL_PATH),
        device=getattr(cfg, "EMBEDDINGS_DEVICE", "cuda"),
        batch_size=getattr(cfg, "EMBEDDINGS_BATCH_SIZE", 4),
        max_length=getattr(cfg, "EMBEDDINGS_MAX_LENGTH", 512),
        hidden_layer=getattr(cfg, "EMBEDDINGS_LAYER", -1),
        pooling=getattr(cfg, "EMBEDDINGS_POOLING", "mean"),
        load_in_4bit=getattr(cfg, "EMBEDDINGS_LOAD_IN_4BIT", True),
    )
