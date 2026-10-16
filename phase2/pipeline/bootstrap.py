                             

                                     
                                  
   

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PHASE2_DIR = PIPELINE_DIR.parent
AGENT_DIR = PHASE2_DIR.parent
PHASE1_DIR = AGENT_DIR / "phase1"

if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

ARTIFACTS_DIR = PHASE1_DIR / "artifacts"

from config import ORIGINAL_LORA_PATH, UNLEARNED_LORA_OUTPUT_PATH              
from core.llm_engine import BASE_MODEL_PATH, CustomQwenLLM              

DEFAULT_PROBE_PATH = ARTIFACTS_DIR / "causal_probes.pkl"
DEFAULT_GRAPH_PATH = ARTIFACTS_DIR / "unified_graph.pkl"
DEFAULT_CORE_DATASET_PATH = ARTIFACTS_DIR / "D_forget_core.jsonl"
DEFAULT_CONTEXT_DATASET_PATH = ARTIFACTS_DIR / "D_forget_context.jsonl"
DEFAULT_QUERY_CONDITIONED_SIGNAL_PATH = ARTIFACTS_DIR / "query_conditioned_unlearning.pkl"
DEFAULT_RETAIN_PATH = Path("/home/likunhao/likunhao52shared/datasets/TOFU/retain90.json")
DEFAULT_LOCALIZATION_PKL = "localized_neurons.pkl"
DEFAULT_LOCALIZATION_JSON = "localized_neurons.json"
DEFAULT_INTERVENTION_JSON = "intervention_plan.json"
DEFAULT_RUN_METADATA_JSON = "stage2_run_metadata.json"
