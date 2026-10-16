                 

from pathlib import Path

SERVER_ID = "50"

BASE_MODEL_PATH    = "/home/likunhao/likunhao52shared/llms/Qwen3.5-9B"
LORA_PATH          = "/home/likunhao/likunhao52shared/tofu_unlearned_lora"
ORIGINAL_LORA_PATH = "/home/likunhao/likunhao52shared/tofu_native_lora/checkpoint-4500"
UNLEARNED_LORA_OUTPUT_PATH = "/home/likunhao/likunhao52shared/tofu_unlearned_lora"

TOFU_DATASET_DIR = Path("/home/likunhao/likunhao52shared/datasets/TOFU")
TOFU_FULL_JSON   = TOFU_DATASET_DIR / "full.json"
TOFU_FORGET05    = TOFU_DATASET_DIR / "forget05.json"
TOFU_FORGET01    = TOFU_DATASET_DIR / "forget01.json"
TOFU_FORGET10    = TOFU_DATASET_DIR / "forget10.json"

EMBEDDINGS_DEVICE       = "cuda"
EMBEDDINGS_BATCH_SIZE   = 4
EMBEDDINGS_MAX_LENGTH   = 512
EMBEDDINGS_LAYER        = -1
EMBEDDINGS_POOLING      = "mean"
EMBEDDINGS_LOAD_IN_4BIT = True
