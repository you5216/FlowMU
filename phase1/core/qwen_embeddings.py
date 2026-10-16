   
                    
                                                  
                                          
                                                  
   
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


@dataclass(frozen=True)
class QwenEmbeddingConfig:
    model_path: str
    device: str = "cuda"
    batch_size: int = 4
    max_length: int = 512
    hidden_layer: int = -1
    pooling: str = "mean"
    load_in_4bit: bool = True


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


class Qwen3EmbeddingBackend:
                                                                      

    def __init__(self, config: QwenEmbeddingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        preferred_device = (self.config.device or "cuda").strip().lower()
        if preferred_device == "cuda" and not torch.cuda.is_available():
            print("[Init] CUDA 不可用，Qwen3.5 embedding 回退至 CPU。")
            preferred_device = "cpu"

        cfg = AutoConfig.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        text_cfg = cfg.text_config
        self.embedding_dim = int(text_cfg.hidden_size)

        common_kwargs = {
            "config": text_cfg,
            "local_files_only": True,
            "trust_remote_code": True,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        if preferred_device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=bool(self.config.load_in_4bit),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return Qwen3_5TextModel.from_pretrained(
                self.config.model_path,
                device_map="auto",
                quantization_config=quantization_config,
                **common_kwargs,
            )

        cpu_dtype = torch.float32
        if getattr(getattr(torch.backends, "cpu", None), "has_bf16", False):
            cpu_dtype = torch.bfloat16
        return Qwen3_5TextModel.from_pretrained(
            self.config.model_path,
            device_map="cpu",
            torch_dtype=cpu_dtype,
            **common_kwargs,
        )

    def _resolve_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("Qwen3.5 embedding model has no parameters.") from exc

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        if self.config.pooling == "last_token":
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, lengths]
        else:
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=1)

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        target_device = self._resolve_model_device()

        for batch in _batched(texts, max(1, self.config.batch_size)):
            encoded = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states[self.config.hidden_layer]
            pooled = self._pool_hidden_states(hidden_states, encoded["attention_mask"])
            vectors.extend(pooled.detach().cpu().to(torch.float32).tolist())

        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError(f"Qwen3.5 embed_query expects str, got {type(text).__name__}")
        return self._embed_texts([text])[0]


def build_qwen_embeddings(
    *,
    model_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    max_length: int = 512,
    hidden_layer: int = -1,
    pooling: str = "mean",
    load_in_4bit: bool = True,
) -> Qwen3EmbeddingBackend:
    config = QwenEmbeddingConfig(
        model_path=model_path,
        device=device,
        batch_size=max(1, int(batch_size)),
        max_length=max(16, int(max_length)),
        hidden_layer=int(hidden_layer),
        pooling=str(pooling).strip().lower() or "mean",
        load_in_4bit=bool(load_in_4bit),
    )
    if config.pooling not in {"mean", "last_token"}:
        raise ValueError(f"Unsupported Qwen3.5 embedding pooling mode: {config.pooling}")
    return Qwen3EmbeddingBackend(config)
