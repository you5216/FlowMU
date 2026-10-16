                                

                                     
                                         
   

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

logger = logging.getLogger("GAT-AFRU.Stage2Methods")


@dataclass
class LocalizationArtifacts:
    strategy: str
    top_fraction: float
    target_neurons: Dict[str, List[int]]
    layer_scores: Dict[str, List[float]]
    total_target_neurons: int
    parameter_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> Dict[str, Any]:
        parameter_mask_summary = {}
        for name, mask in self.parameter_masks.items():
            parameter_mask_summary[name] = {
                "shape": list(mask.shape),
                "num_selected": int(mask.sum()),
                "density": float(mask.mean()) if mask.size else 0.0,
            }
        return {
            "strategy": self.strategy,
            "top_fraction": self.top_fraction,
            "target_neurons": self.target_neurons,
            "layer_scores": self.layer_scores,
            "total_target_neurons": self.total_target_neurons,
            "parameter_mask_summary": parameter_mask_summary,
            "metadata": self.metadata,
        }


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def extract_layer_index(param_name: str) -> Optional[int]:
    for part in param_name.split("."):
        if part.isdigit():
            return int(part)
    return None


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_val = float(values.max())
    min_val = float(values.min())
    if math.isclose(max_val, min_val):
        return np.ones_like(values, dtype=np.float32)
    return ((values - min_val) / (max_val - min_val)).astype(np.float32)


def register_gradient_mask_hooks(model, parameter_masks: Dict[str, np.ndarray]) -> List[Any]:
    handles = []
    if not parameter_masks:
        return handles

    named_params = dict(model.named_parameters())
    for name, mask in parameter_masks.items():
        param = named_params.get(name)
        if param is None or not param.requires_grad:
            continue
        mask_tensor = torch.as_tensor(mask, device=param.device, dtype=param.dtype)
        if mask_tensor.shape != param.shape:
            continue
        handles.append(param.register_hook(lambda grad, m=mask_tensor: grad * m))
    return handles


def remove_gradient_mask_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            continue


def summarize_dataset_rows(dataset) -> Dict[str, Any]:
    summary = {
        "num_rows": int(len(dataset)),
    }
    if hasattr(dataset, "column_names"):
        summary["columns"] = list(dataset.column_names)
    return summary


def build_run_metadata(
    *,
    localization: LocalizationArtifacts,
    dataset_summary: Dict[str, Any],
    config_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_summary": dataset_summary,
        "config_summary": config_summary,
        "localization": localization.to_serializable(),
    }
