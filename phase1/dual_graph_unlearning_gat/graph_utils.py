                      
                                                                           

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch


RAG_NODE_TYPES = {"static_knowledge": 0}
MEMORY_NODE_TYPES = {"entity": 0, "anchor": 1, "chunk": 2, "unknown": 3}
MEMORY_EDGE_TYPES = {"no_edge": 0, "mentioned_in": 1, "next_turn": 2, "relation": 3}


@dataclass
class GraphBatch:
    graph_name: str
    x: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    node_ids: list[str]
    edge_ids: list[tuple[str, str, int | None]]
    community: torch.Tensor | None = None

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            graph_name=self.graph_name,
            x=self.x.to(device),
            node_type=self.node_type.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_weight=self.edge_weight.to(device),
            node_ids=self.node_ids,
            edge_ids=self.edge_ids,
            community=None if self.community is None else self.community.to(device),
        )


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

                            
def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:limit] or "node"


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:n]


def as_float_embedding(value: Any, dim: int | None = None) -> np.ndarray:
    if value is None:
        if dim is None:
            raise ValueError("Missing embedding and no fallback dim was provided.")
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def infer_embedding_dim(*graphs: nx.Graph) -> int:
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            if data.get("embedding") is not None:
                return int(len(data["embedding"]))
    raise ValueError("No node embedding found.")

             
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        denom = np.linalg.norm(arr) + 1e-12
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / denom


def add_louvain_communities_to_memory(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    projection = nx.Graph()
    topology_nodes = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if str(data.get("node_type") or "unknown") != "chunk"
    ]
    projection.add_nodes_from(topology_nodes)
    edge_type_weight = {
        "relation": 1.0,
        "mentioned_in": 0.8,
        "next_turn": 0.2,
    }
    for source, target, _key, data in graph.edges(keys=True, data=True):
        if source not in projection or target not in projection:
            continue
        edge_type = str(data.get("edge_type") or "unknown")
        base = edge_type_weight.get(edge_type, 0.1)
        confidence = float(data.get("confidence", data.get("similarity", 1.0)) or 1.0)
        weight = float(base * max(0.0, min(1.0, confidence)))
        if projection.has_edge(source, target):
            projection[source][target]["weight"] += weight
        else:
            projection.add_edge(source, target, weight=weight)

    try:
        communities = nx.community.louvain_communities(projection, weight="weight", seed=20260608)
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(projection, weight="weight"))

    community_by_node: dict[str, int] = {}
    for idx, nodes in enumerate(communities):
        for node in nodes:
            community_by_node[str(node)] = int(idx)

    anchor_by_chunk: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type") or "") == "anchor" and data.get("chunk_id") is not None:
            anchor_by_chunk[str(data["chunk_id"])] = str(node_id)

    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type") or "unknown") == "chunk":
            anchor_id = anchor_by_chunk.get(str(node_id))
            data["community_id"] = community_by_node.get(anchor_id, -1)
        else:
            data["community_id"] = community_by_node.get(str(node_id), -1)

    for source, target, key, data in graph.edges(keys=True, data=True):
        cu = graph.nodes[source].get("community_id", -1)
        cv = graph.nodes[target].get("community_id", -1)
        data["same_community"] = int(cu == cv and cu != -1)
        data["community_pair"] = f"{cu}:{cv}"

    sizes = [len(c) for c in communities]
    report = {
        "community_count": len(communities),
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "mean_size": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "projection_nodes": projection.number_of_nodes(),
        "projection_edges": projection.number_of_edges(),
        "chunk_nodes_inherit_anchor_community": sum(
            1
            for _, data in graph.nodes(data=True)
            if str(data.get("node_type") or "") == "chunk" and int(data.get("community_id", -1)) >= 0
        ),
    }
    return graph, report


def add_infomap_communities_to_memory(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    from infomap import Infomap

    topology_nodes = [
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if str(data.get("node_type") or "unknown") != "chunk"
    ]
    node_to_int = {node_id: idx for idx, node_id in enumerate(topology_nodes)}
    int_to_node = {idx: node_id for node_id, idx in node_to_int.items()}

    edge_type_weight = {
        "relation": (1.0, 0.35),
        "mentioned_in": (0.8, 0.4),
        "next_turn": (0.45, 0.15),
    }
    flow_edges: dict[tuple[int, int], float] = {}
    skipped_edges = 0
    for source, target, _key, data in graph.edges(keys=True, data=True):
        source_id, target_id = str(source), str(target)
        if source_id not in node_to_int or target_id not in node_to_int:
            skipped_edges += 1
            continue
        edge_type = str(data.get("edge_type") or "unknown")
        forward_base, backward_base = edge_type_weight.get(edge_type, (0.1, 0.05))
        score = float(data.get("confidence", data.get("similarity", 1.0)) or 1.0)
        score = max(0.0, min(1.0, score))
        for left, right, base in (
            (source_id, target_id, forward_base),
            (target_id, source_id, backward_base),
        ):
            weight = float(base * score)
            if weight <= 0.0:
                continue
            key = (node_to_int[left], node_to_int[right])
            flow_edges[key] = flow_edges.get(key, 0.0) + weight

    infomap = Infomap("--directed --silent")
    for node_idx in range(len(topology_nodes)):
        infomap.add_node(node_idx)
    for (source_idx, target_idx), weight in flow_edges.items():
        infomap.add_link(source_idx, target_idx, float(weight))
    infomap.run()

    community_by_node: dict[str, int] = {}
    for node_idx, module_id in infomap.modules:
        node_id = int_to_node.get(int(node_idx))
        if node_id is not None:
            community_by_node[node_id] = int(module_id)

    raw_modules = sorted({module_id for module_id in community_by_node.values()})
    module_mapping = {module_id: idx for idx, module_id in enumerate(raw_modules)}
    community_by_node = {node_id: module_mapping[module_id] for node_id, module_id in community_by_node.items()}

    anchor_by_chunk: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type") or "") == "anchor" and data.get("chunk_id") is not None:
            anchor_by_chunk[str(data["chunk_id"])] = str(node_id)

    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type") or "unknown") == "chunk":
            anchor_id = anchor_by_chunk.get(str(node_id))
            data["community_id"] = community_by_node.get(anchor_id, -1)
        else:
            data["community_id"] = community_by_node.get(str(node_id), -1)
        data["community_algorithm"] = "infomap_directed_flow"

    for source, target, key, data in graph.edges(keys=True, data=True):
        cu = graph.nodes[source].get("community_id", -1)
        cv = graph.nodes[target].get("community_id", -1)
        data["same_community"] = int(cu == cv and cu != -1)
        data["community_pair"] = f"{cu}:{cv}"

    sizes_counter: dict[int, int] = {}
    for node_id, data in graph.nodes(data=True):
        community_id = int(data.get("community_id", -1))
        if community_id >= 0 and str(data.get("node_type") or "") != "chunk":
            sizes_counter[community_id] = sizes_counter.get(community_id, 0) + 1
    sizes = list(sizes_counter.values())
    report = {
        "algorithm": "infomap_directed_flow",
        "community_count": len(sizes_counter),
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "mean_size": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "topology_nodes": len(topology_nodes),
        "flow_edges": len(flow_edges),
        "skipped_original_edges": int(skipped_edges),
        "chunk_nodes_inherit_anchor_community": sum(
            1
            for _, data in graph.nodes(data=True)
            if str(data.get("node_type") or "") == "chunk" and int(data.get("community_id", -1)) >= 0
        ),
        "edge_type_flow_weights": {
            edge_type: {"forward": forward, "backward": backward}
            for edge_type, (forward, backward) in edge_type_weight.items()
        },
    }
    return graph, report


def build_rag_batch(graph: nx.Graph, embedding_dim: int | None = None) -> GraphBatch:
    embedding_dim = embedding_dim or infer_embedding_dim(graph)
    node_ids = [str(node_id) for node_id in graph.nodes]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    x = np.stack([as_float_embedding(graph.nodes[node_id].get("embedding"), embedding_dim) for node_id in node_ids])
    communities = [int(graph.nodes[node_id].get("community", -1)) for node_id in node_ids]

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    edge_ids: list[tuple[str, str, int | None]] = []
    for source, target, data in graph.edges(data=True):
        source_id, target_id = str(source), str(target)
        w = float(data.get("weight", 1.0) or 1.0)
        for left, right in ((source_id, target_id), (target_id, source_id)):
            src.append(node_to_idx[left])
            dst.append(node_to_idx[right])
            weights.append(w)
            edge_ids.append((left, right, None))

    return GraphBatch(
        graph_name="rag",
        x=torch.tensor(x, dtype=torch.float32),
        node_type=torch.zeros(len(node_ids), dtype=torch.long),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.zeros(len(src), dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float32).view(-1, 1),
        node_ids=node_ids,
        edge_ids=edge_ids,
        community=torch.tensor(communities, dtype=torch.long),
    )


def build_memory_batch(graph: nx.MultiDiGraph, embedding_dim: int) -> GraphBatch:
    node_ids = [str(node_id) for node_id in graph.nodes]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    chunk_embedding_by_id = {
        str(node_id): as_float_embedding(data.get("embedding"), embedding_dim)
        for node_id, data in graph.nodes(data=True)
        if str(data.get("node_type") or "unknown") == "chunk"
    }
    xs: list[np.ndarray] = []
    node_types: list[int] = []
    communities: list[int] = []
    for node_id in node_ids:
        data = graph.nodes[node_id]
        if str(data.get("node_type") or "unknown") == "anchor":
                                                                                 
                                                                         
                                                                              
                                                                                
            xs.append(chunk_embedding_by_id.get(str(data.get("chunk_id")), np.zeros(embedding_dim, dtype=np.float32)))
        else:
            xs.append(as_float_embedding(data.get("embedding"), embedding_dim))
        node_types.append(MEMORY_NODE_TYPES.get(str(data.get("node_type") or "unknown"), MEMORY_NODE_TYPES["unknown"]))
        communities.append(int(data.get("community_id", -1)))

    src: list[int] = []
    dst: list[int] = []
    edge_types: list[int] = []
    edge_ids: list[tuple[str, str, int | None]] = []
    weights: list[float] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        source_id, target_id = str(source), str(target)
        edge_type_id = MEMORY_EDGE_TYPES.get(str(data.get("edge_type") or "no_edge"), MEMORY_EDGE_TYPES["no_edge"])
        confidence = float(data.get("confidence", data.get("similarity", 1.0)) or 1.0)
        for left, right in ((source_id, target_id), (target_id, source_id)):
            src.append(node_to_idx[left])
            dst.append(node_to_idx[right])
            edge_types.append(edge_type_id)
            weights.append(confidence)
            edge_ids.append((left, right, int(key) if key is not None else None))

    return GraphBatch(
        graph_name="memory",
        x=torch.tensor(np.stack(xs).astype(np.float32), dtype=torch.float32),
        node_type=torch.tensor(node_types, dtype=torch.long),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float32).view(-1, 1),
        node_ids=node_ids,
        edge_ids=edge_ids,
        community=torch.tensor(communities, dtype=torch.long),
    )

                                          
def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value
