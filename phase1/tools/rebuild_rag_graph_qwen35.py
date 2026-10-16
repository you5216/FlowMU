               
                                                                              
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parent
AGENT_DIR = PHASE1_DIR.parent
PROJECT_ROOT = AGENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.procedure.runtime_support import get_runtime_config              
from agent.phase1.core.qwen_embeddings import build_qwen_embeddings              


def parse_args() -> argparse.Namespace:
    cfg = get_runtime_config()
    artifacts_dir = PHASE1_DIR / "artifacts" / "rag"
    parser = argparse.ArgumentParser(
        description="Rebuild phase1 rag_graph.pkl with the Qwen3.5 embedding backend."
    )
    parser.add_argument("--input-data", default=str(cfg.TOFU_FULL_JSON))
    parser.add_argument("--output-pkl", default=str(artifacts_dir / "rag_graph.pkl"))
    parser.add_argument("--output-gexf", default=str(artifacts_dir / "rag_graph.gexf"))
    parser.add_argument(
        "--metadata-path",
        default=str(artifacts_dir / "rag_graph_build_metadata.json"),
    )
    parser.add_argument("--backup-dir", default=str(artifacts_dir / "backups"))
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--device", default=str(getattr(cfg, "EMBEDDINGS_DEVICE", "cuda")))
    parser.add_argument(
        "--load-in-4bit",
        choices=["0", "1"],
        default="1" if bool(getattr(cfg, "EMBEDDINGS_LOAD_IN_4BIT", True)) else "0",
    )
    parser.add_argument("--batch-size", type=int, default=int(getattr(cfg, "EMBEDDINGS_BATCH_SIZE", 4)))
    parser.add_argument("--max-length", type=int, default=int(getattr(cfg, "EMBEDDINGS_MAX_LENGTH", 512)))
    parser.add_argument("--hidden-layer", type=int, default=int(getattr(cfg, "EMBEDDINGS_LAYER", -1)))
    parser.add_argument("--pooling", default=str(getattr(cfg, "EMBEDDINGS_POOLING", "mean")))
    parser.add_argument("--base-model-path", default=str(cfg.BASE_MODEL_PATH))
    return parser.parse_args()


def backup_if_exists(path: Path, backup_dir: Path, stamp: str) -> str | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.name}.{stamp}.bak"
    shutil.copy2(path, backup_path)
    return str(backup_path)


def load_qa_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "question": str(obj["question"]).strip(),
                    "answer": str(obj["answer"]).strip(),
                }
            )
    if not rows:
        raise ValueError(f"No QA rows found in {path}")
    return rows


def assign_communities(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0:
        return
    communities = nx_comm.louvain_communities(graph, weight="weight")              

                
    golden_ratio_conjugate = 0.618033988749895                             
    hue = 0.0
    colors: list[str] = []
    for _ in range(len(communities)):
        hue = (hue + golden_ratio_conjugate) % 1
        rgb = tuple(int(channel * 255) for channel in __import__("colorsys").hsv_to_rgb(hue, 0.8, 0.9))
        colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    for community_index, nodes in enumerate(communities):
        for node_id in nodes:
            graph.nodes[node_id]["community"] = community_index       
            graph.nodes[node_id]["cluster_color"] = colors[community_index]

                            
def build_semantic_graph(
    qa_rows: list[dict[str, str]],
    embeddings_model,
    *,
    percentile: float,
) -> tuple[nx.Graph, dict[str, float]]:
                                      
    texts = [f"Question: {row['question']}\nAnswer: {row['answer']}" for row in qa_rows]
                                   
    raw_vectors = embeddings_model.embed_documents(texts)
    node_vectors = np.asarray(raw_vectors, dtype=np.float32)
                          
    norms = np.linalg.norm(node_vectors, axis=1, keepdims=True) + 1e-10
    normalized_vectors = node_vectors / norms
    similarity_matrix = normalized_vectors @ normalized_vectors.T
    np.fill_diagonal(similarity_matrix, 1.0)
                           
    upper_values = squareform(similarity_matrix, checks=False)
    dynamic_threshold = max(float(np.percentile(upper_values, percentile)), 0.55)
                       
    distance_matrix = (1.0 - similarity_matrix).astype(np.float32, copy=False)
    np.fill_diagonal(distance_matrix, 0.0)
    mst_sparse = minimum_spanning_tree(csr_matrix(distance_matrix))
    mst_rows, mst_cols = mst_sparse.nonzero()
               
    graph = nx.Graph()
    for index, row in enumerate(qa_rows):
        graph.add_node(
            f"NODE_{index}",
            question=row["question"],
            answer=row["answer"],
            embedding=node_vectors[index],
            type="static_knowledge",
        )
                       
    for src, dst in zip(mst_rows.tolist(), mst_cols.tolist()):
        graph.add_edge(
            f"NODE_{src}",
            f"NODE_{dst}",
            weight=float(similarity_matrix[src, dst]),
        )

    shortcut_mask = np.triu(similarity_matrix >= dynamic_threshold, k=1)
    shortcut_rows, shortcut_cols = np.where(shortcut_mask)
    shortcut_count = 0
    for src, dst in zip(shortcut_rows.tolist(), shortcut_cols.tolist()):
        node_u = f"NODE_{src}"
        node_v = f"NODE_{dst}"
        if graph.has_edge(node_u, node_v):
            continue
        graph.add_edge(node_u, node_v, weight=float(similarity_matrix[src, dst]))
        shortcut_count += 1

                
    stats = {
        "dynamic_threshold": float(dynamic_threshold),
        "mst_edges": int(len(mst_rows)),
        "shortcut_edges": int(shortcut_count),
        "total_edges": int(graph.number_of_edges()),
        "node_count": int(graph.number_of_nodes()),
        "embedding_dim": int(node_vectors.shape[1]),
    }
    return graph, stats

                               
def write_gexf_without_embeddings(graph: nx.Graph, output_path: Path) -> None:
    safe_graph = graph.copy()
    for _, data in safe_graph.nodes(data=True):
        data.pop("embedding", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(safe_graph, output_path)


def main() -> None:
    args = parse_args()
    cfg = get_runtime_config()

    input_data = Path(args.input_data)
    output_pkl = Path(args.output_pkl)
    output_gexf = Path(args.output_gexf)
    metadata_path = Path(args.metadata_path)
    backup_dir = Path(args.backup_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_pkl = backup_if_exists(output_pkl, backup_dir, stamp)
    backup_gexf = backup_if_exists(output_gexf, backup_dir, stamp)

    print(f"[Rebuild] Loading QA rows from {input_data}")
    qa_rows = load_qa_rows(input_data)
    print(f"[Rebuild] Loaded {len(qa_rows)} QA rows")

    print("[Rebuild] Initializing Qwen3.5 embedding backend")
    embeddings = build_qwen_embeddings(
        model_path=args.base_model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        hidden_layer=args.hidden_layer,
        pooling=args.pooling,
        load_in_4bit=args.load_in_4bit == "1",
    )

    print("[Rebuild] Building semantic graph with Qwen3.5 embeddings")
    graph, stats = build_semantic_graph(
        qa_rows,
        embeddings,
        percentile=float(args.percentile),
    )
    print("[Rebuild] Assigning Louvain communities")
    assign_communities(graph)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        import pickle

        pickle.dump(graph, f)
    write_gexf_without_embeddings(graph, output_gexf)

    metadata = {
        "built_at_utc": stamp,
        "source_dataset": str(input_data),
        "output_pkl": str(output_pkl),
        "output_gexf": str(output_gexf),
        "backup_pkl": backup_pkl,
        "backup_gexf": backup_gexf,
        "base_model_path": str(args.base_model_path),
        "encoder_backend": "agent.phase1.core.qwen_embeddings.Qwen3EmbeddingBackend",
        "encoder_model_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel",
        "runtime_config_module": cfg.__name__,
        "device_requested": args.device,
        "load_in_4bit": args.load_in_4bit == "1",
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "hidden_layer": int(args.hidden_layer),
        "pooling": str(args.pooling),
        "percentile": float(args.percentile),
        "node_type": "static_knowledge",
        "stats": {
            **stats,
            "communities": int(len({data.get("community") for _, data in graph.nodes(data=True)})),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
