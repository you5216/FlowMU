                      
   
                                                                      
        
                                                                                                              
                                       
                               
                                          
   
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ANSWER_GEN_DIR = Path(__file__).resolve().parent
_MEMORYGENERATE_DIR = _ANSWER_GEN_DIR.parent
for _path in (_ANSWER_GEN_DIR, _MEMORYGENERATE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from .generator import StandaloneAnswerGenConfig, default_paths, generate_memory_qa
except ImportError:
    from answer_gen.generator import StandaloneAnswerGenConfig, default_paths, generate_memory_qa


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Generate answers from memory_q.json with a standalone "
            "memory + graph-rag + llm pipeline. Static retrieval comes from rag_graph.pkl."
        )
    )
    parser.add_argument("--questions-path", default=str(paths["questions_path"]))
    parser.add_argument("--static-graph-path", default=str(paths["static_graph_path"]))
    parser.add_argument("--output-path", default=str(paths["output_path"]))
    parser.add_argument("--trace-path", default=str(paths["trace_path"]))
    parser.add_argument("--memory-path", default=str(paths["memory_path"]))
    parser.add_argument("--summary-path", default=str(paths["summary_path"]))

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", choices=["0", "1"], default="1")
    parser.add_argument("--use-memory", choices=["0", "1"], default="1")
    parser.add_argument("--reset-memory", choices=["0", "1"], default="1")
    parser.add_argument(
        "--questions-per-scope",
        "--questions-per-author",
        dest="questions_per_scope",
        type=int,
        default=36,
        help="Only used to bucket multi-turn memory when question rows do not include scope_id/author_id.",
    )
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=100)

    parser.add_argument("--answer-backend", choices=["vllm_api", "local"], default="vllm_api")
    parser.add_argument("--answer-vllm-api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-vllm-api-key", default="EMPTY")
    parser.add_argument("--answer-vllm-model-name", default="")
    parser.add_argument("--base-model-path", default="")

    parser.add_argument("--embeddings-device", default="cuda")
    parser.add_argument("--embeddings-batch-size", type=int, default=4)
    parser.add_argument("--embeddings-max-length", type=int, default=512)
    parser.add_argument("--embeddings-hidden-layer", type=int, default=-1)
    parser.add_argument("--embeddings-pooling", default="mean")
    parser.add_argument("--embeddings-load-in-4bit", choices=["0", "1"], default="1")

    parser.add_argument("--llm-max-new-tokens", type=int, default=512)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_memory_qa(
        questions_path=Path(args.questions_path),
        static_graph_path=Path(args.static_graph_path),
        output_path=Path(args.output_path),
        trace_path=Path(args.trace_path),
        memory_path=Path(args.memory_path),
        backend=args.answer_backend,
        base_model_path=args.base_model_path,
        vllm_api_base=args.answer_vllm_api_base,
        vllm_api_key=args.answer_vllm_api_key,
        vllm_model_name=args.answer_vllm_model_name,
        limit=args.limit,
        resume=_as_bool(args.resume),
        config=StandaloneAnswerGenConfig(
            questions_per_scope=args.questions_per_scope,
            rag_top_k=args.rag_top_k,
            memory_top_k=args.memory_top_k,
            save_every=args.save_every,
            use_memory=_as_bool(args.use_memory),
            reset_memory=_as_bool(args.reset_memory),
            embeddings_device=args.embeddings_device,
            embeddings_batch_size=args.embeddings_batch_size,
            embeddings_max_length=args.embeddings_max_length,
            embeddings_hidden_layer=args.embeddings_hidden_layer,
            embeddings_pooling=args.embeddings_pooling,
            embeddings_load_in_4bit=_as_bool(args.embeddings_load_in_4bit),
            llm_max_new_tokens=args.llm_max_new_tokens,
            llm_temperature=args.llm_temperature,
            llm_top_p=args.llm_top_p,
        ),
    )
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
