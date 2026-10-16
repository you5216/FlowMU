                      
                                                                    

                                                                              
                                                                         
   

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase1.core.qwen_embeddings import build_qwen_embeddings              


DATASET_ROOT = Path("/home/likunhao/likunhao52shared/datasets/TOFU/re_split")
DEFAULT_MODEL = Path("/home/likunhao/likunhao52shared/llms/Qwen3.5-9B")
DEFAULT_BASE = Path("/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/final_performance/tofu")

SPLITS = {
    "for01_re99": ("forget01.json", "retain99.json"),
    "for05_re95": ("forget05.json", "retain95.json"),
    "for10_re90": ("forget10.json", "retain90.json"),
}


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset: {path}")
    return [row for row in data if isinstance(row, dict)]


def questions_for_split(dataset_root: Path, split_name: str) -> list[str]:
    forget_name, retain_name = SPLITS[split_name]
    rows = load_json_or_jsonl(dataset_root / forget_name) + load_json_or_jsonl(dataset_root / retain_name)
    return [str(row.get("question") or row.get("instruction") or "") for row in rows]


def hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b"\0")
        digest.update(encoded)
        digest.update(b"\0")
    return digest.hexdigest()


def expected_eval_metadata(
    texts: list[str],
    *,
    model_path: Path,
    max_length: int,
    output_cache_device: str,
) -> dict[str, Any]:
    return {
        "texts_sha256": hash_texts(texts),
        "text_count": len(texts),
        "encoder_model_path": str(model_path),
        "embedding_max_length": int(max_length),
        "embedding_hidden_layer": -1,
        "embedding_pooling": "mean",
        "embedding_4bit": bool(output_cache_device == "cuda"),
    }


def load_cache(path: Path) -> tuple[np.ndarray, dict[str, Any], list[str] | None]:
    with np.load(path, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))
        questions = None
        if "questions" in data:
            questions = [str(item) for item in data["questions"].tolist()]
    return vectors, metadata, questions


def save_cache(
    path: Path,
    *,
    vectors: np.ndarray,
    metadata: dict[str, Any],
    questions: list[str],
    source_metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "questions": np.asarray(questions, dtype=object),
    }
    if source_metadata is not None:
        payload["source_metadata_json"] = json.dumps(source_metadata, ensure_ascii=False, sort_keys=True)
    np.savez_compressed(path, **payload)
    print(f"[Save] {path} vectors={tuple(vectors.shape)} device_meta={metadata['embedding_4bit']}")


def build_full_cache(
    *,
    full_cache: Path,
    full_questions: list[str],
    model_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    expected_hash = hash_texts(full_questions)
    if full_cache.exists() and not overwrite:
        vectors, metadata, cached_questions = load_cache(full_cache)
        if metadata.get("texts_sha256") == expected_hash and int(metadata.get("text_count", -1)) == len(full_questions):
            print(f"[Full] reuse existing full cache: {full_cache}")
            return vectors, metadata, cached_questions or full_questions
        print(f"[Full] existing cache hash mismatch, rebuilding: {full_cache}")

    print(f"[Full] embedding {len(full_questions)} questions on {device}: {full_cache}")
    embedder = build_qwen_embeddings(
        model_path=str(model_path),
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        hidden_layer=-1,
        pooling="mean",
        load_in_4bit=(device == "cuda"),
    )
    vectors = np.asarray(embedder.embed_documents(full_questions), dtype=np.float32)
    metadata = expected_eval_metadata(
        full_questions,
        model_path=model_path,
        max_length=max_length,
        output_cache_device=device,
    )
    save_cache(full_cache, vectors=vectors, metadata=metadata, questions=full_questions)
    return vectors, metadata, full_questions


def build_index(questions: list[str]) -> dict[str, deque[int]]:
    index: dict[str, deque[int]] = defaultdict(deque)
    for idx, question in enumerate(questions):
        index[question].append(idx)
    return index


def derive_split_cache(
    *,
    split_name: str,
    split_questions: list[str],
    full_questions: list[str],
    full_vectors: np.ndarray,
    full_metadata: dict[str, Any],
    output_path: Path,
    model_path: Path,
    max_length: int,
    output_cache_device: str,
    overwrite: bool,
) -> None:
    expected = expected_eval_metadata(
        split_questions,
        model_path=model_path,
        max_length=max_length,
        output_cache_device=output_cache_device,
    )
    if output_path.exists() and not overwrite:
        vectors, metadata, _ = load_cache(output_path)
        if metadata == expected and len(vectors) == len(split_questions):
            print(f"[Split] reuse existing split cache: {output_path}")
            return
        print(f"[Split] stale split cache, rebuilding: {output_path}")

    index = build_index(full_questions)
    selected: list[int] = []
    missing: list[str] = []
    for question in split_questions:
        bucket = index.get(question)
        if not bucket:
            missing.append(question)
            continue
        selected.append(bucket.popleft())
    if missing:
        raise RuntimeError(f"{split_name}: {len(missing)} questions not found in full cache, first={missing[0]!r}")
    vectors = full_vectors[np.asarray(selected, dtype=np.int64)]
    save_cache(
        output_path,
        vectors=vectors,
        metadata=expected,
        questions=split_questions,
        source_metadata=full_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--full-cache", type=Path, default=DEFAULT_BASE / "base/query_embedding_cache/full_eval_for10_order_qwen35_len64_gpu_current.npz")
    parser.add_argument("--source-split", choices=sorted(SPLITS), default="for10_re90")
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=sorted(SPLITS))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-cache-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--overwrite-full", action="store_true")
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_questions = questions_for_split(args.dataset_root, args.source_split)
    full_vectors, full_metadata, cached_questions = build_full_cache(
        full_cache=args.full_cache,
        full_questions=full_questions,
        model_path=args.model_path,
        device=args.device,
        batch_size=max(1, int(args.batch_size)),
        max_length=int(args.max_length),
        overwrite=bool(args.overwrite_full),
    )
    if len(cached_questions) != len(full_vectors):
        raise RuntimeError("Full cache questions/vectors length mismatch.")

    for split_name in args.splits:
        split_questions = questions_for_split(args.dataset_root, split_name)
        output_path = args.base_dir / split_name / "cache" / f"retrieval_qwen_qvec_{split_name}_resplit_full_len64_current.npz"
        derive_split_cache(
            split_name=split_name,
            split_questions=split_questions,
            full_questions=cached_questions,
            full_vectors=full_vectors,
            full_metadata=full_metadata,
            output_path=output_path,
            model_path=args.model_path,
            max_length=int(args.max_length),
            output_cache_device=args.output_cache_device,
            overwrite=bool(args.overwrite_splits),
        )


if __name__ == "__main__":
    main()
