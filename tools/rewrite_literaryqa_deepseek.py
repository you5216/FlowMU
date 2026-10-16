                      
                                                                          

                                                                              
                                                                             
                              

                                                  
                                    
   

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ANNOTATION_DIR = Path(
    "/home/likunhao/likunhao52shared/datasets/LiteraryQA/literaryqa_annotations"
)
DEFAULT_INPUT_DIR = DEFAULT_ANNOTATION_DIR
DEFAULT_MEMORY_DIR = DEFAULT_ANNOTATION_DIR / "memory"
DEFAULT_RAG_DIR = DEFAULT_ANNOTATION_DIR / "rag"
DEFAULT_OUTPUT_DIR = DEFAULT_RAG_DIR
DEFAULT_PROGRESS = DEFAULT_RAG_DIR / "progress.jsonl"
DEFAULT_BOOK_OUTPUT = DEFAULT_RAG_DIR / "books.jsonl"
DEFAULT_FULL_OUTPUT = DEFAULT_RAG_DIR / "full.json"
DEFAULT_MANIFEST = DEFAULT_MEMORY_DIR / "split_manifest_seed42.json"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
SPLITS = ("train", "validation", "test")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def load_documents(input_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for split in SPLITS:
        path = input_dir / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc["_source_split"] = split
                docs.append(doc)

    seen: set[str] = set()
    duplicates: list[str] = []
    for doc in docs:
        doc_id = doc["document_id"]
        if doc_id in seen:
            duplicates.append(doc_id)
        seen.add(doc_id)
    if duplicates:
        raise ValueError(f"Duplicate document_id values found: {duplicates[:10]}")
    return docs


def first_answer_records(doc: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, qa in enumerate(doc.get("qas", [])):
        answers = qa.get("answers", [])
        if not answers:
            continue
        records.append(
            {
                "id": idx,
                "question": normalize_space(qa.get("question")),
                "answer": normalize_space(answers[0]),
            }
        )
    return records


def build_messages(
    *,
    title: str,
    summary: str,
    metadata: dict[str, Any],
    qas: list[dict[str, Any]],
    summary_chars: int,
) -> list[dict[str, str]]:
    compact_qas = [
        {"id": item["id"], "question": item["question"], "answer": item["answer"]}
        for item in qas
    ]
    genre_tags = metadata.get("genre_tags", "")
    publication_date = metadata.get("publication_date", "")
    system = (
        "You are an expert dataset editor for retrieval-augmented generation (RAG). "
        "Rewrite literary QA pairs so each pair is standalone and understandable without "
        "seeing the book, summary, or neighboring QA pairs. Preserve the original meaning."
    )
    user = (
        "Rewrite every QA pair below into a standalone, RAG-friendly QA pair.\n\n"
        "Book context:\n"
        f"- Title: {title}\n"
        f"- Publication date: {publication_date}\n"
        f"- Genre tags: {genre_tags}\n"
        f"- Summary: {truncate_text(summary, summary_chars)}\n\n"
        "Rules:\n"
        "1. Return the same number of QA pairs, in the same order, with the same id values.\n"
        "2. The rewritten question must be self-contained. Replace unclear pronouns and vague references "
        "such as he, she, they, this story, the novel, her brother, the character, or the protagonist with "
        "explicit names, roles, events, and/or the book title whenever the context makes that clear.\n"
        "3. Prefer questions like \"In <book title>, who/what/why ...\" when the original question depends on book context.\n"
        "4. The rewritten answer should also be self-contained. If the original answer is only a name or short phrase, "
        "turn it into a concise complete answer using the rewritten question's subject.\n"
        "5. Do not invent facts. Do not add facts that are not supported by the original QA pair or the book summary.\n"
        "6. Preserve the original answer's meaning. If the context is ambiguous, keep the answer conservative and include the book title rather than guessing.\n"
        "7. Use clear English. Keep each question and answer concise.\n"
        "8. Output only valid JSON. No markdown, no explanations.\n\n"
        "Example style:\n"
        "Original: {\"question\":\"Who does he marry?\",\"answer\":\"Elizabeth\"}\n"
        "Good rewrite: {\"question\":\"In Pride and Prejudice, who does Fitzwilliam Darcy marry?\","
        "\"answer\":\"In Pride and Prejudice, Fitzwilliam Darcy marries Elizabeth Bennet.\"}\n\n"
        "Input QA pairs:\n"
        f"{json.dumps(compact_qas, ensure_ascii=False)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\"qas\":[{\"id\":0,\"question\":\"...\",\"answer\":\"...\"}]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("DeepSeek response is not a JSON object")
    return obj


def validate_rewrite(original_qas: list[dict[str, Any]], response_obj: dict[str, Any]) -> list[dict[str, Any]]:
    rewritten = response_obj.get("qas")
    if not isinstance(rewritten, list):
        raise ValueError("DeepSeek response missing list field: qas")
    if len(rewritten) != len(original_qas):
        raise ValueError(f"Expected {len(original_qas)} qas, got {len(rewritten)}")

    clean: list[dict[str, Any]] = []
    expected_ids = [item["id"] for item in original_qas]
    got_ids = []
    for item in rewritten:
        if not isinstance(item, dict):
            raise ValueError("A rewritten QA item is not an object")
        item_id = item.get("id")
        got_ids.append(item_id)
        question = normalize_space(item.get("question"))
        answer = normalize_space(item.get("answer"))
        if not question or not answer:
            raise ValueError(f"Empty rewritten question/answer for id={item_id}")
        clean.append({"id": item_id, "question": question, "answer": answer})
    if got_ids != expected_ids:
        raise ValueError(f"QA ids/order mismatch. Expected {expected_ids}, got {got_ids}")
    return clean


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    return extract_json_object(content)


def rewrite_book(
    *,
    api_key: str,
    base_url: str,
    model: str,
    doc: dict[str, Any],
    summary_chars: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    original_qas = first_answer_records(doc)
    if not original_qas:
        raise ValueError(f"Document has no first-answer QA records: {doc['document_id']}")
    messages = build_messages(
        title=doc.get("title", ""),
        summary=doc.get("summary", ""),
        metadata=doc.get("metadata", {}),
        qas=original_qas,
        summary_chars=summary_chars,
    )

    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            response_obj = call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            rewritten_qas = validate_rewrite(original_qas, response_obj)
            return {
                "document_id": doc["document_id"],
                "gutenberg_id": doc["gutenberg_id"],
                "title": doc["title"],
                "source_split": doc["_source_split"],
                "qa_count": len(rewritten_qas),
                "qas": rewritten_qas,
            }
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt > retries:
                break
            sleep_s = retry_sleep * attempt
            print(
                f"[Retry] title={doc.get('title')} attempt={attempt}/{retries} "
                f"sleep={sleep_s:.1f}s error={exc}",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"Failed to rewrite document_id={doc['document_id']} title={doc.get('title')}: {last_error}"
    ) from last_error


def load_completed(progress_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not progress_path.exists():
        return completed
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[row["document_id"]] = row
    return completed


def append_progress(progress_path: Path, row: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def flatten_rewritten_docs(rewritten_docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for doc in rewritten_docs:
        for qa in doc["qas"]:
            rows.append({"question": qa["question"], "answer": qa["answer"]})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def finalize_outputs(
    *,
    docs: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    output_dir: Path,
    book_output: Path,
    full_output: Path,
    manifest_path: Path,
) -> None:
    missing = [doc for doc in docs if doc["document_id"] not in completed]
    if missing:
        print(
            f"[Finalize skipped] completed={len(completed)}/{len(docs)} "
            f"missing={len(missing)} first_missing={missing[0]['title']}",
            flush=True,
        )
        return

    rewritten_docs = [completed[doc["document_id"]] for doc in docs]
    write_jsonl(book_output, rewritten_docs)
    write_jsonl(full_output, flatten_rewritten_docs(rewritten_docs))
    print(f"[Wrote] {full_output}", flush=True)
    print(f"[Wrote] {book_output}", flush=True)

    if not manifest_path.exists():
        print(f"[Split skipped] missing manifest: {manifest_path}", flush=True)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc_by_id = {doc["document_id"]: doc for doc in rewritten_docs}
    all_ids = set(doc_by_id)
    for pct_text, forget_books in manifest["forget_books"].items():
        percent = int(pct_text)
        forget_ids = {item["document_id"] for item in forget_books}
        unknown_ids = forget_ids - all_ids
        if unknown_ids:
            raise ValueError(f"Manifest contains unknown document ids: {sorted(unknown_ids)[:5]}")
        retain_ids = all_ids - forget_ids

        forget_docs = [doc for doc in rewritten_docs if doc["document_id"] in forget_ids]
        retain_docs = [doc for doc in rewritten_docs if doc["document_id"] in retain_ids]
        forget_path = output_dir / f"forget{percent:02d}.json"
        retain_path = output_dir / f"retain{100 - percent:02d}.json"
        write_jsonl(forget_path, flatten_rewritten_docs(forget_docs))
        write_jsonl(retain_path, flatten_rewritten_docs(retain_docs))
        print(
            f"[Wrote] {forget_path.name}: {sum(len(d['qas']) for d in forget_docs)} QA; "
            f"{retain_path.name}: {sum(len(d['qas']) for d in retain_docs)} QA",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--book-output", type=Path, default=DEFAULT_BOOK_OUTPUT)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--summary-chars", type=int, default=5000)
    parser.add_argument("--start-book", type=int, default=0)
    parser.add_argument("--limit-books", type=int, default=0)
    parser.add_argument("--only-document-id", default="")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite-progress", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--dry-run-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_documents(args.input_dir)

    if args.overwrite_progress and args.progress.exists():
        args.progress.unlink()
    completed = load_completed(args.progress)

    if args.finalize_only:
        finalize_outputs(
            docs=docs,
            completed=completed,
            output_dir=args.output_dir,
            book_output=args.book_output,
            full_output=args.full_output,
            manifest_path=args.manifest,
        )
        return

    selected_docs = docs
    if args.only_document_id:
        selected_docs = [doc for doc in docs if doc["document_id"] == args.only_document_id]
        if not selected_docs:
            raise ValueError(f"Unknown document_id: {args.only_document_id}")
    else:
        end = len(docs) if args.limit_books <= 0 else min(len(docs), args.start_book + args.limit_books)
        selected_docs = docs[args.start_book:end]

    if args.dry_run_prompt:
        doc = selected_docs[0]
        messages = build_messages(
            title=doc.get("title", ""),
            summary=doc.get("summary", ""),
            metadata=doc.get("metadata", {}),
            qas=first_answer_records(doc),
            summary_chars=args.summary_chars,
        )
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set environment variable {args.api_key_env}.")

    todo = [doc for doc in selected_docs if doc["document_id"] not in completed]
    print(
        f"[Start] docs={len(docs)} completed={len(completed)} selected={len(selected_docs)} "
        f"todo={len(todo)} model={args.model}",
        flush=True,
    )

    for pos, doc in enumerate(todo, start=1):
        rewritten = rewrite_book(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            doc=doc,
            summary_chars=args.summary_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        append_progress(args.progress, rewritten)
        completed[rewritten["document_id"]] = rewritten
        print(
            f"[Done] {pos}/{len(todo)} title={rewritten['title']} qa={rewritten['qa_count']} "
            f"completed={len(completed)}/{len(docs)}",
            flush=True,
        )
        if args.delay > 0:
            time.sleep(args.delay)

    if not args.skip_finalize:
        finalize_outputs(
            docs=docs,
            completed=completed,
            output_dir=args.output_dir,
            book_output=args.book_output,
            full_output=args.full_output,
            manifest_path=args.manifest,
        )


if __name__ == "__main__":
    main()
