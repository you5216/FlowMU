                      
                                                          

                                                                              
                                                                               
                                                                            
                                                                          
                                                                             
                                                  
   

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import getpass
import importlib.util
import json
import os
import random
import time
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


AGENT_ROOT = Path(__file__).resolve().parents[3]
FULL_DIR = AGENT_ROOT / "final_performance/final/litellm_extraction_quality_tofu_full"
OUTPUT_DIR = AGENT_ROOT / "final_performance/final/litellm_extraction_quality_tofu_curated"
FIG_DIR = AGENT_ROOT / "final_performance/final/fig"
FULL_SCRIPT = FULL_DIR / "run_tofu_extraction_quality.py"
STEM = "litellm_extraction_quality_tofu_curated"

SAMPLE_SEED = 42
SUBSET_PLAN = {"award": 180, "work": 20}
DEEPSEEK_OUTPUT = OUTPUT_DIR / "deepseek_v4_pro_extractions.jsonl"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
TEXT_SIZE = 18.5


def load_full_module():
    spec = importlib.util.spec_from_file_location("tofu_full_extraction_quality", FULL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FULL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tofu_full_extraction_quality"] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def existing_sample_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out: set[int] = set()
    for row in load_jsonl(path):
        try:
            out.add(int(row["sample_id"]))
        except Exception:
            continue
    return out


def group_sample_ids(rows: list[dict[str, Any]], silver: dict[int, dict[str, Any]], mod: Any) -> dict[str, list[int]]:
    groups = {"award": [], "work": []}
    for row in rows:
        sample_id = int(row["sample_id"])
        question_norm = mod.norm_text(row["question"])
        _entities, rels = mod.extraction_sets(silver[sample_id].get("extraction") or {})
        predicates = {pred for _subj, pred, _obj in rels}
        if "award" in question_norm or "prize" in question_norm or "won_award" in predicates:
            groups["award"].append(sample_id)
        elif "book" in question_norm or "wrote_work" in predicates:
            groups["work"].append(sample_id)
    return groups


def select_subset(groups: dict[str, list[int]]) -> list[int]:
    rng = random.Random(SAMPLE_SEED)
    selected: list[int] = []
    for group, count in SUBSET_PLAN.items():
        candidates = list(groups[group])
        if len(candidates) < count:
            raise ValueError(f"Not enough {group} samples: need={count}, have={len(candidates)}")
        rng.shuffle(candidates)
        selected.extend(candidates[:count])
    return selected


def write_subset_files(
    rows_by_id: dict[int, dict[str, Any]],
    selected_ids: list[int],
    silver: dict[int, dict[str, Any]],
) -> None:
    with (OUTPUT_DIR / "curated_subset_manifest.jsonl").open("w", encoding="utf-8") as f:
        for sample_id in selected_ids:
            row = rows_by_id[sample_id]
            payload = {
                "sample_id": sample_id,
                "question": row["question"],
                "answer": row["answer"],
                "curated_group": "award"
                if any(key in row["question"].casefold() for key in ["award", "prize"])
                else "work",
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "curated_gold_annotations.jsonl").open("w", encoding="utf-8") as f:
        for sample_id in selected_ids:
            row = rows_by_id[sample_id]
            payload = {
                "sample_id": sample_id,
                "question": row["question"],
                "answer": row["answer"],
                "gold_source": "Qwen3.5-9B silver annotation, curated explicit relation subset",
                "extraction": silver[sample_id].get("extraction") or {},
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if sys.stdin.isatty():
        return getpass.getpass("DeepSeek API key: ").strip()
    return sys.stdin.readline().strip()


def call_deepseek(
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    retries: int,
    retry_delay: float,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise information extraction engine. Return one-line minified JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            message = obj["choices"][0]["message"]
            content = str(message.get("content") or "").strip()
            if not content:
                raise RuntimeError("DeepSeek returned empty message.content")
            return content
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:800]}")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"DeepSeek request failed after {retries + 1} attempts: {last_error}") from last_error


def parse_deepseek_json(text: str) -> Any | None:
    text = text.strip()
    if "```" in text:
        text = text.replace("```json", "```")
        if text.startswith("```"):
            text = text.split("```", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for start, ch in enumerate(text):
        if ch not in "[{":
            continue
        close = "]" if ch == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(text)):
            cur = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif cur == "\\":
                    escape = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
            elif cur == ch:
                depth += 1
            elif cur == close:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : pos + 1])
                    except Exception:
                        break
    return None


def build_deepseek_batch_prompt(rows: list[dict[str, Any]], mod: Any) -> str:
    payload = [
        {
            "sample_id": int(row["sample_id"]),
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in rows
    ]
    predicates = ", ".join(mod.PREDICATES)
    return (
        "Extract compact graph entities and relation triples for each factual QA pair.\n"
        "Return strict JSON only, no markdown and no explanation.\n"
        "The output must be one JSON object with key items.\n"
        "Each item must be: {\"sample_id\": int, \"extraction\": {\"entities\": [{\"name\": str, \"type\": str}], "
        "\"mentioned_entities\": [str], \"relations\": [{\"subject\": str, \"predicate\": str, \"object\": str, \"confidence\": float}]}}.\n"
        "Use at most 5 entities and 2 relations per QA. Entity names must be short canonical strings.\n"
        f"Predicates must be selected from: {predicates}.\n"
        "If the relation is unclear, use related_to. Include the answer as an entity when it is a name, place, date, title, role, genre, award, or concept.\n"
        "Input QA list:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_deepseek_single_prompt(row: dict[str, Any], mod: Any) -> str:
    predicates = ", ".join(mod.PREDICATES)
    return (
        "Extract entities and relation triples from this QA pair. Return JSON only.\n"
        "Schema: {\"entities\":[{\"name\":\"...\",\"type\":\"person|award|book|genre|date|place|concept|unknown\"}],"
        "\"mentioned_entities\":[\"...\"],"
        "\"relations\":[{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"confidence\":0.95}]}\n"
        "Use at most 5 entities and 2 relations. Do not include pronouns.\n"
        f"Allowed predicates: {predicates}.\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}\n"
    )


def parse_deepseek_batch(raw: str, rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    obj = parse_deepseek_json(raw)
    if isinstance(obj, dict):
        items = obj.get("items") or obj.get("extractions") or obj.get("results")
    else:
        items = obj
    if not isinstance(items, list):
        raise ValueError("DeepSeek batch response does not contain a JSON item list.")
    parsed: dict[int, dict[str, Any]] = {}
    expected = {int(row["sample_id"]) for row in rows}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            sample_id = int(item.get("sample_id"))
        except Exception:
            continue
        if sample_id not in expected:
            continue
        extraction = item.get("extraction")
        if not isinstance(extraction, dict):
            extraction = {
                "entities": item.get("entities") if isinstance(item.get("entities"), list) else [],
                "mentioned_entities": item.get("mentioned_entities")
                if isinstance(item.get("mentioned_entities"), list)
                else [],
                "relations": item.get("relations") if isinstance(item.get("relations"), list) else [],
            }
        parsed[sample_id] = extraction
    missing = expected - set(parsed)
    if missing:
        raise ValueError(f"DeepSeek batch response missing sample ids: {sorted(missing)[:8]}")
    return parsed


def call_deepseek_batch(
    api_key: str,
    api_base: str,
    model: str,
    rows: list[dict[str, Any]],
    mod: Any,
    max_tokens: int,
    retries: int,
    retry_delay: float,
) -> tuple[str, dict[int, dict[str, Any]]]:
    raw = call_deepseek(
        api_key=api_key,
        api_base=api_base,
        model=model,
        prompt=build_deepseek_batch_prompt(rows, mod),
        max_tokens=max_tokens,
        retries=retries,
        retry_delay=retry_delay,
    )
    try:
        return raw, parse_deepseek_batch(raw, rows)
    except Exception:
        debug_path = OUTPUT_DIR / "deepseek_batch_failed_raw.txt"
        debug_payload = {
            "sample_ids": [int(row["sample_id"]) for row in rows],
            "raw_completion": raw,
        }
        debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def ensure_deepseek_outputs(
    rows: list[dict[str, Any]],
    mod: Any,
    api_base: str,
    model: str,
    max_tokens: int,
    retries: int,
    retry_delay: float,
    workers: int,
    batch_size: int,
) -> None:
    if count_jsonl(DEEPSEEK_OUTPUT) == len(rows):
        print(f"[DeepSeek-v4] complete: {len(rows)}/{len(rows)}", flush=True)
        return
    api_key = read_deepseek_key()
    if not api_key:
        raise RuntimeError("Missing DeepSeek API key and deepseek_v4_pro_extractions.jsonl is incomplete.")

    done = existing_sample_ids(DEEPSEEK_OUTPUT)
    pending = [row for row in rows if int(row["sample_id"]) not in done]
    DEEPSEEK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if batch_size > 1:
        completed = len(done)
        with DEEPSEEK_OUTPUT.open("a", encoding="utf-8") as f:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                raw, parsed_by_id = call_deepseek_batch(
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    rows=batch,
                    mod=mod,
                    max_tokens=max_tokens,
                    retries=retries,
                    retry_delay=retry_delay,
                )
                for row in batch:
                    sample_id = int(row["sample_id"])
                    f.write(
                        json.dumps(
                            {
                                "sample_id": row["sample_id"],
                                "question": row["question"],
                                "answer": row["answer"],
                                "method": "DeepSeek-v4",
                                "model": model,
                                "parse_source": "deepseek_batch",
                                "raw_completion": raw,
                                "extraction": parsed_by_id[sample_id],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                f.flush()
                completed += len(batch)
                if completed % 20 == 0 or completed == len(rows):
                    print(f"[DeepSeek-v4] processed {completed}/{len(rows)}", flush=True)
        return

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        raw = call_deepseek(
            api_key=api_key,
            api_base=api_base,
            model=model,
            prompt=build_deepseek_single_prompt(row, mod),
            max_tokens=max_tokens,
            retries=retries,
            retry_delay=retry_delay,
        )
        parsed = mod.parse_json_object(raw)
        parse_source = "deepseek"
        if parsed is None:
            parsed = mod.rule_based_extract(row["question"], row["answer"])
            parse_source = "rule_fallback"
        return {
            "sample_id": row["sample_id"],
            "question": row["question"],
            "answer": row["answer"],
            "method": "DeepSeek-v4",
            "model": model,
            "parse_source": parse_source,
            "raw_completion": raw,
            "extraction": parsed,
        }

    with DEEPSEEK_OUTPUT.open("a", encoding="utf-8") as f:
        if workers <= 1:
            for offset, row in enumerate(pending, start=1):
                record = run_one(row)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                completed = len(done) + offset
                if completed % 20 == 0 or completed == len(rows):
                    print(f"[DeepSeek-v4] processed {completed}/{len(rows)}", flush=True)
            return

        completed = len(done)
        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, row) for row in pending]
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"[DeepSeek-v4] skipped failed request: {exc}", flush=True)
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                completed += 1
                if completed % 20 == 0 or completed == len(rows):
                    print(f"[DeepSeek-v4] processed {completed}/{len(rows)}", flush=True)
        if completed != len(rows):
            raise RuntimeError(f"DeepSeek output incomplete: success={completed}/{len(rows)}, failed={failed}")

def save_summary(summary: list[dict[str, Any]]) -> None:
    csv_path = OUTPUT_DIR / f"{STEM}_summary.csv"
    fieldnames = list(summary[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = [
        "| Extractor | Ent-P | Ent-R | Ent-F1 | Rel-P | Rel-R | Rel-F1 | RelPair-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        md.append(
            f"| {row['Extractor']} | {row['Ent-P']*100:.2f} | {row['Ent-R']*100:.2f} | "
            f"{row['Ent-F1']*100:.2f} | {row['Rel-P']*100:.2f} | {row['Rel-R']*100:.2f} | "
            f"{row['Rel-F1']*100:.2f} | {row['RelPair-F1']*100:.2f} |"
        )
    (OUTPUT_DIR / f"{STEM}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def plot_summary(summary: list[dict[str, Any]]) -> None:
    configure_matplotlib()
    methods = [row["Extractor"] for row in summary]
    metrics = ["Ent-F1", "Rel-F1", "RelPair-F1"]
    colors = {
        "Rule-based": "#8F9DAE",
        "LiteLLM": "#C94338",
        "Strong LLM": "#E8954C",
        "DeepSeek-v4": "#9B2148",
    }

    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    center = (len(methods) - 1) / 2.0
    fig, ax = plt.subplots(figsize=(8.6, 4.1), facecolor="white")
    ax.set_facecolor("#F0F0F7")
    for i, method in enumerate(methods):
        values = [float(summary[i][metric]) * 100.0 for metric in metrics]
        ax.bar(
            x + (i - center) * width,
            values,
            width=width,
            color=colors.get(method, "#999999"),
            edgecolor="#4B5563",
            linewidth=0.55,
            label=method,
            zorder=3,
        )
    ax.set_ylim(20, 100)
    ax.set_ylabel("F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Entity", "Relation", "Relation Pair"])
    ax.grid(True, which="major", axis="y", linestyle="--", color="gray", alpha=0.5, linewidth=0.9, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.7, color="#555555", pad=2)
    for spine in ax.spines.values():
        spine.set_alpha(0.0)
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        ncol=4,
        columnspacing=0.9,
        handlelength=1.1,
        borderpad=0.25,
    )
    leg.get_frame().set_facecolor("#F7F7FB")
    leg.get_frame().set_edgecolor("#B7BBC8")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.96)
    fig.tight_layout(pad=0.5)

    for base in (OUTPUT_DIR / STEM, FIG_DIR / STEM):
        fig.savefig(base.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.025, facecolor="white")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025, facecolor="white")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-api-base", default=os.environ.get("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_API_BASE))
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--deepseek-max-tokens", type=int, default=512)
    parser.add_argument("--deepseek-retries", type=int, default=3)
    parser.add_argument("--deepseek-retry-delay", type=float, default=2.0)
    parser.add_argument("--deepseek-workers", type=int, default=8)
    parser.add_argument("--deepseek-batch-size", type=int, default=1)
    parser.add_argument("--skip-deepseek", action="store_true")
    parser.add_argument("--use-existing-deepseek-subset", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_full_module()

    rows = load_jsonl(FULL_DIR / "sample_manifest.jsonl")
    rows_by_id = {int(row["sample_id"]): row for row in rows}
    methods = {
        "Rule-based": mod.load_extractions(FULL_DIR / "rule_based_extractions.jsonl"),
        "LiteLLM": mod.load_extractions(FULL_DIR / "litellm_qwen25_1p5b_extractions.jsonl"),
        "Strong LLM": mod.load_extractions(FULL_DIR / "strong_local_qwen2_7b_extractions.jsonl"),
    }
    silver = mod.load_extractions(FULL_DIR / "silver_qwen35_9b_extractions.jsonl")

    groups = group_sample_ids(rows, silver, mod)
    selected_ids = select_subset(groups)
    selected_rows = [rows_by_id[sample_id] for sample_id in selected_ids]
    deepseek_existing: dict[int, dict[str, Any]] = {}
    if args.use_existing_deepseek_subset:
        deepseek_existing = mod.load_extractions(DEEPSEEK_OUTPUT)
        selected_rows = [row for row in selected_rows if int(row["sample_id"]) in deepseek_existing]
        selected_ids = [int(row["sample_id"]) for row in selected_rows]
        methods["DeepSeek-v4"] = deepseek_existing
    elif not args.skip_deepseek:
        ensure_deepseek_outputs(
            selected_rows,
            mod,
            api_base=args.deepseek_api_base,
            model=args.deepseek_model,
            max_tokens=args.deepseek_max_tokens,
            retries=args.deepseek_retries,
            retry_delay=args.deepseek_retry_delay,
            workers=args.deepseek_workers,
            batch_size=args.deepseek_batch_size,
        )
        methods["DeepSeek-v4"] = mod.load_extractions(DEEPSEEK_OUTPUT)
    summary = mod.evaluate(selected_rows, methods, silver)

    write_subset_files(rows_by_id, selected_ids, silver)
    save_summary(summary)
    plot_summary(summary)

    metadata = {
        "experiment": STEM,
        "dataset": "TOFU",
        "n_samples": len(selected_ids),
        "evaluated_sample_ids": selected_ids,
        "subset_plan": SUBSET_PLAN,
        "sample_seed": SAMPLE_SEED,
        "selection_rule": "Fixed-seed sample from explicit award and work/book QA relation groups.",
        "deepseek_success_subset": args.use_existing_deepseek_subset,
        "gold_source": "Qwen3.5-9B silver annotations for the selected explicit relation subset.",
        "human_annotated": False,
        "manual_audit_note": "curated_gold_annotations.jsonl is the file to manually verify if the paper text must claim human annotation.",
        "source_full_experiment": str(FULL_DIR),
        "methods": {
            "Rule-based": "local regex/rule extractor",
            "LiteLLM": "Qwen2.5-1.5B-Instruct cached extraction",
            "Strong LLM": "Qwen2-7B-Instruct cached extraction",
            "DeepSeek-v4": f"{args.deepseek_model} extraction via DeepSeek API",
        },
        "deepseek_output": str(DEEPSEEK_OUTPUT),
        "deepseek_api_base": args.deepseek_api_base,
        "deepseek_model": args.deepseek_model,
        "deepseek_workers": args.deepseek_workers,
        "deepseek_batch_size": args.deepseek_batch_size,
        "deepseek_key_stored": False,
    }
    (OUTPUT_DIR / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shutil.copy2(OUTPUT_DIR / f"{STEM}_summary.csv", FIG_DIR / f"{STEM}_summary.csv")
    print((OUTPUT_DIR / f"{STEM}_summary.md").read_text(encoding="utf-8"))
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
