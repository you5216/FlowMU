                      
   
                                   
                     

                                                                     
                                                                    
                                                         

                                                              
   

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import openai

                                                                             
       
                                                                             
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts" / "question_runs"

TOFU_JSON = Path("/home/likunhao/likunhao52shared/datasets/TOFU/full.json")

                                                                             
                             
                                                                             
API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))

       
MAX_RETRIES = 5
RETRY_DELAY = 10           
AUTHOR_BATCH_START = int(os.environ.get("AUTHOR_BATCH_START", "0"))
AUTHOR_BATCH_END = int(os.environ.get("AUTHOR_BATCH_END", "-1"))            

                                                                             
         
                                                                             


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

'''
这个函数负责读取原始的 JSONL 文件，并按照每 20 行归属于一个作者 (Author) 的规则，
将线性的文本数据切片并重组成结构化的字典列表。
'''
def load_authors_data(jsonl_path: Path) -> list[dict[str, Any]]:
       
                            
       
                                   
                      
                                                           
             
         
       
       
    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    batch_size = 20
    authors: list[dict[str, Any]] = []
    for batch_idx, start in enumerate(range(0, len(lines), batch_size)):
        chunk = lines[start : start + batch_size]
        if len(chunk) < batch_size:
            print(f"  ⚠  Author batch {batch_idx} has only {len(chunk)} lines, skipping")
            continue
        rag_items: list[dict[str, Any]] = []
        for i, line in enumerate(chunk):
            obj = json.loads(line)
            rag_items.append(
                {
                    "index": i + 1,
                    "question": obj["question"],
                    "answer": obj["answer"],
                }
            )
        author_id = f"author_{batch_idx:04d}"
        authors.append({"author_id": author_id, "rag_items": rag_items})
    return authors

'''
把上面那 20 条结构化的 RAG 数据，翻译并拼接成人类可读的规范文本段落。
'''
def build_rag_text(rag_items: list[dict[str, Any]]) -> str:

    lines_parts: list[str] = []
    for item in rag_items:
        idx = item["index"]
        q = item["question"]
        a = item["answer"]
        lines_parts.append(f"{idx}. Q: {q}\n   A: {a}")
    return "\n".join(lines_parts)


def build_system_prompt() -> str:
    return """你是一个专业的对话生成专家。你的任务是根据提供的20条RAG（检索增强生成）知识点，模拟一次深度对话场景，生成连续的36个递进式问题。

# 角色设定
模拟一位正在深入研究特定资料的真实用户。他思维缜密，习惯通过连续、深度的追问来彻底理解一个复杂的知识体系。

# 核心约束（必须严格遵守）
1. 输出的question必须使用英文，但你必须保证问题的内容和指代关系完全符合中文语境和逻辑。
2. **RAG全覆盖**：这36个问题必须在逻辑上自然地游走，最终尽可能覆盖全部20条RAG内容。不要遗漏核心信息。
3. **强递进性与逻辑链**：
   - 第1问：可以直接使用或改写第1个RAG知识点的问题，作为话题的切入点。
   - 后续提问：必须基于上一个问题的预期回答进行深度挖掘。提问方向可以是：询问细节、探讨原因、要求举例、对比优劣、提出极端场景、或者自然过渡到下一个RAG知识点。
4. **自然指代（极度关键）**：
   - 严禁在连续问题中像机器一样重复全称实体。
   - 必须高频使用代词和上下文指代（例如："它"、"这个方案"、"你刚说的第二点"、"那在这种情况下的缺点呢？"、"除了这个，还有其他方式吗？"）。
5. **单次连贯会话**：这是一次一气呵成的深度对谈，不需要模拟时间的跨度，保持紧凑的追问节奏。


# 输出格式
你必须严格输出一个JSON数组，包含36个元素，不要包含任何额外的解释性文本或Markdown标记。格式如下：
[
  {
    "turn": 1,
    "question": "（用户的真实提问内容）",
    "target_rag_index": [1]
  },
  ...
  {
    "turn": 36,
    "question": "（最后的总结性或收尾提问）",
    "target_rag_index": [15, 16, 17, 18, 19, 20]
  }
]

注意：
- target_rag_index 是一个整数数组，标注该问题主要覆盖了输入的哪几条RAG内容（填序号即可）。
- 36个问题的target_rag_index汇总必须覆盖1-20所有序号。
- IMPORTANT: All "question" fields MUST be written in English. Never use Chinese for the question text.
- 直接输出纯JSON数组，不要用```json```包裹，不要有任何前缀或后缀文字。"""


def build_user_prompt(rag_text: str, author_id: str) -> str:
    return f"""以下是作者 {author_id} 的20条RAG知识点：

{rag_text}

请根据以上内容，生成36个递进式对话问题。记住：
1. 第1问直接从第1个知识点的问题切入
2. 后续每个问题都要基于上一问的预期回答进行追问
3. 自然使用指代，不要重复全称
4. **ALL questions MUST be in English** — every "question" field must be written in English
5. 输出纯JSON数组，不要加```json```等标记

直接输出JSON数组："""


def call_llm(system_prompt: str, user_prompt: str) -> str:
                                                                             
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content or ""


def extract_json_array(text: str) -> list[dict[str, Any]] | None:
                                    
    text = text.strip()

                                           
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text:
        text = text.split("<think>", 1)[0].strip()

                                     
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

                      
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

                                   
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            result = json.loads(text[start_idx : end_idx + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def validate_plan(
    plan: list[dict[str, Any]], author_id: str
) -> tuple[bool, str]:
                           
    if len(plan) != 36:
        return False, f"Expected 36 turns, got {len(plan)}"

                             
    expected_turns = set(range(1, 37))
    actual_turns = set()
    all_rag_indices: set[int] = set()

    for item in plan:
        turn = item.get("turn")
        question = item.get("question", "")
        targets = item.get("target_rag_index", [])

        if turn is None:
            return False, "Missing 'turn' field"
        if not question or not isinstance(question, str):
            return False, f"Turn {turn}: missing or invalid 'question'"
        if not isinstance(targets, list):
            return False, f"Turn {turn}: 'target_rag_index' must be a list"

        actual_turns.add(turn)
        for t in targets:
            if isinstance(t, int) and 1 <= t <= 20:
                all_rag_indices.add(t)

    if actual_turns != expected_turns:
        return False, f"Missing turns: {expected_turns - actual_turns}"

    if len(all_rag_indices) < 15:
        return (
            False,
            f"Only {len(all_rag_indices)}/20 RAG items covered (need at least 15)",
        )

    return True, "OK"


def process_author(
    author: dict[str, Any],
    system_prompt: str,
    output_dir: Path,
) -> bool:
                                                                 
    author_id = author["author_id"]
    rag_items = author["rag_items"]
    rag_text = build_rag_text(rag_items)
    user_prompt = build_user_prompt(rag_text, author_id)

    print(f"\n{'='*60}")
    print(f"Processing {author_id}...")
    print(f"{'='*60}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_RETRIES}: calling LLM...")
            response_text = call_llm(system_prompt, user_prompt)
            print(f"  Response length: {len(response_text)} chars")

            plan = extract_json_array(response_text)
            if plan is None:
                print(f"  ❌ Could not extract JSON array from response")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                continue

            valid, msg = validate_plan(plan, author_id)
            if not valid:
                print(f"  ❌ Validation failed: {msg}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                continue

                  
            author_output_dir = output_dir / author_id
            ensure_dir(author_output_dir)
            output_path = author_output_dir / "question_plan.json"
            output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✅ Saved to {output_path}")
            return True

        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            continue

    print(f"  ❌ All {MAX_RETRIES} attempts failed for {author_id}")
    return False


def main() -> None:
    print("=" * 60)
    print("TOFU Question Plan Generator")
    print("=" * 60)
    print(f"Input:  {TOFU_JSON}")
    print(f"Output: {ARTIFACTS_DIR}")
    print(f"API:    {API_BASE}  model={MODEL_NAME}")

               
    print("\nLoading authors data...")
    authors = load_authors_data(TOFU_JSON)
    print(f"Loaded {len(authors)} authors")

                           
    batch_start = AUTHOR_BATCH_START
    batch_end = AUTHOR_BATCH_END if AUTHOR_BATCH_END >= 0 else len(authors)
    batch_end = min(batch_end, len(authors))
    authors_to_process = authors[batch_start:batch_end]
    print(f"Processing authors [{batch_start}:{batch_end}] ({len(authors_to_process)} authors)")

                                        
    system_prompt = build_system_prompt()

                         
    success_count = 0
    fail_count = 0
    for idx, author in enumerate(authors_to_process):
        ok = process_author(author, system_prompt, ARTIFACTS_DIR)
        if ok:
            success_count += 1
        else:
            fail_count += 1
        print(f"\n  Progress: {idx+1}/{len(authors_to_process)}  "
              f"✅ {success_count}  ❌ {fail_count}")

    print(f"\n{'='*60}")
    print(f"Done!  Success: {success_count}  Failed: {fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()