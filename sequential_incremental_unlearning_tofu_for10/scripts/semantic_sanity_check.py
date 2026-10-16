                      
                                                                               

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


CHECKS = [
    {
        "name": "arithmetic",
        "prompt": "Answer with only the final answer, no reasoning. What is 2 + 2?",
        "need_any": ["4", "four"],
    },
    {
        "name": "capital_france",
        "prompt": "Answer with only the final answer, no reasoning. What is the capital of France?",
        "need_any": ["paris"],
    },
    {
        "name": "hamlet_author",
        "prompt": "Answer with only the final answer, no reasoning. Who wrote Hamlet?",
        "need_any": ["shakespeare"],
    },
]


def chat(api_url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        api_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    results = []
    for check in CHECKS:
        raw = chat(args.api_url, args.model_name, check["prompt"])
        normalized = raw.lower()
        passed = any(token in normalized for token in check["need_any"])
        results.append({**check, "raw_prediction": raw, "prediction": raw, "passed": passed})

    report = {"all_passed": all(item["passed"] for item in results), "checks": results}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
