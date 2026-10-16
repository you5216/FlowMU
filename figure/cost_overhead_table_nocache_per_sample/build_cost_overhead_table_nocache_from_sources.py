                      
                                                                               

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCES_JSON = SCRIPT_DIR / "cost_overhead_table_nocache_sources.json"
STEM = "cost_overhead_table_nocache"
DATASETS = ["TOFU", "LiteraryQA"]
STAGES = ["RAG Fabrication", "Memory Fabrication", "Agent Unlearning", "Inference"]


def read_sources() -> dict[str, Any]:
    return json.loads(SOURCES_JSON.read_text(encoding="utf-8"))


def pivot_rows(sources: dict[str, Any]) -> list[dict[str, float | str]]:
    by_key = {(row["dataset"], row["stage"]): row for row in sources["rows"]}
    out: list[dict[str, float | str]] = []
    for dataset in DATASETS:
        item: dict[str, float | str] = {"Dataset": dataset}
        for stage in STAGES:
            row = by_key[(dataset, stage)]
            prefix = stage.replace(" ", "_")
            item[f"{prefix}_Time_ms_per_sample"] = float(row["time_ms_per_sample"])
            item[f"{prefix}_Memory_MiB_per_sample"] = float(row["memory_mb_per_sample"])
        out.append(item)
    return out


def fmt_time(value: float) -> str:
    return f"{value:.2f}"


def fmt_mem(value: float) -> str:
    return f"{value:.3f}"


def write_csv_json(rows: list[dict[str, float | str]]) -> None:
    columns = [
        "Dataset",
        "RAG_Fabrication_Time_ms_per_sample",
        "RAG_Fabrication_Memory_MiB_per_sample",
        "Memory_Fabrication_Time_ms_per_sample",
        "Memory_Fabrication_Memory_MiB_per_sample",
        "Agent_Unlearning_Time_ms_per_sample",
        "Agent_Unlearning_Memory_MiB_per_sample",
        "Inference_Time_ms_per_sample",
        "Inference_Memory_MiB_per_sample",
    ]
    with (SCRIPT_DIR / f"{STEM}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (SCRIPT_DIR / f"{STEM}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def row_values(row: dict[str, float | str]) -> dict[str, str]:
    return {
        "dataset": str(row["Dataset"]),
        "rag_t": fmt_time(float(row["RAG_Fabrication_Time_ms_per_sample"])),
        "rag_m": fmt_mem(float(row["RAG_Fabrication_Memory_MiB_per_sample"])),
        "mem_t": fmt_time(float(row["Memory_Fabrication_Time_ms_per_sample"])),
        "mem_m": fmt_mem(float(row["Memory_Fabrication_Memory_MiB_per_sample"])),
        "agent_t": fmt_time(float(row["Agent_Unlearning_Time_ms_per_sample"])),
        "agent_m": fmt_mem(float(row["Agent_Unlearning_Memory_MiB_per_sample"])),
        "inf_t": fmt_time(float(row["Inference_Time_ms_per_sample"])),
        "inf_m": fmt_mem(float(row["Inference_Memory_MiB_per_sample"])),
    }


def write_md(rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# FlowMU Cost Overhead Table (No Cache)",
        "",
        "| Dataset | RAG Fabrication Time (ms/sample) | RAG Fabrication Memory (MiB/sample) | Memory Fabrication Time (ms/sample) | Memory Fabrication Memory (MiB/sample) | Agent Unlearning Time (ms/sample) | Agent Unlearning Memory (MiB/sample) | Inference Time (ms/sample) | Inference Memory (MiB/sample) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        v = row_values(row)
        lines.append(
            "| {dataset} | {rag_t} | {rag_m} | {mem_t} | {mem_m} | {agent_t} | {agent_m} | {inf_t} | {inf_m} |".format(
                **v
            )
        )
    lines.extend(
        [
            "",
            "Table: Average time and memory overhead per sample of FlowMU under the no-cache graph-fabrication protocol.",
            "",
            "Source audit: `cost_overhead_table_nocache_sources.json`.",
        ]
    )
    (SCRIPT_DIR / f"{STEM}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(rows: list[dict[str, float | str]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\multirow{3}{*}{Dataset} & \multicolumn{2}{c}{RAG Fabrication} & \multicolumn{2}{c}{Memory Fabrication} & \multicolumn{2}{c}{Agent Unlearning} & \multicolumn{2}{c}{Inference} \\",
        r" & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} \\",
        r" & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MiB/sample)} & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MiB/sample)} & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MiB/sample)} & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MiB/sample)} \\",
        r"\midrule",
    ]
    for row in rows:
        v = row_values(row)
        lines.append(
            "{dataset} & {rag_t} & {rag_m} & {mem_t} & {mem_m} & {agent_t} & {agent_m} & {inf_t} & {inf_m} \\\\".format(
                **v
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Average time and memory overhead per sample of FlowMU under the no-cache graph-fabrication protocol. RAG and Memory denote graph construction costs, while Agent Unlearning and Inference report offline optimization and online serving costs, respectively.}",
            r"\label{tab:flowmu_cost_overhead_nocache}",
            r"\end{table}",
        ]
    )
    (SCRIPT_DIR / f"{STEM}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = pivot_rows(read_sources())
    write_csv_json(rows)
    write_md(rows)
    write_tex(rows)
    for path in [
        SCRIPT_DIR / f"{STEM}.csv",
        SCRIPT_DIR / f"{STEM}.json",
        SCRIPT_DIR / f"{STEM}.md",
        SCRIPT_DIR / f"{STEM}.tex",
    ]:
        print(path)


if __name__ == "__main__":
    main()
