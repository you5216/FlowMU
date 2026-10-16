                      
                                        

                                                                       
                                                                           
   

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = AGENT_ROOT / "final_performance/final/cost_overhead_table_nocache"
OLD_DIR = AGENT_ROOT / "final_performance/final/cost_overhead_table"
RUN_DIR = OUT_DIR / "literaryqa_memory_fabrication_nocache"
REPORT_PATH = OUT_DIR / "reports/literaryqa_memory_fabrication_nocache.json"
BYTES_PER_MB = 1024 * 1024
LITERARYQA_N = 19061


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(AGENT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def artifact_files() -> list[Path]:
    return [
        RUN_DIR / "embedding_cache.pkl",
        RUN_DIR / "generic_memory_graph.pkl",
        RUN_DIR / "generic_memory_graph.json",
        RUN_DIR / "generic_memory_graph_report.json",
    ]


def artifact_mib_total() -> float:
    return sum(path.stat().st_size for path in artifact_files()) / BYTES_PER_MB


def fmt_time(value: float) -> str:
    return f"{value:.2f}"


def fmt_mem(value: float) -> str:
    return f"{value:.3f}"


def load_old_pivot() -> list[dict[str, Any]]:
    return read_json(OLD_DIR / "cost_overhead_table.json")


def load_old_sources() -> dict[str, Any]:
    return read_json(OLD_DIR / "cost_overhead_table_sources.json")


def replace_literaryqa_memory_row(pivot: list[dict[str, Any]], sources: dict[str, Any]) -> None:
    report = read_json(REPORT_PATH)
    wall_time_sec = float(report["wall_time_sec"])
    artifact_mib = artifact_mib_total()
    time_ms_per_sample = wall_time_sec * 1000.0 / LITERARYQA_N
    memory_mb_per_sample = artifact_mib / LITERARYQA_N

    for row in pivot:
        if row["Dataset"] == "LiteraryQA":
            row["Memory_Fabrication_Time_ms_per_sample"] = round(time_ms_per_sample, 6)
            row["Memory_Fabrication_Memory_MB_per_sample"] = round(memory_mb_per_sample, 6)
            break
    else:
        raise ValueError("LiteraryQA row not found in old pivot table")

    for row in sources["rows"]:
        if row["dataset"] == "LiteraryQA" and row["stage"] == "Memory Fabrication":
            row.update(
                {
                    "time_sec": round(wall_time_sec, 6),
                    "time_ms_per_sample": round(time_ms_per_sample, 6),
                    "memory_mb_total": round(artifact_mib, 6),
                    "memory_mb_per_sample": round(memory_mb_per_sample, 6),
                    "time_sources": [
                        {
                            "source": rel(REPORT_PATH),
                            "method": "cost monitor wall_time_sec from fresh no-cache output directory",
                            "start_time_utc": report.get("start_time_utc"),
                            "end_time_utc": report.get("end_time_utc"),
                            "gpu_peak_mem_mb": report.get("gpu_peak_mem_mb"),
                            "gpu_avg_mem_mb": report.get("gpu_avg_mem_mb"),
                        }
                    ],
                    "memory_sources": [
                        {"source": rel(path), "bytes": path.stat().st_size}
                        for path in artifact_files()
                    ],
                    "notes": (
                        "Memory graph construction rerun in a fresh output directory with no "
                        "pre-existing embedding_cache.pkl. Time is monitored wall time; memory "
                        "keeps the same offline artifact-footprint proxy as the original table."
                    ),
                }
            )
            break
    else:
        raise ValueError("LiteraryQA Memory Fabrication source row not found")

    sources["generated_by"] = rel(Path(__file__))
    sources.setdefault("protocol_notes", []).append(
        "No-cache variant: only LiteraryQA Memory Fabrication is replaced by the fresh monitored no-cache run; old 0.91 ms/sample was a cache-reuse artifact mtime span."
    )


def write_outputs(pivot: list[dict[str, Any]], sources: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "Dataset",
        "RAG_Fabrication_Time_ms_per_sample",
        "RAG_Fabrication_Memory_MB_per_sample",
        "Memory_Fabrication_Time_ms_per_sample",
        "Memory_Fabrication_Memory_MB_per_sample",
        "Agent_Unlearning_Time_ms_per_sample",
        "Agent_Unlearning_Memory_MB_per_sample",
        "Inference_Time_ms_per_sample",
        "Inference_Memory_MB_per_sample",
    ]

    with (OUT_DIR / "cost_overhead_table_nocache.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pivot)

    write_json(OUT_DIR / "cost_overhead_table_nocache.json", pivot)
    write_json(OUT_DIR / "cost_overhead_table_nocache_sources.json", sources)

    md_lines = [
        "# FlowMU Cost Overhead Table (No Cache)",
        "",
        "| Dataset | RAG Fabrication Time (ms/sample) | RAG Fabrication Memory (MB/sample) | Memory Fabrication Time (ms/sample) | Memory Fabrication Memory (MB/sample) | Agent Unlearning Time (ms/sample) | Agent Unlearning Memory (MB/sample) | Inference Time (ms/sample) | Inference Memory (MB/sample) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pivot:
        md_lines.append(
            "| {Dataset} | {rag_t} | {rag_m} | {mem_t} | {mem_m} | {agent_t} | {agent_m} | {inf_t} | {inf_m} |".format(
                Dataset=row["Dataset"],
                rag_t=fmt_time(float(row["RAG_Fabrication_Time_ms_per_sample"])),
                rag_m=fmt_mem(float(row["RAG_Fabrication_Memory_MB_per_sample"])),
                mem_t=fmt_time(float(row["Memory_Fabrication_Time_ms_per_sample"])),
                mem_m=fmt_mem(float(row["Memory_Fabrication_Memory_MB_per_sample"])),
                agent_t=fmt_time(float(row["Agent_Unlearning_Time_ms_per_sample"])),
                agent_m=fmt_mem(float(row["Agent_Unlearning_Memory_MB_per_sample"])),
                inf_t=fmt_time(float(row["Inference_Time_ms_per_sample"])),
                inf_m=fmt_mem(float(row["Inference_Memory_MB_per_sample"])),
            )
        )
    md_lines.extend(
        [
            "",
            "Table 4: Average time and memory overhead per sample of FlowMU under the no-cache graph-fabrication protocol. RAG and Memory denote graph construction costs, while Agent Unlearning and Inference report offline optimization and online serving costs, respectively.",
            "",
            "Source audit: `cost_overhead_table_nocache_sources.json`.",
        ]
    )
    (OUT_DIR / "cost_overhead_table_nocache.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\multirow{3}{*}{Dataset} & \multicolumn{2}{c}{RAG Fabrication} & \multicolumn{2}{c}{Memory Fabrication} & \multicolumn{2}{c}{Agent Unlearning} & \multicolumn{2}{c}{Inference} \\",
        r" & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} & \multicolumn{1}{c}{Time} & \multicolumn{1}{c}{Memory} \\",
        r" & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MB/sample)} & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MB/sample)} & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MB/sample)} & \multicolumn{1}{c}{(ms/sample)} & \multicolumn{1}{c}{(MB/sample)} \\",
        r"\midrule",
    ]
    for row in pivot:
        tex_lines.append(
            "{dataset} & {rag_t} & {rag_m} & {mem_t} & {mem_m} & {agent_t} & {agent_m} & {inf_t} & {inf_m} \\\\".format(
                dataset=row["Dataset"],
                rag_t=fmt_time(float(row["RAG_Fabrication_Time_ms_per_sample"])),
                rag_m=fmt_mem(float(row["RAG_Fabrication_Memory_MB_per_sample"])),
                mem_t=fmt_time(float(row["Memory_Fabrication_Time_ms_per_sample"])),
                mem_m=fmt_mem(float(row["Memory_Fabrication_Memory_MB_per_sample"])),
                agent_t=fmt_time(float(row["Agent_Unlearning_Time_ms_per_sample"])),
                agent_m=fmt_mem(float(row["Agent_Unlearning_Memory_MB_per_sample"])),
                inf_t=fmt_time(float(row["Inference_Time_ms_per_sample"])),
                inf_m=fmt_mem(float(row["Inference_Memory_MB_per_sample"])),
            )
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Average time and memory overhead per sample of FlowMU under the no-cache graph-fabrication protocol. RAG and Memory denote graph construction costs, while Agent Unlearning and Inference report offline optimization and online serving costs, respectively.}",
            r"\label{tab:flowmu_cost_overhead_nocache}",
            r"\end{table}",
        ]
    )
    (OUT_DIR / "cost_overhead_table_nocache.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    pivot = load_old_pivot()
    sources = load_old_sources()
    replace_literaryqa_memory_row(pivot, sources)
    write_outputs(pivot, sources)
    print(json.dumps(pivot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
