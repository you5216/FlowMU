                      
                                                            

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "flowmu_ga_tofu_for10_ablation_table.csv"
STEM = OUT_DIR / "flowmu_ga_tofu_for10_ablation_table"
METRICS = ("F-ACC", "R-ACC", "F-RougeL", "R-RougeL")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(rows: list[dict[str, str]]) -> None:
    data = [
        {
            "Task": row["Task"],
            **{metric: float(row[metric]) for metric in METRICS},
        }
        for row in rows
    ]
    STEM.with_suffix(".json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "| Task | F-ACC | R-ACC | F-RougeL | R-RougeL |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Task} | {F-ACC} | {R-ACC} | {F-RougeL} | {R-RougeL} |".format(**row)
        )
    STEM.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_method(name: str) -> str:
    return name.replace("&", r"\&")


def write_latex(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & F-ACC & R-ACC & F-RougeL & R-RougeL \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{task} & {f_acc} & {r_acc} & {f_rougel} & {r_rougel} \\\\".format(
                task=latex_method(row["Task"]),
                f_acc=row["F-ACC"],
                r_acc=row["R-ACC"],
                f_rougel=row["F-RougeL"],
                r_rougel=row["R-RougeL"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    STEM.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_json(rows)
    write_markdown(rows)
    write_latex(rows)
    for suffix in (".json", ".md", ".tex"):
        print(STEM.with_suffix(suffix))


if __name__ == "__main__":
    main()
