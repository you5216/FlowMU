                      
                                                                            

                                                                               
                                                                           
         
   

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "run_lqa_pure_sbu_six_rows_20260706.py"
spec = importlib.util.spec_from_file_location("lqa_six", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_SCRIPT}")
lqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lqa)


SCALES = [0.0, 0.10, 0.25, 0.50]
CKPT_STEP = 100
TARGET_FACC = 0.45


def scale_tag(scale: float) -> str:
    if abs(scale) < 1e-12:
        return "s000"
    return f"s{int(round(scale * 100)):03d}"


def copy_scaled_adapter(ratio: str, scale: float) -> Path:
    src = lqa.BASELINE_ROOT / f"experiments/literaryqa/training/sbu_for{ratio}_lora/checkpoint-{CKPT_STEP}"
    dst = lqa.OUT_ROOT / "sbu_scaled_adapter" / f"sbu_for{ratio}_ckpt{CKPT_STEP}_{scale_tag(scale)}"
    marker = dst / "scale_metadata.json"
    if marker.exists() and (dst / "adapter_model.safetensors").exists():
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if abs(float(meta.get("scale", -1)) - scale) < 1e-9:
            return dst
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "adapter_config.json",
        "README.md",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    tensors = load_file(src / "adapter_model.safetensors")
    scaled = {}
    for key, value in tensors.items():
        scaled[key] = value * scale if "lora_B" in key else value
    save_file(scaled, dst / "adapter_model.safetensors")
    marker.write_text(
        json.dumps(
            {
                "source_adapter": str(src),
                "checkpoint_step": CKPT_STEP,
                "scale": scale,
                "scale_rule": "multiply all lora_B tensors; lora_A unchanged",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return dst


def merge_scaled_adapter(ratio: str, scale: float) -> Path:
    if abs(scale) < 1e-12:
        return lqa.NATIVE_LLM
    adapter = copy_scaled_adapter(ratio, scale)
    merged = lqa.OUT_ROOT / "sbu_scaled_merged" / f"sbu_for{ratio}_ckpt{CKPT_STEP}_{scale_tag(scale)}"
    if (merged / "config.json").exists() and (merged / "model.safetensors.index.json").exists():
        return merged
    lqa.run(
        [
            str(lqa.ENV_PY),
            str(lqa.BASELINE_ROOT / "scripts/merge_lora_literaryqa.py"),
            "--base-model-path",
            str(lqa.NATIVE_LLM),
            "--lora-path",
            str(adapter),
            "--output-dir",
            str(merged),
            "--torch-dtype",
            "float16",
        ],
        cwd=lqa.BASELINE_ROOT,
        stdout=lqa.LOG_ROOT / f"sbu_for{ratio}_ckpt{CKPT_STEP}_{scale_tag(scale)}_merge.log",
    )
    return merged


def filter_results_to_forget(cell_dir: Path) -> None:
    path = cell_dir / "results.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("split") == "forget"]
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def screen_ratio(ratio: str, args: argparse.Namespace) -> tuple[float, dict, Path]:
    summary_path = lqa.OUT_ROOT / "sbu_scaled_screening_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    cfg = lqa.RATIO_CFG[ratio]
    best: tuple[float, dict, Path] | None = None
    for scale in SCALES:
        key = f"for{ratio}_ckpt{CKPT_STEP}_{scale_tag(scale)}"
        merged = merge_scaled_adapter(ratio, scale)
        cell_dir = lqa.OUT_ROOT / "sbu_scaled_screen" / key
        metrics_path = cell_dir / "final_metrics/final_metrics_summary.json"
        if key in summary and metrics_path.exists():
            metrics = summary[key]
        else:
            label = f"sbu_for{ratio}_ckpt{CKPT_STEP}_{scale_tag(scale)}_screen"
            lqa.run_sbu_step6(
                ratio=ratio,
                model_path=merged,
                cell_dir=cell_dir,
                label=label,
                forget_limit=cfg["fl"],
                retain_path=lqa.EMPTY_RETAIN_JSON,
                retain_limit=1,
                require_complete=cfg["fl"],
                num_workers=args.num_workers,
            )
            filter_results_to_forget(cell_dir)
            lqa.run_judge_and_metrics(cell_dir, label, num_workers=args.judge_workers)
            metrics = lqa.load_metrics(metrics_path)
            summary[key] = {
                "ratio": ratio,
                "scale": scale,
                "model_path": str(merged),
                "forget_acc": metrics["forget_acc"],
                "retain_acc": metrics["retain_acc"],
                "forget_rouge_l": metrics["forget_rouge_l"],
                "retain_rouge_l": metrics["retain_rouge_l"],
            }
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lqa.log(f"SBU scaled screen ratio={ratio} scale={scale:.2f} F-ACC={metrics['forget_acc']:.4f}")
        candidate = (scale, metrics, merged)
        if best is None or abs(metrics["forget_acc"] - TARGET_FACC) < abs(best[1]["forget_acc"] - TARGET_FACC):
            best = candidate
        if metrics["forget_acc"] >= TARGET_FACC:
            return candidate
    assert best is not None
    return best


def run_scaled_sbu(args: argparse.Namespace) -> None:
    chosen = {}
    for ratio, cfg in lqa.RATIO_CFG.items():
        scale, screen_metrics, model_path = screen_ratio(ratio, args)
        chosen[ratio] = {"scale": scale, "screen": screen_metrics, "model_path": str(model_path)}
        cell_dir = lqa.OUT_ROOT / "sbu_scaled_final" / f"sbu_for{ratio}"
        label = f"sbu_for{ratio}_ckpt{CKPT_STEP}_{scale_tag(scale)}"
        lqa.run_sbu_step6(
            ratio=ratio,
            model_path=model_path,
            cell_dir=cell_dir,
            label=label,
            forget_limit=cfg["fl"],
            retain_path=cfg["retain"],
            retain_limit=cfg["rl"],
            require_complete=cfg["fl"] + cfg["rl"],
            num_workers=args.num_workers,
        )
        lqa.run_judge_and_metrics(cell_dir, f"sbu_for{ratio}", num_workers=args.judge_workers)
    (lqa.OUT_ROOT / "sbu_scaled_chosen.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_table() -> None:
    rows = []
    for method, root in [
        ("PURE", lqa.OUT_ROOT / "pure_final"),
        ("SBU", lqa.OUT_ROOT / "sbu_scaled_final"),
    ]:
        for ratio in ["01", "05", "10"]:
            path = root / f"{method.lower()}_for{ratio}" / "final_metrics/final_metrics_summary.json"
            if method == "SBU":
                path = root / f"sbu_for{ratio}" / "final_metrics/final_metrics_summary.json"
            if not path.exists():
                continue
            m = lqa.load_metrics(path)
            rows.append(
                {
                    "method": method,
                    "forget_ratio": ratio,
                    "forget_acc": m["forget_acc"],
                    "retain_acc": m["retain_acc"],
                    "forget_rouge_l": m["forget_rouge_l"],
                    "retain_rouge_l": m["retain_rouge_l"],
                    "metrics_path": str(path),
                }
            )
    out = lqa.OUT_ROOT / "final_six_rows_metrics_scaled_sbu.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    lqa.log(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sbu", "table"], default="sbu")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=16)
    args = ap.parse_args()
    lqa.check_paths()
    if args.stage == "sbu":
        run_scaled_sbu(args)
        write_table()
    else:
        write_table()


if __name__ == "__main__":
    main()
