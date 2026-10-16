                      
                   

                                  
                              
   

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _PIPELINE_DIR.parent
_AGENT_DIR = _PHASE2_DIR.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

if __package__ in {None, ""}:
    from phase2.pipeline.bootstrap import BASE_MODEL_PATH, ORIGINAL_LORA_PATH, UNLEARNED_LORA_OUTPUT_PATH
    from phase2.pipeline.config import build_config, parse_args as parse_unlearning_args
    from phase2.pipeline.extract_key_content import (
        execute_key_content_extraction,
        parse_args as parse_key_content_args,
    )
    from phase2.pipeline.parametric_eval import (
        execute_parametric_eval,
        parse_args as parse_eval_args,
    )
    from phase2.pipeline.runner import execute_parameter_unlearning
else:
    from .bootstrap import BASE_MODEL_PATH, ORIGINAL_LORA_PATH, UNLEARNED_LORA_OUTPUT_PATH
    from .config import build_config, parse_args as parse_unlearning_args
    from .extract_key_content import execute_key_content_extraction, parse_args as parse_key_content_args
    from .parametric_eval import execute_parametric_eval, parse_args as parse_eval_args
    from .runner import execute_parameter_unlearning


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("GAT-AFRU.Phase2.Main")


def _default_eval_output_dir() -> str:
    return str(_PHASE2_DIR / "eval_runs" / "main_eval")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified Phase-2 pipeline runner")
    parser.add_argument(
        "--stage",
        default="full",
        choices=["full", "key_content", "unlearning", "eval"],
        help="Which pipeline stage(s) to run.",
    )
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--original-lora-path", default=ORIGINAL_LORA_PATH)
    parser.add_argument("--forget-target-path", default="")
    parser.add_argument("--retain-path", default="/home/likunhao/likunhao52shared/datasets/TOFU/retain90.json")
    parser.add_argument(
        "--key-output-path",
        default="",
        help=(
            "Optional key-content output/annotation path. When empty, the "
            "unlearning stage will not inject --localization-key-annotation-path."
        ),
    )
    parser.add_argument("--key-lora-path", default="")
    parser.add_argument("--key-max-forget-examples", type=int, default=0)
    parser.add_argument("--key-max-retain-examples", type=int, default=0)
    parser.add_argument("--key-max-new-tokens", type=int, default=128)
    parser.add_argument("--key-save-every", type=int, default=1)
    parser.add_argument("--key-resume", action="store_true")
    parser.add_argument("--unlearning-output-dir", default=UNLEARNED_LORA_OUTPUT_PATH)
    parser.add_argument("--skip-key-content", action="store_true")
    parser.add_argument("--skip-unlearning", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-dataset-path", default="")
    parser.add_argument("--eval-dataset-limit", type=int, default=0)
    parser.add_argument("--eval-forget-path", default="")
    parser.add_argument("--eval-retain-path", default="")
    parser.add_argument("--eval-forget-limit", type=int, default=400)
    parser.add_argument("--eval-retain-limit", type=int, default=3600)
    parser.add_argument("--eval-lora-path", default="")
    parser.add_argument("--eval-output-dir", default=_default_eval_output_dir())
    parser.add_argument("--eval-backend", default="local", choices=["local", "vllm_api"])
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args(argv)


def _should_run(stage: str, target: str, skip: bool) -> bool:
    if skip:
        return False
    return stage == "full" or stage == target


def run_key_content_stage(args: argparse.Namespace) -> None:
    key_args = [
        "--base-model-path",
        args.base_model_path,
        "--output-path",
        args.key_output_path,
        "--max-forget-examples",
        str(args.key_max_forget_examples),
        "--max-retain-examples",
        str(args.key_max_retain_examples),
        "--max-new-tokens",
        str(args.key_max_new_tokens),
        "--save-every",
        str(args.key_save_every),
    ]
    if args.key_lora_path:
        key_args += ["--lora-path", args.key_lora_path]
    if args.forget_target_path:
        key_args += ["--forget-target-path", args.forget_target_path]
    if args.retain_path:
        key_args += ["--retain-path", args.retain_path]
    if args.key_resume:
        key_args.append("--resume")
    logger.info("Running key-content extraction...")
    execute_key_content_extraction(parse_key_content_args(key_args))


def run_unlearning_stage(args: argparse.Namespace, passthrough_args: list[str]) -> None:
    unlearning_args = [
        "--base-model-path",
        args.base_model_path,
        "--original-lora-path",
        args.original_lora_path,
        "--output-dir",
        args.unlearning_output_dir,
    ]
    if args.forget_target_path:
        unlearning_args += ["--forget-target-path", args.forget_target_path]
    if args.retain_path:
        unlearning_args += ["--retain-path", args.retain_path]
    if args.key_output_path and "--localization-key-annotation-path" not in passthrough_args:
        unlearning_args += ["--localization-key-annotation-path", args.key_output_path]
    unlearning_args += passthrough_args
    logger.info("Running targeted parameter unlearning...")
    unlearning_ns = parse_unlearning_args(unlearning_args)
    cfg = build_config(unlearning_ns)
    execute_parameter_unlearning(cfg, dry_run=args.dry_run or getattr(unlearning_ns, "dry_run", False))


def run_eval_stage(args: argparse.Namespace) -> None:
    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_lora_path = args.eval_lora_path or args.original_lora_path
    eval_args = [
        "--base-model-path",
        args.base_model_path,
        "--lora-path",
        eval_lora_path,
        "--output-csv",
        str(eval_output_dir / "results.csv"),
        "--output-json",
        str(eval_output_dir / "results.json"),
        "--backend",
        args.eval_backend,
        "--max-new-tokens",
        str(args.eval_max_new_tokens),
        "--temperature",
        str(args.eval_temperature),
    ]
    if args.eval_dataset_path:
        eval_args += ["--dataset-path", args.eval_dataset_path, "--dataset-limit", str(args.eval_dataset_limit)]
    else:
        forget_path = args.eval_forget_path or args.forget_target_path
        retain_path = args.eval_retain_path or args.retain_path
        if not forget_path or not retain_path:
            raise ValueError("Eval stage requires either --eval-dataset-path or both forget/retain paths.")
        eval_args += [
            "--forget-path",
            forget_path,
            "--retain-path",
            retain_path,
            "--forget-limit",
            str(args.eval_forget_limit),
            "--retain-limit",
            str(args.eval_retain_limit),
        ]
    if args.eval_no_resume:
        eval_args.append("--no-resume")
    logger.info("Running parametric evaluation...")
    execute_parametric_eval(parse_eval_args(eval_args))


def main(argv: list[str] | None = None) -> None:
    args, passthrough_args = parse_args(argv)

    if _should_run(args.stage, "key_content", args.skip_key_content):
        run_key_content_stage(args)

    if _should_run(args.stage, "unlearning", args.skip_unlearning):
        run_unlearning_stage(args, passthrough_args)

    if _should_run(args.stage, "eval", args.skip_eval):
        run_eval_stage(args)


if __name__ == "__main__":
    main()
