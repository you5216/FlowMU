                                                 

                                  
                                     
   

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict
from pathlib import Path

from .config import Stage2Config, build_output_artifact_paths, log_plan, validate_inputs
from .masked_optimization import run_masked_parameter_optimization
from .support import build_tokenizer, load_forget_dataset, load_retain_dataset, run_localization, set_global_seed
from .stage2_unlearning import build_run_metadata, save_json, save_pickle, summarize_dataset_rows

logger = logging.getLogger("GAT-AFRU.Phase2.Runner")


def execute_parameter_unlearning(cfg: Stage2Config, dry_run: bool = False) -> None:
    logger.info("Initializing Phase-2 targeted parameter unlearning pipeline...")
    validate_inputs(cfg)
    set_global_seed(cfg.seed)
    log_plan(cfg, logger)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_output_artifact_paths(output_dir)
    if dry_run:
        logger.info("Dry run completed successfully.")
        return

    if cfg.reuse_localization_pkl is not None:
        logger.info("Reusing existing localization artifact: %s", cfg.reuse_localization_pkl)
        with cfg.reuse_localization_pkl.open("rb") as f:
            localization = pickle.load(f)
    else:
        localization = run_localization(cfg)
    save_pickle(artifact_paths["localization_pkl"], localization)
    save_json(artifact_paths["localization_json"], localization.to_serializable())

    forget_dataset = load_forget_dataset(cfg)
    retain_dataset = load_retain_dataset(cfg)
    tokenizer = build_tokenizer(cfg.base_model_path)
    extra_metadata = {
        "forget_dataset_summary": summarize_dataset_rows(forget_dataset),
        "retain_dataset_summary": summarize_dataset_rows(retain_dataset),
    }

    run_masked_parameter_optimization(
        cfg=cfg,
        localization=localization,
        tokenizer=tokenizer,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        run_metadata_extra=extra_metadata,
    )

    run_metadata = build_run_metadata(
        localization=localization,
        dataset_summary=summarize_dataset_rows(forget_dataset),
        config_summary=asdict(cfg),
    )
    run_metadata.update(extra_metadata)
    save_json(artifact_paths["run_metadata_json"], run_metadata)
    logger.info("Unlearned LoRA exported to: %s", cfg.output_dir)
