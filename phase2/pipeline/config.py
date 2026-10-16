                            

                                
                                    
   

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from .bootstrap import (
    BASE_MODEL_PATH,
    DEFAULT_CONTEXT_DATASET_PATH,
    DEFAULT_CORE_DATASET_PATH,
    DEFAULT_GRAPH_PATH,
    DEFAULT_LOCALIZATION_JSON,
    DEFAULT_LOCALIZATION_PKL,
    DEFAULT_RETAIN_PATH,
    DEFAULT_RUN_METADATA_JSON,
    ORIGINAL_LORA_PATH,
    UNLEARNED_LORA_OUTPUT_PATH,
)


@dataclass
class Stage2Config:
    base_model_path: str
    original_lora_path: str
    output_dir: str
    graph_path: Path
    forget_target_path: Optional[Path]
    retain_path: Path
    core_dataset_path: Path
    context_dataset_path: Path
    init_new_lora: bool = False
    init_new_lora_on_frozen: bool = False
    new_lora_r: int = 16
    new_lora_alpha: int = 32
    new_lora_dropout: float = 0.05
    new_lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    seed: int = 42
    learning_rate: float = 1e-4
    num_train_epochs: float = 5.0
    max_train_steps: int = 0
    resume_from_checkpoint: Optional[Path] = None
    logging_steps: int = 5
    max_length: int = 512
    warmup_steps: int = 5
    max_grad_norm: float = 0.3
    top_fraction: float = 0.01
    coarse_top_layers: int = 4
    per_layer_top_neurons: int = 30
    manual_layer_whitelist: Optional[list[int]] = None
    reuse_localization_pkl: Optional[Path] = None
    localization_batch_size: int = 1
    localization_forget_samples: int = 64
    localization_retain_samples: int = 64
    localization_strategy: str = "author_firing_intersection"
    localization_layer_start: int = 0
    localization_layer_end: int = -1
    localization_target_token: str = "keyword_answer"
    activation_fired_threshold: float = 0.1
    forget_author_bucket_size: int = 20
    forget_firing_min_ratio: float = 0.8
    retain_firing_max_ratio: float = 0.05
    author_bucket_top_neurons: int = 20
    contrastive_lambda: float = 1.2
    author_abs_weight: float = 1.0
    author_posneg_weight: float = 0.5
    localization_gradient_checkpointing: bool = True
    localization_key_annotation_path: Optional[Path] = None
    localization_key_token_min_char_len: int = 2
    activation_signed_weight: float = 1.0
    activation_abs_weight: float = 0.25
    localization_verify_enabled: bool = True
    localization_verify_forget_samples: int = 8
    localization_verify_retain_samples: int = 8
    localization_verify_retain_penalty: float = 1.0
    verification_top_neurons_per_layer: int = 20
    final_top_neurons_per_layer: int = 10
    verification_score_std_multiplier: float = 0.5
    verification_min_score: float = 0.002
    relaxed_layer_max_index: int = 20
    relaxed_final_top_neurons_per_layer: int = 14
    relaxed_verification_score_std_multiplier: float = 0.25
    relaxed_verification_min_score: float = 0.0015
    global_max_target_neurons: int = 256
    localization_verify_scale_factor: float = 0.5
    disable_parameter_mask: bool = False
    safe_response: str = "I do not have that information."
    forget_loss_strategy: str = "parameter_random_vector"
    forget_objective_weight: float = 1.0
    forget_target_mode: str = "random_vector"
    forget_ce_weight: float = 0.0
    answer_suppression_path: Optional[Path] = None
    answer_suppression_weight: float = 0.0
    random_vector_forget_weight: float = 1.0
    forget_anchor_weight: float = 1.0
    forget_anchor_max_similarity: float = 0.0
    use_adaptive_forget_objective: bool = True
    random_vector_target_similarity: float = 0.25
    forget_anchor_target_similarity: float = -0.10
    forget_weight_lr: float = 0.05
    forget_weight_min: float = 0.0
    forget_weight_max: float = 12.0
    random_vector_seed: int = 42
    parameter_random_vector_weight: float = 1.0
    parameter_random_vector_scale: float = 0.01
    forget_loss_top_layers: int = 8
    graph_vector_suppress_weight: float = 0.0
    graph_vector_objective: str = "suppress"
    graph_vector_random_weight: float = 1.0
    graph_vector_away_weight: float = 1.0
    graph_vector_keep_weight: float = 0.0
    graph_vector_away_max_similarity: float = 0.0
    graph_vector_keep_max_items: int = 0
    graph_vector_rag_graph_path: Optional[Path] = None
    graph_vector_memory_graph_path: Optional[Path] = None
    graph_vector_batch_size: int = 4
    graph_vector_max_items: int = 0
    graph_vector_anchor_text: str = "Graph concept probe."
    entity_max_neighbors: int = 2
    entity_max_cluster_facts: int = 4
    entity_view_direct_weight: float = 1.0
    entity_view_context_weight: float = 1.0
    entity_view_cluster_weight: float = 1.0
    repr_forget_weight: float = 0.0
    repr_cluster_weight: float = 0.0
    repr_target_max_similarity: float = 0.25
    repr_distance_weight: float = 0.0
    repr_layerwise_weight: float = 0.0
    retain_ce_weight: float = 0.0
    retain_repr_weight: float = 0.0
    retain_kl_weight: float = 1.0
    use_dual_kl_objective: bool = True
    retain_kl_target: float = 0.02
    dual_kl_lr: float = 0.05
    dual_kl_weight_init: float = 1.0
    dual_kl_weight_min: float = 0.2
    retain_topk: int = 32
    forget_nll_calibration_weight: float = 0.0
    forget_nll_calibration_margin: float = 0.75
    forget_nll_calibration_fixed_target: float = 1.0
    forget_nll_calibration_target_mode: str = "retain_ce"
    early_stop_enabled: bool = True
    early_stop_min_steps: int = 200
    early_stop_patience: int = 200
    best_score_retain_alpha: float = 2.0
    best_score_mia_alpha: float = 0.0
    best_retain_kl_max: float = -1.0
    best_retain_kl_fallback_max: float = -1.0
    canonical_proxy_enabled: bool = False
    canonical_proxy_forget_path: Optional[Path] = None
    canonical_proxy_retain_path: Optional[Path] = None
    canonical_proxy_forget_samples: int = 80
    canonical_proxy_retain_samples: int = 200
    canonical_proxy_interval: int = 100
    canonical_proxy_batch_size: int = 2
    canonical_proxy_retain_nll_delta_max: float = 0.30
    canonical_proxy_retain_kl_max: float = 0.05
    canonical_proxy_forget_nll_weight: float = 1.0
    canonical_proxy_forget_logprob_drop_weight: float = 0.5
    canonical_proxy_retain_nll_penalty: float = 4.0
    canonical_proxy_retain_kl_penalty: float = 8.0
    canonical_proxy_hard_retain_filter: bool = True
    forget_ce_checkpoint_thresholds: Optional[list[float]] = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 hierarchical targeted parameter unlearning")
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--original-lora-path", default=ORIGINAL_LORA_PATH)
    parser.add_argument("--output-dir", default=UNLEARNED_LORA_OUTPUT_PATH)
    parser.add_argument(
        "--init-new-lora",
        action="store_true",
        help="Initialize a fresh unlearning LoRA on --base-model-path instead of loading --original-lora-path.",
    )
    parser.add_argument("--new-lora-r", type=int, default=16)
    parser.add_argument("--new-lora-alpha", type=int, default=32)
    parser.add_argument("--new-lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--new-lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names used when --init-new-lora creates a fresh adapter.",
    )
    parser.add_argument("--graph-path", default=str(DEFAULT_GRAPH_PATH))
    parser.add_argument("--forget-target-path", default=None)
    parser.add_argument("--retain-path", default=str(DEFAULT_RETAIN_PATH))
    parser.add_argument("--core-dataset-path", default=str(DEFAULT_CORE_DATASET_PATH))
    parser.add_argument("--context-dataset-path", default=str(DEFAULT_CONTEXT_DATASET_PATH))
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optional hard cap on optimizer steps while keeping the scheduler length from --num-train-epochs.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help=(
            "Resume GA/RMU-style masked optimization from a LoRA checkpoint. "
            "If checkpoint/training_state.pt exists, optimizer/scheduler/global_step are restored too."
        ),
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--coarse-top-layers", type=int, default=4)
    parser.add_argument(
        "--per-layer-top-neurons",
        type=int,
        default=30,
        help="For the final localization path, keep at most this many coarse candidate neurons per layer before perturbation verification.",
    )
    parser.add_argument(
        "--reuse-localization-pkl",
        default="",
        help="Optional path to an existing localized_neurons.pkl. When set, Phase-2 skips localization and reuses this artifact.",
    )
    parser.add_argument(
        "--manual-layer-whitelist",
        default="",
        help="Optional comma-separated decoder layer indices to force as localization candidates, e.g. 3 or 3,11",
    )
    parser.add_argument("--localization-batch-size", type=int, default=1)
    parser.add_argument("--localization-forget-samples", type=int, default=64)
    parser.add_argument("--localization-retain-samples", type=int, default=64)
    parser.add_argument(
        "--localization-strategy",
        default="author_firing_intersection",
        choices=["author_firing_intersection", "activation_contrast"],
        help="Final localization method. author_firing_intersection follows the 20-sample author bucket SOP.",
    )
    parser.add_argument(
        "--localization-layer-start",
        type=int,
        default=0,
        help="Inclusive decoder layer start index for localization. Default 0 means start from the first layer.",
    )
    parser.add_argument(
        "--localization-layer-end",
        type=int,
        default=-1,
        help="Inclusive decoder layer end index for localization. Default -1 means automatically use the last available decoder layer.",
    )
    parser.add_argument(
        "--localization-target-token",
        default="keyword_answer",
        choices=["keyword_answer", "raw_qa"],
        help="Localization token-selection mode: keyword_answer only aggregates answer tokens matched from key phrases; raw_qa aggregates the full question+answer span.",
    )
    parser.add_argument("--activation-fired-threshold", type=float, default=0.01)
    parser.add_argument("--forget-author-bucket-size", type=int, default=20)
    parser.add_argument("--forget-firing-min-ratio", type=float, default=0.8)
    parser.add_argument("--retain-firing-max-ratio", type=float, default=0.05)
    parser.add_argument("--author-bucket-top-neurons", type=int, default=20)
    parser.add_argument("--contrastive-lambda", type=float, default=1.2)
    parser.add_argument("--author-abs-weight", type=float, default=1.0)
    parser.add_argument("--author-posneg-weight", type=float, default=0.5)
    parser.add_argument(
        "--localization-key-annotation-path",
        default="",
        help="Optional JSON file produced by extract_key_content.py. When provided, localization prefers its key phrases for answer-token masking.",
    )
    parser.add_argument(
        "--disable-localization-key-token-only",
        action="store_true",
        help="Deprecated compatibility flag. Use --localization-target-token raw_qa for full-span localization.",
    )
    parser.add_argument("--localization-key-token-min-char-len", type=int, default=2)
    parser.add_argument("--activation-signed-weight", type=float, default=1.0)
    parser.add_argument("--activation-abs-weight", type=float, default=0.25)
    parser.add_argument(
        "--disable-localization-verify",
        action="store_true",
        help="Disable perturbation-based neuron verification after coarse activation screening.",
    )
    parser.add_argument("--localization-verify-forget-samples", type=int, default=8)
    parser.add_argument("--localization-verify-retain-samples", type=int, default=8)
    parser.add_argument("--localization-verify-retain-penalty", type=float, default=1.0)
    parser.add_argument("--verification-top-neurons-per-layer", type=int, default=20)
    parser.add_argument(
        "--final-top-neurons-per-layer",
        type=int,
        default=10,
        help="After perturbation verification, keep at most this many rows per layer before global pruning.",
    )
    parser.add_argument(
        "--verification-score-std-multiplier",
        type=float,
        default=0.5,
        help="Layerwise verification threshold = max(min_score, mean + std_multiplier * std).",
    )
    parser.add_argument(
        "--verification-min-score",
        type=float,
        default=0.002,
        help="Absolute minimum verification score required after perturbation screening.",
    )
    parser.add_argument(
        "--relaxed-layer-max-index",
        type=int,
        default=20,
        help="Layers with index <= this value use relaxed final verification filtering.",
    )
    parser.add_argument(
        "--relaxed-final-top-neurons-per-layer",
        type=int,
        default=14,
        help="Per-layer final cap for relaxed early/mid layers.",
    )
    parser.add_argument(
        "--relaxed-verification-score-std-multiplier",
        type=float,
        default=0.25,
        help="Relaxed layerwise std multiplier for early/mid layers.",
    )
    parser.add_argument(
        "--relaxed-verification-min-score",
        type=float,
        default=0.0015,
        help="Relaxed absolute verification score floor for early/mid layers.",
    )
    parser.add_argument(
        "--global-max-target-neurons",
        type=int,
        default=256,
        help="Global cap on the final verified neuron count after layerwise filtering.",
    )
    parser.add_argument("--localization-verify-scale-factor", type=float, default=0.5)
    parser.add_argument(
        "--disable-parameter-mask",
        action="store_true",
        help="Ablation switch: update all trainable LoRA parameters instead of only localized masked parameters.",
    )
    parser.add_argument(
        "--disable-localization-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing during localization. Enabled by default to reduce activation memory.",
    )
    parser.add_argument("--safe-response", default="I do not have that information.")
    parser.add_argument(
        "--forget-loss-strategy",
        default="parameter_random_vector",
        choices=[
            "gradient_ascent",
            "gradient_ascent_projected",
            "representation_erasure",
            "random_vector",
            "parameter_random_vector",
            "random_answer_ce",
        ],
        help="Forget-side objective: token-level negative CE, feature-level zero-vector erasure, hidden-state random-vector attraction, masked-parameter random-vector attraction, or random-answer token CE.",
    )
    parser.add_argument("--forget-objective-weight", type=float, default=1.0)
    parser.add_argument(
        "--forget-target-mode",
        default="random_vector",
        choices=["safe_response", "random_vector"],
        help="Forget-side target type. Final method uses random_vector; safe_response is kept only as a legacy reference path.",
    )
    parser.add_argument("--forget-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--answer-suppression-path",
        default="",
        help=(
            "Optional graph-derived QA JSON/JSONL for a separate answer-suppression loss. "
            "When set with --answer-suppression-weight > 0, its answer CE is maximized."
        ),
    )
    parser.add_argument("--answer-suppression-weight", type=float, default=0.0)
    parser.add_argument("--random-vector-forget-weight", type=float, default=1.0)
    parser.add_argument("--forget-anchor-weight", type=float, default=1.0)
    parser.add_argument("--forget-anchor-max-similarity", type=float, default=0.0)
    parser.add_argument(
        "--disable-adaptive-forget-objective",
        action="store_true",
        help="Disable online adaptation of random-vector and anchor forget weights.",
    )
    parser.add_argument("--random-vector-target-similarity", type=float, default=0.25)
    parser.add_argument("--forget-anchor-target-similarity", type=float, default=-0.10)
    parser.add_argument("--forget-weight-lr", type=float, default=0.05)
    parser.add_argument("--forget-weight-min", type=float, default=0.0)
    parser.add_argument("--forget-weight-max", type=float, default=12.0)
    parser.add_argument("--random-vector-seed", type=int, default=42)
    parser.add_argument("--parameter-random-vector-weight", type=float, default=1.0)
    parser.add_argument("--parameter-random-vector-scale", type=float, default=0.01)
    parser.add_argument(
        "--forget-loss-top-layers",
        type=int,
        default=8,
        help="Use only the strongest top-K localized layers for the layerwise random-vector forget loss.",
    )
    parser.add_argument(
        "--graph-vector-suppress-weight",
        type=float,
        default=0.0,
        help="Weight for graph-vector suppression on localized MLP neurons.",
    )
    parser.add_argument(
        "--graph-vector-objective",
        default="suppress",
        choices=["suppress", "final_repr", "neuron_repr"],
        help=(
            "Graph-vector objective: suppress=old activation-to-zero baseline; "
            "final_repr=scheme A on final hidden representation; "
            "neuron_repr=scheme B on localized neuron activations."
        ),
    )
    parser.add_argument("--graph-vector-random-weight", type=float, default=1.0)
    parser.add_argument("--graph-vector-away-weight", type=float, default=1.0)
    parser.add_argument("--graph-vector-keep-weight", type=float, default=0.0)
    parser.add_argument("--graph-vector-away-max-similarity", type=float, default=0.0)
    parser.add_argument(
        "--graph-vector-keep-max-items",
        type=int,
        default=0,
        help="Maximum non-unlearned graph vectors used for keep loss. 0 means use all candidates.",
    )
    parser.add_argument(
        "--graph-vector-rag-graph-path",
        default="",
        help="Optional rag_unlearning_graph.pkl path. Empty means read rag_graph_path from reused localization metadata.",
    )
    parser.add_argument(
        "--graph-vector-memory-graph-path",
        default="",
        help="Optional memory_unlearning_graph.pkl path. Empty means read memory_graph_path from reused localization metadata.",
    )
    parser.add_argument("--graph-vector-batch-size", type=int, default=4)
    parser.add_argument(
        "--graph-vector-max-items",
        type=int,
        default=0,
        help="Maximum forget graph vectors used for suppression. 0 means use all unlearned graph vectors.",
    )
    parser.add_argument("--graph-vector-anchor-text", default="Graph concept probe.")
    parser.add_argument("--entity-max-neighbors", type=int, default=2)
    parser.add_argument("--entity-max-cluster-facts", type=int, default=4)
    parser.add_argument("--entity-view-direct-weight", type=float, default=1.0)
    parser.add_argument("--entity-view-context-weight", type=float, default=1.0)
    parser.add_argument("--entity-view-cluster-weight", type=float, default=1.0)
    parser.add_argument("--repr-forget-weight", type=float, default=0.0)
    parser.add_argument("--repr-cluster-weight", type=float, default=0.0)
    parser.add_argument("--repr-target-max-similarity", type=float, default=0.25)
    parser.add_argument("--repr-distance-weight", type=float, default=0.0)
    parser.add_argument("--repr-layerwise-weight", type=float, default=0.0)
    parser.add_argument("--retain-ce-weight", type=float, default=0.0)
    parser.add_argument("--retain-repr-weight", type=float, default=0.0)
    parser.add_argument("--retain-kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--disable-dual-kl-objective",
        action="store_true",
        help="Disable constrained retain-KL dual update and fall back to a fixed retain-kl-weight.",
    )
    parser.add_argument("--retain-kl-target", type=float, default=0.02)
    parser.add_argument("--dual-kl-lr", type=float, default=0.05)
    parser.add_argument("--dual-kl-weight-init", type=float, default=1.0)
    parser.add_argument(
        "--dual-kl-weight-min",
        type=float,
        default=0.2,
        help="Minimum floor for the adaptive retain-KL dual weight so retain protection never fully collapses.",
    )
    parser.add_argument("--retain-topk", type=int, default=32)
    parser.add_argument(
        "--forget-nll-calibration-weight",
        type=float,
        default=0.0,
        help="MIA-aware regularization weight. Penalizes forget answer NLL only when it exceeds the calibration target.",
    )
    parser.add_argument(
        "--forget-nll-calibration-margin",
        type=float,
        default=0.75,
        help="Margin added to retain CE when --forget-nll-calibration-target-mode=retain_ce.",
    )
    parser.add_argument(
        "--forget-nll-calibration-fixed-target",
        type=float,
        default=1.0,
        help="Fixed forget NLL ceiling when --forget-nll-calibration-target-mode=fixed.",
    )
    parser.add_argument(
        "--forget-nll-calibration-target-mode",
        default="retain_ce",
        choices=["retain_ce", "fixed"],
        help="Target used by forget NLL calibration: retain batch CE + margin, or a fixed NLL ceiling.",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Disable best-score early stopping and always train for the full scheduled steps.",
    )
    parser.add_argument("--early-stop-min-steps", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=200)
    parser.add_argument(
        "--best-score-retain-alpha",
        type=float,
        default=2.0,
        help="Tradeoff coefficient in best-step score: rand_sim - alpha * max(retain_kl - target, 0).",
    )
    parser.add_argument(
        "--best-score-mia-alpha",
        type=float,
        default=0.0,
        help="Optional best-step penalty for MIA risk: max(forget_ce - calibration_target, 0).",
    )
    parser.add_argument(
        "--best-retain-kl-max",
        type=float,
        default=-1.0,
        help="Hard retain-KL guard for best checkpoint selection. Disabled when < 0.",
    )
    parser.add_argument(
        "--best-retain-kl-fallback-max",
        type=float,
        default=-1.0,
        help="Fallback best-checkpoint retain-KL guard used only before any primary-guard step exists.",
    )
    parser.add_argument(
        "--enable-canonical-proxy",
        action="store_true",
        help="Select best checkpoints with a fixed canonical QA proxy instead of training-batch proxy metrics.",
    )
    parser.add_argument(
        "--canonical-proxy-forget-path",
        default="",
        help="Canonical forget QA JSON/JSONL used for proxy NLL tracking, e.g. TOFU forget10.json.",
    )
    parser.add_argument(
        "--canonical-proxy-retain-path",
        default="",
        help="Canonical retain QA JSON/JSONL used for proxy NLL/KL tracking, e.g. TOFU retain90.json.",
    )
    parser.add_argument("--canonical-proxy-forget-samples", type=int, default=80)
    parser.add_argument("--canonical-proxy-retain-samples", type=int, default=200)
    parser.add_argument("--canonical-proxy-interval", type=int, default=100)
    parser.add_argument("--canonical-proxy-batch-size", type=int, default=2)
    parser.add_argument(
        "--canonical-proxy-retain-nll-delta-max",
        type=float,
        default=0.30,
        help="Hard retain guard: max allowed canonical retain reference-answer NLL increase.",
    )
    parser.add_argument(
        "--canonical-proxy-retain-kl-max",
        type=float,
        default=0.05,
        help="Hard retain guard: max allowed canonical retain top-k KL against the base model.",
    )
    parser.add_argument("--canonical-proxy-forget-nll-weight", type=float, default=1.0)
    parser.add_argument("--canonical-proxy-forget-logprob-drop-weight", type=float, default=0.5)
    parser.add_argument("--canonical-proxy-retain-nll-penalty", type=float, default=4.0)
    parser.add_argument("--canonical-proxy-retain-kl-penalty", type=float, default=8.0)
    parser.add_argument(
        "--disable-canonical-proxy-hard-retain-filter",
        action="store_true",
        help="Keep proxy score continuous instead of discarding checkpoints that violate retain guards.",
    )
    parser.add_argument(
        "--forget-ce-checkpoint-thresholds",
        default="",
        help=(
            "Comma-separated forget CE thresholds. During training, the first step whose forget_ce "
            "reaches each threshold is saved under output_dir/ce_threshold_checkpoints/."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Stage2Config:
    whitelist = None
    if str(args.manual_layer_whitelist).strip():
        whitelist = sorted(
            {
                int(part.strip())
                for part in str(args.manual_layer_whitelist).split(",")
                if part.strip()
            }
        )
    ce_thresholds = None
    if str(args.forget_ce_checkpoint_thresholds).strip():
        ce_thresholds = sorted(
            {
                float(part.strip())
                for part in str(args.forget_ce_checkpoint_thresholds).split(",")
                if part.strip()
            }
        )
    return Stage2Config(
        base_model_path=args.base_model_path,
        original_lora_path=args.original_lora_path,
        output_dir=args.output_dir,
        graph_path=Path(args.graph_path),
        init_new_lora=args.init_new_lora,
        new_lora_r=args.new_lora_r,
        new_lora_alpha=args.new_lora_alpha,
        new_lora_dropout=args.new_lora_dropout,
        new_lora_target_modules=args.new_lora_target_modules,
        forget_target_path=Path(args.forget_target_path) if args.forget_target_path else None,
        retain_path=Path(args.retain_path),
        core_dataset_path=Path(args.core_dataset_path),
        context_dataset_path=Path(args.context_dataset_path),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        resume_from_checkpoint=Path(args.resume_from_checkpoint) if str(args.resume_from_checkpoint).strip() else None,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        top_fraction=args.top_fraction,
        coarse_top_layers=args.coarse_top_layers,
        per_layer_top_neurons=args.per_layer_top_neurons,
        reuse_localization_pkl=Path(args.reuse_localization_pkl) if str(args.reuse_localization_pkl).strip() else None,
        manual_layer_whitelist=whitelist,
        localization_batch_size=args.localization_batch_size,
        localization_forget_samples=args.localization_forget_samples,
        localization_retain_samples=args.localization_retain_samples,
        localization_strategy=args.localization_strategy,
        localization_layer_start=args.localization_layer_start,
        localization_layer_end=args.localization_layer_end,
        localization_target_token=args.localization_target_token,
        activation_fired_threshold=args.activation_fired_threshold,
        forget_author_bucket_size=args.forget_author_bucket_size,
        forget_firing_min_ratio=args.forget_firing_min_ratio,
        retain_firing_max_ratio=args.retain_firing_max_ratio,
        author_bucket_top_neurons=args.author_bucket_top_neurons,
        contrastive_lambda=args.contrastive_lambda,
        author_abs_weight=args.author_abs_weight,
        author_posneg_weight=args.author_posneg_weight,
        localization_gradient_checkpointing=not args.disable_localization_gradient_checkpointing,
        localization_key_annotation_path=Path(args.localization_key_annotation_path) if str(args.localization_key_annotation_path).strip() else None,
        localization_key_token_min_char_len=args.localization_key_token_min_char_len,
        activation_signed_weight=args.activation_signed_weight,
        activation_abs_weight=args.activation_abs_weight,
        localization_verify_enabled=not args.disable_localization_verify,
        localization_verify_forget_samples=args.localization_verify_forget_samples,
        localization_verify_retain_samples=args.localization_verify_retain_samples,
        localization_verify_retain_penalty=args.localization_verify_retain_penalty,
        verification_top_neurons_per_layer=args.verification_top_neurons_per_layer,
        final_top_neurons_per_layer=args.final_top_neurons_per_layer,
        verification_score_std_multiplier=args.verification_score_std_multiplier,
        verification_min_score=args.verification_min_score,
        relaxed_layer_max_index=args.relaxed_layer_max_index,
        relaxed_final_top_neurons_per_layer=args.relaxed_final_top_neurons_per_layer,
        relaxed_verification_score_std_multiplier=args.relaxed_verification_score_std_multiplier,
        relaxed_verification_min_score=args.relaxed_verification_min_score,
        global_max_target_neurons=args.global_max_target_neurons,
        localization_verify_scale_factor=args.localization_verify_scale_factor,
        disable_parameter_mask=args.disable_parameter_mask,
        safe_response=args.safe_response,
        forget_loss_strategy=args.forget_loss_strategy,
        forget_objective_weight=args.forget_objective_weight,
        forget_target_mode=args.forget_target_mode,
        forget_ce_weight=args.forget_ce_weight,
        answer_suppression_path=Path(args.answer_suppression_path) if str(args.answer_suppression_path).strip() else None,
        answer_suppression_weight=args.answer_suppression_weight,
        random_vector_forget_weight=args.random_vector_forget_weight,
        forget_anchor_weight=args.forget_anchor_weight,
        forget_anchor_max_similarity=args.forget_anchor_max_similarity,
        use_adaptive_forget_objective=not args.disable_adaptive_forget_objective,
        random_vector_target_similarity=args.random_vector_target_similarity,
        forget_anchor_target_similarity=args.forget_anchor_target_similarity,
        forget_weight_lr=args.forget_weight_lr,
        forget_weight_min=args.forget_weight_min,
        forget_weight_max=args.forget_weight_max,
        random_vector_seed=args.random_vector_seed,
        parameter_random_vector_weight=args.parameter_random_vector_weight,
        parameter_random_vector_scale=args.parameter_random_vector_scale,
        forget_loss_top_layers=args.forget_loss_top_layers,
        graph_vector_suppress_weight=args.graph_vector_suppress_weight,
        graph_vector_objective=args.graph_vector_objective,
        graph_vector_random_weight=args.graph_vector_random_weight,
        graph_vector_away_weight=args.graph_vector_away_weight,
        graph_vector_keep_weight=args.graph_vector_keep_weight,
        graph_vector_away_max_similarity=args.graph_vector_away_max_similarity,
        graph_vector_keep_max_items=args.graph_vector_keep_max_items,
        graph_vector_rag_graph_path=Path(args.graph_vector_rag_graph_path) if str(args.graph_vector_rag_graph_path).strip() else None,
        graph_vector_memory_graph_path=Path(args.graph_vector_memory_graph_path) if str(args.graph_vector_memory_graph_path).strip() else None,
        graph_vector_batch_size=args.graph_vector_batch_size,
        graph_vector_max_items=args.graph_vector_max_items,
        graph_vector_anchor_text=args.graph_vector_anchor_text,
        entity_max_neighbors=args.entity_max_neighbors,
        entity_max_cluster_facts=args.entity_max_cluster_facts,
        entity_view_direct_weight=args.entity_view_direct_weight,
        entity_view_context_weight=args.entity_view_context_weight,
        entity_view_cluster_weight=args.entity_view_cluster_weight,
        repr_forget_weight=args.repr_forget_weight,
        repr_cluster_weight=args.repr_cluster_weight,
        repr_target_max_similarity=args.repr_target_max_similarity,
        repr_distance_weight=args.repr_distance_weight,
        repr_layerwise_weight=args.repr_layerwise_weight,
        retain_ce_weight=args.retain_ce_weight,
        retain_repr_weight=args.retain_repr_weight,
        retain_kl_weight=args.retain_kl_weight,
        use_dual_kl_objective=not args.disable_dual_kl_objective,
        retain_kl_target=args.retain_kl_target,
        dual_kl_lr=args.dual_kl_lr,
        dual_kl_weight_init=args.dual_kl_weight_init,
        dual_kl_weight_min=args.dual_kl_weight_min,
        retain_topk=args.retain_topk,
        forget_nll_calibration_weight=args.forget_nll_calibration_weight,
        forget_nll_calibration_margin=args.forget_nll_calibration_margin,
        forget_nll_calibration_fixed_target=args.forget_nll_calibration_fixed_target,
        forget_nll_calibration_target_mode=args.forget_nll_calibration_target_mode,
        early_stop_enabled=not args.disable_early_stop,
        early_stop_min_steps=args.early_stop_min_steps,
        early_stop_patience=args.early_stop_patience,
        best_score_retain_alpha=args.best_score_retain_alpha,
        best_score_mia_alpha=args.best_score_mia_alpha,
        best_retain_kl_max=args.best_retain_kl_max,
        best_retain_kl_fallback_max=args.best_retain_kl_fallback_max,
        canonical_proxy_enabled=args.enable_canonical_proxy,
        canonical_proxy_forget_path=Path(args.canonical_proxy_forget_path) if str(args.canonical_proxy_forget_path).strip() else None,
        canonical_proxy_retain_path=Path(args.canonical_proxy_retain_path) if str(args.canonical_proxy_retain_path).strip() else None,
        canonical_proxy_forget_samples=args.canonical_proxy_forget_samples,
        canonical_proxy_retain_samples=args.canonical_proxy_retain_samples,
        canonical_proxy_interval=args.canonical_proxy_interval,
        canonical_proxy_batch_size=args.canonical_proxy_batch_size,
        canonical_proxy_retain_nll_delta_max=args.canonical_proxy_retain_nll_delta_max,
        canonical_proxy_retain_kl_max=args.canonical_proxy_retain_kl_max,
        canonical_proxy_forget_nll_weight=args.canonical_proxy_forget_nll_weight,
        canonical_proxy_forget_logprob_drop_weight=args.canonical_proxy_forget_logprob_drop_weight,
        canonical_proxy_retain_nll_penalty=args.canonical_proxy_retain_nll_penalty,
        canonical_proxy_retain_kl_penalty=args.canonical_proxy_retain_kl_penalty,
        canonical_proxy_hard_retain_filter=not args.disable_canonical_proxy_hard_retain_filter,
        forget_ce_checkpoint_thresholds=ce_thresholds,
    )


def validate_inputs(cfg: Stage2Config) -> None:
    missing = []
    if not Path(cfg.base_model_path).exists():
        missing.append(f"base model: {cfg.base_model_path}")
    if not cfg.init_new_lora and not Path(cfg.original_lora_path).exists():
        missing.append(f"original lora: {cfg.original_lora_path}")
    if cfg.resume_from_checkpoint is not None and not cfg.resume_from_checkpoint.exists():
        missing.append(f"resume checkpoint: {cfg.resume_from_checkpoint}")
    if cfg.init_new_lora:
        if cfg.new_lora_r <= 0:
            raise ValueError("--new-lora-r must be > 0.")
        if cfg.new_lora_alpha <= 0:
            raise ValueError("--new-lora-alpha must be > 0.")
        if cfg.new_lora_dropout < 0:
            raise ValueError("--new-lora-dropout must be >= 0.")
        if not str(cfg.new_lora_target_modules).strip():
            raise ValueError("--new-lora-target-modules must not be empty when --init-new-lora is set.")

                                                                        
                                                                            
                                        
     
    if cfg.forget_target_path is None:
        if not cfg.graph_path.exists():
            missing.append(f"graph: {cfg.graph_path}")
    if cfg.forget_target_path is not None and not cfg.forget_target_path.exists():
        missing.append(f"forget targets: {cfg.forget_target_path}")
    if cfg.answer_suppression_path is not None and not cfg.answer_suppression_path.exists():
        missing.append(f"answer suppression dataset: {cfg.answer_suppression_path}")
    if not cfg.retain_path.exists():
        missing.append(f"retain dataset: {cfg.retain_path}")
    if cfg.forget_target_path is None and not cfg.core_dataset_path.exists():
        missing.append(f"core dataset: {cfg.core_dataset_path}")
    if cfg.forget_target_path is None and not cfg.context_dataset_path.exists():
        missing.append(f"context dataset: {cfg.context_dataset_path}")
    if cfg.top_fraction <= 0 or cfg.top_fraction > 1:
        raise ValueError("--top-fraction must be within (0, 1].")
    if cfg.coarse_top_layers <= 0:
        raise ValueError("--coarse-top-layers must be > 0.")
    if cfg.per_layer_top_neurons <= 0:
        raise ValueError("--per-layer-top-neurons must be > 0.")
    if cfg.reuse_localization_pkl is not None and not cfg.reuse_localization_pkl.exists():
        raise FileNotFoundError(f"reuse localization pkl does not exist: {cfg.reuse_localization_pkl}")
    if cfg.manual_layer_whitelist is not None and any(layer_idx < 0 for layer_idx in cfg.manual_layer_whitelist):
        raise ValueError("--manual-layer-whitelist must contain non-negative integers.")
    if cfg.localization_batch_size <= 0:
        raise ValueError("--localization-batch-size must be > 0.")
    if cfg.localization_strategy not in {"author_firing_intersection", "activation_contrast"}:
        raise ValueError("--localization-strategy must be one of: author_firing_intersection, activation_contrast.")
    if cfg.localization_layer_start < 0:
        raise ValueError("--localization-layer-start must be >= 0.")
    if cfg.localization_layer_end != -1 and cfg.localization_layer_end < cfg.localization_layer_start:
        raise ValueError("--localization-layer-end must be -1 or >= --localization-layer-start.")
    if cfg.localization_target_token not in {"keyword_answer", "raw_qa"}:
        raise ValueError("--localization-target-token must be one of: keyword_answer, raw_qa.")
    if cfg.activation_fired_threshold < 0:
        raise ValueError("--activation-fired-threshold must be >= 0.")
    if cfg.forget_author_bucket_size <= 0:
        raise ValueError("--forget-author-bucket-size must be > 0.")
    if cfg.forget_firing_min_ratio <= 0 or cfg.forget_firing_min_ratio > 1:
        raise ValueError("--forget-firing-min-ratio must be within (0, 1].")
    if cfg.retain_firing_max_ratio < 0 or cfg.retain_firing_max_ratio > 1:
        raise ValueError("--retain-firing-max-ratio must be within [0, 1].")
    if cfg.author_bucket_top_neurons <= 0:
        raise ValueError("--author-bucket-top-neurons must be > 0.")
    if cfg.author_abs_weight < 0 or cfg.author_posneg_weight < 0:
        raise ValueError("--author-abs-weight and --author-posneg-weight must be >= 0.")
    if cfg.localization_key_token_min_char_len < 1:
        raise ValueError("--localization-key-token-min-char-len must be >= 1.")
    if cfg.localization_key_annotation_path is not None and not cfg.localization_key_annotation_path.exists():
        raise FileNotFoundError(f"localization key annotation path does not exist: {cfg.localization_key_annotation_path}")
    if cfg.seed < 0:
        raise ValueError("--seed must be >= 0.")
    if cfg.localization_forget_samples <= 0 or cfg.localization_retain_samples <= 0:
        raise ValueError("--localization-forget-samples and --localization-retain-samples must be > 0.")
    if cfg.localization_verify_forget_samples <= 0 or cfg.localization_verify_retain_samples <= 0:
        raise ValueError("--localization-verify-forget-samples and --localization-verify-retain-samples must be > 0.")
    if cfg.contrastive_lambda < 0:
        raise ValueError("--contrastive-lambda must be >= 0.")
    if cfg.forget_loss_top_layers <= 0:
        raise ValueError("--forget-loss-top-layers must be > 0.")
    if cfg.activation_signed_weight < 0 or cfg.activation_abs_weight < 0:
        raise ValueError("--activation-signed-weight and --activation-abs-weight must be >= 0.")
    if cfg.localization_verify_retain_penalty < 0:
        raise ValueError("--localization-verify-retain-penalty must be >= 0.")
    if cfg.verification_top_neurons_per_layer <= 0:
        raise ValueError("--verification-top-neurons-per-layer must be > 0.")
    if cfg.verification_top_neurons_per_layer > cfg.per_layer_top_neurons:
        raise ValueError("--verification-top-neurons-per-layer must be <= --per-layer-top-neurons.")
    if cfg.final_top_neurons_per_layer <= 0:
        raise ValueError("--final-top-neurons-per-layer must be > 0.")
    if cfg.final_top_neurons_per_layer > cfg.verification_top_neurons_per_layer:
        raise ValueError("--final-top-neurons-per-layer must be <= --verification-top-neurons-per-layer.")
    if cfg.verification_score_std_multiplier < 0:
        raise ValueError("--verification-score-std-multiplier must be >= 0.")
    if cfg.verification_min_score < 0:
        raise ValueError("--verification-min-score must be >= 0.")
    if cfg.relaxed_layer_max_index < 0:
        raise ValueError("--relaxed-layer-max-index must be >= 0.")
    if cfg.relaxed_final_top_neurons_per_layer <= 0:
        raise ValueError("--relaxed-final-top-neurons-per-layer must be > 0.")
    if cfg.relaxed_final_top_neurons_per_layer < cfg.final_top_neurons_per_layer:
        raise ValueError("--relaxed-final-top-neurons-per-layer must be >= --final-top-neurons-per-layer.")
    if cfg.relaxed_final_top_neurons_per_layer > cfg.verification_top_neurons_per_layer:
        raise ValueError("--relaxed-final-top-neurons-per-layer must be <= --verification-top-neurons-per-layer.")
    if cfg.relaxed_verification_score_std_multiplier < 0:
        raise ValueError("--relaxed-verification-score-std-multiplier must be >= 0.")
    if cfg.relaxed_verification_min_score < 0:
        raise ValueError("--relaxed-verification-min-score must be >= 0.")
    if cfg.relaxed_verification_min_score > cfg.verification_min_score:
        raise ValueError("--relaxed-verification-min-score must be <= --verification-min-score.")
    if cfg.global_max_target_neurons <= 0:
        raise ValueError("--global-max-target-neurons must be > 0.")
    if cfg.localization_verify_scale_factor < 0:
        raise ValueError("--localization-verify-scale-factor must be >= 0.")
    if cfg.forget_target_mode not in {"safe_response", "random_vector"}:
        raise ValueError("--forget-target-mode must be one of: safe_response, random_vector.")
    if cfg.forget_loss_strategy not in {
        "gradient_ascent",
        "gradient_ascent_projected",
        "representation_erasure",
        "random_vector",
        "parameter_random_vector",
        "random_answer_ce",
    }:
        raise ValueError("--forget-loss-strategy must be one of: gradient_ascent, gradient_ascent_projected, representation_erasure, random_vector, parameter_random_vector, random_answer_ce.")
    if cfg.forget_objective_weight < 0:
        raise ValueError("--forget-objective-weight must be >= 0.")
    if (
        cfg.forget_ce_weight < 0
        or cfg.answer_suppression_weight < 0
        or cfg.random_vector_forget_weight < 0
        or cfg.forget_anchor_weight < 0
        or cfg.graph_vector_suppress_weight < 0
        or cfg.graph_vector_random_weight < 0
        or cfg.graph_vector_away_weight < 0
        or cfg.graph_vector_keep_weight < 0
        or cfg.retain_ce_weight < 0
        or cfg.retain_kl_weight < 0
        or cfg.retain_repr_weight < 0
    ):
        raise ValueError(
            "--forget-ce-weight, --answer-suppression-weight, --random-vector-forget-weight, --forget-anchor-weight, --graph-vector-suppress-weight, "
            "--graph-vector-random-weight, --graph-vector-away-weight, --graph-vector-keep-weight, "
            "--retain-ce-weight, --retain-kl-weight, and --retain-repr-weight must be >= 0."
        )
    if cfg.graph_vector_objective not in {"suppress", "final_repr", "neuron_repr"}:
        raise ValueError("--graph-vector-objective must be one of: suppress, final_repr, neuron_repr.")
    if cfg.graph_vector_away_max_similarity >= 1.0:
        raise ValueError("--graph-vector-away-max-similarity must be < 1.0.")
    if cfg.forget_anchor_max_similarity >= 1.0:
        raise ValueError("--forget-anchor-max-similarity must be < 1.0.")
    if cfg.entity_max_neighbors < 0 or cfg.entity_max_cluster_facts <= 0:
        raise ValueError("--entity-max-neighbors must be >= 0 and --entity-max-cluster-facts must be > 0.")
    if min(cfg.entity_view_direct_weight, cfg.entity_view_context_weight, cfg.entity_view_cluster_weight) < 0:
        raise ValueError("--entity-view-* weights must be >= 0.")
    if min(cfg.repr_forget_weight, cfg.repr_cluster_weight, cfg.repr_distance_weight, cfg.repr_layerwise_weight) < 0:
        raise ValueError("--repr-* weights must be >= 0.")
    if cfg.repr_target_max_similarity >= 1.0:
        raise ValueError("--repr-target-max-similarity must be < 1.0.")
    if cfg.random_vector_target_similarity >= 1.0:
        raise ValueError("--random-vector-target-similarity must be < 1.0.")
    if cfg.forget_anchor_target_similarity >= 1.0:
        raise ValueError("--forget-anchor-target-similarity must be < 1.0.")
    if cfg.forget_weight_lr < 0:
        raise ValueError("--forget-weight-lr must be >= 0.")
    if cfg.forget_weight_min < 0 or cfg.forget_weight_max < 0:
        raise ValueError("--forget-weight-min and --forget-weight-max must be >= 0.")
    if cfg.forget_weight_min > cfg.forget_weight_max:
        raise ValueError("--forget-weight-min must be <= --forget-weight-max.")
    if cfg.parameter_random_vector_weight < 0:
        raise ValueError("--parameter-random-vector-weight must be >= 0.")
    if cfg.parameter_random_vector_scale < 0:
        raise ValueError("--parameter-random-vector-scale must be >= 0.")
    if cfg.graph_vector_batch_size <= 0:
        raise ValueError("--graph-vector-batch-size must be > 0.")
    if cfg.graph_vector_max_items < 0:
        raise ValueError("--graph-vector-max-items must be >= 0.")
    if cfg.graph_vector_keep_max_items < 0:
        raise ValueError("--graph-vector-keep-max-items must be >= 0.")
    if cfg.graph_vector_rag_graph_path is not None and not cfg.graph_vector_rag_graph_path.exists():
        missing.append(f"graph-vector rag graph: {cfg.graph_vector_rag_graph_path}")
    if cfg.graph_vector_memory_graph_path is not None and not cfg.graph_vector_memory_graph_path.exists():
        missing.append(f"graph-vector memory graph: {cfg.graph_vector_memory_graph_path}")
    if cfg.retain_kl_target < 0:
        raise ValueError("--retain-kl-target must be >= 0.")
    if (
        cfg.best_retain_kl_max >= 0
        and cfg.best_retain_kl_fallback_max >= 0
        and cfg.best_retain_kl_fallback_max < cfg.best_retain_kl_max
    ):
        raise ValueError("--best-retain-kl-fallback-max must be >= --best-retain-kl-max.")
    if cfg.forget_nll_calibration_weight < 0:
        raise ValueError("--forget-nll-calibration-weight must be >= 0.")
    if cfg.forget_nll_calibration_margin < 0:
        raise ValueError("--forget-nll-calibration-margin must be >= 0.")
    if cfg.forget_nll_calibration_fixed_target < 0:
        raise ValueError("--forget-nll-calibration-fixed-target must be >= 0.")
    if cfg.best_score_mia_alpha < 0:
        raise ValueError("--best-score-mia-alpha must be >= 0.")
    if cfg.dual_kl_lr < 0 or cfg.dual_kl_weight_init < 0 or cfg.dual_kl_weight_min < 0:
        raise ValueError("--dual-kl-lr, --dual-kl-weight-init, and --dual-kl-weight-min must be >= 0.")
    if cfg.dual_kl_weight_min > cfg.dual_kl_weight_init:
        raise ValueError("--dual-kl-weight-min must be <= --dual-kl-weight-init.")
    if cfg.retain_topk <= 0:
        raise ValueError("--retain-topk must be > 0.")
    if cfg.early_stop_min_steps < 0:
        raise ValueError("--early-stop-min-steps must be >= 0.")
    if cfg.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be > 0.")
    if cfg.best_score_retain_alpha < 0:
        raise ValueError("--best-score-retain-alpha must be >= 0.")
    if cfg.canonical_proxy_enabled:
        if cfg.canonical_proxy_forget_path is None or cfg.canonical_proxy_retain_path is None:
            raise ValueError("--enable-canonical-proxy requires --canonical-proxy-forget-path and --canonical-proxy-retain-path.")
        if not cfg.canonical_proxy_forget_path.exists():
            missing.append(f"canonical proxy forget: {cfg.canonical_proxy_forget_path}")
        if not cfg.canonical_proxy_retain_path.exists():
            missing.append(f"canonical proxy retain: {cfg.canonical_proxy_retain_path}")
        if cfg.canonical_proxy_forget_samples <= 0 or cfg.canonical_proxy_retain_samples <= 0:
            raise ValueError("--canonical-proxy-*-samples must be > 0.")
        if cfg.canonical_proxy_interval <= 0:
            raise ValueError("--canonical-proxy-interval must be > 0.")
        if cfg.canonical_proxy_batch_size <= 0:
            raise ValueError("--canonical-proxy-batch-size must be > 0.")
        if cfg.canonical_proxy_retain_nll_delta_max < 0 or cfg.canonical_proxy_retain_kl_max < 0:
            raise ValueError("--canonical-proxy retain thresholds must be >= 0.")
        if min(
            cfg.canonical_proxy_forget_nll_weight,
            cfg.canonical_proxy_forget_logprob_drop_weight,
            cfg.canonical_proxy_retain_nll_penalty,
            cfg.canonical_proxy_retain_kl_penalty,
        ) < 0:
            raise ValueError("--canonical-proxy weights and penalties must be >= 0.")
    if cfg.forget_ce_checkpoint_thresholds is not None:
        if any(threshold <= 0 for threshold in cfg.forget_ce_checkpoint_thresholds):
            raise ValueError("--forget-ce-checkpoint-thresholds values must be > 0.")
    if missing:
        raise FileNotFoundError("Missing required Phase-2 inputs:\n- " + "\n- ".join(missing))


def build_output_artifact_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "localization_pkl": output_dir / DEFAULT_LOCALIZATION_PKL,
        "localization_json": output_dir / DEFAULT_LOCALIZATION_JSON,
        "run_metadata_json": output_dir / DEFAULT_RUN_METADATA_JSON,
    }


def log_plan(cfg: Stage2Config, logger: logging.Logger) -> None:
    logger.info("Phase-2 configuration summary:")
    for key, value in asdict(cfg).items():
        logger.info("  %s=%s", key, value)
