from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "task_mode": "exclusive",
        "num_classes": 2,
        "region_num_classes": 2,
        "embed_dim": 96,
        "depths": [2, 2, 2, 2],
        "num_heads_pyramid": [3, 6, 12, 24],
        "common_dim": 192,
        "decoder_dim": 160,
        "transformer_depth": 2,
        "num_heads": 8,
        "window_size": 7,
        "edge_tau": 2.0,
        "channel_rank": 16,
        "erga_hidden_dim": 128,
        "use_prior_attention": False,
        "use_edge_to_region_constraint": False,
        "unet_base_channels": 32,
        "classic_unet_base_channels": 64,
        "alpha": 0.5,
        "beta": 0.3,
        "inject_edge_skip": True,
        "ablation": {
            "baseline": "erga",
            "cross_level_fusion": "gated",
            "attention_branches": "both",
            "relation_guidance": "full",
            "erga_fusion": True,
            "bottleneck_enhancement": "full",
            "use_plain_unet_baseline": False,
            "disable_cross_level_fusion": False,
            "cross_level_fusion_mode": "gated",
            "disable_feature_enhancement": False,
            "disable_edge_branch": False,
            "disable_region_branch": False,
            "disable_overlap_prior": False,
            "disable_boundary_generators": False,
            "disable_region_to_edge_constraint": False,
            "disable_icdc": False,
            "disable_coarse_sdm": False,
            "disable_erga": False,
            "disable_lightweight_attention": False,
            "disable_transformer": False,
            "disable_prior_injection": False,
            "disable_curriculum_gradient": False,
            "disable_uncertainty_head": False,
        },
    },
    "loss": {
        "lambda_seg": 1.0,
        "lambda_edge": 0.2,
        "lambda_shared": 0.1,
        "lambda_inner": 0.1,
        "lambda_cons": 0.05,
        "eps": 1.0e-6,
    },
}


def get_image_size_tag(cfg: Dict[str, Any]) -> str:
    image_size = cfg.get("data", {}).get("image_size", [512, 512])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return "unknown"
    height, width = int(image_size[0]), int(image_size[1])
    return str(height) if height == width else f"{height}x{width}"


def append_size_tag(name: str, size_tag: str) -> str:
    suffix = f"_{size_tag}"
    return name if name.endswith(suffix) else f"{name}{suffix}"


def apply_size_variant_paths(cfg: Dict[str, Any], config_stem: str) -> None:
    size_tag = get_image_size_tag(cfg)
    variant_name = append_size_tag(config_stem, size_tag)
    train_cfg = cfg.setdefault("train", {})
    infer_cfg = cfg.setdefault("inference", {})

    train_cfg["save_dir"] = f"checkpoints/{size_tag}/{variant_name}"
    train_cfg["log_dir"] = f"logs/{size_tag}/{variant_name}"
    train_cfg["csv_log_path"] = f"logs/{size_tag}/{variant_name}/train_log.csv"
    train_cfg["best_metrics_path"] = f"checkpoints/{size_tag}/{variant_name}/best_metrics.json"
    train_cfg["best_model_path"] = f"checkpoints/{size_tag}/{variant_name}/best_model.pth"
    train_cfg["last_model_path"] = f"checkpoints/{size_tag}/{variant_name}/last_model.pth"
    infer_cfg["prediction_dir"] = f"predictions/{size_tag}/{variant_name}"
    infer_cfg["result_json"] = f"results/{size_tag}/{variant_name}_metrics.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_ablation_config(cfg: Dict[str, Any], raw_cfg: Dict[str, Any]) -> None:
    """Expand compact ablation controls into the legacy booleans used by modules."""
    ablation = cfg["model"].setdefault("ablation", {})
    raw_ablation = raw_cfg.get("model", {}).get("ablation", {})

    if "baseline" in raw_ablation:
        ablation["use_plain_unet_baseline"] = raw_ablation["baseline"] == "plain_unet"
    elif "use_plain_unet_baseline" in raw_ablation:
        ablation["baseline"] = "plain_unet" if bool(raw_ablation["use_plain_unet_baseline"]) else "erga"

    if "cross_level_fusion" in raw_ablation:
        fusion = raw_ablation["cross_level_fusion"]
        ablation["cross_level_fusion_mode"] = fusion
        ablation["disable_cross_level_fusion"] = fusion == "f3_only"
    elif "cross_level_fusion_mode" in raw_ablation or "disable_cross_level_fusion" in raw_ablation:
        fusion = ablation.get("cross_level_fusion_mode", "gated")
        if bool(ablation.get("disable_cross_level_fusion", False)):
            fusion = "f3_only"
        ablation["cross_level_fusion"] = fusion

    if "attention_branches" in raw_ablation:
        branches = raw_ablation["attention_branches"]
        ablation["disable_edge_branch"] = branches not in {"both", "edge_only"}
        ablation["disable_region_branch"] = branches not in {"both", "region_only"}
    elif "disable_edge_branch" in raw_ablation or "disable_region_branch" in raw_ablation:
        edge_on = not bool(ablation.get("disable_edge_branch", False))
        region_on = not bool(ablation.get("disable_region_branch", False))
        if edge_on and region_on:
            ablation["attention_branches"] = "both"
        elif edge_on:
            ablation["attention_branches"] = "edge_only"
        elif region_on:
            ablation["attention_branches"] = "region_only"
        else:
            ablation["attention_branches"] = "none"

    if "relation_guidance" in raw_ablation:
        relation = raw_ablation["relation_guidance"]
        ablation["disable_overlap_prior"] = relation == "none"
        ablation["disable_region_to_edge_constraint"] = relation in {"prior_only", "none"}
        ablation["disable_prior_injection"] = relation in {"prior_only", "no_decoder_injection", "none"}
    elif any(key in raw_ablation for key in ("disable_overlap_prior", "disable_region_to_edge_constraint", "disable_prior_injection")):
        overlap_on = not bool(ablation.get("disable_overlap_prior", False))
        r2e_on = not bool(ablation.get("disable_region_to_edge_constraint", False))
        injection_on = not bool(ablation.get("disable_prior_injection", False))
        if not overlap_on:
            ablation["relation_guidance"] = "none"
        elif r2e_on and injection_on:
            ablation["relation_guidance"] = "full"
        elif r2e_on:
            ablation["relation_guidance"] = "no_decoder_injection"
        else:
            ablation["relation_guidance"] = "prior_only"

    if "erga_fusion" in raw_ablation:
        ablation["disable_erga"] = not bool(raw_ablation["erga_fusion"])
    elif "disable_erga" in raw_ablation:
        ablation["erga_fusion"] = not bool(ablation.get("disable_erga", False))

    if "bottleneck_enhancement" in raw_ablation:
        enhancement = raw_ablation["bottleneck_enhancement"]
        ablation["disable_lightweight_attention"] = enhancement not in {"full", "lightweight_attention"}
        ablation["disable_transformer"] = enhancement not in {"full", "transformer"}
    elif "disable_lightweight_attention" in raw_ablation or "disable_transformer" in raw_ablation:
        light_on = not bool(ablation.get("disable_lightweight_attention", False))
        transformer_on = not bool(ablation.get("disable_transformer", False))
        if light_on and transformer_on:
            ablation["bottleneck_enhancement"] = "full"
        elif light_on:
            ablation["bottleneck_enhancement"] = "lightweight_attention"
        elif transformer_on:
            ablation["bottleneck_enhancement"] = "transformer"
        else:
            ablation["bottleneck_enhancement"] = "none"


def _validate_config(cfg: Dict[str, Any]) -> None:
    required_sections = ["data", "model", "loss", "train", "inference"]
    missing_sections = [section for section in required_sections if section not in cfg]
    if missing_sections:
        raise KeyError(f"Missing config sections: {missing_sections}")

    task_mode = cfg["model"]["task_mode"]
    data_mode = cfg["data"]["mode"]
    valid_task_modes = {"binary", "multiclass", "multilabel", "exclusive", "independent"}
    if task_mode not in valid_task_modes:
        raise ValueError(f"Unsupported model.task_mode: {task_mode}. Valid: {valid_task_modes}")

    # Normalize for validation purposes.
    _map = {"binary": "exclusive", "multiclass": "exclusive", "multilabel": "independent"}
    normalized_mode = _map.get(task_mode, task_mode)

    # region_mode is now optional (auto-inferred), but validate if present.
    region_mode = cfg["model"].get("region_mode")
    if region_mode is not None and region_mode not in {"single_label", "multi_label"}:
        raise ValueError(f"Unsupported model.region_mode: {region_mode}")
    if data_mode not in {"multiclass", "multilabel", "multi_source"}:
        raise ValueError(f"Unsupported data.mode: {data_mode}")

    if data_mode == "multi_source":
        return

    if len(cfg["model"]["depths"]) != 4:
        raise ValueError("model.depths must contain 4 stages")
    if len(cfg["model"]["num_heads_pyramid"]) != 4:
        raise ValueError("model.num_heads_pyramid must contain 4 stages")

    common_dim = int(cfg["model"]["common_dim"])
    num_heads = int(cfg["model"]["num_heads"])
    if common_dim % num_heads != 0:
        raise ValueError(f"model.common_dim ({common_dim}) must be divisible by model.num_heads ({num_heads})")

    ablation_cfg = cfg["model"].get("ablation", {})
    valid_baselines = {"erga", "plain_unet"}
    if ablation_cfg.get("baseline", "erga") not in valid_baselines:
        raise ValueError(f"Unsupported model.ablation.baseline: {ablation_cfg.get('baseline')}. Valid: {valid_baselines}")

    fusion_mode = ablation_cfg.get("cross_level_fusion_mode", "gated")
    valid_fusion_modes = {"gated", "sum", "mean", "f3_only"}
    if fusion_mode not in valid_fusion_modes:
        raise ValueError(f"Unsupported model.ablation.cross_level_fusion_mode: {fusion_mode}. Valid: {valid_fusion_modes}")
    if ablation_cfg.get("cross_level_fusion", fusion_mode) not in valid_fusion_modes:
        raise ValueError(f"Unsupported model.ablation.cross_level_fusion: {ablation_cfg.get('cross_level_fusion')}. Valid: {valid_fusion_modes}")

    valid_attention = {"both", "edge_only", "region_only", "none"}
    if ablation_cfg.get("attention_branches", "both") not in valid_attention:
        raise ValueError(f"Unsupported model.ablation.attention_branches: {ablation_cfg.get('attention_branches')}. Valid: {valid_attention}")

    valid_relation = {"full", "prior_only", "no_decoder_injection", "none"}
    if ablation_cfg.get("relation_guidance", "full") not in valid_relation:
        raise ValueError(f"Unsupported model.ablation.relation_guidance: {ablation_cfg.get('relation_guidance')}. Valid: {valid_relation}")

    valid_enhancement = {"full", "lightweight_attention", "transformer", "none"}
    if ablation_cfg.get("bottleneck_enhancement", "full") not in valid_enhancement:
        raise ValueError(f"Unsupported model.ablation.bottleneck_enhancement: {ablation_cfg.get('bottleneck_enhancement')}. Valid: {valid_enhancement}")

    num_classes = int(cfg["model"]["num_classes"])
    if task_mode == "binary" and num_classes != 1:
        raise ValueError("model.num_classes must be 1 when model.task_mode is 'binary'")
    if normalized_mode == "exclusive" and task_mode != "binary" and num_classes < 1:
        raise ValueError("model.num_classes must be >= 1 for exclusive segmentation")
    if normalized_mode == "independent" and num_classes < 1:
        raise ValueError("model.num_classes must be >= 1 for independent segmentation")

    data_num_classes = int(cfg["data"]["num_classes"])
    model_num_classes = num_classes
    region_num_classes = int(cfg["model"]["region_num_classes"])
    class_names = cfg["data"].get("class_names", [])
    if class_names and len(class_names) != data_num_classes:
        raise ValueError("len(data.class_names) must match data.num_classes")

    if normalized_mode == "exclusive" and model_num_classes > 1 and model_num_classes != data_num_classes:
        raise ValueError("For exclusive segmentation with num_classes>1, model.num_classes must match data.num_classes")
    if normalized_mode == "independent" and data_mode not in ("multilabel", "multiclass"):
        raise ValueError("For independent segmentation, data.mode must be 'multilabel' or 'multiclass'")
    if normalized_mode == "independent" and model_num_classes != data_num_classes:
        raise ValueError("For independent segmentation, model.num_classes must match data.num_classes")

    if region_num_classes < model_num_classes:
        raise ValueError("model.region_num_classes must be >= model.num_classes")


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if raw_cfg is None:
        raise ValueError(f"Empty config file: {config_path}")
    cfg = _deep_merge(DEFAULT_CONFIG, raw_cfg)
    if not path.stem.startswith("_temp_kfold_"):
        apply_size_variant_paths(cfg, path.stem)
    _normalize_ablation_config(cfg, raw_cfg)
    _validate_config(cfg)
    return cfg


def save_json(path: str | Path, content: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
