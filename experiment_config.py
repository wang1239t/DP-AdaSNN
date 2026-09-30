"""Lightweight helpers for local YAML experiment entrypoints."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = ROOT / "ultralytics" / "cfg" / "default.yaml"

def yaml_load(file):
    """Load a YAML file as a dict."""
    with open(file, errors="ignore", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)

def experiment_path(filename):
    """Return an experiment YAML path inside ultralytics/cfg/experiments."""
    return ROOT / "ultralytics" / "cfg" / "experiments" / filename

def load_experiment_cfg(cfg_path=None):
    """Merge default.yaml with an optional experiment YAML."""
    cfg = dict(DEFAULT_CFG_DICT)
    overrides = {}
    if cfg_path:
        cfg_file = Path(cfg_path)
        overrides = yaml_load(cfg_file)
        cfg.update(overrides)
        cfg["cfg"] = str(cfg_file)
    cfg["_overrides"] = overrides
    return cfg

def apply_runtime_env(cfg):
    """Apply script-level environment settings without importing Ultralytics."""

    if os.name == 'nt':
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    cuda_visible_devices = cfg.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

def yolo_kwargs(cfg, exclude=()):
    """Return experiment overrides safe to pass to YOLO train/val/predict calls."""
    skipped = {"model", "cfg", *exclude}
    source = cfg.get("_overrides", cfg)
    return {k: v for k, v in source.items() if k not in skipped and v is not None}

def event_image_size(cfg):
    """Return configured event image size as (height, width)."""
    size = cfg.get("event_image_size") or [240, 304]
    if len(size) != 2:
        raise ValueError(f"event_image_size must be [height, width], got {size}")
    return int(size[0]), int(size[1])

def data_names(data):
    """Load class names from a dataset YAML without validating dataset paths."""
    if not data:
        return {}
    data_path = Path(data)
    if not data_path.exists():
        candidate = ROOT / "ultralytics" / "cfg" / "datasets" / str(data)
        data_path = candidate if candidate.exists() else data_path
    if not data_path.exists() or data_path.suffix not in {".yaml", ".yml"}:
        return {}
    names = yaml_load(data_path).get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return names or {}
