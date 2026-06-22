"""YAML config loading + PolicyClient factory.

No ROS2 imports; usable by both headless replay and a future robot node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RUN_MODES = frozenset({"model", "replay", "replay_debug"})

DEFAULT_CONFIG = {
    "mode": "model",
    "policy_server_url": "tcp://127.0.0.1:5555",
    "policy_zmq_timeout_ms": 120_000,
    # TO DEBUG: adjust to the LeRobot V3 dataset root used for replay modes.
    "dataset_root": "/media/jushen/Leslie-liu/galbot_dataset/lerobot_v3",
    "dataset_repo_id": "unitree_g1",
    "episode_index": 0,
    "action_rate": 30.0,
    "replay_max_steps": 0,
    # TO DEBUG: camera short-name must match the model's input_features keys.
    "obs_camera_key": None,
    "image_save": {
        "enabled": False,
        "directory": "debug_policy_images",
        "interval": 1,
        "max_frames": 0,
        "use_timestamp_subdir": True,
    },
    "joints": {
        "enabled": False,
        "directory": "debug_joints",
        "use_timestamp_subdir": True,
    },
}


def load_config(config_path: str | Path | None, logger: Any | None = None) -> dict:
    """Load and merge YAML config with defaults."""
    if config_path is None:
        config = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    merged = {**DEFAULT_CONFIG, **config}

    img_def = dict(DEFAULT_CONFIG.get("image_save") or {})
    img_user = config.get("image_save") if isinstance(config.get("image_save"), dict) else {}
    merged["image_save"] = {**img_def, **img_user}

    jt_def = dict(DEFAULT_CONFIG.get("joints") or {})
    jt_user = config.get("joints") if isinstance(config.get("joints"), dict) else {}
    merged["joints"] = {**jt_def, **jt_user}

    mode = merged.get("mode")
    if mode not in RUN_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(RUN_MODES)}")

    if logger is not None:
        logger.info(f"Configuration loaded from: {config_path}")

    return merged


def make_policy_client(config: dict, logger: Any | None = None):
    """Factory for PolicyClient from merged config."""
    from policy_client import PolicyClient

    return PolicyClient(
        server_url=config["policy_server_url"],
        timeout_ms=int(config.get("policy_zmq_timeout_ms", 120_000)),
        logger=logger,
        image_save=config.get("image_save"),
        joints=config.get("joints"),
    )
