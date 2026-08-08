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
    "dataset_repo_id": "galbot_g1",
    "episode_index": 0,
    "action_rate": 30.0,
    "replay_max_steps": 0,
    # TO DEBUG: camera short-name must match the model's input_features keys.
    "obs_camera_key": None,
    # mode=model 真机控制端参数（默认值取自真机调通脚本，见 robot_interface.py）
    "robot": {
        # 真机(Orin)上 galbot_sdk 的系统路径；本机已安装 SDK 则置 null
        "sdk_lib_path": "/userdata/update/manual_update/lib",
        "camera_warmup_sec": 5.0,
        "rgb_timeout_sec": 10.0,
        "sensors": {
            "right_arm": "RIGHT_ARM_CAMERA",
            "front_head_right": "HEAD_RIGHT_CAMERA",
        },
        "max_speed": 0.05,
        "timeout": 1.0,
        "gripper_speed": 0.1,
        "gripper_force": 10.0,
        "gripper_min_raw": 35.0,
        "gripper_max_raw": 100.0,
    },
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

    rb_def = dict(DEFAULT_CONFIG.get("robot") or {})
    rb_user = config.get("robot") if isinstance(config.get("robot"), dict) else {}
    merged["robot"] = {**rb_def, **rb_user}

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
