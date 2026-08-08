"""LeRobot policy inference wrapper (runs in the Python 3.12 + LeRobot env).

Loads a pretrained policy and exposes ``inference(obs)`` / ``reset()``.
``obs`` dict contract:
  images: dict[str, np.ndarray]  # short camera name -> uint8 RGB HWC
  arm_gripper_joints: np.ndarray | list[float]  # state vector
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from galbot.train._compat import apply_lerobot_compat_patch

apply_lerobot_compat_patch()

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy

# --- pi05 / VLA 类策略的部署设置（ACT 等无语言条件的策略会忽略） ---
# 任务指令：作为语言条件注入每个观测。切换任务时只需改这一行。
PI05_TASK = "task1"
# 本地 PaliGemma tokenizer 路径（离线加载）。早期 checkpoint 的 policy_preprocessor.json
# 未保存 tokenizer 名，需在此显式指定；新 checkpoint 自带 tokenizer_name 时可置 None。
PI05_TOKENIZER_PATH: str | None = "/media/jushen/Leslie-liu/leslie-liu/pretrained/paligemma-3b-pt-224"


class PolicyAgent:
    """Thin wrapper around a pretrained LeRobot policy for real-robot inference."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.policy_type: str = ""
        self._cam_aliases: dict[str, str] = {}  # policy 侧相机短名 -> 数据集相机短名
        self._missing_cam_notified: set[str] = set()
        self.policy = self._load_policy()
        self.device = next(self.policy.parameters()).device

        self._image_keys: dict[str, tuple[int, int]] = {}
        self._state_key: str | None = None
        self._state_dim: int = 0
        self._action_dim: int = 0
        self._parse_model_config()

        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self._load_pre_post_processors()

    def _load_policy(self) -> PreTrainedPolicy:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        print(f"[PolicyAgent] Loading model from: {self.model_path}")
        cfg = PreTrainedConfig.from_pretrained(self.model_path)
        # 策略类型（"act" / "pi05" / ...）从 checkpoint config 自动识别，无需手动切换。
        self.policy_type = cfg.type
        policy = get_policy_class(cfg.type).from_pretrained(self.model_path, config=cfg)
        print(f"[PolicyAgent] Model loaded on {policy.config.device} (type={self.policy_type})")
        return policy

    def _parse_model_config(self) -> None:
        cfg = self.policy.config

        for key, feat in cfg.input_features.items():
            if feat.type is FeatureType.VISUAL:
                _, h, w = feat.shape
                self._image_keys[key] = (w, h)
            elif feat.type is FeatureType.STATE:
                self._state_key = key
                self._state_dim = feat.shape[0]

        for _, feat in cfg.output_features.items():
            if feat.type is FeatureType.ACTION:
                self._action_dim = feat.shape[0]

        cam_list = ", ".join(f"{k} {v}" for k, v in self._image_keys.items())
        print(f"[PolicyAgent] Cameras  : {cam_list}")
        print(f"[PolicyAgent] State dim: {self._state_dim}  Action dim: {self._action_dim}")

    def _load_pre_post_processors(self) -> None:
        """Load LeRobot pre/post processors if exported next to the checkpoint."""
        pre_json = self.model_path / "policy_preprocessor.json"
        post_json = self.model_path / "policy_postprocessor.json"
        if not (pre_json.is_file() and post_json.is_file()):
            print(
                "[PolicyAgent] No policy_preprocessor.json / policy_postprocessor.json; "
                "inference uses raw select_action (no dataset normalize / no action denorm).",
                flush=True,
            )
            return
        # 相机短名别名：训练时 rename_map 把数据集相机名映射成了 policy 侧名称，
        # 允许客户端/replay 继续发送数据集短名（如 front_head_right -> base_0_rgb）。
        try:
            for step in json.loads(pre_json.read_text()).get("steps", []):
                if step.get("registry_name") == "rename_observations_processor":
                    rename_map = (step.get("config") or {}).get("rename_map") or {}
                    self._cam_aliases = {
                        v.rsplit(".", maxsplit=1)[-1]: k.rsplit(".", maxsplit=1)[-1]
                        for k, v in rename_map.items()
                    }
        except (json.JSONDecodeError, OSError):
            pass

        dev = str(self.device)
        pre_overrides: dict[str, Any] = {
            "device_processor": {"device": dev, "float_dtype": None},
            "rename_observations_processor": {"rename_map": {}},
        }
        if self.policy_type == "pi05" and PI05_TOKENIZER_PATH:
            # 早期 checkpoint 的 policy_preprocessor.json 未保存 tokenizer 名，此处补上本地路径。
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": PI05_TOKENIZER_PATH}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.model_path),
            preprocessor_overrides=pre_overrides,
            postprocessor_overrides={"device_processor": {"device": dev, "float_dtype": None}},
        )
        print(
            "[PolicyAgent] Loaded LeRobot preprocessor + postprocessor.",
            flush=True,
        )

    def inference(self, obs: dict) -> torch.Tensor:
        batch = self._prepare_batch(obs)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        return action

    def reset(self) -> None:
        self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

    def _prepare_batch(self, obs: dict) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        imgs = obs.get("images")
        if not isinstance(imgs, dict) or not imgs:
            raise ValueError("obs must have non-empty dict obs['images'] (short_name -> uint8 RGB HWC)")

        # TO DEBUG: camera short-name mapping between robot SDK and model checkpoint.
        for model_key, (target_w, target_h) in self._image_keys.items():
            cam_name = model_key.rsplit(".", maxsplit=1)[-1]
            img = imgs.get(cam_name)
            if img is None and cam_name in self._cam_aliases:
                img = imgs.get(self._cam_aliases[cam_name])
            if img is None:
                if self.policy_type == "pi05":
                    # pi05 对缺失相机内部以零图 + mask=0 处理（openpi 行为），跳过即可。
                    if cam_name not in self._missing_cam_notified:
                        print(f"[PolicyAgent] camera {cam_name!r} not provided; model will mask it.", flush=True)
                        self._missing_cam_notified.add(cam_name)
                    continue
                need = ", ".join(k.rsplit(".", maxsplit=1)[-1] for k in self._image_keys)
                avail = ", ".join(sorted(imgs.keys()))
                raise ValueError(
                    f"obs['images'] missing {cam_name!r} (model expects: {need}); got keys: [{avail}]"
                )
            img = cv2.resize(img, dsize=(target_w, target_h))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[model_key] = img_t.unsqueeze(0).to(self.device, non_blocking=True)

        if self._state_key is not None:
            state = obs["arm_gripper_joints"]
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
            batch[self._state_key] = state_t.unsqueeze(0).to(self.device, non_blocking=True)

        if self.policy_type == "pi05":
            # VLA 语言指令；preprocessor 的 AddBatchDimension 会包装成 [str]。
            batch["task"] = PI05_TASK

        return batch
