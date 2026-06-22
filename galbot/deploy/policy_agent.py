"""LeRobot policy inference wrapper (runs in the Python 3.12 + LeRobot env).

Loads a pretrained policy and exposes ``inference(obs)`` / ``reset()``.
``obs`` dict contract:
  images: dict[str, np.ndarray]  # short camera name -> uint8 RGB HWC
  arm_gripper_joints: np.ndarray | list[float]  # state vector
"""

from __future__ import annotations

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


class PolicyAgent:
    """Thin wrapper around a pretrained LeRobot policy for real-robot inference."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
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
        print(f"[PolicyAgent] Loading model from: {self.model_path}")
        policy = PreTrainedPolicy.from_pretrained(self.model_path)
        print(f"[PolicyAgent] Model loaded on {policy.config.device}")
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
        dev = str(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.model_path),
            preprocessor_overrides={
                "device_processor": {"device": dev, "float_dtype": None},
                "rename_observations_processor": {"rename_map": {}},
            },
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
            if cam_name not in imgs:
                need = ", ".join(k.rsplit(".", maxsplit=1)[-1] for k in self._image_keys)
                avail = ", ".join(sorted(imgs.keys()))
                raise ValueError(
                    f"obs['images'] missing {cam_name!r} (model expects: {need}); got keys: [{avail}]"
                )
            img = cv2.resize(imgs[cam_name], dsize=(target_w, target_h))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[model_key] = img_t.unsqueeze(0).to(self.device, non_blocking=True)

        if self._state_key is not None:
            state = obs["arm_gripper_joints"]
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
            batch[self._state_key] = state_t.unsqueeze(0).to(self.device, non_blocking=True)

        return batch
