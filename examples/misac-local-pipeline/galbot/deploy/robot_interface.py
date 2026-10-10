"""Galbot G1 robot control interface using GalbotSDK.

This module is intentionally dependency-free at import time: ``galbot_sdk`` is
imported lazily inside ``connect()`` so the file can be syntax-checked and
inspected even when the SDK is not installed in the current environment.

All observation / action mappings are marked with ``# TO DEBUG`` and must be
validated against the real G1 robot SDK and the trained model's input/output
features.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GalbotRobotInterface:
    """Thin wrapper around ``galbot_sdk.GalbotSDK`` for policy inference loops.

    Expected SDK contract (from G1 developer docs):
      - ``GalbotSDK.connect(...)`` returns a robot handle.
      - ``handle.get_observation()`` returns a dict with joint states and RGB images.
      - ``handle.set_joint_positions(targets)`` commands arm/hand joints.
      - ``handle.set_base_velocity(vx, vy, w)`` commands the chassis.

    # TO DEBUG: the exact method names, return shapes, and coordinate conventions
    must be confirmed against the installed ``galbot_sdk`` package.
    """

    def __init__(self) -> None:
        self._robot: Any | None = None
        self._connected = False

    def connect(self, **kwargs) -> None:
        """Connect to the robot via GalbotSDK.

        # TO DEBUG: pass the correct connection arguments (IP, robot_id, etc.)
        as required by your deployment setup.
        """
        # Lazy import so this module can be loaded without galbot_sdk installed.
        from galbot_sdk import GalbotSDK

        self._robot = GalbotSDK.connect(**kwargs)
        self._connected = True
        print(f"[GalbotRobotInterface] connected: {self._robot}")

    def disconnect(self) -> None:
        """Release robot connection if the SDK exposes a close/disconnect method."""
        if self._robot is None:
            return
        # TO DEBUG: verify the SDK's lifecycle method name (close / disconnect / stop).
        close_fn = getattr(self._robot, "disconnect", None) or getattr(self._robot, "close", None)
        if callable(close_fn):
            close_fn()
        self._robot = None
        self._connected = False
        print("[GalbotRobotInterface] disconnected")

    def is_connected(self) -> bool:
        return self._connected and self._robot is not None

    def get_observation(self) -> dict[str, Any]:
        """Fetch robot observation and convert to PolicyClient format.

        Returns:
            {
                "images": {short_cam_name: uint8 RGB HWC ndarray, ...},
                "arm_gripper_joints": float32 ndarray of state,
            }

        # TO DEBUG: the exact key names, image formats, and state ordering from
        ``galbot_sdk`` must match the trained policy's input_features.
        """
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")

        raw_obs = self._robot.get_observation()

        # ------------------------------------------------------------------
        # TO DEBUG: adjust field names to match GalbotSDK observation schema.
        # The names below are placeholders inferred from the G1 SDK docs.
        # ------------------------------------------------------------------
        raw_images = raw_obs.get("images") or raw_obs.get("cameras") or {}
        raw_state = raw_obs.get("joint_states") or raw_obs.get("joint_positions") or raw_obs.get("state")

        images: dict[str, np.ndarray] = {}
        for cam_name, img in raw_images.items():
            img_arr = np.asarray(img)
            # TO DEBUG: verify SDK returns RGB; convert BGR -> RGB if necessary.
            if img_arr.ndim == 3 and img_arr.shape[-1] == 3:
                images[str(cam_name)] = img_arr.astype(np.uint8)
            else:
                raise ValueError(f"unexpected image shape for {cam_name}: {img_arr.shape}")

        if raw_state is None:
            raise RuntimeError("observation does not contain joint/state data")
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)

        return {"images": images, "arm_gripper_joints": state}

    def apply_action(self, action: np.ndarray) -> None:
        """Send a policy action to the robot.

        # TO DEBUG: the action vector layout must be aligned with the robot's
        joint command order. The split between arm joints and gripper/base below
        is a placeholder.
        """
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 0:
            raise ValueError("empty action")

        # ------------------------------------------------------------------
        # TO DEBUG: split the action vector according to the robot's DOF layout.
        # The numbers below are placeholders and must match robot_config.json.
        # ------------------------------------------------------------------
        arm_dim = 7  # left arm
        gripper_dim = 1
        base_dim = 0

        joints = action[:arm_dim].tolist()
        if len(action) > arm_dim:
            gripper = action[arm_dim : arm_dim + gripper_dim].tolist()
        else:
            gripper = None

        # TO DEBUG: confirm set_joint_positions accepts list/array and units (rad/deg).
        self._robot.set_joint_positions(joints)
        if gripper is not None:
            # TO DEBUG: confirm gripper command method name and value range.
            set_gripper = getattr(self._robot, "set_gripper_positions", None) or getattr(
                self._robot, "set_hand_positions", None
            )
            if callable(set_gripper):
                set_gripper(gripper)

        # TO DEBUG: enable only if the policy outputs chassis velocity commands.
        if base_dim > 0 and len(action) >= arm_dim + gripper_dim + base_dim:
            vx, vy, w = action[arm_dim + gripper_dim : arm_dim + gripper_dim + 3].tolist()
            self._robot.set_base_velocity(vx, vy, w)

    def reset(self) -> None:
        """Optional robot-side reset (home pose, clear faults, etc.)."""
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")
        # TO DEBUG: implement reset behavior if the SDK provides it (e.g. go_home).
        reset_fn = getattr(self._robot, "reset", None) or getattr(self._robot, "go_home", None)
        if callable(reset_fn):
            reset_fn()
