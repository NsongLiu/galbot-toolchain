"""LeRobot V3 dataset -> ZMQ replay loop (no robot control dependencies).

Reads observations from a LeRobot V3 episode and sends each step to
``policy_server``, logging the returned action. This validates the full wire +
model pipeline without real hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config_loader import load_config, make_policy_client


class _StdioLogger:
    def info(self, msg: str) -> None:
        print(msg)

    def warning(self, msg: str) -> None:
        print(f"[WARN] {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)


def _load_episode_observations(cfg: dict, logger):
    """Load RGB + state observations from a LeRobot V3 episode."""
    # TO DEBUG: this import requires the LeRobot / xhum-new environment.
    # The robot control environment should NOT need this module.
    from galbot.train._compat import apply_lerobot_compat_patch

    apply_lerobot_compat_patch()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        cfg["dataset_repo_id"],
        root=cfg["dataset_root"],
        episodes=[int(cfg["episode_index"])],
        video_backend=cfg.get("video_backend", "pyav"),
    )

    # TO DEBUG: derive camera short-name mapping from model config or YAML.
    obs_camera_key = cfg.get("obs_camera_key")
    camera_keys = ds.meta.camera_keys
    if not camera_keys:
        raise RuntimeError("dataset has no camera keys")

    obs_list = []
    for frame_idx in range(ds.num_frames):
        item = ds[frame_idx]
        images = {}
        for cam_key in camera_keys:
            # cam_key is e.g. "observation.images.left_arm"; short name is "left_arm".
            short = cam_key.rsplit(".", maxsplit=1)[-1]
            if obs_camera_key is not None and short != obs_camera_key:
                continue
            img = item[cam_key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            # TO DEBUG: verify pixel range and channel order (LeRobot stores CHW tensor).
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            images[short] = img

        state = item.get("observation.state")
        if state is None:
            raise RuntimeError("dataset item missing observation.state")
        if hasattr(state, "numpy"):
            state = state.numpy()

        obs_list.append({"images": images, "arm_gripper_joints": state})

    return obs_list


def run(config_path: str) -> int:
    log = _StdioLogger()
    cfg = load_config(config_path, log)
    if cfg.get("mode") not in {"replay", "replay_debug"}:
        log.error(f"mode must be 'replay' or 'replay_debug' (got {cfg.get('mode')!r})")
        return 1

    url = cfg.get("policy_server_url", "")
    if not url:
        log.error("replay: set policy_server_url in YAML")
        return 1

    try:
        obs_list = _load_episode_observations(cfg, log)
    except Exception as e:
        log.error(f"load episode observations failed: {e}")
        return 1

    if not obs_list:
        log.error("empty observation list")
        return 1

    max_steps = int(cfg.get("replay_max_steps", 0) or 0)
    max_steps = len(obs_list) if max_steps <= 0 else min(max_steps, len(obs_list))

    period = 1.0 / max(float(cfg.get("action_rate", 30.0)), 1e-6)
    zmq_to = int(cfg.get("policy_zmq_timeout_ms", 120_000))
    log.info(
        f"replay: {max_steps} ZMQ inference steps  policy_server={url}  "
        f"policy_zmq_timeout_ms={zmq_to}"
    )

    client = make_policy_client(cfg, log)
    try:
        client.reset()
        for i in range(max_steps):
            t0 = time.perf_counter()
            action = client.inference(obs_list[i])
            dt = time.perf_counter() - t0
            row = action[0]
            log.info(
                f"step {i + 1}/{max_steps}  wall={dt:.3f}s  action_shape={tuple(action.shape)}  "
                f"|a|_mean={float(np.mean(np.abs(row))):.4f}"
            )
            time.sleep(period)
    finally:
        client.close()

    log.info("replay: finished OK")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LeRobot V3 dataset -> ZMQ replay (no robot)")
    p.add_argument("--config", type=str, required=True)
    raise SystemExit(run(p.parse_args().config))
