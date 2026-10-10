"""Verify a local LeRobot fine-tune artifact for Galbot.

Checks:
  1. LeRobot can load the dataset.
  2. A saved policy checkpoint can be loaded by PolicyAgent.
  3. A dummy observation can run through local inference and returns the
     expected action shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Galbot LeRobot dataset/checkpoint smoke outputs.")
    parser.add_argument("--dataset-root", required=True, help="Path to a LeRobot V3 dataset root")
    parser.add_argument("--repo-id", required=True, help="Dataset repo_id stored in the LeRobot metadata")
    parser.add_argument("--model-path", help="Path to checkpoints/.../pretrained_model")
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def _verify_dataset(repo_id: str, dataset_root: str, video_backend: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=dataset_root, video_backend=video_backend)
    if ds.num_frames <= 0:
        raise RuntimeError("dataset has no frames")
    if ds.num_episodes <= 0:
        raise RuntimeError("dataset has no episodes")

    item = ds[0]
    state = item["observation.state"]
    action = item["action"]
    print(
        "VERIFY DATASET OK "
        f"frames={ds.num_frames} episodes={ds.num_episodes} "
        f"state_shape={tuple(state.shape)} action_shape={tuple(action.shape)} "
        f"cameras={ds.meta.camera_keys}"
    )
    return ds


def _verify_model(model_path: str) -> None:
    from galbot.deploy.policy_agent import PolicyAgent

    agent = PolicyAgent(model_path)
    images = {}
    for model_key, (target_w, target_h) in agent._image_keys.items():
        cam_name = model_key.rsplit(".", maxsplit=1)[-1]
        images[cam_name] = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    obs = {
        "images": images,
        "arm_gripper_joints": np.zeros(agent._state_dim, dtype=np.float32),
    }
    agent.reset()
    action = agent.inference(obs)
    got_shape = tuple(action.shape)
    expected_shape = (1, agent._action_dim)
    if got_shape != expected_shape:
        raise RuntimeError(f"unexpected action shape: got {got_shape}, expected {expected_shape}")
    print(f"VERIFY MODEL OK model_path={model_path} action_shape={got_shape}")


def main() -> int:
    args = _parse_args()
    _verify_dataset(args.repo_id, args.dataset_root, args.video_backend)
    if args.model_path:
        _verify_model(args.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
