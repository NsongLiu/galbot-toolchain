"""Replay a dataset episode through a trained checkpoint and compare predicted vs GT actions.

Usage:
    PYTHONPATH=. python eval_trajectory.py \
        --checkpoint <ckpt>/pretrained_model \
        --dataset-root <lerobot_v3>/<dataset> --repo-id <repo_id> \
        --episode 0 --out eval_ep0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from galbot.train._compat import apply_lerobot_compat_patch

apply_lerobot_compat_patch()

from galbot.deploy.policy_agent import PolicyAgent
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to <ckpt>/pretrained_model")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--task", default=None, help="VLA 语言指令（pi05 用；缺省用 policy_agent 内置值）")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full episode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agent = PolicyAgent(args.checkpoint, task=args.task)
    agent.reset()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[args.episode], video_backend="pyav")
    action_names = ds.meta.features["action"].get("names") or [f"joint{i}" for i in range(ds.meta.features["action"].shape[0])]
    n = len(ds) if args.max_steps <= 0 else min(len(ds), args.max_steps)
    print(f"Episode {args.episode}: {n} frames, action joints: {action_names}")

    preds, gts = [], []
    for i in range(n):
        item = ds[i]
        images = {}
        for cam_key in ds.meta.camera_keys:
            img = item[cam_key]  # (3,H,W) float [0,1]
            images[cam_key.rsplit(".", maxsplit=1)[-1]] = (
                (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            )
        obs = {"images": images, "arm_gripper_joints": item["observation.state"].numpy()}
        with torch.no_grad():
            action = agent.inference(obs)
        preds.append(action.squeeze(0).cpu().float().numpy())
        gts.append(item["action"].numpy())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}")

    preds = np.stack(preds)
    gts = np.stack(gts)
    mae = np.abs(preds - gts).mean(axis=0)
    print("\n=== Trajectory tracking (MAE, dataset units) ===")
    for name, m in zip(action_names, mae, strict=True):
        print(f"  {name:24s} {m:.4f}")
    print(f"  {'overall':24s} {mae.mean():.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_j = len(action_names)
    fig, axes = plt.subplots((n_j + 1) // 2, 2, figsize=(14, 2.2 * ((n_j + 1) // 2)), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes.flat)):
        if j >= n_j:
            ax.axis("off")
            continue
        ax.plot(gts[:, j], label="GT", lw=1.5)
        ax.plot(preds[:, j], label="pred", lw=1.1, alpha=0.85)
        ax.set_title(f"{action_names[j]}  MAE={mae[j]:.4f}", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(f"{Path(args.checkpoint).parent.name}  episode {args.episode}")
    fig.tight_layout()
    out = args.out or f"eval_ep{args.episode}.png"
    fig.savefig(out, dpi=120)
    print(f"\nplot saved: {out}")


if __name__ == "__main__":
    main()
