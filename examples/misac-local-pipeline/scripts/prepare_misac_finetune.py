"""Prepare a stratified MiSAC fine-tune split and Galbot training config."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MiSAC train/eval split and training JSON.")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "assets" / "lerobot_datasets" / "example"),
    )
    parser.add_argument("--repo-id", default="example_right_arm")
    parser.add_argument("--manifest", help="Defaults to <dataset-root>/meta/conversion_manifest.json")
    parser.add_argument(
        "--base-model-path",
        default=str(
            PROJECT_ROOT
            / "assets"
            / "models"
            / "base_act"
            / "checkpoints"
            / "last"
            / "pretrained_model"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "assets" / "models" / "act_finetune"),
    )
    parser.add_argument(
        "--split-out",
        default=str(PROJECT_ROOT / "assets" / "config" / "example_split.json"),
    )
    parser.add_argument(
        "--config-out",
        default=str(PROJECT_ROOT / "assets" / "config" / "example_finetune.json"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=250)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    return parser.parse_args()


def _stratified_split(episodes: list[dict], eval_fraction: float, seed: int) -> dict:
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.get("task", "unknown")].append(int(episode["episode_index"]))

    rng = random.Random(seed)
    split_tasks = {}
    train_episodes: list[int] = []
    eval_episodes: list[int] = []

    for task, episode_ids in sorted(by_task.items()):
        shuffled = list(sorted(episode_ids))
        rng.shuffle(shuffled)
        eval_count = max(1, round(len(shuffled) * eval_fraction)) if len(shuffled) > 1 else 0
        eval_ids = sorted(shuffled[:eval_count])
        train_ids = sorted(shuffled[eval_count:])
        if not train_ids and eval_ids:
            train_ids.append(eval_ids.pop())
        split_tasks[task] = {"train": train_ids, "eval": eval_ids}
        train_episodes.extend(train_ids)
        eval_episodes.extend(eval_ids)

    return {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "train_episodes": sorted(train_episodes),
        "eval_episodes": sorted(eval_episodes),
        "tasks": split_tasks,
    }


def main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest) if args.manifest else dataset_root / "meta" / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]

    split = _stratified_split(episodes, args.eval_fraction, args.seed)
    split.update(
        {
            "dataset_root": str(dataset_root),
            "repo_id": args.repo_id,
            "manifest": str(manifest_path),
        }
    )

    train_cfg = {
        "dataset": {
            "repo_id": args.repo_id,
            "root": str(dataset_root).replace("\\", "/"),
            "episodes": split["train_episodes"],
        },
        "policy": {
            "type": "act",
            "path": str(Path(args.base_model_path)).replace("\\", "/"),
            "load_processors_from_path": False,
            "push_to_hub": False,
        },
        "training": {
            "output_dir": str(Path(args.output_dir)).replace("\\", "/"),
            "batch_size": args.batch_size,
            "steps": args.steps,
            "num_workers": args.num_workers,
            "save_freq": args.save_freq,
            "log_freq": args.log_freq,
            "seed": args.seed,
            "resume": False,
            "resume_dir": None,
        },
        "use_imagenet_stats": True,
        "video_backend": "pyav",
    }

    split_out = Path(args.split_out)
    config_out = Path(args.config_out)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    config_out.write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")

    print(f"WROTE split={split_out}")
    print(f"WROTE config={config_out}")
    print(f"train_episodes={len(split['train_episodes'])} eval_episodes={len(split['eval_episodes'])}")
    for task, task_split in split["tasks"].items():
        print(f"{task}: train={len(task_split['train'])} eval={len(task_split['eval'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
