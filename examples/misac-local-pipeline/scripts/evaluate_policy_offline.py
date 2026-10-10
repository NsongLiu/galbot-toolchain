"""Offline evaluation for a Galbot LeRobot policy checkpoint.

The primary metric is open-loop normalized action MAE: the policy predicts an
ACT action chunk without seeing the target action, and the prediction is
compared with the held-out action chunk in the same normalized space used by
training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from galbot.train._compat import apply_lerobot_compat_patch, apply_lerobot_runtime_compat_patch

apply_lerobot_compat_patch()
apply_lerobot_runtime_compat_patch()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Galbot policy on held-out LeRobot episodes.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split-json", help="JSON from prepare_misac_finetune.py")
    parser.add_argument("--split-name", choices=["train", "eval"], default="eval")
    parser.add_argument("--episodes", nargs="*", type=int, help="Episode ids; overrides --split-json")
    parser.add_argument("--manifest", help="Defaults to <dataset-root>/meta/conversion_manifest.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--processor-source", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Use a deterministic shuffled sample.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--include-train-forward-loss", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _load_episodes(args: argparse.Namespace) -> list[int]:
    if args.episodes:
        return sorted(args.episodes)
    if not args.split_json:
        raise ValueError("Either --episodes or --split-json is required.")
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    key = f"{args.split_name}_episodes"
    return sorted(int(x) for x in split[key])


def _load_task_map(dataset_root: Path, manifest_arg: str | None) -> dict[int, str]:
    manifest_path = Path(manifest_arg) if manifest_arg else dataset_root / "meta" / "conversion_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {int(e["episode_index"]): e.get("task", "unknown") for e in manifest.get("episodes", [])}


def _episode_groups(episodes: list[int], task_map: dict[int, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for episode in episodes:
        groups[task_map.get(episode, "unknown")].append(episode)
    return dict(sorted(groups.items()))


def _empty_acc(action_dim: int | None = None) -> dict[str, Any]:
    return {
        "num_batches": 0,
        "num_samples": 0,
        "valid_action_values": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "zero_sum_abs": 0.0,
        "train_forward_loss_sum": 0.0,
        "train_forward_l1_sum": 0.0,
        "train_forward_batches": 0,
        "action_dim": action_dim,
        "sum_abs_by_dim": None,
        "zero_sum_abs_by_dim": None,
        "valid_by_dim": None,
    }


def _merge_acc(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "num_batches",
        "num_samples",
        "valid_action_values",
        "sum_abs",
        "sum_sq",
        "zero_sum_abs",
        "train_forward_loss_sum",
        "train_forward_l1_sum",
        "train_forward_batches",
    ):
        dst[key] += src[key]
    if dst["action_dim"] is None:
        dst["action_dim"] = src["action_dim"]
    for key in ("sum_abs_by_dim", "zero_sum_abs_by_dim", "valid_by_dim"):
        if src[key] is None:
            continue
        if dst[key] is None:
            dst[key] = src[key].clone()
        else:
            dst[key] += src[key]


def _finalize(acc: dict[str, Any]) -> dict[str, Any]:
    count = max(float(acc["valid_action_values"]), 1.0)
    result = {
        "num_batches": int(acc["num_batches"]),
        "num_samples": int(acc["num_samples"]),
        "valid_action_values": int(acc["valid_action_values"]),
        "open_loop_l1": acc["sum_abs"] / count,
        "open_loop_rmse": math.sqrt(acc["sum_sq"] / count),
        "zero_action_l1": acc["zero_sum_abs"] / count,
    }
    if result["zero_action_l1"] > 0:
        result["relative_improvement_vs_zero_pct"] = (
            (result["zero_action_l1"] - result["open_loop_l1"]) / result["zero_action_l1"] * 100.0
        )
    if acc["train_forward_batches"]:
        denom = float(acc["train_forward_batches"])
        result["train_forward_loss"] = acc["train_forward_loss_sum"] / denom
        result["train_forward_l1_loss"] = acc["train_forward_l1_sum"] / denom
    if acc["sum_abs_by_dim"] is not None:
        valid_by_dim = torch.clamp(acc["valid_by_dim"], min=1.0)
        result["open_loop_l1_by_dim"] = (acc["sum_abs_by_dim"] / valid_by_dim).tolist()
        result["zero_action_l1_by_dim"] = (acc["zero_sum_abs_by_dim"] / valid_by_dim).tolist()
    return result


def _evaluate_episode_subset(
    *,
    repo_id: str,
    dataset_root: Path,
    episodes: list[int],
    delta_timestamps: dict,
    video_backend: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    shuffle: bool,
    seed: int,
    preprocessor,
    policy,
    include_train_forward_loss: bool,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=next(policy.parameters()).device.type == "cuda",
        drop_last=False,
        generator=generator if shuffle else None,
    )
    acc = _empty_acc()
    policy.eval()
    preprocessor.reset()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        samples = int(batch["action"].shape[0])
        with torch.inference_mode():
            processed = preprocessor(batch)
            policy.eval()
            pred = policy.predict_action_chunk(processed)
            target = processed["action"]
            valid = (~processed["action_is_pad"]).unsqueeze(-1)
            valid = valid.expand_as(target)
            diff = pred - target
            abs_diff = diff.abs() * valid
            sq_diff = diff.pow(2) * valid
            zero_abs = target.abs() * valid

            if acc["action_dim"] is None:
                acc["action_dim"] = int(target.shape[-1])
                acc["sum_abs_by_dim"] = torch.zeros(acc["action_dim"], dtype=torch.float64)
                acc["zero_sum_abs_by_dim"] = torch.zeros(acc["action_dim"], dtype=torch.float64)
                acc["valid_by_dim"] = torch.zeros(acc["action_dim"], dtype=torch.float64)

            acc["num_batches"] += 1
            acc["num_samples"] += samples
            acc["valid_action_values"] += float(valid.sum().item())
            acc["sum_abs"] += float(abs_diff.sum().item())
            acc["sum_sq"] += float(sq_diff.sum().item())
            acc["zero_sum_abs"] += float(zero_abs.sum().item())
            acc["sum_abs_by_dim"] += abs_diff.sum(dim=(0, 1)).double().cpu()
            acc["zero_sum_abs_by_dim"] += zero_abs.sum(dim=(0, 1)).double().cpu()
            acc["valid_by_dim"] += valid.sum(dim=(0, 1)).double().cpu()

            if include_train_forward_loss:
                policy.train()
                loss, out = policy.forward(processed)
                acc["train_forward_loss_sum"] += float(loss.item())
                acc["train_forward_l1_sum"] += float(out.get("l1_loss", 0.0))
                acc["train_forward_batches"] += 1
                policy.eval()

    return acc


def main() -> int:
    args = _parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path)
    episodes = _load_episodes(args)
    task_map = _load_task_map(dataset_root, args.manifest)
    episode_groups = _episode_groups(episodes, task_map)

    cfg = PreTrainedConfig.from_pretrained(model_path)
    cfg.pretrained_path = model_path
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        cfg.device = "cpu"
    elif args.device != "auto":
        cfg.device = args.device
    elif str(getattr(cfg, "device", "")).startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    for key in ds_meta.camera_keys:
        for stat_name, stat_val in IMAGENET_STATS.items():
            ds_meta.stats[key][stat_name] = torch.tensor(stat_val, dtype=torch.float32)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    processor_pretrained_path = model_path if args.processor_source == "model" else None
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=processor_pretrained_path,
        dataset_stats=ds_meta.stats,
    )

    overall_acc = _empty_acc()
    by_task: dict[str, dict[str, Any]] = {}
    for task, task_episodes in episode_groups.items():
        acc = _evaluate_episode_subset(
            repo_id=args.repo_id,
            dataset_root=dataset_root,
            episodes=task_episodes,
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            shuffle=args.shuffle,
            seed=args.seed + sum(ord(ch) for ch in task),
            preprocessor=preprocessor,
            policy=policy,
            include_train_forward_loss=args.include_train_forward_loss,
        )
        _merge_acc(overall_acc, acc)
        by_task[task] = _finalize(acc)
        print(
            f"{task}: episodes={len(task_episodes)} "
            f"open_loop_l1={by_task[task]['open_loop_l1']:.6f} "
            f"zero_l1={by_task[task]['zero_action_l1']:.6f}"
        )

    result = {
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "model_path": str(model_path),
        "processor_source": args.processor_source,
        "split_name": args.split_name,
        "episodes": episodes,
        "episode_groups": episode_groups,
        "overall": _finalize(overall_acc),
        "by_task": by_task,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"WROTE metrics={output_json}")
    print(
        "OVERALL "
        f"open_loop_l1={result['overall']['open_loop_l1']:.6f} "
        f"rmse={result['overall']['open_loop_rmse']:.6f} "
        f"zero_l1={result['overall']['zero_action_l1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
