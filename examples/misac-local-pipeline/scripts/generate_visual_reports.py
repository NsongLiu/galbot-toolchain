"""Generate reproducible visual reports for the local Galbot MiSAC pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_with_project_python() -> None:
    project_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not project_python.is_file():
        return
    if Path(sys.executable).resolve() == project_python.resolve():
        return
    if importlib.util.find_spec("lerobot") is not None:
        return

    env = os.environ.copy()
    env["GALBOT_VISUAL_REPORT_REEXEC"] = "1"
    print(
        f"[runtime] Current Python has no lerobot: {sys.executable}\n"
        f"[runtime] Re-running with project Python: {project_python}",
        flush=True,
    )
    completed = subprocess.run([str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(completed.returncode)


if os.environ.get("GALBOT_VISUAL_REPORT_REEXEC") != "1":
    _maybe_reexec_with_project_python()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from galbot.train._compat import apply_lerobot_compat_patch, apply_lerobot_runtime_compat_patch

apply_lerobot_compat_patch()
apply_lerobot_runtime_compat_patch()


def _project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Galbot MiSAC PNG/CSV visual reports.")
    parser.add_argument("--dataset-root", default="assets/lerobot_datasets/example")
    parser.add_argument("--repo-id", default="example_right_arm")
    parser.add_argument("--split-json", default="assets/config/example_split.json")
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument(
        "--model-path",
        default="assets/models/act_finetune/checkpoints/001000/pretrained_model",
    )
    parser.add_argument(
        "--base-model-path",
        default="assets/models/base_act/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--metrics-suffix", default="auto", help="auto, sample2x8, or sample32x8.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Which eval sample to visualize.")
    parser.add_argument("--max-trajectory-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_file(eval_dir: Path, label: str, suffix: str) -> Path:
    if label == "Initial":
        return eval_dir / f"misac_eval_initial_{suffix}.json"
    return eval_dir / f"misac_eval_finetune_{int(label):06d}_{suffix}.json"


def _choose_suffix(eval_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    for suffix in ("sample32x8", "sample2x8"):
        if _metric_file(eval_dir, "Initial", suffix).is_file() and _metric_file(eval_dir, "1000", suffix).is_file():
            return suffix
    raise FileNotFoundError(f"No complete metrics set found in {eval_dir}")


def _load_metric_rows(eval_dir: Path, suffix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ("Initial", "250", "500", "750", "1000"):
        path = _metric_file(eval_dir, label, suffix)
        if not path.is_file():
            continue
        data = _load_json(path)
        overall = data["overall"]
        rows.append(
            {
                "label": label,
                "path": path,
                "data": data,
                "l1": float(overall["open_loop_l1"]),
                "rmse": float(overall["open_loop_rmse"]),
                "zero": float(overall["zero_action_l1"]),
                "by_dim": overall.get("open_loop_l1_by_dim", []),
                "zero_by_dim": overall.get("zero_action_l1_by_dim", []),
                "by_task": data.get("by_task", {}),
            }
        )
    if len(rows) < 2:
        raise RuntimeError(f"Need at least initial and final metrics in {eval_dir}; found {len(rows)}")
    return rows


def _write_metrics_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "open_loop_l1", "rmse", "zero_action_l1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "checkpoint": row["label"],
                    "open_loop_l1": f"{row['l1']:.9f}",
                    "rmse": f"{row['rmse']:.9f}",
                    "zero_action_l1": f"{row['zero']:.9f}",
                }
            )


def _save_overview(rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    l1 = [r["l1"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    zero = [r["zero"] for r in rows]
    best = min(rows, key=lambda r: r["l1"])
    first = rows[0]
    improvement = (first["l1"] - best["l1"]) / first["l1"] * 100 if first["l1"] else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Galbot MiSAC fine-tune overview", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(labels, l1, marker="o", color="#2458a6", linewidth=2.5, label="open_loop_l1")
    ax.plot(labels, rmse, marker="s", color="#b45f06", linewidth=2.0, label="rmse")
    ax.set_title(f"Checkpoint metrics; best={best['label']}, improvement={improvement:.1f}%")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("error; lower is better")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    width = 0.38
    x = list(range(len(labels)))
    ax.bar([i - width / 2 for i in x], l1, width=width, color="#4f7cac", label="policy")
    ax.bar([i + width / 2 for i in x], zero, width=width, color="#8b8f97", label="zero-action baseline")
    ax.set_xticks(x, labels)
    ax.set_title("Policy error vs zero-action baseline")
    ax.set_ylabel("open_loop_l1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    final = rows[-1]
    ax = axes[1, 0]
    tasks = sorted(final["by_task"])
    task_l1 = [final["by_task"][task]["open_loop_l1"] for task in tasks]
    task_zero = [final["by_task"][task]["zero_action_l1"] for task in tasks]
    tx = list(range(len(tasks)))
    ax.bar([i - width / 2 for i in tx], task_l1, width=width, color="#2f855a", label="1000-step policy")
    ax.bar([i + width / 2 for i in tx], task_zero, width=width, color="#8b8f97", label="zero-action baseline")
    ax.set_xticks(tx, tasks)
    ax.set_title("Held-out task breakdown")
    ax.set_ylabel("open_loop_l1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    dims = list(range(1, len(final["by_dim"]) + 1))
    ax.plot(dims, final["by_dim"], marker="o", color="#2b6cb0", linewidth=2.2, label="1000-step policy")
    if final["zero_by_dim"]:
        ax.plot(dims, final["zero_by_dim"], marker="x", color="#718096", linewidth=1.8, label="zero baseline")
    ax.set_title("1000-step error by action dimension")
    ax.set_xlabel("right_arm action dimension")
    ax.set_ylabel("L1")
    ax.set_xticks(dims)
    ax.grid(alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_validation_flow(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title("Galbot MiSAC validation flow", fontsize=18, fontweight="bold", pad=18)

    boxes = [
        ("MiSAC MCAP\nraw episodes", 0.06, 0.70, "#e6f0ff"),
        ("convert_mcap_to_lerobot.py\nright_arm + 224px image", 0.25, 0.70, "#edf7ed"),
        ("LeRobot dataset\nstate, image, action chunk", 0.46, 0.70, "#fff4df"),
        ("prepare_misac_finetune.py\ntrain / eval split", 0.67, 0.70, "#f1ecff"),
        ("galbot.train.train\nACT fine-tune checkpoints", 0.25, 0.35, "#f9e8e8"),
        ("evaluate_policy_offline.py\npredict chunk without target action", 0.50, 0.35, "#e8f5f7"),
        ("visual reports\nmetrics + trajectory comparison", 0.75, 0.35, "#f1f5f9"),
    ]
    for text, x, y, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="#334155", linewidth=1.2),
            transform=ax.transAxes,
        )

    arrows = [
        ((0.14, 0.70), (0.20, 0.70)),
        ((0.34, 0.70), (0.39, 0.70)),
        ((0.55, 0.70), (0.61, 0.70)),
        ((0.67, 0.60), (0.33, 0.45)),
        ((0.42, 0.35), (0.44, 0.35)),
        ((0.62, 0.35), (0.67, 0.35)),
        ((0.75, 0.60), (0.58, 0.45)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.7),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )

    ax.text(
        0.50,
        0.08,
        "Offline validation compares predicted future action chunks against held-out demonstration action chunks.",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
        transform=ax.transAxes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _load_eval_episodes(split_json: Path) -> list[int]:
    split = _load_json(split_json)
    return [int(x) for x in split["eval_episodes"]]


def _load_action_names(dataset_root: Path) -> list[str]:
    info = _load_json(dataset_root / "meta" / "info.json")
    names = info["features"]["action"].get("names")
    if names:
        return [str(x) for x in names]
    shape = info["features"]["action"]["shape"]
    return [f"action_{i + 1}" for i in range(int(shape[0]))]


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _load_sample_batch(
    *,
    repo_id: str,
    dataset_root: Path,
    split_json: Path,
    delta_timestamps: dict[str, Any],
    video_backend: str,
    sample_offset: int,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = _load_eval_episodes(split_json)
    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    if sample_offset < 0 or sample_offset >= len(ds):
        raise IndexError(f"sample_offset {sample_offset} outside eval dataset length {len(ds)}")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    for idx, batch in enumerate(dataloader):
        if idx == sample_offset:
            return batch
    raise RuntimeError(f"Could not load sample_offset {sample_offset}")


def _predict_chunk(
    *,
    repo_id: str,
    dataset_root: Path,
    model_path: Path,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import IMAGENET_STATS
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(model_path)
    cfg.pretrained_path = model_path

    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    for key in ds_meta.camera_keys:
        for stat_name, stat_val in IMAGENET_STATS.items():
            ds_meta.stats[key][stat_name] = torch.tensor(stat_val, dtype=torch.float32)

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=None, dataset_stats=ds_meta.stats)

    policy.eval()
    preprocessor.reset()
    with torch.inference_mode():
        processed = preprocessor(_clone_batch(batch))
        pred = policy.predict_action_chunk(processed).detach().cpu()[0]
        target = processed["action"].detach().cpu()[0]
        valid = (~processed["action_is_pad"]).detach().cpu()[0].bool()
    meta = {"chunk_size": int(pred.shape[0]), "action_dim": int(pred.shape[-1])}
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, target, valid, meta


def _resolve_delta_timestamps(repo_id: str, dataset_root: Path, model_path: Path) -> dict[str, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
    from lerobot.policies.factory import make_policy  # noqa: F401

    cfg = PreTrainedConfig.from_pretrained(model_path)
    cfg.pretrained_path = model_path
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    for key in ds_meta.camera_keys:
        for stat_name, stat_val in IMAGENET_STATS.items():
            ds_meta.stats[key][stat_name] = torch.tensor(stat_val, dtype=torch.float32)
    return resolve_delta_timestamps(cfg, ds_meta)


def _tensor_scalar(batch: dict[str, Any], key: str) -> int | float | None:
    value = batch.get(key)
    if value is None or not torch.is_tensor(value):
        return None
    return value.detach().cpu().reshape(-1)[0].item()


def _save_trajectory_comparison(
    *,
    repo_id: str,
    dataset_root: Path,
    split_json: Path,
    model_path: Path,
    base_model_path: Path | None,
    report_dir: Path,
    sample_offset: int,
    max_steps: int,
    video_backend: str,
) -> None:
    delta_timestamps = _resolve_delta_timestamps(repo_id, dataset_root, model_path)
    batch = _load_sample_batch(
        repo_id=repo_id,
        dataset_root=dataset_root,
        split_json=split_json,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        sample_offset=sample_offset,
    )

    pred, target, valid, meta = _predict_chunk(
        repo_id=repo_id,
        dataset_root=dataset_root,
        model_path=model_path,
        batch=batch,
    )
    base_pred = None
    if base_model_path and base_model_path.is_dir():
        base_pred, _, _, _ = _predict_chunk(
            repo_id=repo_id,
            dataset_root=dataset_root,
            model_path=base_model_path,
            batch=batch,
        )

    valid_steps = torch.nonzero(valid, as_tuple=False).flatten()
    if len(valid_steps) == 0:
        raise RuntimeError("Selected sample has no valid action steps")
    n_steps = min(int(valid_steps[-1].item()) + 1, max_steps, pred.shape[0])
    steps = list(range(n_steps))
    action_names = _load_action_names(dataset_root)
    action_dim = min(pred.shape[-1], len(action_names))

    abs_err = (pred[:n_steps, :action_dim] - target[:n_steps, :action_dim]).abs()
    mean_l1 = float(abs_err.mean().item())
    by_dim_l1 = abs_err.mean(dim=0).tolist()
    base_mean_l1 = None
    if base_pred is not None:
        base_mean_l1 = float((base_pred[:n_steps, :action_dim] - target[:n_steps, :action_dim]).abs().mean().item())

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    axes_flat = axes.flatten()
    fig.suptitle(
        "Predicted vs true action chunk on held-out eval sample\n"
        "normalized action space; lower overlap error is better",
        fontsize=18,
        fontweight="bold",
    )

    for dim in range(action_dim):
        ax = axes_flat[dim]
        ax.plot(steps, target[:n_steps, dim].numpy(), color="#111827", linewidth=2.2, label="true")
        ax.plot(steps, pred[:n_steps, dim].numpy(), color="#2563eb", linewidth=2.0, label="fine-tuned pred")
        if base_pred is not None:
            ax.plot(
                steps,
                base_pred[:n_steps, dim].numpy(),
                color="#d97706",
                linewidth=1.6,
                linestyle="--",
                label="initial pred",
            )
        ax.set_title(f"{action_names[dim]}  L1={by_dim_l1[dim]:.3f}")
        ax.set_xlabel("future step in predicted chunk")
        ax.set_ylabel("normalized action")
        ax.grid(alpha=0.22)
        if dim == 0:
            ax.legend(loc="best")

    ax = axes_flat[-1]
    ax.plot(steps, abs_err.mean(dim=1).numpy(), color="#dc2626", linewidth=2.2, label="fine-tuned mean abs err")
    if base_pred is not None:
        base_step_err = (base_pred[:n_steps, :action_dim] - target[:n_steps, :action_dim]).abs().mean(dim=1)
        ax.plot(steps, base_step_err.numpy(), color="#d97706", linewidth=1.8, linestyle="--", label="initial mean abs err")
    title = f"mean L1 over dims: fine-tuned={mean_l1:.3f}"
    if base_mean_l1 is not None:
        title += f", initial={base_mean_l1:.3f}"
    ax.set_title(title)
    ax.set_xlabel("future step in predicted chunk")
    ax.set_ylabel("mean absolute error")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    episode_index = _tensor_scalar(batch, "episode_index")
    frame_index = _tensor_scalar(batch, "frame_index")
    fig.text(
        0.5,
        0.004,
        f"sample_offset={sample_offset}, episode_index={episode_index}, frame_index={frame_index}, "
        f"valid_steps={int(valid[:n_steps].sum().item())}, chunk_size={meta['chunk_size']}",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    png_path = report_dir / "trajectory_prediction_vs_true.png"
    fig.savefig(png_path, dpi=170)
    plt.close(fig)

    json_path = report_dir / "trajectory_prediction_vs_true_sample.json"
    sample = {
        "sample_offset": sample_offset,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "space": "normalized_action",
        "plotted_steps": n_steps,
        "action_names": action_names[:action_dim],
        "fine_tuned_mean_l1": mean_l1,
        "initial_mean_l1": base_mean_l1,
        "fine_tuned_l1_by_dim": by_dim_l1,
        "true": target[:n_steps, :action_dim].tolist(),
        "fine_tuned_pred": pred[:n_steps, :action_dim].tolist(),
        "initial_pred": base_pred[:n_steps, :action_dim].tolist() if base_pred is not None else None,
    }
    json_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    dataset_root = _project_path(args.dataset_root)
    split_json = _project_path(args.split_json)
    eval_dir = _project_path(args.eval_dir)
    report_dir = _project_path(args.report_dir)
    model_path = _project_path(args.model_path)
    base_model_path = _project_path(args.base_model_path) if args.base_model_path else None
    report_dir.mkdir(parents=True, exist_ok=True)

    suffix = _choose_suffix(eval_dir, args.metrics_suffix)
    rows = _load_metric_rows(eval_dir, suffix)
    _write_metrics_csv(rows, report_dir / "galbot_misac_metrics_summary.csv")
    _save_overview(rows, report_dir / "galbot_misac_finetune_overview.png")
    _save_validation_flow(report_dir / "galbot_misac_validation_flow.png")
    _save_trajectory_comparison(
        repo_id=args.repo_id,
        dataset_root=dataset_root,
        split_json=split_json,
        model_path=model_path,
        base_model_path=base_model_path,
        report_dir=report_dir,
        sample_offset=args.sample_offset,
        max_steps=args.max_trajectory_steps,
        video_backend=args.video_backend,
    )

    print(f"WROTE {report_dir / 'galbot_misac_metrics_summary.csv'}")
    print(f"WROTE {report_dir / 'galbot_misac_finetune_overview.png'}")
    print(f"WROTE {report_dir / 'galbot_misac_validation_flow.png'}")
    print(f"WROTE {report_dir / 'trajectory_prediction_vs_true.png'}")
    print(f"WROTE {report_dir / 'trajectory_prediction_vs_true_sample.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
