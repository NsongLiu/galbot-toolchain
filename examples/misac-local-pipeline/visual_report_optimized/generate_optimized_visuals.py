"""Optimized visual report generation for Galbot MiSAC ACT fine-tuning.

This is intentionally separate from ``scripts/generate_visual_reports.py`` so
the first version remains reproducible while this version explores a clearer
reporting flow and better trajectory sample selection.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent


def _maybe_reexec_with_project_python() -> None:
    project_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not project_python.is_file():
        return
    if Path(sys.executable).resolve() == project_python.resolve():
        return
    if importlib.util.find_spec("lerobot") is not None:
        return

    env = os.environ.copy()
    env["GALBOT_OPT_VISUAL_REEXEC"] = "1"
    print(
        f"[runtime] Current Python has no lerobot: {sys.executable}\n"
        f"[runtime] Re-running with project Python: {project_python}",
        flush=True,
    )
    completed = subprocess.run([str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(completed.returncode)


if os.environ.get("GALBOT_OPT_VISUAL_REEXEC") != "1":
    _maybe_reexec_with_project_python()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from galbot.train._compat import apply_lerobot_compat_patch, apply_lerobot_runtime_compat_patch

apply_lerobot_compat_patch()
apply_lerobot_runtime_compat_patch()


@dataclass
class MetricRow:
    label: str
    step: int
    l1: float
    rmse: float
    zero_l1: float
    by_task: dict[str, Any]


@dataclass
class ScanRecord:
    sample_offset: int
    episode_index: int
    frame_index: int
    final_l1: float
    initial_l1: float
    improvement: float
    improvement_pct: float
    motion: float
    final_by_dim: list[float]
    initial_by_dim: list[float]
    target: np.ndarray
    final_pred: np.ndarray
    initial_pred: np.ndarray
    valid_steps: int


def _project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimized Galbot MiSAC visual reports.")
    parser.add_argument("--dataset-root", default="assets/lerobot_datasets/example")
    parser.add_argument("--repo-id", default="example_right_arm")
    parser.add_argument("--split-json", default="assets/config/example_split.json")
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "outputs"))
    parser.add_argument(
        "--model-path",
        default="assets/models/act_finetune/checkpoints/001000/pretrained_model",
    )
    parser.add_argument(
        "--base-model-path",
        default="assets/models/base_act/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--metrics-suffix", default="sample32x8")
    parser.add_argument("--scan-samples", type=int, default=192)
    parser.add_argument("--scan-batch-size", type=int, default=8)
    parser.add_argument("--max-trajectory-steps", type=int, default=100)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--training-loss-csv",
        help="Optional CSV with step,training_loss columns. No real loss trace is bundled.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_file(eval_dir: Path, label: str, suffix: str) -> Path:
    if label == "Initial":
        return eval_dir / f"misac_eval_initial_{suffix}.json"
    return eval_dir / f"misac_eval_finetune_{int(label):06d}_{suffix}.json"


def _load_metrics(eval_dir: Path, suffix: str) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for label, step in [("Initial", 0), ("250", 250), ("500", 500), ("750", 750), ("1000", 1000)]:
        path = _metric_file(eval_dir, label, suffix)
        if not path.is_file():
            raise FileNotFoundError(f"Missing metric file: {path}")
        data = _load_json(path)
        overall = data["overall"]
        rows.append(
            MetricRow(
                label=label,
                step=step,
                l1=float(overall["open_loop_l1"]),
                rmse=float(overall["open_loop_rmse"]),
                zero_l1=float(overall["zero_action_l1"]),
                by_task=data.get("by_task", {}),
            )
        )
    return rows


def _load_training_loss(path: Path | None) -> list[tuple[int, float]]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        trace = []
        for row in rows:
            trace.append((int(row["step"]), float(row["training_loss"])))
    return trace


def _annotate_points(ax, xs, ys, fmt="{:.3f}", every: set[int] | None = None, dy: float = 0.0) -> None:
    for idx, (x, y) in enumerate(zip(xs, ys)):
        if every is not None and idx not in every:
            continue
        ax.text(x, y + dy, fmt.format(y), ha="center", va="bottom", fontsize=8, color="#444")


def _save_overview(rows: list[MetricRow], output_dir: Path, loss_trace: list[tuple[int, float]]) -> None:
    steps = [r.step for r in rows]
    l1 = [r.l1 for r in rows]
    rmse = [r.rmse for r in rows]
    zero = rows[-1].zero_l1

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    fig.suptitle("Galbot MiSAC ACT Finetune - Offline Evidence", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, l1, marker="o", color="#1f77b4", linewidth=2.5, label="Policy open-loop L1")
    ax.axhline(zero, color="#ff7f0e", linestyle="--", linewidth=2.0, label="Zero-action baseline")
    _annotate_points(ax, steps, l1, fmt="{:.3f}", every={0, 1, 2, 3, 4}, dy=0.004)
    ax.set_title("Validation L1 by checkpoint")
    ax.set_xlabel("Finetune step")
    ax.set_ylabel("Normalized action L1 (lower is better)")
    ax.set_ylim(min(l1) - 0.025, max(max(l1), zero) + 0.025)
    ax.grid(True, color="#c7c7c7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(steps, rmse, marker="s", color="#b64b4b", linewidth=2.5)
    _annotate_points(ax, steps, rmse, fmt="{:.3f}", every={0, 1, 2, 3, 4}, dy=0.004)
    ax.set_title("Validation RMSE by checkpoint")
    ax.set_xlabel("Finetune step")
    ax.set_ylabel("Normalized action RMSE (lower is better)")
    ax.set_ylim(min(rmse) - 0.025, max(rmse) + 0.025)
    ax.grid(True, color="#c7c7c7", linewidth=0.8, alpha=0.8)

    ax = axes[1, 0]
    if loss_trace:
        loss_steps = [x for x, _ in loss_trace]
        losses = [y for _, y in loss_trace]
        ax.plot(loss_steps, losses, marker="o", color="#4f8359", linewidth=2.2, markersize=4)
        ax.set_title("Training loss from supplied CSV")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Training loss")
        ax.grid(True, color="#c7c7c7", linewidth=0.8, alpha=0.8)
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "No training-loss trace is bundled.\nPass --training-loss-csv to plot your own run.",
            ha="center",
            va="center",
            fontsize=12,
            color="#475569",
            transform=ax.transAxes,
        )

    ax = axes[1, 1]
    initial = rows[0]
    final = rows[-1]
    tasks = sorted(final.by_task)
    init_vals = [float(initial.by_task[t]["open_loop_l1"]) for t in tasks]
    final_vals = [float(final.by_task[t]["open_loop_l1"]) for t in tasks]
    x = np.arange(len(tasks))
    width = 0.36
    ax.bar(x - width / 2, init_vals, width, label="Initial", color="#1f77b4")
    ax.bar(x + width / 2, final_vals, width, label="1000 steps", color="#ff7f0e")
    for i, (a, b) in enumerate(zip(init_vals, final_vals)):
        drop = (a - b) / a * 100 if a else 0
        ax.text(i, max(a, b) + 0.025, f"-{drop:.1f}%", ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_title("All held-out tasks improve")
    ax.set_xlabel("Task")
    ax.set_ylabel("Open-loop L1")
    ax.set_xticks(x, tasks)
    ax.set_ylim(0, max(init_vals + final_vals) * 1.15)
    ax.grid(True, axis="y", color="#c7c7c7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right")

    fig.savefig(output_dir / "optimized_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_validation_flow(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.set_title("Galbot MiSAC Local Validation Flow", fontsize=18, fontweight="bold", pad=16)

    boxes = {}
    labels = [
        ("raw", "MiSAC MCAP\nraw episodes", "#e8f1ff"),
        ("convert", "MCAP -> LeRobot\nright_arm + 224px RGB", "#e9f8ee"),
        ("dataset", "LeRobot V3 Dataset\nstate + image + action", "#fff3d8"),
        ("split", "Train / Eval Split\n57 / 15 episodes", "#f0eaff"),
        ("train", "ACT Fine-tune\n0 -> 1000 steps", "#fdeaea"),
        ("eval", "Offline Eval\npredict future chunk", "#e8f7fa"),
        ("scan", "Sample Scan\nshowcase / median / hard", "#eef2f7"),
        ("report", "Visual Reports\nmetrics + trajectories", "#edf7ed"),
    ]
    start_x = 0.45
    gap = 0.20
    width = 1.95
    height = 1.05
    y = 2.45
    for idx, (key, text, color) in enumerate(labels):
        boxes[key] = (start_x + idx * (width + gap), y, width, height, text, color)

    def draw_box(key: str) -> None:
        x, y, w, h, text, color = boxes[key]
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.6,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, textwrap.fill(text, width=19), ha="center", va="center", fontsize=9.5)

    for key in boxes:
        draw_box(key)

    def right_mid(key: str) -> tuple[float, float]:
        x, y, w, h, *_ = boxes[key]
        return x + w, y + h / 2

    def left_mid(key: str) -> tuple[float, float]:
        x, y, _, h, *_ = boxes[key]
        return x, y + h / 2

    def bottom_mid(key: str) -> tuple[float, float]:
        x, y, w, *_ = boxes[key]
        return x + w / 2, y

    def top_mid(key: str) -> tuple[float, float]:
        x, y, w, h, *_ = boxes[key]
        return x + w / 2, y + h

    def arrow(start, end) -> None:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.9, color="#334155"))

    for left, right in zip([x[0] for x in labels[:-1]], [x[0] for x in labels[1:]]):
        arrow(right_mid(left), left_mid(right))

    ax.text(
        9.0,
        1.25,
        "Core check: current image/state -> ACT predicts future action chunk -> compare with held-out true action chunk.",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )
    fig.savefig(output_dir / "optimized_validation_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_eval_episodes(split_json: Path) -> list[int]:
    split = _load_json(split_json)
    return [int(x) for x in split["eval_episodes"]]


def _load_action_names(dataset_root: Path) -> list[str]:
    info = _load_json(dataset_root / "meta" / "info.json")
    return [str(x) for x in info["features"]["action"].get("names", [])]


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


def _load_policy(repo_id: str, dataset_root: Path, model_path: Path):
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
    return policy, preprocessor


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def _predict(policy, preprocessor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preprocessor.reset()
    with torch.inference_mode():
        processed = preprocessor(_clone_batch(batch))
        pred = policy.predict_action_chunk(processed).detach().cpu()
        target = processed["action"].detach().cpu()
        valid = (~processed["action_is_pad"]).detach().cpu().bool()
    return pred, target, valid


def _sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    max_steps: int,
) -> tuple[np.ndarray, list[float], list[int], list[float]]:
    batch_size = pred.shape[0]
    sample_l1 = []
    steps_used = []
    motion = []
    by_dim_rows = []
    for i in range(batch_size):
        valid_idx = torch.nonzero(valid[i], as_tuple=False).flatten()
        if len(valid_idx) == 0:
            n_steps = min(max_steps, pred.shape[1])
        else:
            n_steps = min(int(valid_idx[-1].item()) + 1, max_steps, pred.shape[1])
        diff = (pred[i, :n_steps] - target[i, :n_steps]).abs()
        sample_l1.append(float(diff.mean().item()))
        steps_used.append(n_steps)
        by_dim_rows.append(diff.mean(dim=0).tolist())
        if n_steps > 1:
            motion.append(float((target[i, 1:n_steps] - target[i, : n_steps - 1]).abs().mean().item()))
        else:
            motion.append(0.0)
    return np.array(sample_l1), by_dim_rows, steps_used, motion


def _scan_samples(
    *,
    repo_id: str,
    dataset_root: Path,
    split_json: Path,
    model_path: Path,
    base_model_path: Path,
    scan_samples: int,
    batch_size: int,
    max_steps: int,
    video_backend: str,
) -> list[ScanRecord]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = _resolve_delta_timestamps(repo_id, dataset_root, model_path)
    episodes = _load_eval_episodes(split_json)
    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    total = min(scan_samples, len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    final_policy, final_pre = _load_policy(repo_id, dataset_root, model_path)
    base_policy, base_pre = _load_policy(repo_id, dataset_root, base_model_path)

    records: list[ScanRecord] = []
    offset = 0
    for batch in loader:
        if offset >= total:
            break
        pred_final, target, valid = _predict(final_policy, final_pre, batch)
        pred_base, _, _ = _predict(base_policy, base_pre, batch)
        final_l1, final_by_dim, steps_used, motion = _sample_metrics(pred_final, target, valid, max_steps)
        init_l1, init_by_dim, _, _ = _sample_metrics(pred_base, target, valid, max_steps)

        bsz = pred_final.shape[0]
        for i in range(bsz):
            if offset + i >= total:
                break
            n_steps = steps_used[i]
            improvement = float(init_l1[i] - final_l1[i])
            improvement_pct = improvement / float(init_l1[i]) * 100 if init_l1[i] else 0.0
            records.append(
                ScanRecord(
                    sample_offset=offset + i,
                    episode_index=int(batch["episode_index"][i].detach().cpu().item()),
                    frame_index=int(batch["frame_index"][i].detach().cpu().item()),
                    final_l1=float(final_l1[i]),
                    initial_l1=float(init_l1[i]),
                    improvement=improvement,
                    improvement_pct=improvement_pct,
                    motion=float(motion[i]),
                    final_by_dim=[float(x) for x in final_by_dim[i]],
                    initial_by_dim=[float(x) for x in init_by_dim[i]],
                    target=target[i, :n_steps].numpy(),
                    final_pred=pred_final[i, :n_steps].numpy(),
                    initial_pred=pred_base[i, :n_steps].numpy(),
                    valid_steps=n_steps,
                )
            )
        offset += bsz
        print(f"[scan] {min(offset, total)}/{total} samples", flush=True)

    del final_policy, base_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _select_records(records: list[ScanRecord]) -> dict[str, ScanRecord]:
    if not records:
        raise RuntimeError("No scan records were produced")
    final_l1 = np.array([r.final_l1 for r in records])
    motion = np.array([r.motion for r in records])
    final_q40 = float(np.quantile(final_l1, 0.40))
    candidates = [r for r in records if r.final_l1 <= final_q40 and r.improvement > 0]
    if not candidates:
        candidates = [r for r in records if r.improvement > 0] or records

    best = max(candidates, key=lambda r: (r.improvement_pct - 25.0 * r.final_l1, r.improvement_pct))
    median_value = float(np.median(final_l1))
    median = min(records, key=lambda r: abs(r.final_l1 - median_value))
    hard = max(records, key=lambda r: r.final_l1)
    easy = min(records, key=lambda r: r.final_l1)
    return {"best": best, "median": median, "hard": hard, "easy": easy}


def _write_scan_outputs(records: list[ScanRecord], selected: dict[str, ScanRecord], output_dir: Path) -> None:
    with (output_dir / "sample_scan_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_offset",
                "episode_index",
                "frame_index",
                "initial_l1",
                "final_l1",
                "improvement",
                "improvement_pct",
                "motion",
                "valid_steps",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "sample_offset": r.sample_offset,
                    "episode_index": r.episode_index,
                    "frame_index": r.frame_index,
                    "initial_l1": f"{r.initial_l1:.6f}",
                    "final_l1": f"{r.final_l1:.6f}",
                    "improvement": f"{r.improvement:.6f}",
                    "improvement_pct": f"{r.improvement_pct:.3f}",
                    "motion": f"{r.motion:.6f}",
                    "valid_steps": r.valid_steps,
                }
            )

    payload = {
        name: {
            "sample_offset": r.sample_offset,
            "episode_index": r.episode_index,
            "frame_index": r.frame_index,
            "initial_l1": r.initial_l1,
            "final_l1": r.final_l1,
            "improvement": r.improvement,
            "improvement_pct": r.improvement_pct,
            "motion": r.motion,
            "valid_steps": r.valid_steps,
            "final_l1_by_dim": r.final_by_dim,
            "initial_l1_by_dim": r.initial_by_dim,
        }
        for name, r in selected.items()
    }
    (output_dir / "sample_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _save_best_trajectory(record: ScanRecord, action_names: list[str], output_dir: Path) -> None:
    action_dim = record.target.shape[1]
    final_by_dim = np.array(record.final_by_dim)
    init_by_dim = np.array(record.initial_by_dim)
    improvement_by_dim = init_by_dim - final_by_dim
    active = np.std(record.target, axis=0)
    rank = np.argsort(-(improvement_by_dim + 0.15 * active))[:4]

    steps = np.arange(record.valid_steps)
    final_step_err = np.mean(np.abs(record.final_pred - record.target), axis=1)
    init_step_err = np.mean(np.abs(record.initial_pred - record.target), axis=1)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 1.0])
    fig.suptitle(
        "Optimized trajectory view: selected held-out sample with clear fine-tune improvement",
        fontsize=16,
        fontweight="bold",
    )

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(steps, _smooth(init_step_err), color="#d97706", linestyle="--", linewidth=2.2, label="initial error")
    ax.plot(steps, _smooth(final_step_err), color="#dc2626", linewidth=2.4, label="fine-tuned error")
    ax.fill_between(steps, _smooth(final_step_err), _smooth(init_step_err), color="#22c55e", alpha=0.18)
    ax.set_title(
        f"Mean error over future chunk: {record.initial_l1:.3f} -> {record.final_l1:.3f} "
        f"({record.improvement_pct:.1f}% better)"
    )
    ax.set_xlabel("future step")
    ax.set_ylabel("mean absolute error")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(gs[0, 2:])
    x = np.arange(action_dim)
    width = 0.38
    ax.bar(x - width / 2, init_by_dim, width, label="initial", color="#d97706")
    ax.bar(x + width / 2, final_by_dim, width, label="fine-tuned", color="#2563eb")
    ax.set_title("Error by action dimension")
    ax.set_ylabel("L1")
    ax.set_xticks(x, [name.replace("right_arm_", "") for name in action_names], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for slot, dim in enumerate(rank):
        row = 1 + slot // 2
        col = (slot % 2) * 2
        ax = fig.add_subplot(gs[row, col : col + 2])
        ax.plot(steps, record.target[:, dim], color="#111827", linewidth=2.5, label="true")
        ax.plot(steps, record.final_pred[:, dim], color="#2563eb", linewidth=2.1, label="fine-tuned pred")
        ax.plot(steps, record.initial_pred[:, dim], color="#d97706", linestyle="--", linewidth=1.8, label="initial pred")
        ax.set_title(
            f"{action_names[dim]} | L1 {record.initial_by_dim[dim]:.3f} -> {record.final_by_dim[dim]:.3f}"
        )
        ax.set_xlabel("future step")
        ax.set_ylabel("normalized action")
        ax.grid(alpha=0.25)
        if slot == 0:
            ax.legend(loc="best")

    fig.text(
        0.5,
        0.01,
        f"Selected by sample scan. sample_offset={record.sample_offset}, episode={record.episode_index}, "
        f"frame={record.frame_index}, valid_steps={record.valid_steps}",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(output_dir / "optimized_trajectory_best.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_triptych(selected: dict[str, ScanRecord], output_dir: Path) -> None:
    order = [("best", selected["best"]), ("median", selected["median"]), ("hard", selected["hard"])]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Trajectory sample scan: best / median / hard cases", fontsize=16, fontweight="bold")

    for col, (name, r) in enumerate(order):
        steps = np.arange(r.valid_steps)
        final_step_err = np.mean(np.abs(r.final_pred - r.target), axis=1)
        init_step_err = np.mean(np.abs(r.initial_pred - r.target), axis=1)

        ax = axes[0, col]
        ax.plot(steps, _smooth(init_step_err), color="#d97706", linestyle="--", linewidth=2.0, label="initial")
        ax.plot(steps, _smooth(final_step_err), color="#2563eb", linewidth=2.2, label="fine-tuned")
        ax.set_title(
            f"{name}: {r.initial_l1:.3f} -> {r.final_l1:.3f}\n"
            f"improvement {r.improvement_pct:.1f}%"
        )
        ax.set_xlabel("future step")
        ax.set_ylabel("mean abs error")
        ax.grid(alpha=0.25)
        if col == 0:
            ax.legend()

        ax = axes[1, col]
        ax.bar(["initial", "fine-tuned"], [r.initial_l1, r.final_l1], color=["#d97706", "#2563eb"])
        ax.set_ylim(0, max(r.initial_l1, r.final_l1) * 1.25)
        ax.set_ylabel("sample L1")
        ax.set_title(f"episode={r.episode_index}, frame={r.frame_index}, offset={r.sample_offset}")
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(output_dir / "optimized_trajectory_triptych.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    dataset_root = _project_path(args.dataset_root)
    split_json = _project_path(args.split_json)
    eval_dir = _project_path(args.eval_dir)
    model_path = _project_path(args.model_path)
    base_model_path = _project_path(args.base_model_path)
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(eval_dir, args.metrics_suffix)
    loss_trace = _load_training_loss(_project_path(args.training_loss_csv) if args.training_loss_csv else None)
    _save_overview(metrics, output_dir, loss_trace)
    _save_validation_flow(output_dir)

    print("[scan] loading models and scanning held-out eval samples", flush=True)
    records = _scan_samples(
        repo_id=args.repo_id,
        dataset_root=dataset_root,
        split_json=split_json,
        model_path=model_path,
        base_model_path=base_model_path,
        scan_samples=args.scan_samples,
        batch_size=args.scan_batch_size,
        max_steps=args.max_trajectory_steps,
        video_backend=args.video_backend,
    )
    selected = _select_records(records)
    _write_scan_outputs(records, selected, output_dir)
    action_names = _load_action_names(dataset_root)
    _save_best_trajectory(selected["best"], action_names, output_dir)
    _save_triptych(selected, output_dir)

    for path in [
        "optimized_overview.png",
        "optimized_validation_flow.png",
        "optimized_trajectory_best.png",
        "optimized_trajectory_triptych.png",
        "sample_scan_summary.csv",
        "sample_selection.json",
    ]:
        print(f"WROTE {output_dir / path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
