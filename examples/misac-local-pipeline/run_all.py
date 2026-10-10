"""Run the local Galbot MiSAC conversion, fine-tune, evaluation, and smoke checks.

Default behavior reuses existing expensive artifacts if present. Use
``--force-convert`` or ``--force-train`` to recompute them.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "pipeline_config.json"
PROJECT_PATH_KEYS = {
    "mcap_dir",
    "dataset_root",
    "robot_config",
    "split_json",
    "train_config",
    "base_model",
    "train_output_dir",
    "eval_dir",
    "report_dir",
}


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _inside_project(path: Path) -> bool:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    return resolved == root or root in resolved.parents


def _resolve_config(cfg: dict) -> dict:
    resolved = dict(cfg)
    for key in PROJECT_PATH_KEYS:
        resolved[key] = str(_path(cfg[key]))
        if not _inside_project(Path(resolved[key])):
            raise RuntimeError(f"Config path must stay inside {PROJECT_ROOT}: {key}={resolved[key]}")

    configured_python = str(cfg.get("python", "")).strip()
    if configured_python:
        py_path = Path(configured_python)
        if not py_path.is_absolute():
            py_path = PROJECT_ROOT / py_path
        resolved["python"] = str(py_path.resolve())
    else:
        resolved["python"] = sys.executable
    return resolved


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    if not _inside_project(resolved) or resolved == PROJECT_ROOT.resolve():
        raise RuntimeError(f"Refusing to delete outside project outputs/assets: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print("\n[run]", " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd}")
    print(f"[ok] {elapsed:.1f}s", flush=True)


def _convert_if_needed(cfg: dict, force: bool) -> None:
    dataset_root = _path(cfg["dataset_root"])
    if force:
        _safe_rmtree(dataset_root)
    if (dataset_root / "meta" / "info.json").is_file() and not force:
        print(f"[skip] Converted dataset exists: {dataset_root}")
        return

    py = cfg["python"]
    cmd = [
        py,
        str(PROJECT_ROOT / "convert_mcap_to_lerobot.py"),
        "--mcap-dir",
        str(_path(cfg["mcap_dir"])),
        "--out-dir",
        str(dataset_root),
        "--repo-id",
        cfg["repo_id"],
        "--action-groups",
        "right_arm",
        "--master-camera",
        "right_arm",
        "--camera-keys",
        "right_arm",
        "--image-width",
        "224",
        "--image-height",
        "224",
        "--skip-empty-actions",
        "--skip-errors",
        "--recursive",
        "--use-images",
        "--robot-config",
        str(_path(cfg["robot_config"])),
    ]
    _run(cmd)


def _prepare_split_and_config(cfg: dict) -> None:
    py = cfg["python"]
    cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "prepare_misac_finetune.py"),
        "--dataset-root",
        str(_path(cfg["dataset_root"])),
        "--repo-id",
        cfg["repo_id"],
        "--base-model-path",
        str(_path(cfg["base_model"])),
        "--output-dir",
        str(_path(cfg["train_output_dir"])),
        "--split-out",
        str(_path(cfg["split_json"])),
        "--config-out",
        str(_path(cfg["train_config"])),
        "--steps",
        str(cfg["steps"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--num-workers",
        "0",
        "--save-freq",
        "250",
        "--log-freq",
        "50",
        "--seed",
        str(cfg["seed"]),
    ]
    _run(cmd)


def _train_if_needed(cfg: dict, force: bool, skip: bool) -> None:
    if skip:
        print("[skip] Training skipped by --skip-train")
        return
    out_dir = _path(cfg["train_output_dir"])
    final_model = out_dir / "checkpoints" / "001000" / "pretrained_model"
    if force:
        _safe_rmtree(out_dir)
    if (final_model / "config.json").is_file() and not force:
        print(f"[skip] Final checkpoint exists: {final_model}")
        return
    _run([cfg["python"], "-m", "galbot.train.train", "--config", str(_path(cfg["train_config"]))])


def _eval_one(cfg: dict, model_path: Path, output_json: Path, seed: int, max_batches: int) -> None:
    cmd = [
        cfg["python"],
        str(PROJECT_ROOT / "scripts" / "evaluate_policy_offline.py"),
        "--dataset-root",
        str(_path(cfg["dataset_root"])),
        "--repo-id",
        cfg["repo_id"],
        "--model-path",
        str(model_path),
        "--split-json",
        str(_path(cfg["split_json"])),
        "--split-name",
        "eval",
        "--processor-source",
        "dataset",
        "--batch-size",
        "8",
        "--shuffle",
        "--seed",
        str(seed),
        "--max-batches",
        str(max_batches),
        "--output-json",
        str(output_json),
    ]
    _run(cmd)


def _evaluate(cfg: dict, quick: bool) -> list[Path]:
    eval_dir = _path(cfg["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    max_batches = 2 if quick else int(cfg["eval_batches"])
    paths: list[Path] = []

    initial_out = eval_dir / f"misac_eval_initial_sample{max_batches}x8.json"
    _eval_one(cfg, _path(cfg["base_model"]), initial_out, int(cfg["seed"]), max_batches)
    paths.append(initial_out)

    ckpt_root = _path(cfg["train_output_dir"]) / "checkpoints"
    for step in ("000250", "000500", "000750", "001000"):
        model_path = ckpt_root / step / "pretrained_model"
        if not (model_path / "config.json").is_file():
            print(f"[skip] Missing checkpoint for evaluation: {model_path}")
            continue
        out = eval_dir / f"misac_eval_finetune_{step}_sample{max_batches}x8.json"
        _eval_one(cfg, model_path, out, int(cfg["seed"]), max_batches)
        paths.append(out)

    final_model = ckpt_root / "001000" / "pretrained_model"
    if (final_model / "config.json").is_file() and not quick:
        initial_second = eval_dir / "misac_eval_initial_seed2026_sample32x8.json"
        final_second = eval_dir / "misac_eval_finetune_001000_seed2026_sample32x8.json"
        _eval_one(cfg, _path(cfg["base_model"]), initial_second, int(cfg["second_seed"]), max_batches)
        _eval_one(cfg, final_model, final_second, int(cfg["second_seed"]), max_batches)
        paths.extend([initial_second, final_second])
    return paths


def _smoke_verify(cfg: dict, skip_deploy: bool) -> None:
    final_model = _path(cfg["train_output_dir"]) / "checkpoints" / "001000" / "pretrained_model"
    if not (final_model / "config.json").is_file():
        print(f"[skip] No final model for smoke verify: {final_model}")
        return
    _run(
        [
            cfg["python"],
            str(PROJECT_ROOT / "scripts" / "verify_finetune_smoke.py"),
            "--dataset-root",
            str(_path(cfg["dataset_root"])),
            "--repo-id",
            cfg["repo_id"],
            "--model-path",
            str(final_model),
        ]
    )
    if not skip_deploy:
        _run(
            [
                cfg["python"],
                str(PROJECT_ROOT / "scripts" / "verify_policy_server_roundtrip.py"),
                "--model-path",
                str(final_model),
                "--bind",
                "tcp://127.0.0.1:57993",
            ]
        )


def _load_metric(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(cfg: dict, quick: bool) -> None:
    report_dir = _path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = _path(cfg["eval_dir"])
    suffix = "sample2x8" if quick else "sample32x8"
    metric_files = [
        ("Initial", eval_dir / f"misac_eval_initial_{suffix}.json"),
        ("250", eval_dir / f"misac_eval_finetune_000250_{suffix}.json"),
        ("500", eval_dir / f"misac_eval_finetune_000500_{suffix}.json"),
        ("750", eval_dir / f"misac_eval_finetune_000750_{suffix}.json"),
        ("1000", eval_dir / f"misac_eval_finetune_001000_{suffix}.json"),
    ]
    rows = []
    for label, path in metric_files:
        data = _load_metric(path)
        if data:
            rows.append(
                {
                    "label": label,
                    "l1": data["overall"]["open_loop_l1"],
                    "rmse": data["overall"]["open_loop_rmse"],
                    "zero": data["overall"]["zero_action_l1"],
                    "by_task": data.get("by_task", {}),
                }
            )
    if not rows:
        print("[warn] No metrics found; report not written.")
        return

    best = min(rows, key=lambda x: x["l1"])
    first = rows[0]
    improvement = (first["l1"] - best["l1"]) / first["l1"] * 100.0 if first["l1"] else 0.0
    md = report_dir / "latest_summary.md"
    html_path = report_dir / "latest_report.html"
    summary = {
        "dataset_root": cfg["dataset_root"],
        "train_output_dir": cfg["train_output_dir"],
        "best_label": best["label"],
        "initial_open_loop_l1": first["l1"],
        "best_open_loop_l1": best["l1"],
        "improvement_pct": improvement,
        "rows": rows,
    }
    (report_dir / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md.write_text(
        "\n".join(
            [
                "# Galbot MiSAC local pipeline summary",
                "",
                f"- Dataset: `{cfg['dataset_root']}`",
                f"- Model output: `{cfg['train_output_dir']}`",
                f"- Best checkpoint: `{best['label']}`",
                f"- Open-loop L1: `{first['l1']:.6f}` -> `{best['l1']:.6f}`",
                f"- Improvement: `{improvement:.1f}%`",
                "",
                "| checkpoint | open_loop_l1 | rmse | zero_action_l1 |",
                "|---|---:|---:|---:|",
                *[
                    f"| {r['label']} | {r['l1']:.6f} | {r['rmse']:.6f} | {r['zero']:.6f} |"
                    for r in rows
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    max_l1 = max(max(r["l1"] for r in rows), max(r["zero"] for r in rows))
    bars = []
    for i, r in enumerate(rows):
        width = 520 * r["l1"] / max_l1
        y = 60 + i * 38
        bars.append(
            f'<text x="12" y="{y + 18}" font-size="13">{html.escape(r["label"])}</text>'
            f'<rect x="90" y="{y}" width="{width:.1f}" height="22" fill="#4f7cac"></rect>'
            f'<text x="{100 + width:.1f}" y="{y + 16}" font-size="12">{r["l1"]:.3f}</text>'
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Galbot MiSAC Pipeline Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2937; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Galbot MiSAC local pipeline report</h1>
  <p>Open-loop L1 improved from <b>{first['l1']:.6f}</b> to <b>{best['l1']:.6f}</b> ({improvement:.1f}%).</p>
  <svg width="700" height="{90 + len(rows) * 38}" role="img" aria-label="Open-loop L1 by checkpoint">
    <text x="12" y="24" font-size="16" font-weight="600">Open-loop L1 by checkpoint; lower is better</text>
    {''.join(bars)}
  </svg>
  <table>
    <thead><tr><th>checkpoint</th><th>open_loop_l1</th><th>rmse</th><th>zero_action_l1</th></tr></thead>
    <tbody>
      {''.join(f"<tr><td>{html.escape(r['label'])}</td><td>{r['l1']:.6f}</td><td>{r['rmse']:.6f}</td><td>{r['zero']:.6f}</td></tr>" for r in rows)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"[report] {md}")
    print(f"[report] {html_path}")
    print(f"[summary] open_loop_l1 {first['l1']:.6f} -> {best['l1']:.6f} ({improvement:.1f}%)")


def _generate_visual_reports(cfg: dict, quick: bool, skip: bool) -> None:
    if skip:
        print("[skip] Visual report generation skipped by --skip-visuals")
        return
    suffix = "sample2x8" if quick else "sample32x8"
    final_model = _path(cfg["train_output_dir"]) / "checkpoints" / "001000" / "pretrained_model"
    cmd = [
        cfg["python"],
        str(PROJECT_ROOT / "scripts" / "generate_visual_reports.py"),
        "--dataset-root",
        str(_path(cfg["dataset_root"])),
        "--repo-id",
        cfg["repo_id"],
        "--split-json",
        str(_path(cfg["split_json"])),
        "--eval-dir",
        str(_path(cfg["eval_dir"])),
        "--report-dir",
        str(_path(cfg["report_dir"])),
        "--model-path",
        str(final_model),
        "--base-model-path",
        str(_path(cfg["base_model"])),
        "--metrics-suffix",
        suffix,
        "--seed",
        str(cfg["seed"]),
    ]
    _run(cmd)


def main() -> int:
    cfg = _resolve_config(_load_config())
    parser = argparse.ArgumentParser(description="Run local Galbot MiSAC pipeline.")
    parser.add_argument("--python", default=cfg["python"], help="Python executable for Galbot/LeRobot.")
    parser.add_argument("--force-convert", action="store_true", help="Rebuild converted LeRobot dataset.")
    parser.add_argument("--force-train", action="store_true", help="Delete and retrain the fine-tune output.")
    parser.add_argument("--skip-train", action="store_true", help="Do not train; reuse existing checkpoint.")
    parser.add_argument("--skip-deploy-verify", action="store_true", help="Skip ZMQ policy_server roundtrip.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip PNG/CSV visual report generation.")
    parser.add_argument("--quick", action="store_true", help="Use 2 eval batches per task for a fast smoke run.")
    args = parser.parse_args()
    cfg["python"] = str(_path(args.python)) if args.python else sys.executable

    print(f"[project] {PROJECT_ROOT}")
    print(f"[python]  {cfg['python']}")
    _convert_if_needed(cfg, args.force_convert)
    _prepare_split_and_config(cfg)
    _train_if_needed(cfg, args.force_train, args.skip_train)
    _evaluate(cfg, args.quick)
    _smoke_verify(cfg, args.skip_deploy_verify)
    _write_report(cfg, args.quick)
    _generate_visual_reports(cfg, args.quick, args.skip_visuals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
