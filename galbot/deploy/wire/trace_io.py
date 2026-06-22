"""Debug-dump helpers shared by the wire's two endpoints.

Runtime deps limited to numpy + stdlib. Optional PNG encoders are imported
inside the writer so a missing dep degrades gracefully.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Write uint8 RGB (H, W, 3) to PNG; fall back to ``.npy`` on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        import cv2

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise OSError("cv2.imwrite returned False")
        return
    except Exception:
        pass
    try:
        from PIL import Image

        Image.fromarray(arr, mode="RGB").save(path)
        return
    except Exception:
        pass
    np.save(path.with_suffix(".npy"), arr)


def safe_cam_token(name: str) -> str:
    """Filesystem-safe token from a camera short name."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return s[:120] if s else "cam"


def coerce_nonneg_int(val: Any, default: int) -> int:
    """Non-negative int with bool guard."""
    if val is None:
        return default
    if isinstance(val, bool):
        return default
    return int(val)


def maybe_save_obs_rgb_pngs(
    cfg: dict[str, Any] | None,
    obs: Mapping[str, Any],
    seq: int,
    *,
    log_error: Callable[[str], None] | None = None,
) -> None:
    """Write one PNG per camera in ``obs['images']`` when rate limits allow."""
    if not cfg:
        return
    if cfg["max_frames"] and cfg["saved"] >= cfg["max_frames"]:
        return
    if seq % cfg["interval"] != 0:
        return
    images = obs.get("images")
    if not isinstance(images, dict) or not images:
        return

    err = log_error if log_error is not None else (lambda m: print(m, flush=True))
    out_dir: Path = cfg["dir"]
    wrote = 0
    for cam_name, img in images.items():
        token = safe_cam_token(str(cam_name))
        path = out_dir / f"{token}_{seq:08d}.png"
        try:
            write_rgb_png(path, img)
            wrote += 1
        except Exception as e:
            err(f"image_save: failed to write {path}: {e}")
    if wrote:
        cfg["saved"] += 1


def save_joints_vector(
    root: Path,
    seq: int,
    joints: np.ndarray,
    *,
    log_error: Callable[[str], None] | None = None,
) -> None:
    """Save ``arm_gripper_joints`` as float32 (N,) under ``root/state_{seq:08d}.npy``."""
    err = log_error if log_error is not None else (lambda m: print(m, flush=True))
    root.mkdir(parents=True, exist_ok=True)
    vec = np.asarray(joints, dtype=np.float32).reshape(-1)
    path = root / f"state_{seq:08d}.npy"
    try:
        np.save(path, vec)
    except Exception as e:
        err(f"joints: failed to write {path}: {e}")
