"""ZMQ observation multipart encode/decode — the wire contract.

Shared between ``policy_server.py`` (LeRobot / torch) and ``policy_client.py``
(headless robot / replay). Only stdlib + numpy allowed here.

Wire meta (JSON) fields:
  version : int, must equal ``PROTOCOL_VERSION``.
  op      : ``"infer"`` (default) | ``"reset"``.
  state   : flat list of floats  (infer only)
  images  : ``{short_cam: [H, W, 3], ...}`` in the same order as the following
            binary frames                 (infer only)

Multipart framing:
  infer REQ  -> [meta_json, img0_bytes, img1_bytes, ...]
  reset REQ  -> [meta_json]
  infer REP  -> [meta_json, action_f32_bytes]
  reset REP  -> [meta_json, b""]
  error REP  -> [meta_json, b""]
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
OP_INFER = "infer"
OP_RESET = "reset"


def _coerce_uint8_rgb(name: str, arr: np.ndarray) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 RGB, scaling from float if necessary."""
    x = np.asarray(arr)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"image {name!r} must be (H, W, 3) RGB HWC, got shape {x.shape}")
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32, copy=False)
    if xf.size and float(xf.max()) <= 1.0 + 1e-5:
        xf = xf * 255.0
    return np.clip(xf, 0, 255).astype(np.uint8)


def obs_to_multipart(obs: dict[str, Any]) -> list[bytes]:
    """Build an inference REQ multipart: ``[meta_json, img0_bytes, ...]``."""
    images = obs["images"]
    state = np.asarray(obs["arm_gripper_joints"], dtype=np.float32).reshape(-1)
    encoded: dict[str, np.ndarray] = {
        name: _coerce_uint8_rgb(name, img) for name, img in images.items()
    }
    meta: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "op": OP_INFER,
        "state": state.tolist(),
        "images": {name: list(img.shape) for name, img in encoded.items()},
    }
    parts: list[bytes] = [json.dumps(meta).encode("utf-8")]
    for _, img in encoded.items():
        parts.append(img.tobytes(order="C"))
    return parts


def reset_to_multipart() -> list[bytes]:
    """Build a reset REQ multipart."""
    meta = {"version": PROTOCOL_VERSION, "op": OP_RESET}
    return [json.dumps(meta).encode("utf-8")]


def multipart_to_meta(parts: list[bytes]) -> dict[str, Any]:
    """Parse the first frame (meta JSON) and reject mismatched protocol versions."""
    if not parts:
        raise ValueError("empty message")
    meta = json.loads(parts[0].decode("utf-8"))
    if meta.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {meta.get('version')}")
    return meta


def multipart_to_obs(parts: list[bytes], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decode an inference REQ multipart into ``{images, arm_gripper_joints}``."""
    if meta is None:
        meta = multipart_to_meta(parts)
    if "state" not in meta or "images" not in meta:
        raise ValueError("infer op requires 'state' and 'images' in meta")

    state = np.asarray(meta["state"], dtype=np.float32)
    images_meta = meta["images"]
    if not isinstance(images_meta, dict):
        raise ValueError("meta.images must be a dict")

    obs_images: dict[str, np.ndarray] = {}
    idx = 1
    for cam_name, shape in images_meta.items():
        if len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError(f"camera {cam_name!r}: shape must be [H, W, 3], got {shape}")
        if idx >= len(parts):
            raise ValueError(f"missing image frame for camera {cam_name}")
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
        buf = parts[idx]
        idx += 1
        arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
        if arr.size != h * w * c:
            raise ValueError(f"image size mismatch for {cam_name}: expected {h*w*c}, got {arr.size}")
        obs_images[cam_name] = arr.reshape(h, w, c)

    return {"images": obs_images, "arm_gripper_joints": state}
