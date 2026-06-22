"""Remote policy server (Python 3.12 + LeRobot).

Robot / motion-control process talks to this over ZeroMQ REQ/REP so torch and
LeRobot stay out of the robot's conda environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq  # type: ignore[import-not-found]  # requires pyzmq installed at runtime

# Ensure sibling modules and the wire package are importable without pip install.
_HERE = Path(__file__).resolve().parent
_WIRE = _HERE / "wire"
for _p in (_HERE, _WIRE):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from galbot.deploy.wire.obs_codec import (
        OP_INFER,
        OP_RESET,
        PROTOCOL_VERSION,
        multipart_to_meta,
        multipart_to_obs,
    )
    from galbot.deploy.wire.trace_io import maybe_save_obs_rgb_pngs, save_joints_vector
except ImportError:
    from obs_codec import OP_INFER, OP_RESET, PROTOCOL_VERSION, multipart_to_meta, multipart_to_obs
    from trace_io import maybe_save_obs_rgb_pngs, save_joints_vector

from policy_agent import PolicyAgent


def _encode_action_reply(tensor) -> tuple[bytes, bytes]:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    meta = {"version": PROTOCOL_VERSION, "op": OP_INFER, "shape": list(arr.shape)}
    return json.dumps(meta).encode("utf-8"), arr.tobytes()


def _encode_reset_ack() -> tuple[bytes, bytes]:
    meta = {"version": PROTOCOL_VERSION, "op": "reset_ack"}
    return json.dumps(meta).encode("utf-8"), b""


def _encode_error_reply(err: str) -> tuple[bytes, bytes]:
    meta = {"version": PROTOCOL_VERSION, "error": err, "shape": []}
    return json.dumps(meta).encode("utf-8"), b""


def _server_log_error(msg: str) -> None:
    print(f"[policy_server] {msg}", flush=True)


def _build_joint_trace_dir(raw: str | None, flat: bool) -> Path | None:
    if not raw:
        return None
    base = Path(raw).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base
    if not flat:
        base = base / time.strftime("%Y%m%d_%H%M%S")
    out = base / "server_post_decode"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_image_save_cfg(raw_dir: str | None, flat: bool, interval: int, max_frames: int) -> dict[str, Any] | None:
    if not raw_dir:
        return None
    base = Path(raw_dir).expanduser()
    if not base.is_absolute():
        base = Path.cwd() / base
    if not flat:
        base = base / time.strftime("%Y%m%d_%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "dir": base,
        "interval": max(1, int(interval)),
        "max_frames": max(0, int(max_frames)),
        "saved": 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galbot ZMQ policy server (Py3.12 + LeRobot)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained_model directory")
    parser.add_argument("--bind", type=str, default="tcp://0.0.0.0:5555", help="ZMQ bind address")
    parser.add_argument("--linger", type=int, default=0, help="ZMQ socket linger (ms)")
    parser.add_argument("--save_images_dir", type=str, default=None, help="Save decoded RGB PNGs here")
    parser.add_argument("--save_images_interval", type=int, default=1, help="Save every N-th request")
    parser.add_argument("--save_images_max", type=int, default=0, help="Stop after this many saved steps")
    parser.add_argument("--save_images_flat", action="store_true", help="No timestamp subdir")
    parser.add_argument("--joint_trace_dir", type=str, default=None, help="Save decoded joints here")
    parser.add_argument("--joint_trace_flat", action="store_true", help="No timestamp subdir")
    return parser.parse_args()


def _bind_rep_socket(ctx: zmq.Context, bind: str, linger: int) -> zmq.Socket:
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, linger)
    sock.bind(bind)
    return sock


def main():
    args = _parse_args()

    ctx = zmq.Context()
    sock = _bind_rep_socket(ctx, args.bind, args.linger)
    print(f"[policy_server] bound {args.bind}", flush=True)

    agent = PolicyAgent(args.model_path)
    print("[policy_server] ready for requests", flush=True)

    joint_trace_root = _build_joint_trace_dir(args.joint_trace_dir, args.joint_trace_flat)
    save_cfg = _build_image_save_cfg(
        args.save_images_dir,
        args.save_images_flat,
        args.save_images_interval,
        args.save_images_max,
    )

    infer_seq = 0
    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.ZMQError as e:
            _server_log_error(f"recv failed ({e}); rebinding REP socket")
            try:
                sock.close(linger=0)
            except Exception:
                pass
            sock = _bind_rep_socket(ctx, args.bind, args.linger)
            continue

        reply_meta: bytes
        reply_body: bytes
        try:
            meta = multipart_to_meta(parts)
            op = meta.get("op", OP_INFER)
            if op == OP_RESET:
                agent.reset()
                print(f"[policy_server] reset acknowledged (infer_seq={infer_seq})", flush=True)
                reply_meta, reply_body = _encode_reset_ack()
            elif op == OP_INFER:
                obs = multipart_to_obs(parts, meta)
                infer_seq += 1
                if joint_trace_root is not None:
                    save_joints_vector(
                        joint_trace_root, infer_seq, obs["arm_gripper_joints"], log_error=_server_log_error
                    )
                maybe_save_obs_rgb_pngs(save_cfg, obs, infer_seq, log_error=_server_log_error)
                out = agent.inference(obs)
                reply_meta, reply_body = _encode_action_reply(out)
            else:
                raise ValueError(f"unknown op: {op!r}")
        except Exception as e:
            reply_meta, reply_body = _encode_error_reply(str(e))
            print(f"[policy_server] error: {e}", flush=True)

        try:
            sock.send_multipart([reply_meta, reply_body])
        except zmq.ZMQError as e:
            _server_log_error(f"send failed ({e}); rebinding REP socket")
            try:
                sock.close(linger=0)
            except Exception:
                pass
            sock = _bind_rep_socket(ctx, args.bind, args.linger)


if __name__ == "__main__":
    main()
