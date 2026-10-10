"""Local deployment smoke test without LeRobot or a real robot.

This validates the PolicyClient <-> ZMQ REP wire path with a tiny in-process
mock policy server. It is intentionally independent from torch / LeRobot so it
can run in a lightweight robot-control environment.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

zmq = None
PolicyClient = None
OP_INFER = None
OP_RESET = None
PROTOCOL_VERSION = None
multipart_to_meta = None
multipart_to_obs = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local mock PolicyClient <-> ZMQ smoke test.")
    return parser.parse_args()


def _load_runtime_dependencies() -> None:
    global zmq
    global PolicyClient
    global OP_INFER
    global OP_RESET
    global PROTOCOL_VERSION
    global multipart_to_meta
    global multipart_to_obs

    import zmq as zmq_module
    from galbot.deploy.policy_client import PolicyClient as PolicyClientClass
    from galbot.deploy.wire.obs_codec import (
        OP_INFER as OP_INFER_VALUE,
        OP_RESET as OP_RESET_VALUE,
        PROTOCOL_VERSION as PROTOCOL_VERSION_VALUE,
        multipart_to_meta as multipart_to_meta_fn,
        multipart_to_obs as multipart_to_obs_fn,
    )

    zmq = zmq_module
    PolicyClient = PolicyClientClass
    OP_INFER = OP_INFER_VALUE
    OP_RESET = OP_RESET_VALUE
    PROTOCOL_VERSION = PROTOCOL_VERSION_VALUE
    multipart_to_meta = multipart_to_meta_fn
    multipart_to_obs = multipart_to_obs_fn


def _serve_mock(ready: queue.Queue[str], done: threading.Event, action_dim: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind("tcp://127.0.0.1:*")
    ready.put(sock.getsockopt_string(zmq.LAST_ENDPOINT))

    try:
        while not done.is_set():
            try:
                parts = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                done.wait(0.01)
                continue

            try:
                meta = multipart_to_meta(parts)
                op = meta.get("op", OP_INFER)
                if op == OP_RESET:
                    reply = {"version": PROTOCOL_VERSION, "op": "reset_ack"}
                    sock.send_multipart([json.dumps(reply).encode("utf-8"), b""])
                    continue
                if op != OP_INFER:
                    raise ValueError(f"unexpected op: {op!r}")

                obs = multipart_to_obs(parts, meta)
                if "left_arm" not in obs["images"]:
                    raise ValueError("dummy observation lost left_arm image")
                if obs["arm_gripper_joints"].shape != (29,):
                    raise ValueError("dummy state vector shape mismatch")

                action = np.arange(action_dim, dtype=np.float32).reshape(1, action_dim)
                reply = {
                    "version": PROTOCOL_VERSION,
                    "op": OP_INFER,
                    "shape": list(action.shape),
                }
                sock.send_multipart([json.dumps(reply).encode("utf-8"), action.tobytes()])
            except Exception as exc:
                reply = {"version": PROTOCOL_VERSION, "error": str(exc), "shape": []}
                sock.send_multipart([json.dumps(reply).encode("utf-8"), b""])
    finally:
        sock.close(linger=0)
        ctx.term()


def main() -> int:
    _parse_args()
    _load_runtime_dependencies()
    ready: queue.Queue[str] = queue.Queue()
    done = threading.Event()
    server = threading.Thread(target=_serve_mock, args=(ready, done, 8), daemon=True)
    server.start()
    endpoint = ready.get(timeout=5)

    obs = {
        "images": {"left_arm": np.zeros((16, 16, 3), dtype=np.uint8)},
        "arm_gripper_joints": np.linspace(0.0, 1.0, 29, dtype=np.float32),
    }

    client = PolicyClient(server_url=endpoint, timeout_ms=5_000)
    try:
        client.reset()
        action = client.inference(obs)
    finally:
        client.close()
        done.set()
        server.join(timeout=5)

    expected = np.arange(8, dtype=np.float32).reshape(1, 8)
    if action.shape != expected.shape or not np.allclose(action, expected):
        raise RuntimeError(f"unexpected action reply: shape={action.shape}, values={action}")

    print(f"SMOKE OK endpoint={endpoint} action_shape={tuple(action.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
