"""Start a real policy_server and verify one PolicyClient inference round-trip."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify policy_server + PolicyClient with a dummy observation.")
    parser.add_argument("--model-path", required=True, help="Path to checkpoints/.../pretrained_model")
    parser.add_argument("--bind", default="tcp://127.0.0.1:57991")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from galbot.deploy.policy_client import PolicyClient

    cmd = [
        sys.executable,
        str(REPO_ROOT / "galbot" / "deploy" / "policy_server.py"),
        "--model_path",
        args.model_path,
        "--bind",
        args.bind,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = False
        started = time.time()
        while time.time() - started < args.timeout_ms / 1000:
            line = proc.stdout.readline() if proc.stdout is not None else ""
            if line:
                print(line.rstrip())
                if "ready for requests" in line:
                    ready = True
                    break
            if proc.poll() is not None:
                raise RuntimeError(f"policy_server exited early with code {proc.returncode}")
        if not ready:
            raise RuntimeError("policy_server did not become ready")

        client = PolicyClient(server_url=args.bind, timeout_ms=args.timeout_ms)
        try:
            client.reset()
            obs = {
                "images": {"right_arm": np.zeros((224, 224, 3), dtype=np.uint8)},
                "arm_gripper_joints": np.zeros(29, dtype=np.float32),
            }
            action = client.inference(obs)
            print(f"VERIFY REAL ZMQ OK action_shape={tuple(action.shape)}")
        finally:
            client.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
