"""Minimal example of a robot control loop using PolicyClient + GalbotSDK.

This script runs on the robot control side (which may use a different conda
environment from xhum-new). It only needs:
  - pyzmq, numpy, PyYAML
  - galbot_sdk (installed in the robot control environment)

Replace the dummy connection arguments and action layout in
``robot_interface.py`` with values from the real G1 deployment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config_loader import load_config, make_policy_client
from robot_interface import GalbotRobotInterface


class _StdioLogger:
    def info(self, msg: str) -> None:
        print(msg)

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="Galbot G1 policy control loop example")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=100, help="Number of control steps")
    args = p.parse_args()

    log = _StdioLogger()
    cfg = load_config(args.config, log)

    # TO DEBUG: supply the correct GalbotSDK connection kwargs for your robot.
    robot = GalbotRobotInterface()
    robot.connect()

    client = make_policy_client(cfg, log)
    try:
        client.reset()
        robot.reset()

        action_rate = float(cfg.get("action_rate", 30.0))
        period = 1.0 / max(action_rate, 1e-6)

        for step in range(args.steps):
            t0 = time.perf_counter()

            obs = robot.get_observation()
            action = client.inference(obs)
            robot.apply_action(action[0])

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            log.info(f"step {step + 1}/{args.steps}  dt={elapsed:.3f}s")
    finally:
        client.close()
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
