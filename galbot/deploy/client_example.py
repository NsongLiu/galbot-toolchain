"""G1 真机部署控制循环：GalbotSDK 观测 -> ZMQ 推理 -> 动作下发。

运行在机器人控制端（Orin），只依赖 pyzmq / numpy / opencv / PyYAML / galbot_sdk；
策略推理由 policy_server（LeRobot 环境）完成。

安全模式（沿用真机调通脚本的约定）：
  - 默认 dry-run：连接机器人、读取观测、请求推理并打印动作，但不下发任何运动指令；
  - 实机执行需同时加 ``--execute --acknowledge DEPLOY_MODEL``。

用法示例：
  python client_example.py --config galbot/deploy/configs/my_deploy.yaml --steps 300
  python client_example.py --config ... --steps 300 \
      --execute --acknowledge DEPLOY_MODEL
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config_loader import load_config, make_policy_client
from robot_interface import EXECUTE_ACK_TOKEN, GalbotRobotInterface


class _StdioLogger:
    def info(self, msg: str) -> None:
        print(msg)

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)


def _make_robot(cfg: dict) -> GalbotRobotInterface:
    """按 YAML 的 robot: 段构造接口（缺省值即真机调通参数）。"""
    robot_cfg = cfg.get("robot") or {}
    return GalbotRobotInterface(
        sensors=robot_cfg.get("sensors"),
        sdk_lib_path=robot_cfg.get("sdk_lib_path"),
        camera_warmup_sec=float(robot_cfg.get("camera_warmup_sec", 5.0)),
        rgb_timeout_sec=float(robot_cfg.get("rgb_timeout_sec", 10.0)),
        max_speed=float(robot_cfg.get("max_speed", 0.05)),
        timeout=float(robot_cfg.get("timeout", 1.0)),
        gripper_speed=float(robot_cfg.get("gripper_speed", 0.1)),
        gripper_force=float(robot_cfg.get("gripper_force", 10.0)),
        gripper_min_raw=float(robot_cfg.get("gripper_min_raw", 35.0)),
        gripper_max_raw=float(robot_cfg.get("gripper_max_raw", 100.0)),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Galbot G1 真机策略控制循环")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=120, help="控制步数")
    p.add_argument("--execute", action="store_true", help="实际下发动作（默认 dry-run）")
    p.add_argument("--acknowledge", default="", help=f"执行确认口令：{EXECUTE_ACK_TOKEN}")
    p.add_argument(
        "--start-right-arm",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="循环开始前先阻塞式移动右臂到该 7 关节位姿（仅 execute 模式生效）",
    )
    args = p.parse_args()

    if args.steps <= 0:
        p.error("--steps must be positive")

    log = _StdioLogger()
    cfg = load_config(args.config, log)
    if cfg.get("mode") != "model":
        log.error(f"真机控制循环要求 mode: model（当前 {cfg.get('mode')!r}）；"
                  "replay 请使用 replay.py")
        return 1

    execute = bool(args.execute)
    if execute and args.acknowledge != EXECUTE_ACK_TOKEN:
        log.error(f"--execute 需要 --acknowledge {EXECUTE_ACK_TOKEN}")
        return 1

    robot = _make_robot(cfg)
    client = make_policy_client(cfg, log)

    action_rate = float(cfg.get("action_rate", 30.0))
    period = 1.0 / max(action_rate, 1e-6)
    log.info(
        f"[deploy] mode={'EXECUTE' if execute else 'DRY-RUN'} steps={args.steps} "
        f"rate={action_rate}Hz server={cfg['policy_server_url']}"
    )

    try:
        robot.connect()
        client.reset()

        if execute and args.start_right_arm is not None:
            log.info(f"[deploy] move-to-start={args.start_right_arm}")
            robot.move_arm_to([float(v) for v in args.start_right_arm])

        for step in range(args.steps):
            step_started = time.perf_counter()

            obs = robot.get_observation()
            action = np.asarray(client.inference(obs), dtype=np.float32)
            row = action[0]

            if execute:
                robot.apply_action(row)

            arm_str = ", ".join(f"{v:.3f}" for v in row[:7])
            log.info(
                f"[deploy] step {step + 1}/{args.steps} "
                f"dt={time.perf_counter() - step_started:.3f}s "
                f"arm=[{arm_str}] gripper={float(row[7]):.1f}"
            )

            # 阻塞式动作指令本身消耗节拍；此处仅补足剩余周期。
            sleep_time = period - (time.perf_counter() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        client.close()
        robot.disconnect()

    log.info("[deploy] finished OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
