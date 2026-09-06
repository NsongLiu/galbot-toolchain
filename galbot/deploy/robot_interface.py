"""Galbot G1 真机控制接口（基于 galbot_sdk.g1 的真实 API）。

所有 SDK 调用方式均移植自在 G1 真机（Orin）上调试通过的脚本：
  - debug/standalone_infer_8d_chunked.py   （推理 + 右臂/夹爪执行）
  - debug/reset_galbot_to_mcap_first_frame.py （SDK 路径、关机序列、夹爪单位）

``galbot_sdk`` 在 ``connect()`` 内延迟导入（真机上位于系统路径
``/userdata/update/manual_update/lib``，可用 ``sdk_lib_path`` 注入），
因此本模块在未安装 SDK 的环境（如策略服务器）中也能正常 import。

观测契约（与 wire/obs_codec 一致）：
  images: dict[str, np.ndarray]           # 相机短名 -> uint8 RGB HWC
  arm_gripper_joints: np.ndarray (29,)    # 与 robot_config.json 的 state_joint_order 一致
动作契约：8 维 = 右臂 7 关节(rad) + 右夹爪(raw 0..100)。
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 关节布局常量（与 robot_config.json / convert_mcap_to_lerobot.py 一致，真机已验证）
# ---------------------------------------------------------------------------
# get_joint_positions 按此分组顺序返回 27 个值（gripper 单位为米）。
STATE_GROUPS = [
    "right_arm",    # 7
    "chassis",      # 4
    "left_arm",     # 7
    "head",         # 2
    "leg",          # 5
    "right_gripper",  # 1（米）
    "left_gripper",   # 1（米）
]
SDK_STATE_LEN = 27
ARM_JOINT_NAMES = [f"right_arm_joint{i}" for i in range(1, 8)]
ACTION_DIM = 8  # 7 臂关节 + 1 右夹爪
STATE_DIM = 29

_RIGHT_GRIPPER_IDX = 25
_LEFT_GRIPPER_IDX = 26
KEY_DEFAULT = 0.0  # 29 维 state 中 left_key / right_key 占位值

GRIPPER_M_TO_RAW = 1000.0  # SDK 夹爪单位为米；数据集/策略使用 0..100 原始尺度

# 相机短名 -> SensorType 枚举名（真机验证默认值）
DEFAULT_SENSORS = {
    "right_arm": "RIGHT_ARM_CAMERA",
    "front_head_right": "HEAD_RIGHT_CAMERA",
}

# 实机执行确认口令（沿用真机调通脚本的安全约定）
EXECUTE_ACK_TOKEN = "DEPLOY_MODEL"


def _positions(items: Any) -> np.ndarray:
    """SDK 关节项（带 .position 属性或纯数值）-> float32 向量。"""
    return np.asarray(
        [
            float(item.position) if hasattr(item, "position") else float(item)
            for item in items
        ],
        dtype=np.float32,
    )


def _get_rgb_with_timeout(robot: Any, sensor: Any, timeout_sec: float) -> Any:
    """get_rgb_data 可能阻塞，用守护线程 + 超时包装（移植自真机脚本）。"""
    q: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            q.put(robot.get_rgb_data(sensor))
        except Exception as exc:  # noqa: BLE001 - 透传给主线程
            q.put(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise RuntimeError(f"get_rgb_data timeout after {timeout_sec}s")

    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result


def _decode_rgb(payload: Any, sensor_name: str) -> np.ndarray:
    """SDK RGB payload（dict{data: bytes} 或裸 bytes）解码为 uint8 RGB HWC。"""
    import cv2  # 延迟导入：仅机器人端需要 opencv

    if payload is None:
        raise RuntimeError(f"SDK RGB payload is None ({sensor_name})")
    if isinstance(payload, dict):
        encoded = payload.get("data")
        if encoded is None:
            raise RuntimeError(
                f"SDK RGB payload has no 'data' field ({sensor_name}); "
                f"keys={list(payload.keys())}"
            )
    elif isinstance(payload, bytes):
        encoded = payload
    else:
        raise RuntimeError(
            f"unexpected SDK payload type ({sensor_name}): {type(payload).__name__}"
        )

    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to decode SDK RGB payload ({sensor_name})")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class GalbotRobotInterface:
    """``galbot_sdk.g1.GalbotRobot`` 的薄封装，供策略推理控制循环使用。

    参数默认值均取自真机调通脚本（standalone_infer_8d_chunked.py）。
    """

    def __init__(
        self,
        *,
        sensors: dict[str, str] | None = None,
        sdk_lib_path: str | None = None,
        camera_warmup_sec: float = 5.0,
        rgb_timeout_sec: float = 10.0,
        max_speed: float = 0.05,
        timeout: float = 1.0,
        gripper_speed: float = 0.1,
        gripper_force: float = 10.0,
        gripper_min_raw: float = 35.0,
        gripper_max_raw: float = 100.0,
    ) -> None:
        self._sensor_names = dict(sensors or DEFAULT_SENSORS)
        self._sdk_lib_path = sdk_lib_path or None
        self._camera_warmup_sec = float(camera_warmup_sec)
        self._rgb_timeout_sec = float(rgb_timeout_sec)
        self._max_speed = float(max_speed)
        self._timeout = float(timeout)
        self._gripper_speed = float(gripper_speed)
        self._gripper_force = float(gripper_force)
        self._gripper_min_raw = float(gripper_min_raw)
        self._gripper_max_raw = float(gripper_max_raw)

        self._robot: Any | None = None
        self._cam_sensors: dict[str, Any] = {}  # 相机短名 -> SensorType 枚举
        self._control_status: Any = None
        self._g1_joint_group: Any = None
        # 真机脚本行为：夹爪首条命令阻塞；一旦失败则后续全部改为非阻塞。
        self._gripper_blocking = True

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """初始化机器人并枚举相机（真机上需数秒等待图像流就绪）。"""
        if self._sdk_lib_path and self._sdk_lib_path not in sys.path:
            sys.path.insert(0, self._sdk_lib_path)

        from galbot_sdk.g1 import (  # noqa: PLC0415 - 延迟导入 SDK
            ControlStatus,
            G1JointGroup,
            GalbotRobot,
            SensorType,
        )

        self._control_status = ControlStatus
        self._g1_joint_group = G1JointGroup

        cam_sensors: dict[str, Any] = {}
        for cam_name, enum_name in self._sensor_names.items():
            sensor = getattr(SensorType, enum_name, None)
            if sensor is None:
                choices = ", ".join(
                    sorted(key for key in dir(SensorType) if "CAMERA" in key)
                )
                raise RuntimeError(
                    f"SensorType has no {enum_name!r} (camera {cam_name!r}); "
                    f"available camera enums: {choices}"
                )
            cam_sensors[cam_name] = sensor

        self._robot = GalbotRobot()
        if not self._robot.init(set(cam_sensors.values())):
            raise RuntimeError("robot.init() returned False")
        self._cam_sensors = cam_sensors

        if self._camera_warmup_sec > 0:
            time.sleep(self._camera_warmup_sec)  # 等待相机出流（真机必需）
        print(
            f"[GalbotRobotInterface] connected; cameras={sorted(cam_sensors)}",
            flush=True,
        )

    def disconnect(self) -> None:
        """真机验证的关机序列：request_shutdown -> wait_for_shutdown -> destroy。"""
        if self._robot is None:
            return
        for fn_name in ("request_shutdown", "wait_for_shutdown", "destroy"):
            try:
                getattr(self._robot, fn_name)()
            except Exception as exc:  # noqa: BLE001 - 关机阶段不抛出
                print(
                    f"[GalbotRobotInterface] {fn_name} warning: {exc}",
                    flush=True,
                )
        self._robot = None
        self._cam_sensors = {}
        print("[GalbotRobotInterface] disconnected", flush=True)

    def is_connected(self) -> bool:
        return self._robot is not None

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """读取 29 维状态 + 双相机 RGB，组成 PolicyClient 观测。"""
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")

        # 27 个 SDK 关节值（第 25/26 为左右夹爪，单位米）。
        raw_pos = self._robot.get_joint_positions(STATE_GROUPS, [])
        pos = _positions(raw_pos).ravel()
        if pos.size < SDK_STATE_LEN:
            # 真机上宁可中断也不把零状态喂给策略（会产生任意动作）。
            raise RuntimeError(
                f"get_joint_positions returned {pos.size} values, "
                f"expected {SDK_STATE_LEN}"
            )

        # 29 维 state：前 25 关节 + 右夹爪(m->raw) + 2 个 key 占位 0 + 左夹爪(m->raw)。
        state = np.concatenate(
            [
                pos[:_RIGHT_GRIPPER_IDX],
                [
                    pos[_RIGHT_GRIPPER_IDX] * GRIPPER_M_TO_RAW,
                    KEY_DEFAULT,
                    KEY_DEFAULT,
                    pos[_LEFT_GRIPPER_IDX] * GRIPPER_M_TO_RAW,
                ],
            ]
        ).astype(np.float32)
        assert state.shape == (STATE_DIM,)

        images = {
            cam_name: _decode_rgb(
                _get_rgb_with_timeout(self._robot, sensor, self._rgb_timeout_sec),
                cam_name,
            )
            for cam_name, sensor in self._cam_sensors.items()
        }
        return {"images": images, "arm_gripper_joints": state}

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------

    def apply_action(self, action: np.ndarray) -> None:
        """下发 8 维动作：右臂 7 关节(rad) + 右夹爪(raw 0..100)。"""
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != ACTION_DIM:
            raise ValueError(f"action must be ({ACTION_DIM},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or Inf")

        arm = action[:7].tolist()
        status = self._robot.set_joint_positions(
            arm, [], ARM_JOINT_NAMES, True, self._max_speed, self._timeout
        )
        if status != self._control_status.SUCCESS:
            print(f"[GalbotRobotInterface] arm move status={status}", flush=True)

        # 夹爪：raw 限幅（真机安全范围 35..100）后换算为米下发。
        gripper_raw = float(
            np.clip(action[7], self._gripper_min_raw, self._gripper_max_raw)
        )
        status = self._robot.set_gripper_command(
            self._g1_joint_group.right_gripper,
            gripper_raw / GRIPPER_M_TO_RAW,
            self._gripper_speed,
            self._gripper_force,
            self._gripper_blocking,
        )
        if status != self._control_status.SUCCESS:
            print(f"[GalbotRobotInterface] gripper move status={status}", flush=True)
            self._gripper_blocking = False  # 真机脚本：失败后改非阻塞

    def move_arm_to(
        self,
        target: list[float] | np.ndarray,
        *,
        speed: float | None = None,
        timeout: float = 15.0,
    ) -> None:
        """阻塞式移动右臂到指定 7 关节位姿（用于部署前归位）。"""
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")

        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if target_arr.size != 7:
            raise ValueError(f"arm target must have 7 joints, got {target_arr.size}")
        status = self._robot.set_joint_positions(
            target_arr.tolist(),
            [],
            ARM_JOINT_NAMES,
            True,
            self._max_speed if speed is None else float(speed),
            float(timeout),
        )
        if status != self._control_status.SUCCESS:
            raise RuntimeError(f"move_arm_to failed: status={status}")

    # ------------------------------------------------------------------
    # 底盘（速度控制，供键盘遥控等场景使用）
    # ------------------------------------------------------------------

    def acquire_chassis_twist_controller(self) -> bool:
        """尝试获取底盘速度控制器（chassis_twist_ctrl）。

        SDK 文档要求速度控制前获取该控制器；真机未验证,故失败仅告警。
        """
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")
        for fn_name in ("acquire_controller", "switch_controller"):
            fn = getattr(self._robot, fn_name, None)
            if fn is None:
                continue
            try:
                status = fn("chassis_twist_ctrl")
            except Exception as exc:  # noqa: BLE001 - 旧版 SDK 可能无此接口
                print(
                    f"[GalbotRobotInterface] {fn_name} warning: {exc}",
                    flush=True,
                )
                continue
            if status == self._control_status.SUCCESS:
                print(
                    f"[GalbotRobotInterface] chassis_twist_ctrl acquired "
                    f"via {fn_name}",
                    flush=True,
                )
                return True
            print(
                f"[GalbotRobotInterface] {fn_name} status={status}",
                flush=True,
            )
        return False

    def set_base_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration_s: float = 0.2,
    ) -> None:
        """下发底盘速度（基坐标系：vx 前/后 m/s, vy 左/右 m/s, wz 偏航 rad/s）。

        ``duration_s`` 为看门狗时长：超时未收到新命令底盘自动停止。
        """
        if self._robot is None:
            raise RuntimeError("robot not connected; call connect() first")
        status = self._robot.set_base_velocity(
            [float(vx), float(vy), 0.0], [0.0, 0.0, float(wz)], float(duration_s)
        )
        if status != self._control_status.SUCCESS:
            print(
                f"[GalbotRobotInterface] set_base_velocity status={status}",
                flush=True,
            )

    def stop_base(self) -> None:
        """立即停止底盘运动（关机/退出前必调）。"""
        if self._robot is None:
            return
        try:
            self._robot.stop_base()
        except Exception as exc:  # noqa: BLE001 - 停止阶段不抛出
            print(f"[GalbotRobotInterface] stop_base warning: {exc}", flush=True)
