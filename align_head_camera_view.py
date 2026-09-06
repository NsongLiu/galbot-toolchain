#!/usr/bin/env python
"""头部右相机视图对齐工具。

以某个 mcap 数据中的头部右相机图像为基准，实时显示机器人当前头部右相机
画面，通过键盘脉冲式控制 G1 底盘平移/旋转，将两个画面对齐（用于复现采集
时的机身位姿）。

按键（焦点需在图像窗口上）：
  w/s     前进 / 后退          a/d     左移 / 右移
  q/e     左转 / 右转          space   立即停止
  +/-     速度放大 / 缩小       v       切换视图（默认半透明叠加，循环：叠加 -> 差分 -> 并排）
  ESC/x   退出（退出前自动停止底盘）

移动为脉冲式：每次按键以下发 --pulse 秒的速度命令，超时自动停止；按住不放
（系统键重复）即为连续运动。

用法示例：
  # 开发机上先从 mcap 提取基准图（无需机器人）：
  python align_head_camera_view.py --mcap episode.mcap --extract-only --save-ref ref.png

  # 真机上运行（SDK 位于 Orin 系统路径）：
  python align_head_camera_view.py --image ref.png
  python align_head_camera_view.py --mcap episode.mcap --frame 0   # 若真机装有 mcap 库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HEAD_RIGHT_TOPIC = "/front_head_camera/right_color/image_raw"
WIN_NAME = "align_head_camera_view"

# 视图模式：并排 / 半透明叠加 / 差分放大
VIEW_MODES = ["side", "overlay", "diff"]


def decode_bgr(data: bytes) -> np.ndarray:
    """压缩图像字节（JPEG/PNG）-> BGR numpy 数组（供 cv2.imshow）。"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode compressed image")
    return img


def load_reference_from_mcap(mcap_path: Path, topic: str, frame_idx: int) -> np.ndarray:
    """流式扫描 mcap，取该 topic 的第 frame_idx 帧（-1 为最后一帧）。"""
    from mcap_protobuf.reader import read_protobuf_messages  # 延迟导入：真机可能没装

    last: bytes | None = None
    count = 0
    for m in read_protobuf_messages(str(mcap_path), topics=[topic]):
        if frame_idx >= 0 and count == frame_idx:
            return decode_bgr(m.proto_msg.data)
        last = m.proto_msg.data
        count += 1
    if frame_idx == -1 and last is not None:
        return decode_bgr(last)
    raise ValueError(
        f"topic {topic} has {count} frame(s) in {mcap_path}; "
        f"frame index {frame_idx} out of range (use -1 for the last frame)"
    )


def compose_view(ref: np.ndarray, live: np.ndarray, mode: str) -> np.ndarray:
    """按模式合成显示画面（live 先缩放到 ref 尺寸）。"""
    if live.shape[:2] != ref.shape[:2]:
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
    if mode == "overlay":
        return cv2.addWeighted(ref, 0.5, live, 0.5, 0.0)
    if mode == "diff":
        return np.clip(cv2.absdiff(ref, live).astype(np.int16) * 3, 0, 255).astype(
            np.uint8
        )
    view = np.hstack([ref, live])
    cv2.putText(view, "REF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(
        view, "LIVE", (ref.shape[1] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
    )
    return view


def draw_hud(view: np.ndarray, mode: str, speed_scale: float, robot_on: bool) -> None:
    """在画面底部叠加状态栏。"""
    text = (
        f"[{mode}] speed x{speed_scale:.2f} robot={'on' if robot_on else 'DRY'} | "
        "w/s/a/d move q/e turn space stop +/- speed v view ESC quit"
    )
    cv2.rectangle(view, (0, view.shape[0] - 28), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        view, text, (8, view.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mcap", type=Path, help="基准图像来源 mcap 文件")
    src.add_argument("--image", type=Path, help="直接使用图片文件作为基准（png/jpg）")
    p.add_argument("--topic", default=HEAD_RIGHT_TOPIC, help="mcap 中头部右相机 topic")
    p.add_argument("--frame", type=int, default=0, help="取该 topic 的第 N 帧（-1 = 最后一帧）")
    p.add_argument("--save-ref", type=Path, default=None, help="把基准图另存为 png（便于拷贝到真机）")
    p.add_argument("--extract-only", action="store_true", help="只提取基准图，不连接机器人")
    p.add_argument("--sdk-lib-path", default="/userdata/update/manual_update/lib", help="真机 SDK 库路径")
    p.add_argument("--camera-warmup", type=float, default=5.0, help="相机出流等待秒数")
    p.add_argument("--linear-speed", type=float, default=0.05, help="平移速度 m/s")
    p.add_argument("--angular-speed", type=float, default=0.15, help="旋转速度 rad/s")
    p.add_argument("--pulse", type=float, default=0.2, help="单次脉冲时长 s（看门狗超时）")
    p.add_argument("--no-robot", action="store_true", help="空跑模式：不连接机器人，live 画面用基准图代替")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- 基准图像 ----
    if args.mcap:
        ref = load_reference_from_mcap(args.mcap, args.topic, args.frame)
        print(f"[ref] {args.mcap} topic={args.topic} frame={args.frame} shape={ref.shape}")
    else:
        ref = cv2.imread(str(args.image))
        if ref is None:
            raise SystemExit(f"failed to read image: {args.image}")
        print(f"[ref] {args.image} shape={ref.shape}")
    if args.save_ref:
        cv2.imwrite(str(args.save_ref), ref)
        print(f"[ref] saved to {args.save_ref}")
    if args.extract_only:
        return

    # ---- 机器人连接 ----
    robot = None
    if not args.no_robot:
        from galbot.deploy.robot_interface import GalbotRobotInterface

        robot = GalbotRobotInterface(
            sensors={"front_head_right": "HEAD_RIGHT_CAMERA"},
            sdk_lib_path=args.sdk_lib_path,
            camera_warmup_sec=args.camera_warmup,
        )
        robot.connect()
        if not robot.acquire_chassis_twist_controller():
            print(
                "[warn] chassis_twist_ctrl 获取失败（SDK 版本可能不同），"
                "仍将尝试直接下发速度命令",
                flush=True,
            )

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    print("[keys] w/s 前后 a/d 左右 q/e 旋转 space 停止 +/- 调速 v 视图 ESC/x 退出", flush=True)

    lin, ang = args.linear_speed, args.angular_speed
    speed_scale = 1.0
    mode_idx = VIEW_MODES.index("overlay")  # 默认半透明叠加，对齐最直观
    # 按键 -> (vx, vy, wz) 方向
    key_dirs = {
        ord("w"): (1.0, 0.0, 0.0), ord("s"): (-1.0, 0.0, 0.0),
        ord("a"): (0.0, 1.0, 0.0), ord("d"): (0.0, -1.0, 0.0),
        ord("q"): (0.0, 0.0, 1.0), ord("e"): (0.0, 0.0, -1.0),
    }

    try:
        while True:
            if robot is not None:
                obs = robot.get_observation()
                live = cv2.cvtColor(obs["images"]["front_head_right"], cv2.COLOR_RGB2BGR)
            else:
                live = ref.copy()

            view = compose_view(ref, live, VIEW_MODES[mode_idx])
            draw_hud(view, VIEW_MODES[mode_idx], speed_scale, robot is not None)
            cv2.imshow(WIN_NAME, view)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("x")):  # ESC / x
                break
            if key == ord("v"):
                mode_idx = (mode_idx + 1) % len(VIEW_MODES)
            elif key in (ord("+"), ord("=")):
                speed_scale = min(speed_scale * 1.5, 3.0)
            elif key in (ord("-"), ord("_")):
                speed_scale = max(speed_scale / 1.5, 0.1)
            elif key == ord(" "):
                if robot is not None:
                    robot.stop_base()
            elif key in key_dirs:
                vx, vy, wz = (d * speed_scale for d in key_dirs[key])
                cmd = (vx * lin, vy * lin, wz * ang, args.pulse)
                if robot is not None:
                    robot.set_base_velocity(*cmd[:3], duration_s=cmd[3])
                else:
                    print(f"[dry] set_base_velocity vx={cmd[0]:.3f} vy={cmd[1]:.3f} wz={cmd[2]:.3f}")
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if robot is not None:
            robot.stop_base()
            robot.disconnect()
        cv2.destroyAllWindows()
        print("[done] base stopped, robot disconnected", flush=True)


if __name__ == "__main__":
    main()
