#!/usr/bin/env python
"""Convert Galbot G1 MCAP protobuf recordings to a LeRobot V3 dataset.

Streaming conversion（参照 xhum hdf5_to_lerobot.py 的做法）：
  - 每个 episode 只单遍扫描 MCAP：数值流（sensor/target）直接解析，
    相机帧保留压缩字节，仅解码时间轴实际选中的帧；
  - JPEG 解码用线程池并行（--decode-workers）；
  - 视频走 LeRobot streaming encoder（帧直接进编码器，不再先写临时 PNG 再读回）。

差一帧 bug 防护：LeRobot streaming encoder 的队列满 100ms 后会丢帧而不是阻塞
（video_utils.py feed_frame），默认队列 30 会让视频帧数少于 parquet 帧数。
本脚本按 MCAP summary 统计把队列设为能容纳最长 episode（--encoder-queue-maxsize
可覆盖），并在转换结束后用 ffprobe 逐相机核对视频帧数与 parquet 总帧数。
"""

import argparse
import bisect
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.reader import read_protobuf_messages

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Disable OpenCV internal threading to avoid deadlocks/contention when we run
# many decode workers in a ThreadPoolExecutor.
cv2.setNumThreads(0)

SENSOR_TOPIC = "singorix/wbcs/sensor"
TARGET_TOPIC = "singorix/wbcs/target"

# Fixed group order observed in the sensor messages. Total = 29 joints.
STATE_GROUPS = [
    ("right_arm", 7),
    ("chassis", 4),
    ("left_arm", 7),
    ("head", 2),
    ("leg", 5),
    ("right_gripper", 1),
    ("left_key", 1),
    ("right_key", 1),
    ("left_gripper", 1),
]

# Default action groups and their joint counts (left arm teleoperation).
# Can be overridden via --action-groups, e.g. "right_arm,7" "right_gripper,1".
DEFAULT_ACTION_GROUPS = [
    ("left_arm", 7),
    ("left_gripper", 1),
]

# Available color camera topics and the feature keys they map to.
DEFAULT_CAMERA_TOPICS = {
    "left_arm": "/left_arm_camera/color/image_raw",
    "right_arm": "/right_arm_camera/color/image_raw",
    "front_head_left": "/front_head_camera/left_color/image_raw",
    "front_head_right": "/front_head_camera/right_color/image_raw",
}


def parse_action_groups(arg_list: list[str] | None) -> list[tuple[str, int]]:
    """Parse --action-groups CLI values like ['right_arm,7', 'right_gripper,1']."""
    if not arg_list:
        return DEFAULT_ACTION_GROUPS
    groups = []
    for item in arg_list:
        for part in item.split():
            name, count = part.rsplit(",", 1)
            groups.append((name.strip(), int(count.strip())))
    return groups


def select_camera_topics(camera_keys: list[str] | None) -> dict[str, str]:
    """Return the subset of CAMERA_TOPICS to use for conversion."""
    if not camera_keys:
        return dict(DEFAULT_CAMERA_TOPICS)
    return {k: DEFAULT_CAMERA_TOPICS[k] for k in camera_keys if k in DEFAULT_CAMERA_TOPICS}


def choose_master_camera(camera_topics: dict[str, str]) -> str:
    """Pick a master camera to drive the frame timeline.

    Prefer wrist cameras over head cameras, and right over left.
    """
    priority = ["right_arm", "left_arm", "front_head_right", "front_head_left"]
    for key in priority:
        if key in camera_topics:
            return key
    return next(iter(camera_topics))


def build_state_vector(sensor_msg) -> np.ndarray:
    """Flatten a SingoriXSensor message into a 29-dim state vector."""
    vec = []
    for group, count in STATE_GROUPS:
        group_state = sensor_msg.joint_sensor_map[group]
        # Sensor always provides positions for every joint in the group.
        positions = list(group_state.position)
        if len(positions) != count:
            raise ValueError(
                f"Unexpected number of joints for group {group}: "
                f"expected {count}, got {len(positions)}"
            )
        vec.extend(positions)
    return np.array(vec, dtype=np.float32)


def parse_target_positions(target_msg, group: str) -> list[float]:
    """Extract target positions for a single action group from a target message."""
    group_target = target_msg.target_group_trajectory_map[group]
    if not group_target.group_commands:
        return []
    # Use the first trajectory point; it represents the immediate target.
    return [cmd.position for cmd in group_target.group_commands[0].joint_commands]


def decode_image(data: bytes) -> np.ndarray:
    """Decode compressed image bytes (JPEG/PNG) to an RGB numpy array."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode compressed image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def build_robot_config(
    mcap_path: Path,
    robot_type: str,
    action_groups: list[tuple[str, int]],
) -> dict:
    """Build a robot joint configuration JSON from the first sensor message."""
    for m in read_protobuf_messages(mcap_path, topics=[SENSOR_TOPIC]):
        sensor_msg = m.proto_msg
        break
    else:
        raise ValueError(f"No sensor message found in {mcap_path}")

    groups = {}
    state_joint_order = []
    for group_name, expected_count in STATE_GROUPS:
        group_state = sensor_msg.joint_sensor_map[group_name]
        names = list(group_state.name)
        if len(names) != expected_count:
            raise ValueError(
                f"Unexpected joint count for {group_name}: expected {expected_count}, got {len(names)}"
            )
        groups[group_name] = {
            "count": len(names),
            "joint_names": names,
        }
        state_joint_order.extend(names)

    action_joint_order = []
    for group_name, expected_count in action_groups:
        names = groups[group_name]["joint_names"]
        if len(names) != expected_count:
            raise ValueError(
                f"Unexpected joint count for action group {group_name}: expected {expected_count}, got {len(names)}"
            )
        action_joint_order.extend(names)

    return {
        "robot_type": robot_type,
        "total_state_joints": len(state_joint_order),
        "total_action_joints": len(action_joint_order),
        "state_joint_order": state_joint_order,
        "action_joint_order": action_joint_order,
        "groups": groups,
    }


def read_mcap_file(
    mcap_path: Path,
    action_groups: list[tuple[str, int]],
    camera_topics: dict[str, str],
):
    """Single-pass streaming read of one MCAP file.

    Numeric streams (sensor/target) are parsed directly; camera frames are kept
    as compressed bytes and decoded later only for the frames the timeline
    actually selects. read_protobuf_messages yields in log-time order
    (log_time_order=True by default), so every per-topic stream is sorted.
    """
    topic_to_key = {topic: key for key, topic in camera_topics.items()}
    topics = [SENSOR_TOPIC, TARGET_TOPIC, *topic_to_key]

    sensor_times = []
    sensor_states = []
    target_times = {group: [] for group, _ in action_groups}
    target_values = {group: [] for group, _ in action_groups}
    # key -> {log_time_ns: compressed bytes}; dedup by timestamp keeps the first
    # frame (some arm camera recordings contain duplicate timestamps).
    camera_frames = {key: {} for key in camera_topics}

    for m in read_protobuf_messages(mcap_path, topics=topics):
        key = topic_to_key.get(m.topic)
        if key is not None:
            camera_frames[key].setdefault(m.log_time_ns, m.proto_msg.data)
        elif m.topic == SENSOR_TOPIC:
            sensor_times.append(m.log_time_ns)
            sensor_states.append(build_state_vector(m.proto_msg))
        else:  # TARGET_TOPIC
            for group, _ in action_groups:
                if group in m.proto_msg.target_group_trajectory_map:
                    positions = parse_target_positions(m.proto_msg, group)
                    if positions:
                        target_times[group].append(m.log_time_ns)
                        target_values[group].append(positions)

    camera_times = {}
    camera_payloads = {}
    for key, frames in camera_frames.items():
        times = sorted(frames.keys())
        camera_times[key] = times
        camera_payloads[key] = [frames[t] for t in times]

    return {
        "sensor_times": sensor_times,
        "sensor_states": sensor_states,
        "target_times": target_times,
        "target_values": target_values,
        "camera_times": camera_times,
        "camera_payloads": camera_payloads,
    }


def nearest_index(times: list[int], t_ns: int) -> int:
    """Return the index of the nearest time in a sorted list."""
    idx = bisect.bisect_left(times, t_ns)
    if idx == 0:
        return 0
    if idx == len(times):
        return len(times) - 1
    if abs(times[idx] - t_ns) < abs(times[idx - 1] - t_ns):
        return idx
    return idx - 1


def latest_index_leq(times: list[int], t_ns: int) -> int | None:
    """Return the index of the latest time <= t_ns, or None if none exists."""
    idx = bisect.bisect_right(times, t_ns) - 1
    if idx < 0:
        return None
    return idx


def get_state_at_time(sensor_times, sensor_states, t_ns: int) -> np.ndarray:
    idx = nearest_index(sensor_times, t_ns)
    return sensor_states[idx]


def get_action_at_time(
    target_times, target_values, t_ns: int, action_groups: list[tuple[str, int]]
) -> np.ndarray:
    """Build action vector from the latest target per group at or before t_ns."""
    vec = []
    for group, count in action_groups:
        idx = latest_index_leq(target_times[group], t_ns)
        if idx is None:
            vec.extend([0.0] * count)
        else:
            positions = target_values[group][idx]
            if len(positions) != count:
                raise ValueError(
                    f"Unexpected target size for {group}: expected {count}, got {len(positions)}"
                )
            vec.extend(positions)
    return np.array(vec, dtype=np.float32)


def select_camera_indices(times: list[int], master_times: list[int], key: str, tolerance_ns: int) -> list[int]:
    """Per master timestamp, the index of the nearest camera frame within tolerance."""
    indices = []
    for t_ns in master_times:
        idx = nearest_index(times, t_ns)
        if abs(times[idx] - t_ns) > tolerance_ns:
            raise ValueError(
                f"No {key} camera frame within tolerance for timestamp {t_ns}"
            )
        indices.append(idx)
    return indices


# 单侧最多允许裁剪的边界帧数（30fps 下约 0.5s）；超过则视为真实数据问题。
MAX_BOUNDARY_TRIM = 15


def trim_unalignable_boundary_frames(
    master_times: list[int],
    camera_times: dict[str, list[int]],
    tolerance_ns: int,
) -> tuple[list[int], int, int]:
    """Trim leading/trailing master frames that no camera can cover within tolerance.

    采集端两路相机的启停时刻不齐（如头部相机流晚启动 ~34ms），导致 episode 首/尾
    个别主帧找不到容差内的配对帧。这些边界帧机器人一般尚未开始/已结束动作，直接
    裁剪；中间帧失配或单侧裁剪超过 MAX_BOUNDARY_TRIM 仍报错（真实丢帧不被掩盖）。

    Returns:
        (trimmed master_times, dropped_head, dropped_tail)
    """
    n = len(master_times)
    alignable = [True] * n
    for key, times in camera_times.items():
        if not times:
            raise ValueError(f"no frames for camera {key}")
        for i, t_ns in enumerate(master_times):
            idx = nearest_index(times, t_ns)
            if abs(times[idx] - t_ns) > tolerance_ns:
                alignable[i] = False

    if all(alignable):
        return master_times, 0, 0
    if not any(alignable):
        raise ValueError("no master frame is alignable across cameras")

    first = alignable.index(True)                      # 首个可对齐帧（前缀长度）
    last = n - alignable[::-1].index(True)             # 末个可对齐帧的后一位
    if not all(alignable[first:last]):
        bad = [i for i in range(first, last) if not alignable[i]]
        raise ValueError(
            f"interior master frames are not alignable ({len(bad)} frames, "
            f"first at index {bad[0]}); refusing to trim interior data"
        )
    n_head, n_tail = first, n - last
    if n_head > MAX_BOUNDARY_TRIM or n_tail > MAX_BOUNDARY_TRIM:
        raise ValueError(
            f"unalignable boundary too long: head={n_head} tail={n_tail} "
            f"(max {MAX_BOUNDARY_TRIM} per side); check camera streams"
        )
    return master_times[first:last], n_head, n_tail


def get_image_shapes(camera_payloads: dict[str, list[bytes]]) -> dict[str, tuple[int, int, int]]:
    """Infer image shapes by decoding the first frame of each camera."""
    shapes = {}
    for key, payloads in camera_payloads.items():
        if not payloads:
            raise ValueError(f"No images for camera {key}")
        shapes[key] = tuple(decode_image(payloads[0]).shape)
    return shapes


def plan_encoder_queue(mcap_files: list[Path], master_topic: str, cap: int = 4096) -> int:
    """Pick an encoder queue size that makes LeRobot's frame-drop path unreachable.

    The streaming encoder drops a frame whenever its queue stays full for 100ms
    (video_utils.py feed_frame). A queue that can hold the longest episode never
    fills, so no frame is dropped. Episode lengths come from the MCAP summary's
    per-channel message counts — a metadata-only read. The count is taken before
    timestamp dedup, so it is a safe upper bound.
    """
    longest = 0
    for path in mcap_files:
        try:
            with open(path, "rb") as f:
                summary = make_reader(f).get_summary()
        except OSError:
            continue
        if summary is None or summary.statistics is None:
            continue
        counts = summary.statistics.channel_message_counts
        for ch_id, channel in summary.channels.items():
            if channel.topic == master_topic:
                longest = max(longest, counts.get(ch_id, 0))

    if longest == 0:
        print("WARNING: could not read MCAP summary statistics; using encoder queue of 1024 frames.")
        return 1024
    if longest > cap:
        print(
            f"WARNING: longest episode is {longest} frames but the encoder queue is capped at {cap};"
            " frames may still be dropped. The post-conversion check will report it."
        )
        return cap
    print(f"Encoder queue: {longest} frames (longest episode).")
    return longest


def _count_frames(mp4: Path, *, exact: bool) -> int:
    """Frame count of one mp4. ``exact`` decodes every packet instead of trusting the header."""
    entry = "stream=nb_read_frames" if exact else "stream=nb_frames"
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if exact:
        cmd.append("-count_frames")
    cmd += ["-show_entries", entry, "-of", "default=nw=1:nk=1", str(mp4)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def verify_video_frames(dataset_root: Path) -> bool:
    """Check that every camera's video holds exactly as many frames as the parquet data.

    LeRobot's streaming encoder silently drops frames when its queue is full, so a
    conversion can look successful while a camera is short a few frames (差一帧 bug).
    Counting them back out is the only reliable way to notice.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"WARNING: verify: info.json not found at {info_path}, skipping check")
        return True

    with open(info_path) as f:
        expected = json.load(f).get("total_frames")
    if not expected:
        print("WARNING: verify: total_frames missing from info.json, skipping check")
        return True

    ok = True
    for cam_dir in sorted((dataset_root / "videos").glob("*")):
        if not cam_dir.is_dir():
            continue
        mp4s = sorted(cam_dir.rglob("*.mp4"))
        counts = [_count_frames(m, exact=False) for m in mp4s]
        # Fall back to decoding for any file whose header does not carry a count.
        for idx, (m, n) in enumerate(zip(mp4s, counts, strict=True)):
            if n < 0:
                counts[idx] = _count_frames(m, exact=True)
        total = sum(max(n, 0) for n in counts)

        if total != expected:
            # Only pay for a full decode when the headers disagree, to avoid false alarms.
            total = sum(max(_count_frames(m, exact=True), 0) for m in mp4s)

        if total == expected:
            print(f"verify: {cam_dir.name} {total}/{expected} frames OK")
        else:
            ok = False
            print(f"ERROR: verify: {cam_dir.name} {total}/{expected} frames — {expected - total} MISSING")
    return ok


def process_episode(
    data: dict,
    dataset: LeRobotDataset,
    task: str,
    action_groups: list[tuple[str, int]],
    camera_topics: dict[str, str],
    tolerance_ns: int,
    decode_workers: int,
) -> int:
    """Decode selected frames and add one episode to the dataset."""
    master_key = choose_master_camera(camera_topics)
    master_times = data["camera_times"][master_key]
    if not master_times:
        raise ValueError("No master camera frames in episode")

    # 裁剪首/尾无相机配对（超容差）的边界帧；中间帧失配仍抛错。
    master_times, n_head, n_tail = trim_unalignable_boundary_frames(
        master_times, data["camera_times"], tolerance_ns
    )
    if n_head or n_tail:
        print(
            f"  trimmed {n_head} leading + {n_tail} trailing master frame(s) "
            "with no camera match within tolerance"
        )

    # Per camera, the frame indices selected by the master timeline (same
    # nearest-within-tolerance semantics as the non-streaming converter).
    selected = {
        key: select_camera_indices(data["camera_times"][key], master_times, key, tolerance_ns)
        for key in camera_topics
    }

    # Decode only the selected frames; dedup repeats first, then map back in
    # timeline order. One thread pool is shared by all cameras of the episode.
    decoded = {}
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for key in camera_topics:
            payloads = data["camera_payloads"][key]
            unique_indices = sorted(set(selected[key]))
            buffers = [payloads[i] for i in unique_indices]
            images = list(pool.map(decode_image, buffers))
            lut = dict(zip(unique_indices, images))
            decoded[key] = [lut[i] for i in selected[key]]

    for frame_idx, t_ns in enumerate(master_times):
        frame = {
            "task": task,
            "observation.state": get_state_at_time(
                data["sensor_times"], data["sensor_states"], t_ns
            ),
            "action": get_action_at_time(
                data["target_times"], data["target_values"], t_ns, action_groups
            ),
        }
        for key in camera_topics:
            frame[f"observation.images.{key}"] = decoded[key][frame_idx]
        dataset.add_frame(frame)

    return len(master_times)


def convert_mcap_to_lerobot(
    mcap_dir: Path,
    out_dir: Path,
    repo_id: str,
    fps: int,
    task: str,
    robot_type: str,
    use_videos: bool,
    vcodec: str,
    robot_config_path: Path,
    action_groups: list[tuple[str, int]],
    camera_topics: dict[str, str],
    streaming_encoding: bool,
    encoder_queue_maxsize: int,
    decode_workers: int,
):
    mcap_files = sorted([p for p in mcap_dir.rglob("*.mcap") if "SYNC" in p.name])
    if not mcap_files:
        raise FileNotFoundError(f"No SYNC .mcap files found in {mcap_dir}")

    print(f"Found {len(mcap_files)} SYNC MCAP file(s) in {mcap_dir}")

    # Build and write a robot joint configuration JSON from the first SYNC file.
    robot_config = build_robot_config(mcap_files[0], robot_type, action_groups)
    robot_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(robot_config_path, "w", encoding="utf-8") as f:
        json.dump(robot_config, f, indent=2, ensure_ascii=False)
    print(f"Robot config written to {robot_config_path}")

    if streaming_encoding and use_videos and encoder_queue_maxsize <= 0:
        master_topic = camera_topics[choose_master_camera(camera_topics)]
        encoder_queue_maxsize = plan_encoder_queue(mcap_files, master_topic)
    elif encoder_queue_maxsize <= 0:
        encoder_queue_maxsize = 30  # LeRobot default; unused without streaming encoding

    # Read the first file to infer image shapes before creating the dataset.
    first_data = read_mcap_file(mcap_files[0], action_groups, camera_topics)
    image_shapes = get_image_shapes(first_data["camera_payloads"])

    action_dim = sum(count for _, count in action_groups)
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": robot_config["action_joint_order"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (sum(count for _, count in STATE_GROUPS),),
            "names": None,
        },
    }
    for key, shape in image_shapes.items():
        features[f"observation.images.{key}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_dir,
        robot_type=robot_type,
        use_videos=use_videos,
        vcodec=vcodec,
        # 帧直接进视频编码器，不再先写临时 PNG 再读回；帧数据与默认路径一致，
        # 仅图像统计量由全量累计（默认路径为采样估计）。
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
    )

    half_interval_ns = int(1e9 / fps)

    for episode_idx, mcap_path in enumerate(mcap_files):
        print(f"\nConverting episode {episode_idx}: {mcap_path.name}")
        t0 = time.perf_counter()
        data = first_data if episode_idx == 0 else read_mcap_file(
            mcap_path, action_groups, camera_topics
        )
        t_read = time.perf_counter() - t0

        num_frames = process_episode(
            data,
            dataset,
            task,
            action_groups,
            camera_topics,
            half_interval_ns,
            decode_workers,
        )
        dataset.save_episode()
        print(
            f"  Saved episode {episode_idx} with {num_frames} frames "
            f"(read {t_read:.1f}s, total {time.perf_counter() - t0:.1f}s)"
        )

    dataset.finalize()
    print(f"\nDataset saved to {out_dir}")

    if use_videos and not verify_video_frames(out_dir):
        print(
            "ERROR: frame count mismatch — the encoder dropped frames. Re-run with a larger"
            " --encoder-queue-maxsize, or with --no-streaming-encoding."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Galbot G1 MCAP recordings to a LeRobot V3 dataset."
    )
    parser.add_argument(
        "--mcap-dir",
        type=Path,
        default=Path("/media/jushen/Leslie-liu/galbot_dataset/tmp"),
        help="Directory containing .mcap files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/media/jushen/Leslie-liu/galbot_dataset/lerobot_v3"),
        help="Output directory for the LeRobot V3 dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="galbot/g1_recordings",
        help="Repo ID stored in the dataset metadata",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="teleoperate the left arm",
        help="Task description stored for every frame",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="galbot_g1",
        help="Robot type written to dataset metadata",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Store cameras as PNG images instead of encoded videos",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec when use-images is not set",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path(__file__).parent / "robot_config.json",
        help="Path to write the robot joint configuration JSON",
    )
    parser.add_argument(
        "--action-groups",
        nargs="+",
        default=None,
        help=(
            "Action groups to extract from singorix/wbcs/target, formatted as 'name,count'. "
            "Defaults to left_arm,7 left_gripper,1. Example: --action-groups right_arm,7 right_gripper,1"
        ),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Camera feature keys to include. Available: left_arm, right_arm, front_head_left, "
            "front_head_right. Defaults to all cameras. Example: --cameras right_arm front_head_right"
        ),
    )
    parser.add_argument(
        "--no-streaming-encoding",
        dest="streaming_encoding",
        action="store_false",
        help=(
            "Fall back to LeRobot's default path, which buffers every frame as a temporary"
            " PNG before encoding. Much slower and writes far more scratch data."
        ),
    )
    parser.set_defaults(streaming_encoding=True)
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=0,
        help=(
            "Frames buffered per camera when streaming to the encoder. LeRobot drops frames"
            " instead of blocking once this queue is full, so it must be able to hold a whole"
            " episode. Default 0 sizes it from the longest episode via MCAP summary statistics."
        ),
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help=(
            "Thread count for parallel JPEG decode per episode (0 = auto: min(8, CPU count))."
            " Does not parallelize LeRobot dataset writes; use 1 if memory is tight."
        ),
    )
    args = parser.parse_args()

    # Suppress ffmpeg/libav "moov atom" info logs
    os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
    logging.getLogger("libav").setLevel(logging.ERROR)

    action_groups = parse_action_groups(args.action_groups)
    camera_topics = select_camera_topics(args.cameras)
    decode_workers = args.decode_workers
    if decode_workers <= 0:
        decode_workers = min(8, (os.cpu_count() or 4))

    convert_mcap_to_lerobot(
        mcap_dir=args.mcap_dir,
        out_dir=args.out_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        robot_type=args.robot_type,
        use_videos=not args.use_images,
        vcodec=args.vcodec,
        robot_config_path=args.robot_config,
        action_groups=action_groups,
        camera_topics=camera_topics,
        streaming_encoding=args.streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        decode_workers=decode_workers,
    )


if __name__ == "__main__":
    main()
