#!/usr/bin/env python
"""Convert Galbot G1 MCAP protobuf recordings to a LeRobot V3 dataset."""

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from mcap_protobuf.reader import read_protobuf_messages

from lerobot.datasets.lerobot_dataset import LeRobotDataset


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

# Action groups and their joint counts.
ACTION_GROUPS = [
    ("left_arm", 7),
    ("left_gripper", 1),
]

# Color camera topics and the feature keys they map to.
CAMERA_TOPICS = {
    "left_arm": "/left_arm_camera/color/image_raw",
    "right_arm": "/right_arm_camera/color/image_raw",
    "front_head_left": "/front_head_camera/left_color/image_raw",
    "front_head_right": "/front_head_camera/right_color/image_raw",
}


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


def decode_image(compressed_image_msg) -> np.ndarray:
    """Decode a CompressedImage protobuf message to an RGB numpy array."""
    data = compressed_image_msg.data
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode compressed image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def build_robot_config(mcap_path: Path, robot_type: str) -> dict:
    """Build a robot joint configuration JSON from the first sensor message."""
    for m in read_protobuf_messages(mcap_path, topics=["singorix/wbcs/sensor"]):
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
    for group_name, expected_count in ACTION_GROUPS:
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


def read_mcap_file(mcap_path: Path):
    """Read one MCAP file and return structured data for conversion."""
    # Sensor: full body state.
    sensor_times = []
    sensor_states = []
    for m in read_protobuf_messages(mcap_path, topics=["singorix/wbcs/sensor"]):
        sensor_times.append(m.log_time_ns)
        sensor_states.append(build_state_vector(m.proto_msg))

    # Targets: per-group action commands.
    target_times = {group: [] for group, _ in ACTION_GROUPS}
    target_values = {group: [] for group, _ in ACTION_GROUPS}
    for m in read_protobuf_messages(mcap_path, topics=["singorix/wbcs/target"]):
        for group, _ in ACTION_GROUPS:
            if group in m.proto_msg.target_group_trajectory_map:
                positions = parse_target_positions(m.proto_msg, group)
                if positions:
                    target_times[group].append(m.log_time_ns)
                    target_values[group].append(positions)

    # Color cameras. Deduplicate by timestamp, keeping the first frame for each
    # timestamp. Some arm camera recordings contain duplicate timestamps, so we
    # use the unique timestamps as the canonical timeline.
    camera_times = {}
    camera_images = {}
    for key, topic in CAMERA_TOPICS.items():
        frame_by_time = {}
        for m in read_protobuf_messages(mcap_path, topics=[topic]):
            t = m.log_time_ns
            if t not in frame_by_time:
                frame_by_time[t] = decode_image(m.proto_msg)
        times = sorted(frame_by_time.keys())
        camera_times[key] = times
        camera_images[key] = [frame_by_time[t] for t in times]

    return {
        "sensor_times": sensor_times,
        "sensor_states": sensor_states,
        "target_times": target_times,
        "target_values": target_values,
        "camera_times": camera_times,
        "camera_images": camera_images,
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


def get_action_at_time(target_times, target_values, t_ns: int) -> np.ndarray:
    """Build action vector from the latest target per group at or before t_ns."""
    vec = []
    for group, count in ACTION_GROUPS:
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


def get_camera_image_at_time(camera_times, camera_images, key: str, t_ns: int, tolerance_ns: int):
    """Return the image for a camera nearest to t_ns, within tolerance."""
    times = camera_times[key]
    images = camera_images[key]
    idx = nearest_index(times, t_ns)
    if abs(times[idx] - t_ns) > tolerance_ns:
        raise ValueError(
            f"No {key} camera frame within tolerance for timestamp {t_ns}"
        )
    return images[idx]


def get_image_shapes(camera_images: dict) -> dict[str, tuple[int, int, int]]:
    """Infer image shapes from the first decoded frame of each camera."""
    shapes = {}
    for key, images in camera_images.items():
        if not images:
            raise ValueError(f"No images for camera {key}")
        shapes[key] = tuple(images[0].shape)
    return shapes


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
):
    mcap_files = sorted([p for p in mcap_dir.glob("*.mcap") if "SYNC" in p.name])
    if not mcap_files:
        raise FileNotFoundError(f"No SYNC .mcap files found in {mcap_dir}")

    print(f"Found {len(mcap_files)} SYNC MCAP file(s) in {mcap_dir}")

    # Build and write a robot joint configuration JSON from the first SYNC file.
    robot_config = build_robot_config(mcap_files[0], robot_type)
    robot_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(robot_config_path, "w", encoding="utf-8") as f:
        json.dump(robot_config, f, indent=2, ensure_ascii=False)
    print(f"Robot config written to {robot_config_path}")

    # Read the first file to infer image shapes before creating the dataset.
    first_data = read_mcap_file(mcap_files[0])
    image_shapes = get_image_shapes(first_data["camera_images"])

    features = {
        "action": {
            "dtype": "float32",
            "shape": (sum(count for _, count in ACTION_GROUPS),),
            "names": [f"left_arm_joint{i}" for i in range(1, 8)] + ["left_gripper_joint1"],
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
    )

    half_interval_ns = int(1e9 / fps)

    for episode_idx, mcap_path in enumerate(mcap_files):
        print(f"\nConverting episode {episode_idx}: {mcap_path.name}")
        data = read_mcap_file(mcap_path)

        master_key = "left_arm"
        master_times = data["camera_times"][master_key]
        if not master_times:
            raise ValueError(f"No master camera frames in {mcap_path.name}")

        start_ns = master_times[0]
        for frame_idx, t_ns in enumerate(master_times):
            state = get_state_at_time(
                data["sensor_times"], data["sensor_states"], t_ns
            )
            action = get_action_at_time(
                data["target_times"], data["target_values"], t_ns
            )

            frame = {
                "task": task,
                "observation.state": state,
                "action": action,
            }

            for key in CAMERA_TOPICS:
                frame[f"observation.images.{key}"] = get_camera_image_at_time(
                    data["camera_times"],
                    data["camera_images"],
                    key,
                    t_ns,
                    half_interval_ns,
                )

            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  Saved episode {episode_idx} with {len(master_times)} frames")

    dataset.finalize()
    print(f"\nDataset saved to {out_dir}")


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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
