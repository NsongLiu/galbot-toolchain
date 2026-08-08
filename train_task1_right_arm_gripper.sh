#!/usr/bin/env bash
set -eu

# Train ACT on the task1 right-arm+right-gripper LeRobot V3 dataset.
# Usage:
#   ./train_task1_right_arm_gripper.sh                       # 1 GPU, default config
#   ./train_task1_right_arm_gripper.sh 4                     # 4 GPUs, default config
#   ./train_task1_right_arm_gripper.sh 4 path/to/config.json # 4 GPUs, custom config

export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_CONFIG="$REPO_ROOT/galbot/train/configs/act_task1_right_arm_gripper.json"

# Parse positional args: [num_gpus] [config]
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="$1"
    CONFIG="${2:-$DEFAULT_CONFIG}"
else
    NUM_GPUS=1
    CONFIG="${1:-$DEFAULT_CONFIG}"
fi

if [ "$NUM_GPUS" -le 1 ]; then
    exec python -m galbot.train.train --config "$CONFIG"
else
    exec accelerate launch --multi_gpu --num_processes "$NUM_GPUS" -m galbot.train.train --config "$CONFIG"
fi
