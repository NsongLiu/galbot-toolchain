#!/usr/bin/env bash
set -eu

# Train PI0.5 (full fine-tuning, chunk50) jointly on task8 + task9 right-arm+right-gripper
# LeRobot V3 datasets, initialized from the task8-trained PI0.5 (chunk50) weights.
# Usage:
#   ./train_task8_task9_right_arm_gripper_pi05_chunk50.sh                       # 1 GPU
#   ./train_task8_task9_right_arm_gripper_pi05_chunk50.sh 4                     # 4 GPUs
#   ./train_task8_task9_right_arm_gripper_pi05_chunk50.sh 4 path/to/config.json # custom config

export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"

# 训练全程离线：tokenizer / 权重均从本地磁盘加载，避免访问 HuggingFace 超时。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_CONFIG="$REPO_ROOT/galbot/train/configs/pi05_task8_task9_right_arm_gripper_chunk50_scratch.json"

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
