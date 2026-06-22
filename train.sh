#!/usr/bin/env bash
set -eu

# Train a policy on the converted Galbot G1 LeRobot V3 dataset.
# Usage:
#   ./train.sh                    # use default config
#   ./train.sh path/to/config.json # use a custom config

export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-$REPO_ROOT/galbot/train/configs/act_example.json}"

exec python -m galbot.train.train --config "$CONFIG"
