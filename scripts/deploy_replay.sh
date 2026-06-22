#!/usr/bin/env bash
set -euo pipefail

# Replay a LeRobot V3 episode through the policy server.
# Usage:
#   ./scripts/deploy_replay.sh --config galbot/deploy/configs/deploy_example.yaml

export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-$REPO_ROOT/galbot/deploy/configs/deploy_example.yaml}"
shift || true

exec python "$REPO_ROOT/galbot/deploy/replay.py" --config "$CONFIG" "$@"
