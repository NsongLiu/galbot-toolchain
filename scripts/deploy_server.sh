#!/usr/bin/env bash
set -euo pipefail

# Start the Galbot policy server (Python 3.12 + LeRobot env).
# Usage:
#   ./scripts/deploy_server.sh --model_path /path/to/checkpoints/last/pretrained_model
#
# Extra args are forwarded to policy_server.py.

export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python "$REPO_ROOT/galbot/deploy/policy_server.py" "$@"
