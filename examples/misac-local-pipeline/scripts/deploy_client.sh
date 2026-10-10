#!/usr/bin/env bash
set -euo pipefail

# Example: headless PolicyClient loop with GalbotSDK (robot control side).
# This file is a template — the robot control environment may differ from
# the policy environment. Ensure galbot_sdk, pyzmq, numpy, and PyYAML are installed there.
#
# Usage:
#   ./scripts/deploy_client.sh --config galbot/deploy/configs/deploy_example.yaml

# Activate the robot control environment before running this template.
# Activate the compatible training or robot environment before running this script.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python "$REPO_ROOT/galbot/deploy/client_example.py" "$@"
