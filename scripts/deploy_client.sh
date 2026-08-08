#!/usr/bin/env bash
set -euo pipefail

# G1 真机控制循环（机器人控制端，通常为 Orin）。
# 控制环境可能与 xhum-new 不同：只需 pyzmq / numpy / opencv / PyYAML，
# 以及 galbot_sdk（真机上位于 /userdata/update/manual_update/lib，
# 由 YAML 的 robot.sdk_lib_path 自动注入 sys.path）。
#
# 用法：
#   dry-run（不下发动作）:
#     ./scripts/deploy_client.sh --config galbot/deploy/configs/my_deploy.yaml --steps 50
#   实机执行:
#     ./scripts/deploy_client.sh --config galbot/deploy/configs/my_deploy.yaml --steps 300 \
#       --execute --acknowledge DEPLOY_MODEL

# TO DEBUG: activate or point PATH to the robot control conda env (not xhum-new).
export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python "$REPO_ROOT/galbot/deploy/client_example.py" "$@"
