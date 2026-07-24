"""Smoke test: load PI0.5 base, wrap with LoRA, verify trainable params & forward."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from galbot.train._compat import apply_lerobot_compat_patch
apply_lerobot_compat_patch()

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy

with open(REPO_ROOT / "galbot/train/configs/pi05_task1_right_arm_gripper_lora.json") as f:
    cfg = json.load(f)

policy_cfg = cfg["policy"]
peft_cfg = cfg["peft"]

print("=" * 60)
print("[1/4] Loading PI0.5 config from:", policy_cfg["path"])
pol_cfg = PreTrainedConfig.from_pretrained(policy_cfg["path"])
pol_cfg.pretrained_path = Path(policy_cfg["path"])
pol_cfg.push_to_hub = False

# Apply overrides (same as train.py)
from galbot.train.train import _apply_policy_overrides
_apply_policy_overrides(pol_cfg, policy_cfg)

print("[2/4] Loading dataset metadata:", cfg["dataset"]["root"])
ds_meta = LeRobotDatasetMetadata(cfg["dataset"]["repo_id"], root=cfg["dataset"]["root"])

print("[3/4] Building policy (this may take ~1 min)...")
policy = make_policy(cfg=pol_cfg, ds_meta=ds_meta, rename_map=policy_cfg.get("rename_map"))

n_total_before = sum(p.numel() for p in policy.parameters())
n_train_before = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"  Before PEFT: trainable={n_train_before:,} / total={n_total_before:,}")

print("[4/4] Wrapping with PEFT (LoRA)...")
policy = policy.wrap_with_peft(peft_cli_overrides=dict(peft_cfg))

n_total_after = sum(p.numel() for p in policy.parameters())
n_train_after = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"  After PEFT:  trainable={n_train_after:,} / total={n_total_after:,}")
print(f"  Trainable ratio: {100*n_train_after/n_total_after:.2f}%")
print(f"  use_peft flag: {policy.config.use_peft}")

assert n_train_after < n_train_before, "LoRA should reduce trainable params"
assert policy.config.use_peft is True

print("=" * 60)
print("SMOKE TEST PASSED")
