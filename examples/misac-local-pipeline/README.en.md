# Galbot Multimodal Reproduction

[中文](README.md) | [English](README.en.md)

## Research Background

This is a local adaptation of the **multimodal data-conversion, ACT training and policy-inference portions** of [NsongLiu/galbot-toolchain, dev-les](https://github.com/NsongLiu/galbot-toolchain/tree/dev-les), for learning and reproduction exercises.

Multimodal inputs here mean camera RGB images and robot joint/state observations. LeRobot ACTPolicy predicts action chunks. The example selects the right-arm camera and actions; other cameras depend on the supplied data and policy configuration. This is not a new multimodal model, a complete reproduction of the upstream toolchain, or a validated closed-loop robot system.

## Adaptation and Original Works

| Component | Source and local scope |
|---|---|
| MCAP → LeRobot V3 | Upstream conversion adapted for local image/state/action processing and alignment |
| `galbot/train/` | Upstream LeRobot training wrapper and compatibility handling; LeRobot supplies the policy |
| `galbot/deploy/` | Upstream policy server/client and observation encoding; local inference does not establish robot deployment |
| `run_all.py` and `scripts/` | Local workflow, split/config generation, offline evaluation, inference checks and reports |
| `visual_report_optimized/` | Local report plotting without bundled real experiment evidence |

- Original toolchain: [NsongLiu/galbot-toolchain](https://github.com/NsongLiu/galbot-toolchain), branch `dev-les`. The upstream head inspected was `b1ed20d591a05d433250fe02a9d527527d009d94`; this is a review reference, not a claim that all local files match that revision.
- ACT: Tony Z. Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (2023); [original implementation](https://github.com/tonyzhaozh/act).
- Framework: [Hugging Face LeRobot](https://github.com/huggingface/lerobot). Check policy and dataset interfaces against the version in use.

Cite the original works when using their methods. Adaptation, citations and a learning-use statement do not grant rights for third-party data, weights or robot SDKs. No additional MIT license is asserted for this repository; use and redistribution must respect the applicable permissions for the original toolchain, LeRobot, ACT, datasets, weights and robot SDKs.

## Validation and Results

The code provides offline action-chunk L1/RMSE evaluation and local ZMQ round-trip checks. Offline action error does not measure robot task success, which requires authorized closed-loop robot or simulation evaluation.

Five synthetic tests cover episode splits, configuration overrides, camera/action selection and deletion boundaries. This review did not retrain ACT, evaluate real recordings, connect a robot, resolve a new full environment or complete a comprehensive dependency CVE audit.

A mock ZMQ round-trip check also passed without LeRobot, a checkpoint or a robot; it validates observation encoding, reset and action-reply formats only. `galbot/deploy/robot_interface.py` is an SDK interface template whose connection arguments, joint mappings and control methods must be checked against the actual GalbotSDK. It is not a verified robot-control program.

The current tree contains core code, neutral examples and documentation. Recordings, checkpoints, real splits, metrics/plots, personal paths and internal document links are excluded. Users must supply resources they are authorized to use; local datasets are not presented as public downloads.

## Code Structure

- [convert_mcap_to_lerobot.py](convert_mcap_to_lerobot.py): multimodal recording conversion.
- [galbot/train/](galbot/train/) and [galbot/deploy/](galbot/deploy/): training and inference wrappers.
- [pipeline_config.json](pipeline_config.json): relative example paths, not bundled assets.
- [scripts/prepare_misac_finetune.py](scripts/prepare_misac_finetune.py): splits/configs generated from a supplied conversion manifest.
- [scripts/evaluate_policy_offline.py](scripts/evaluate_policy_offline.py): offline action prediction evaluation.
- [tests/](tests/): synthetic contracts without real recordings or a robot.

Historical script names remain for caller compatibility and do not imply downloadable assets. Generated split/config and experiment outputs are ignored by Git.

## Usage

Run from the repository root. Configure compatible LeRobot/PyTorch and conversion/deployment dependencies using upstream guidance, [policy-side requirements](galbot/deploy/requirements_policy.txt) and [robot-side requirements](galbot/deploy/requirements_robot.txt). These sides may require different environments. Upstream currently describes Python 3.12 and LeRobot v0.5.1; the full stack is not claimed to work on Python 3.9.

Syntax, CLI and synthetic checks used Python 3.9 with NumPy, OpenCV and PyYAML:

```powershell
py -3.9 -m unittest discover -s tests -v
py -3.9 run_all.py --help
```

For the complete pipeline, edit `pipeline_config.json`, supply authorized recordings, a base checkpoint and a matching robot configuration, then run `run_all.py` in a compatible project environment. `--quick` reduces evaluation batches but still requires assets; a fresh clone alone cannot run real-data evaluation. The workflow generates its own split and training configuration.

Options include `--force-convert`, `--force-train`, `--skip-train` and `--skip-deploy-verify`. Forced conversion/training clears the relevant configured directory; data/model/output paths must remain inside the project, and deletion of the root or external paths is rejected.

The combined workflow retains the 1000-step example: it looks for `000250`, `000500`, `000750` and `001000` checkpoints and uses `001000` for final checks. The standalone configuration generator accepts other step counts, but changing only `pipeline_config.json` does not update the workflow's final checkpoint. Changes to steps, save intervals, evaluation batches or the second seed require checking checkpoint/report filename assumptions as well; those combinations are untested.

Image/joint tracing writes input observations. Default `debug_policy_images/`, `debug_joints/` and `outputs/` directories are ignored by Git; place custom tracing under `outputs/`. Generated traces are not distributable source code.

The policy server defaults to loopback and does not authenticate clients. Network or robot deployment requires an explicit trusted interface and your own access-control/safety process. No robot control was executed in this review.
