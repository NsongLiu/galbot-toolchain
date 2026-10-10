# Galbot MiSAC Local Pipeline

[中文](README.md) | [English](README.en.md)

## 研究背景

本仓库基于 [NsongLiu/galbot-toolchain 的 dev-les 分支](https://github.com/NsongLiu/galbot-toolchain/tree/dev-les)，对其中的**多模态数据转换、ACT 训练和策略推理链路**做本地适配，用于学习与复现练习。

这里的“多模态”指相机 RGB 图像与机器人关节/状态联合输入，由 LeRobot 的 ACTPolicy 预测动作 chunk。示例选择右臂相机和右臂动作组；其他相机取决于实际数据与模型配置。本仓库没有提出新的多模态模型，也不代表原工具链全部功能或真实机器人闭环任务的完整复现。

## 项目适配与原工作

| 内容 | 来源与本地范围 |
|---|---|
| MCAP → LeRobot V3 | 沿用并适配原工具链的数据转换，处理图像、状态、动作与时间对齐 |
| `galbot/train/` | 原工具链的 LeRobot 训练封装及兼容处理；策略实现来自 LeRobot |
| `galbot/deploy/` | 原工具链的策略服务、客户端与观测编码；本地推理验证不等于机器人部署验证 |
| `run_all.py` 与 `scripts/` | 本地流程编排、划分配置生成、离线评估、推理检查和报告代码 |
| `visual_report_optimized/` | 本地离线报告绘制；不随代码附带真实实验结果 |

- 原工具链：[NsongLiu/galbot-toolchain](https://github.com/NsongLiu/galbot-toolchain)，参考分支 `dev-les`。本次核对的上游分支头为 `b1ed20d591a05d433250fe02a9d527527d009d94`；这是核对版本，不代表本地所有文件都与该版本一致。
- ACT 论文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)，Tony Z. Zhao 等（2023）；[原作者代码](https://github.com/tonyzhaozh/act)。
- 框架：[Hugging Face LeRobot](https://github.com/huggingface/lerobot)。ACT 模型与数据集接口应按所用 LeRobot 版本核对。

使用相关方法时请引用原工作。适配、学习声明和引用不替代第三方源码、数据、权重与机器人 SDK 的许可。本仓库没有另行添加 MIT；使用或再分发时请同时遵守原工具链、LeRobot、ACT、数据集、权重和机器人 SDK 的相应许可与权限要求。

## 验证与结果范围

提供离线动作 chunk 的 L1/RMSE 评估和本地 ZMQ 往返验证入口。离线动作误差不能解释为真实机器人任务成功率；真实成功率需要经授权的机器人或仿真闭环评测。

当前版本的五项合成测试覆盖 episode 划分、配置覆盖、动作/相机选择及目录删除保护。没有重新训练 ACT、运行真实数据评测或连接机器人；也没有完成全部依赖组合及漏洞数据库审计。

另已通过不依赖 LeRobot/模型/机器人的 mock ZMQ 往返检查，只验证观测编码、reset 和动作回复的通信格式。`galbot/deploy/robot_interface.py` 是待核对 GalbotSDK 的接口模板，连接参数、关节映射和控制方法需按实际 SDK 确认，不作为已验证的机器人控制程序。

当前文件只保留核心代码、通用示例和介绍。数据录制、训练权重、真实划分、评测数值/图表、个人机器路径和内部文档入口不随当前项目文件分发。数据与权重由使用者在取得相应权限后自行提供，不将任何本地数据集当作公开数据集介绍。

## 代码结构

| 路径 | 内容 |
|---|---|
| [convert_mcap_to_lerobot.py](convert_mcap_to_lerobot.py) | 图像、状态与动作数据转换 |
| [galbot/train/](galbot/train/) | LeRobot 训练封装 |
| [galbot/deploy/](galbot/deploy/) | 策略推理、客户端和 replay 接口 |
| [pipeline_config.json](pipeline_config.json) | 项目相对路径示例，不是已有数据或权重 |
| [scripts/prepare_misac_finetune.py](scripts/prepare_misac_finetune.py) | 从使用者提供的转换 manifest 生成划分与训练配置 |
| [scripts/evaluate_policy_offline.py](scripts/evaluate_policy_offline.py) | 离线动作预测评估 |
| [tests/](tests/) | 无真实数据/机器人资源的合成测试 |

历史脚本名称为兼容已有调用而保留，不意味着相应数据或模型可以公开下载。运行生成的划分、配置与输出已加入忽略规则。

## 使用方法

从仓库根目录运行。先配置兼容的 LeRobot、PyTorch 及转换/部署依赖；参见原工具链安装说明和 [策略侧依赖](galbot/deploy/requirements_policy.txt) / [机器人侧依赖](galbot/deploy/requirements_robot.txt)。两侧环境可能不同，完整训练/推理环境没有在此次整理中重新安装或验证。上游当前说明采用 Python 3.12 与 LeRobot v0.5.1，完整依赖栈不保证兼容 Python 3.9。

本次语法、CLI 与合成测试使用 Python 3.9；测试需要 NumPy、OpenCV 和 PyYAML：

```powershell
py -3.9 -m unittest discover -s tests -v
py -3.9 run_all.py --help
```

完整流程先编辑 `pipeline_config.json`，准备自己有权使用的录制数据、基础模型和机器人配置，再在兼容的项目运行环境中执行 `run_all.py`。`--quick` 只减少评估批次数，仍然需要这些资源；新 clone 不能仅凭代码完成真实数据评测。主流程会生成划分与训练配置，无需把真实 episode 列表提交到 Git。

主要选项：`--force-convert` 重新转换、`--force-train` 重新训练、`--skip-train` 使用已有模型、`--skip-deploy-verify` 跳过 ZMQ 检查。重新转换/训练会清理对应配置目录；所有数据/模型/输出路径必须位于项目根目录内，根目录及项目外路径会被拒绝删除。

当前总入口沿用 1000 步示例：按 `000250`、`000500`、`000750`、`001000` 查找 checkpoint，最终验证固定使用 `001000`。单独的配置生成脚本支持其他步数，但只改 `pipeline_config.json` 的 `steps` 不能让总入口适配新的终点。修改步数、保存间隔、评估批数或第二随机种子时，需要同时核对总入口和报告脚本的 checkpoint/文件名约定；这些组合没有验证。

启用图片或关节调试记录会保存输入观测。默认 `debug_policy_images/`、`debug_joints/` 和 `outputs/` 已被 Git 忽略；自定义调试目录建议放在 `outputs/` 下。生成记录不属于可分发源码。

策略服务默认绑定本机回环地址，客户端没有身份认证。网络或机器人部署需要明确配置可信接口、访问控制和自己的安全流程；本次没有执行机器人控制。
