# Galbot-G1 机器人工具链

基于 [LeRobot](https://github.com/huggingface/lerobot) 的 Galbot-G1 数据转换与模型训练工具链。

## 环境要求

使用项目指定的 `xhum-new` conda 环境：

```bash
conda activate xhum-new
export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"
```

## 1. 数据转换

将 Galbot-G1 机器人采集的 MCAP 数据转换为 LeRobot V3 格式数据集。

### 默认转换

源数据默认路径：`/media/jushen/Leslie-liu/galbot_dataset/tmp`  
目标数据集默认路径：`/media/jushen/Leslie-liu/galbot_dataset/lerobot_v3`

```bash
python convert_mcap_to_lerobot.py
```

### 指定源/目标路径

```bash
python convert_mcap_to_lerobot.py \
  --mcap-dir /media/jushen/Leslie-liu/galbot_dataset/tmp \
  --out-dir /media/jushen/Leslie-liu/galbot_dataset/lerobot_v3
```

### 转换规则

- `FIN`：源数据
- `SYNC`：对齐数据（有效数据，**仅转换此类文件**）
- `CANCELED`：失败数据
- `UNQUALIFED`：操作异常存下的数据

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mcap-dir` | MCAP 源数据目录 | `/media/jushen/Leslie-liu/galbot_dataset/tmp` |
| `--out-dir` | LeRobot V3 目标数据集目录 | `/media/jushen/Leslie-liu/galbot_dataset/lerobot_v3` |
| `--repo-id` | 数据集元数据中的 repo ID | `galbot/g1_recordings` |
| `--fps` | 目标帧率 | `30` |
| `--vcodec` | 视频编码格式，可选 `h264`/`hevc`/`libsvtav1` | `h264` |
| `--use-images` | 将相机数据存为 PNG 图片而非视频 | 未启用 |
| `--task` | 任务描述 | `teleoperate the left arm` |
| `--robot-type` | 机器人类型 | `unitree_g1` |
| `--robot-config` | 机器人关节配置 JSON 输出路径 | `./robot_config.json` |

查看全部参数：

```bash
python convert_mcap_to_lerobot.py --help
```

### 输出产物

转换完成后会得到：

- `lerobot_v3/`：LeRobot V3 格式数据集（`data/`、`meta/`、`videos/`）
- `robot_config.json`：机器人关节分组与顺序配置

## 2. 模型训练

通过外置的 `galbot/train` 模块调用 LeRobot 接口进行训练，**不修改 LeRobot 源码**。

### 快速开始

使用默认配置（ACT 策略，训练 LeRobot V3 数据集）：

```bash
./train.sh
```

或使用 runner 脚本：

```bash
./scripts/galbot-run galbot.train.train --config galbot/train/configs/act_example.json
```

或手动指定 Python 路径：

```bash
PYTHONPATH=. /media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin/python -m galbot.train.train \
  --config galbot/train/configs/act_example.json
```

### 配置文件

参考 `galbot/train/configs/act_example.json`：

```json
{
  "dataset": {
    "repo_id": "unitree_g1",
    "root": "/media/jushen/Leslie-liu/galbot_dataset/lerobot_v3",
    "episodes": null
  },
  "policy": {
    "type": "act",
    "path": null,
    "push_to_hub": false
  },
  "training": {
    "output_dir": "/media/jushen/Leslie-liu/galbot_dataset/outputs/act_run_001",
    "batch_size": 8,
    "steps": 20000,
    "num_workers": 4,
    "save_freq": 5000,
    "log_freq": 100,
    "seed": 1000,
    "resume": false,
    "resume_dir": null
  },
  "use_imagenet_stats": true,
  "video_backend": "pyav"
}
```

### 配置项说明

| 字段 | 说明 |
|------|------|
| `dataset.repo_id` | 数据集标识 |
| `dataset.root` | LeRobot V3 数据集根目录 |
| `dataset.episodes` | 指定使用的 episode 索引列表，`null` 表示全部 |
| `policy.type` | 策略类型，如 `act`、`diffusion`、`vqbet` 等 |
| `policy.path` | 预训练模型路径，`null` 表示从零开始 |
| `policy.push_to_hub` | 是否推送到 Hugging Face Hub |
| `training.output_dir` | 模型输出目录 |
| `training.steps` | 总训练步数 |
| `training.batch_size` | 单卡 batch size |
| `training.num_workers` | DataLoader worker 数 |
| `training.save_freq` | 每隔多少步保存 checkpoint |
| `training.log_freq` | 每隔多少步打印日志 |
| `training.seed` | 随机种子 |
| `training.resume` | 是否从 checkpoint 恢复 |
| `training.resume_dir` | 恢复时指定 checkpoint 目录 |
| `use_imagenet_stats` | 是否用 ImageNet 统计量归一化图像 |
| `video_backend` | 视频解码后端，通常 `pyav` |

### 输出产物

训练完成后会在 `output_dir` 下生成：

- `checkpoints/<step>/pretrained_model/`：策略权重、配置与 processor
- `checkpoints/<step>/training_state/`：优化器、scheduler 与训练步数
- `checkpoints/last/`：指向最新 checkpoint 的符号链接

## 3. 推理部署（server-client 框架）

推理部署采用 **解耦架构**：
- **`policy_server`** 运行在 `xhum-new`（Python 3.12 + LeRobot + torch）环境，加载训练好的策略并通过 ZeroMQ 提供推理服务。
- **机器人控制端** 运行在自己的环境（可能不同 conda / Python 版本），通过 `PolicyClient` 发送观测并接收动作，不依赖 torch / LeRobot。
- **`replay`** 模式从 LeRobot V3 数据集读取轨迹观测，发送给 `policy_server` 验证 wire + model 链路，无需真实机器人。

### 文件结构

```
galbot/deploy/
├── policy_agent.py        # LeRobot 策略封装（xhum-new 环境）
├── policy_server.py       # ZMQ REP 服务端（xhum-new 环境）
├── policy_client.py       # ZMQ REQ 客户端（机器人控制环境，仅需 numpy + pyzmq）
├── replay.py              # LeRobot V3 数据集 -> ZMQ 回放（xhum-new 环境）
├── client_example.py      # 最小客户端示例（连接 GalbotSDK + PolicyClient）
├── robot_interface.py     # GalbotSDK 控制接口封装（get_observation / apply_action）
├── config_loader.py       # YAML 配置加载
├── wire/
│   ├── obs_codec.py       # ZMQ multipart 编解码协议
│   └── trace_io.py        # 存图 / 关节向量落盘
└── configs/
    └── deploy_example.yaml
```

### 快速开始

> **依赖提示**：部署模块使用 ZeroMQ 通信。请在 `xhum-new` 环境安装 `galbot/deploy/requirements_policy.txt`；在机器人控制环境安装 `galbot/deploy/requirements_robot.txt`。

#### 1) 启动策略服务

```bash
./scripts/deploy_server.sh \
  --model_path /media/jushen/Leslie-liu/galbot_dataset/outputs/act_run_001/checkpoints/last/pretrained_model \
  --bind tcp://127.0.0.1:5555
```

#### 2) 轨迹回放验证

另起终端：

```bash
./scripts/deploy_replay.sh --config galbot/deploy/configs/deploy_example.yaml
```

#### 3) 机器人控制端接入

参考 `galbot/deploy/client_example.py` 或 `galbot/deploy/policy_client.py`：

```python
from policy_client import PolicyClient

client = PolicyClient(server_url="tcp://127.0.0.1:5555")
client.reset()
action = client.inference({
    "images": {"left_arm": rgb_uint8_hwc},
    "arm_gripper_joints": state_vec,
})
```

### 部署配置

参考 `galbot/deploy/configs/deploy_example.yaml`：

```yaml
mode: replay_debug
policy_server_url: tcp://127.0.0.1:5555
policy_zmq_timeout_ms: 120000

# replay 模式必填
dataset_root: /media/jushen/Leslie-liu/galbot_dataset/lerobot_v3
dataset_repo_id: unitree_g1
episode_index: 0
action_rate: 30.0

# TO DEBUG: 若模型只期望单个相机，设置 obs_camera_key 与 checkpoint 的短名一致
# obs_camera_key: left_arm
```

### 配置项说明

| 字段 | 说明 |
|------|------|
| `mode` | `model` / `replay` / `replay_debug` |
| `policy_server_url` | ZMQ 服务端地址，必须与 `--bind` 一致 |
| `policy_zmq_timeout_ms` | ZMQ 收发超时（毫秒），0 表示不超时 |
| `dataset_root` | replay 用的 LeRobot V3 数据集根目录 |
| `dataset_repo_id` | 数据集标识 |
| `episode_index` | replay 使用的 episode 索引 |
| `action_rate` | replay 步进频率（Hz） |
| `replay_max_steps` | replay 最大步数，0 表示整集 |
| `obs_camera_key` | 指定发送到模型的相机短名；不设置则发送所有相机 |
| `image_save` | 客户端存图配置 |
| `joints` | 关节向量落盘配置 |

### 调试标记（TO DEBUG）

部署代码中所有需要结合实际机器人 SDK 或真实数据校验的位置均标注了 `# TO DEBUG:`，主要包括：
- 相机短名与模型 `input_features` 的映射（`policy_agent.py`、`replay.py`）。
- 图像通道顺序、像素范围、resize 后形状。
- `client_example.py` 中的 dummy observation 需替换为真实机器人 SDK 数据。
- `dataset_root` / `episode_index` / `obs_camera_key` 需按实际数据集调整。

### 安全提示

- ZMQ TCP 默认无认证，单机部署请使用 `127.0.0.1`。
- 跨机器请使用防火墙或 SSH 隧道。
