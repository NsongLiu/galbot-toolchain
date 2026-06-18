# Galbot-G1 MCAP to LeRobot V3 数据转换

将 Galbot-G1 机器人采集的 MCAP 数据转换为 LeRobot V3 格式数据集。

## 环境要求

使用项目指定的 `xlerobot` conda 环境：

```bash
conda activate xlerobot
export PATH="/root/miniconda3/envs/xlerobot/bin:$PATH"
```

## 快速开始

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

## 转换规则

- `FIN`：源数据
- `SYNC`：对齐数据（有效数据，**仅转换此类文件**）
- `CANCELED`：失败数据
- `UNQUALIFED`：操作异常存下的数据

## 命令行参数

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

## 输出产物

转换完成后会得到：

- `lerobot_v3/`：LeRobot V3 格式数据集（`data/`、`meta/`、`videos/`）
- `robot_config.json`：机器人关节分组与顺序配置
