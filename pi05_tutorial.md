# PI0.5 训练教程（Galbot-G1 / task1_right_arm_gripper）

本文档说明如何基于 LeRobot v0.5.1 在 Galbot-G1 右臂+夹爪数据集上训练 **PI0.5** 策略，包含两种模式：

- **全参数微调（Full Fine-tuning）**：`pi05_task1_right_arm_gripper.json`
- **LoRA 参数高效微调（PEFT/LoRA）**：`pi05_task1_right_arm_gripper_lora.json`

## 路径约定

下文使用以下占位符，请按实际环境替换：

| 占位符 | 含义 |
|--------|------|
| `<CONDA_ENV>` | 安装了 torch 与 LeRobot v0.5.1 的 conda 环境 |
| `<LEROBOT_SRC>` | LeRobot v0.5.1 源码目录（仅作参考，不修改） |
| `<PRETRAINED_DIR>` | 预训练权重目录（含 `pi05_base`、`paligemma-3b-pt-224`） |
| `<WORKSPACE>` | 数据集工作目录（存放转换后的数据集与训练输出） |

---

## 1. 环境与依赖

### 1.1 Conda 环境

```bash
conda activate <CONDA_ENV>
```

要求：Python 3.10+、torch（CUDA）、LeRobot v0.5.1（见 `README.md`「前置要求」）。

### 1.2 LeRobot 依赖

**无需修改 LeRobot 源码**。本工具链直接调用 `<CONDA_ENV>` 中已安装的 LeRobot（v0.5.1，源码可参考 `<LEROBOT_SRC>`），其中：

- `lerobot.policies.pi05.modeling_pi05.PI05Policy._get_default_peft_targets()` 已内置 LoRA 默认目标模块
- `lerobot.policies.pretrained.PreTrainedPolicy.wrap_with_peft()` 提供统一的 PEFT 入口
- 注意要先安装pi环境的依赖，可参考`https://huggingface.co/docs/lerobot/pi05`

### 1.3 兼容性补丁（`galbot/train/_compat.py`）

`galbot/train/train.py` 在导入任何 LeRobot 策略模块之前调用 `apply_lerobot_compat_patch()`，以运行时补丁方式处理已知兼容问题（**不修改 LeRobot 源码、不修改权重文件**）：

| 补丁 | 作用 | 触发条件 |
|------|------|---------|
| dataclass 兼容补丁 | 绕过 Python 3.12 对 `GR00TN15Config` dataclass 字段顺序（`init=False` 字段无默认值）的报错 | 导入 lerobot policies 时 |
| PI0.5 视觉键重映射补丁 `_patch_pi05_vision_tower_key_remap()` | 包装 `PI05Policy._fix_pytorch_state_dict_keys`，修复 checkpoint 视觉权重键名不匹配（见 §2.1） | 仅当检测到不匹配时介入，否则原样透传 |

补丁设计原则：**先检测、再修复**——确认 checkpoint 键含 `vision_tower.vision_model.` 中缀且模型期望 `vision_tower.*` 时才做重映射并打印日志；任何不满足条件的情况都不改变原有加载行为。

### 1.4 离线运行

训练全程可以不访问 HuggingFace：

- 启动脚本 `train_task1_right_arm_gripper_pi05.sh` 默认导出 `HF_HUB_OFFLINE=1`（可用环境变量覆盖）。
- PaliGemma tokenizer 从本地目录加载（见 §2），不走 `google/paligemma-3b-pt-224` 在线下载。注意：该本地路径目前在 `galbot/train/train.py` 的 PI0.5 分支中以 `local_tokenizer_path` 指定，迁移到新环境时请改为本机实际路径（`<PRETRAINED_DIR>/paligemma-3b-pt-224`）。
- PI0.5 的 pre/post processor 在 `galbot/train/train.py` 中**本地逐步组装**，而非调用 LeRobot 的 `make_pi05_pre_post_processors()`——后者内部会联网加载 tokenizer，离线环境下直接报 `OSError: Can't load the configuration of 'google/paligemma-3b-pt-224'`。postprocessor 由 `UnnormalizerProcessorStep + DeviceProcessorStep` 两步构成，与官方实现语义一致。

### 1.5 PEFT 依赖（仅 LoRA 模式需要）

若环境中未安装 `peft` 包，启用 LoRA 训练前必须安装：

```bash
# 如遇网络问题可先配置代理（可选）
# export http_proxy=http://<proxy-host>:<port>
# export https_proxy=http://<proxy-host>:<port>

pip install "peft>=0.18.0,<1.0.0"
```

> LeRobot `pyproject.toml` 中声明的可选依赖为 `peft-dep = ["peft>=0.18.0,<1.0.0"]`，版本需保持在该区间。

全参数微调无需安装 `peft`。

---

## 2. 预训练权重

两种模式共用同一份 PI0.5 基座权重：

| 路径 | 用途 | 来源 |
|------|------|------|
| `<PRETRAINED_DIR>/pi05_base` | PI0.5 主干权重（VLM + action expert） | HuggingFace `lerobot/pi05_base` |
| `<PRETRAINED_DIR>/paligemma-3b-pt-224` | PaliGemma tokenizer（本地加载，避免访问 HuggingFace） | HuggingFace `google/paligemma-3b-pt-224`（仅需 tokenizer 相关文件） |

`pi05_base/` 目录应包含：

```
pi05_base/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

**LoRA 模式对权重的额外要求**：`policy.path` 必须指向已训练好的 PI0.5 checkpoint（`pi05_base`）。LeRobot 在 `use_peft=True` 且 `pretrained_path` 为空时会直接报错——LoRA 不支持从零开始训练。

### 2.1 视觉编码器键名不匹配问题（已由补丁处理）

部分 `pi05_base/model.safetensors` 中 SigLIP 视觉塔（437 个权重）的键名带 `vision_model.` 中缀：

```
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder...   # checkpoint 实际键名
paligemma_with_expert.paligemma.model.vision_tower.encoder...                # 当前模型期望键名
```

不加处理时，加载日志会报 `Missing key(s) in state_dict: ...vision_tower...`（437 个），**视觉编码器保持随机初始化**，VLA 性能严重受损（且除视觉塔外的其余权重仍会正常加载，问题不易察觉）。

解决方式：`galbot/train/_compat.py` 中的运行时补丁（见 §1.3）在加载时检测并重映射这 437 个键，**不修改 checkpoint 文件**。训练启动时应在日志中看到：

```
PI05 vision tower key mismatch detected; remapped 437 keys ('vision_tower.vision_model.' -> 'vision_tower.')
All keys loaded successfully!
```

若只看到 `All keys loaded successfully!` 而没有 remap 日志，说明 checkpoint 键名本身已与模型一致（如官方原版权重），补丁按设计未介入。

---

## 3. 数据集

默认使用 MCAP 转换得到的 LeRobot V3 数据集（repo_id `task1_right_arm_gripper`，48 episodes，仅右臂+右夹爪动作，头部右相机+右腕相机）：

```
<WORKSPACE>/lerobot_v3/task1_right_arm_gripper
```

如需重新生成，参考 `README.md` 第 1 节数据转换流程。

---

## 4. 训练配置

### 4.1 全参数微调（基线）

配置文件：`galbot/train/configs/pi05_task1_right_arm_gripper.json`

关键字段：

```json
{
  "dataset": {
    "repo_id": "task1_right_arm_gripper",
    "root": "<WORKSPACE>/lerobot_v3/task1_right_arm_gripper"
  },
  "policy": {
    "type": "pi05",
    "path": "<PRETRAINED_DIR>/pi05_base",
    "optimizer_lr": 2.5e-5,
    "freeze_vision_encoder": false,
    "train_expert_only": false,
    "gradient_checkpointing": true,
    "dtype": "bfloat16",
    "chunk_size": 50,
    "n_action_steps": 50
  },
  "training": {
    "output_dir": "<WORKSPACE>/outputs/pi05_task1_right_arm_gripper"
  }
}
```

启动训练：

```bash
./train_task1_right_arm_gripper_pi05.sh
# 或多卡
./train_task1_right_arm_gripper_pi05.sh 4
```

### 4.2 LoRA 微调

配置文件：`galbot/train/configs/pi05_task1_right_arm_gripper_lora.json`

在全参数配置的基础上**新增顶层 `peft` 段**：

```json
{
  "policy": { "..." : "..." },
  "peft": {
    "method_type": "LORA",
    "r": 16,
    "target_modules": null,
    "full_training_modules": null,
    "init_type": null
  }
}
```

`galbot/train/train.py` 在 `make_policy()` 之后检测到 `peft` 段即调用 `policy.wrap_with_peft(peft_cli_overrides=...)`，自动冻结主干并注入 LoRA adapter。

#### `peft` 字段说明

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `method_type` | PEFT 方法，当前仅测试过 `LORA`；PeftType 中的其他方法理论可用 | `LORA` |
| `r` | LoRA 秩，越大可训练参数越多、越接近全量微调 | `16` |
| `target_modules` | 目标模块正则；`null` 时使用 PI0.5 内置默认值 | `null` |
| `full_training_modules` | 不参与 LoRA、直接全量训练的模块名（映射到 `modules_to_save`） | `null` |
| `init_type` | LoRA 初始化方式（映射到 `init_lora_weights`） | `null` |

#### PI0.5 默认 LoRA 目标

`target_modules=null` 时，`PI05Policy._get_default_peft_targets()` 注入：

```
(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj
 | model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))
```

即仅对 **action expert（Gemma expert）的 self-attention Q/V 投影** 以及 **state/action 投影层** 加 LoRA，VLM 主干保持冻结。

#### LoRA 训练超参调整

相对全量微调，LoRA 通常使用更大的学习率与更短的 warmup：

| 字段 | 全参数 | LoRA |
|------|--------|------|
| `optimizer_lr` | `2.5e-5` | `1e-4` |
| `scheduler_warmup_steps` | `1000` | `500` |
| `scheduler_decay_steps` | `30000` | `20000` |
| `scheduler_decay_lr` | `2.5e-6` | `1e-5` |

#### 启动 LoRA 训练

```bash
./train_task1_right_arm_gripper_pi05.sh 1 \
  galbot/train/configs/pi05_task1_right_arm_gripper_lora.json

# 多卡
./train_task1_right_arm_gripper_pi05.sh 4 \
  galbot/train/configs/pi05_task1_right_arm_gripper_lora.json
```

---

## 5. Checkpoint 结构

两种模式训练产物目录一致：

```
output_dir/
└── checkpoints/
    ├── <step>/
    │   ├── pretrained_model/         # 策略权重 + 配置 + processor
    │   │   ├── config.json
    │   │   ├── model.safetensors     # 全量模式：完整权重；LoRA 模式：adapter 权重
    │   │   ├── adapter_config.json   # 仅 LoRA 模式
    │   │   ├── adapter_model.safetensors  # 仅 LoRA 模式
    │   │   └── train_config.json
    │   └── training_state/           # optimizer/scheduler/step
    └── last -> <step>
```

**LoRA 恢复推理**：`policy.path` 指向 LoRA checkpoint 目录即可。`config.json` 中 `use_peft=true`，`make_policy` 会读取 `adapter_config.json` 中的 `base_model_name_or_path` 自动加载基座 + adapter。

---

## 6. 常见问题

### Q1: 报错 `ModuleNotFoundError: No module named 'peft'`

未安装 `peft`。参见 §1.5 安装。

### Q2: 报错 `Instantiating a policy with use_peft=True without a checkpoint is not supported`

`policy.path` 为空或未指向有效 PI0.5 checkpoint。LoRA 必须基于已有权重微调。

### Q3: 报错 `OSError: Can't load the configuration of 'google/paligemma-3b-pt-224'`

有代码路径尝试在线加载 PaliGemma tokenizer。确认：① 启动脚本已导出 `HF_HUB_OFFLINE=1`；② 使用的是最新 `galbot/train/train.py`——PI0.5 的 pre/post processor 已在本地组装（见 §1.4），不会再调用联网的 `make_pi05_pre_post_processors()`；③ `train.py` 中的 `local_tokenizer_path` 指向本机实际 tokenizer 目录。

### Q4: 如何确认视觉编码器权重加载成功？

看训练启动日志（见 §2.1）：出现 `remapped 437 keys` + `All keys loaded successfully!` 即正常。若出现 437 个 `vision_tower` 相关的 `Missing key(s)`，说明兼容补丁未生效——确认 `galbot/train/train.py` 顶部调用了 `apply_lerobot_compat_patch()`，且不要在调用它之前导入任何 lerobot policies 模块。

### Q5: LoRA 训出来的模型能直接部署吗？

可以。`policy_server` 加载 LoRA checkpoint 时会自动识别 `use_peft` 标记并组装「基座 + adapter」，部署流程与全量微调一致（参考 `README.md` 第 3 节）。

### Q6: 想把 LoRA 合并回基座权重？

```python
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

cfg = PreTrainedConfig.from_pretrained("<lora_ckpt_dir>")
ds_meta = LeRobotDatasetMetadata(
    "task1_right_arm_gripper",
    root="<WORKSPACE>/lerobot_v3/task1_right_arm_gripper",
)
policy = make_policy(cfg=cfg, ds_meta=ds_meta)  # 返回 PeftModel
merged = policy.merge_and_unload()               # 合并 LoRA 到基座
merged.save_pretrained("<merged_out_dir>")
```

---

## 7. 相关文件

| 路径 | 说明 |
|------|------|
| `galbot/train/train.py` | 训练入口；PI0.5 pre/post processor 本地组装、`peft` 段处理逻辑 |
| `galbot/train/_compat.py` | 运行时兼容补丁：Python 3.12 dataclass、PI0.5 视觉键重映射（§1.3、§2.1） |
| `galbot/train/configs/pi05_task1_right_arm_gripper.json` | 全参数微调配置 |
| `galbot/train/configs/pi05_task1_right_arm_gripper_lora.json` | LoRA 微调配置 |
| `train_task1_right_arm_gripper_pi05.sh` | 训练启动脚本（单/多卡，默认 `HF_HUB_OFFLINE=1`） |
