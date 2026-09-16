"""Galbot 自定义 LeRobot processor step（train 与 deploy 共享）。

步骤通过 ``ProcessorStepRegistry`` 注册，随 checkpoint 的
``policy_preprocessor.json`` 一起序列化/反序列化。部署端（policy_agent）
只需 import 本模块完成注册，即可透明加载含自定义步骤的管线。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="slice_state_processor")
class SliceStateProcessorStep(ObservationProcessorStep):
    """按关节索引切片 ``observation.state``（最后一维）。

    用于只输入部分关节状态（如仅右臂 7 维）而非全身 29 维。须放在
    Normalizer 之前，并将归一化统计量按相同索引切片（见 train.py）。
    """

    indices: list[int] = field(default_factory=list)
    state_key: str = "observation.state"

    def observation(self, observation):
        if self.indices and self.state_key in observation:
            state = observation[self.state_key]
            # 幂等：输入已是切片后维度（如部署端直接发 8 维右臂状态）则原样透传。
            if state.shape[-1] == len(self.indices):
                return observation
            if state.shape[-1] <= max(self.indices):
                raise ValueError(
                    f"state 维度 {state.shape[-1]} 与切片索引 {self.indices} 不兼容"
                )
            observation[self.state_key] = state[..., self.indices]
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"indices": list(self.indices), "state_key": self.state_key}

    def transform_features(self, features):
        obs = features.get(PipelineFeatureType.OBSERVATION) or {}
        feat = obs.get(self.state_key)
        if feat is None or not self.indices:
            return features
        new_features = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {
            **obs,
            self.state_key: replace(feat, shape=(len(self.indices),)),
        }
        return new_features


@dataclass
@ProcessorStepRegistry.register(name="crop_image_processor")
class CropImageProcessorStep(ObservationProcessorStep):
    """对指定相机图像做固定区域裁剪，让策略把更多像素分配给操作区域。

    任务操作区域只占整幅画面一部分时（如货架中间层），先裁剪再送入策略
    （pi05 内部再 resize_with_pad 到 224x224），等效提升该区域的分辨率。

    - ``crops``: 图像键 -> [top, left, height, width]。键名为 rename 之后的
      policy 侧键名，本步骤须放在 RenameObservationsProcessorStep 之后。
    - ``source_shapes``: 图像键 -> [H, W]，裁剪框定义时对应的图像分辨率
      （一般为数据集原始分辨率，由 train.py 自动填充）。运行时输入分辨率
      不一致（如部署端先按 input_features 缩放过的帧）则按比例缩放裁剪框，
      保证训练/部署裁剪内容一致；缺省认为与输入分辨率一致。

    幂等：输入已是裁剪后尺寸则原样透传。
    """

    crops: dict[str, list[int]] = field(default_factory=dict)
    source_shapes: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, box in self.crops.items():
            if len(box) != 4:
                raise ValueError(f"crop box for {key!r} must be [top, left, height, width], got {box}")
            top, left, height, width = (int(v) for v in box)
            if top < 0 or left < 0 or height <= 0 or width <= 0:
                raise ValueError(f"illegal crop box for {key!r}: {box}")
            self.crops[key] = [top, left, height, width]

    @staticmethod
    def _layout(img) -> tuple[int, int, bool]:
        """返回 (H, W, is_hwc)。管线内图像为 CHW 张量；numpy HWC 仅作兜底。"""
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] in (1, 3):
            return int(img.shape[0]), int(img.shape[1]), True
        return int(img.shape[-2]), int(img.shape[-1]), False

    def _scaled_box(self, key: str, h_img: int, w_img: int) -> tuple[int, int, int, int]:
        top, left, height, width = self.crops[key]
        src = self.source_shapes.get(key)
        if src and (int(src[0]), int(src[1])) != (h_img, w_img):
            sy, sx = h_img / int(src[0]), w_img / int(src[1])
            top, height = round(top * sy), round(height * sy)
            left, width = round(left * sx), round(width * sx)
            # 缩放取整最多差 1px，收敛到界内；尺寸本身非法说明配置有误，抛错。
            height = min(height, h_img - top)
            width = min(width, w_img - left)
        if top + height > h_img or left + width > w_img or height <= 0 or width <= 0:
            raise ValueError(
                f"crop box {self.crops[key]} for {key!r} does not fit image {(h_img, w_img)}"
            )
        return top, left, height, width

    def observation(self, observation):
        for key, box in self.crops.items():
            img = observation.get(key)
            if img is None:
                continue
            h_img, w_img, is_hwc = self._layout(img)
            if (h_img, w_img) == (box[2], box[3]):
                continue  # 已是裁剪后尺寸，幂等透传
            top, left, height, width = self._scaled_box(key, h_img, w_img)
            if is_hwc:
                observation[key] = img[top : top + height, left : left + width, :]
            else:
                observation[key] = img[..., top : top + height, left : left + width]
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "crops": {k: list(v) for k, v in self.crops.items()},
            "source_shapes": {k: list(v) for k, v in self.source_shapes.items()},
        }

    def transform_features(self, features):
        obs = features.get(PipelineFeatureType.OBSERVATION) or {}
        updates = {}
        for key, (_, _, height, width) in self.crops.items():
            feat = obs.get(key)
            if feat is not None and len(feat.shape) >= 2:
                updates[key] = replace(feat, shape=(*feat.shape[:-2], height, width))
        if not updates:
            return features
        new_features = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {**obs, **updates}
        return new_features
