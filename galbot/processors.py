"""Galbot 自定义 LeRobot processor step（train 与 deploy 共享）。

步骤通过 ``ProcessorStepRegistry`` 注册，随 checkpoint 的
``policy_preprocessor.json`` 一起序列化/反序列化。部署端（policy_agent）
只需 import 本模块完成注册，即可透明加载含自定义步骤的管线。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

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
