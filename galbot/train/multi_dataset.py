"""多数据集联合训练支持。

将多个特征结构一致的 LeRobotDataset 合并为单一数据集外观（ConcatDataset +
聚合 meta），供 `galbot.train.train` 在配置使用 `datasets`（列表）时启用。
"""

from __future__ import annotations

import torch

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class _MultiMetaAdapter:
    """提供与 LeRobotDatasetMetadata 相同的只读接口（features/stats/fps/camera_keys/info）。

    要求各子数据集特征结构一致（同一机器人、同一转换配置）。
    """

    def __init__(self, datasets: list[LeRobotDataset]) -> None:
        first = datasets[0].meta
        for d in datasets[1:]:
            if set(d.meta.features) != set(first.features) or d.meta.fps != first.fps:
                raise ValueError(
                    f"数据集 {d.repo_id} 的特征或 fps 与 {datasets[0].repo_id} 不一致，无法联合训练"
                )
        self.info = dict(first.info)
        self.info["total_episodes"] = sum(d.num_episodes for d in datasets)
        self.info["total_frames"] = sum(d.num_frames for d in datasets)
        # 归一化统计量按帧数加权合并（mean/std 正确聚合，min/max 取极值）。
        self.stats = aggregate_stats([d.meta.stats for d in datasets])

    @property
    def features(self) -> dict:
        return self.info["features"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def camera_keys(self) -> list[str]:
        return [k for k, ft in self.features.items() if ft["dtype"] in ("video", "image")]


class MultiLeRobotDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset + 聚合 .meta；样本按子数据集顺序寻址，shuffle 后均匀混合。"""

    def __init__(self, datasets: list[LeRobotDataset]) -> None:
        super().__init__(datasets)
        self.meta = _MultiMetaAdapter(datasets)

    @property
    def num_frames(self) -> int:
        return sum(d.num_frames for d in self.datasets)

    @property
    def num_episodes(self) -> int:
        return sum(d.num_episodes for d in self.datasets)

    def __repr__(self) -> str:
        ids = [d.repo_id for d in self.datasets]
        return (
            f"MultiLeRobotDataset(repo_ids={ids}, "
            f"num_frames={self.num_frames}, num_episodes={self.num_episodes})"
        )
