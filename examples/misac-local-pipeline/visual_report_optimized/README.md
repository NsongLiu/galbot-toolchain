# Offline Visual Reports

这是本地离线动作预测报告代码，与 [基础报告脚本](../scripts/generate_visual_reports.py) 分开保存。使用者自行提供有权使用的评估记录、数据集和 checkpoint；代码不附带真实结果。

- `overview` 绘制 checkpoint 指标对照。
- `validation_flow` 展示转换、划分、训练、评估流程。
- `trajectory` 在提供的 held-out 样本中扫描并输出 best / median / hard 对照。

`best` 表示经过选择的样本，不代表整个测试集表现；报告应说明选择规则和对照样本。轨迹默认使用归一化 action 空间，与离线 L1/RMSE 保持同一口径。离线误差不等于机器人任务成功率。

从仓库根目录使用兼容的 LeRobot 环境运行 `visual_report_optimized/generate_optimized_visuals.py`；参数需指向自己的数据、划分、基础模型和微调模型。默认示例路径见 [pipeline_config.json](../pipeline_config.json)。输出位于本目录的 `outputs/`，已被 Git 忽略。

源码不内置真实训练损失。可用 `--training-loss-csv outputs/training_loss.csv` 提供自己的记录，CSV 表头为 `step,training_loss`；不提供时概览标注缺少日志。L1/RMSE 曲线只读取使用者提供的评估 JSON，图标题不预设性能改善。
