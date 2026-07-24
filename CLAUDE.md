# Galbot-G1 机器人工具链
银河通用 G1 机器人开发部署工具链（基于 Lerobot 框架 v5.0.1） 

## 项目规范
- 环境说明：使用`conda activate xhum-new && export PATH="/media/jushen/Leslie-liu/miniconda/envs/xhum-new/bin:$PATH"`的环境来进行开发，具体可参考该环境内指向的 lerobot 源代码来进行基于 lerobot 框架的开发，但是不许修改该环境中指向的 lerobot 源代码
- 功能实现：这是一个从零开始的项目，开发时注意简洁高效，以最小的代码量与依赖实现指定功能
- 分支使用：进行修改前首先切换到`dev-les`分支上进行修改
- 网络问题：如果遇到网络问题可以使用`export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000`来挂代理

## 参考资料
- G1 机器人二开资料：https://developer.galbot.com/docs/SDK/1.8.1/g1/zh
- G1 机器人快速使用手册：https://developer.galbot.com/docs/g1/2.2.4/zh/g1

## 功能开发
- 数据转换: 将采集的mcap数据转换为Lerobot V3格式数据，当前数据路径为/media/jushen/Leslie-liu/galbot_dataset
  - 转换规则: `FIN`:源数据;`SYNC`:对齐数据(有效数据);`CANCELED`:失败数据;UNQUALIFED:操作异常存下的数据;
  - 转换脚本: `convert_mcap_to_lerobot.py`（默认输出至 `/media/jushen/Leslie-liu/galbot_dataset/lerobot_v3`，仅处理 `SYNC` 文件）
  - 机器人配置: `robot_config.json`（由转换脚本自动生成，记录各关节分组与顺序）
