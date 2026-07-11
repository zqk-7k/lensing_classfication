# 强引力透镜引力波的校准配对验证

本仓库提供论文所用的核心代码、冻结的评估清单、逐配对预测分数和统计结果。研究比较两种方法：直接处理峰值对齐应变片段的一维 PI-ResNet，以及使用常 Q 变换图像的 CQT--DeiT（SEMD-inspired）基线。

本项目解决的是“已识别事件之间的配对验证”，不是完整的搜寻流水线、真实噪声检验或 catalog-level FAR 分析。

主要结果、置信区间和物理诊断见 [docs/RESULTS.md](docs/RESULTS.md)，数据与大文件说明见 [docs/ARTIFACTS.md](docs/ARTIFACTS.md)，完整执行顺序见 [experiments/reproducibility/README.md](experiments/reproducibility/README.md)。代码及正式技术文档以英文版本为准。
