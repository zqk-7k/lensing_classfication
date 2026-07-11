# 数据集目录

- `data/ligo_full/`：当前默认数据根目录，包含 PM、SIS、Unlensed 和四组预生成时频图。
- `data/ligo_reduced/`：较小规模的 PM/SIS/Unlensed 数据副本，数组尺寸与 full 版本不同。
- `data/ligo_bf_sis/`：仅 SIS 的小规模 BF 数据。
- `data/final_v3/`：历史 final-v3 数据，保留用于结果复现。
- `data/preprocessed/`：从旧代码目录移出的额外 PM/SIS noisy CQT 图。

默认情况下，分类器、SEMD 和 matching 都读取 `data/ligo_full/`。可通过 `GW_DATA_ROOT` 覆盖。

这些目录体积大且存在不同采样规模。本轮未进行基于内容的全量哈希去重，以避免把不同实验规模误判为重复数据。
