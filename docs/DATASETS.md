# 数据集

当前默认数据根目录为 `data/ligo_full/`，也可通过 `GW_DATA_ROOT` 覆盖。

| 目录 | 内容 | 代表性规模 |
| --- | --- | --- |
| `data/ligo_full/` | PM、SIS、Unlensed 及预生成时频图 | 10,000 个事件，数组长度 98,304 |
| `data/ligo_reduced/` | PM、SIS、Unlensed 小规模数据 | 2,500 个事件，数组长度 98,304 |
| `data/ligo_bf_sis/` | SIS 小规模 BF 数据 | 2,500 个事件 |
| `data/final_v3/` | PM、SIS、Unlensed 的 v3 数据 | 2,500 个事件，数组长度 98,304 |
| `data/preprocessed/` | 额外的 PM/SIS noisy CQT 图 | 图像数据 |

PM、SIS 与 Unlensed 的 CSV 文件是生成参数、样本索引和源事件元数据，属于数据集组成部分，不是实验结果文档，不能单独删除。

不同目录虽然存在同名文件，但样本规模可能不同。选择数据集时必须记录完整 `GW_DATA_ROOT`，不可只记录 PM/SIS 名称。
