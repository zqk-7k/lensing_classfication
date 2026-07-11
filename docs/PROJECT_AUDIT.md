# 项目审计与清理记录

日期：2026-07-11

## 清理前问题

- 根目录约 864GB，代码、数据、缓存、日志和实验输出混放。
- 不存在 Git 仓库，任何删除都缺少版本历史保护。
- 同一套分类器与 SEMD 代码在 `classifier_project`、`ET_vs_LIGO`、`ET_vs_LIGO_bf` 及嵌套目录中重复出现 3–4 次。
- 大量配置绑定 `/root/autodl-tmp/qkzhang` 或已删除的旧目录，换目录后无法运行。
- 存在约 94GB 名义大小的中断 `.tmp.npy` 文件、`pymp-*` socket、`__pycache__`、训练日志和多轮临时图。
- `main.py`、`main2.py`、`config2.py`、`config3.py` 这类文件名无法表达用途，多份实验脚本只有少量差异。

## 保留内容

- 最新的 PM/SIS/unlensed 数据生成脚本。
- 最新的 1D 分类器训练、消融、评估与 sweep 代码。
- 完整 SEMD 预处理、训练和评估链路。
- Siamese matching 的 DDP 主链路和必要评估代码。
- 七个最终论文图脚本；它们归档在 `scripts/figures/`，不作为训练入口。
- 所有有效 `.npy`/CSV 数据集，只做目录归并，不做内容去重。
- SEMD 四个关键 checkpoint、最新分类器/消融历史运行、两份 sweep 汇总表和两张主要论文图。

## 删除内容

- 旧 notebook、早期分类器、重复 SEMD 副本和嵌套的完整代码副本。
- `__pycache__`、`.pyc`、`pymp-*`、临时目录、`.log`、`.out` 和 `.tmp.npy`。
- 已由最终脚本或汇总表替代的中间绘图目录、逐轮日志和旧输出。
- 无明确入口价值的编号变体脚本；原文件仍可从清理前备份恢复。

## 回滚材料

清理前的代码、脚本、Markdown、文本和 CSV 已打包到项目同级目录：

- `classcify-gw-lensing-pairs-main_precleanup_code_20260711.tar.gz`
- `classcify-gw-lensing-pairs-main_precleanup_manifest_20260711.sha256`

该备份不包含数百 GB 的 NPY 数据，但这些有效数据在清理后仍保留于 `data/`。

## 已知技术债

- 四条链路仍采用脚本式配置，下一步宜统一为 CLI/YAML 配置与公共数据索引。
- `scripts/figures/` 是历史复现脚本，部分仍依赖旧运行目录；在重做论文图前应单独参数化。
- 多个大型数据集样本规模不同，不能仅按文件名判断重复，因此本轮没有删除有效数据。
- 本轮只做轻量静态验证，不以 833GB 数据启动训练或完整推理。
