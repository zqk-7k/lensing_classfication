# GW Lensing Pair Classification

本目录已于 2026-07-11 完成第一轮清理，当前只保留四条核心研究链路：引力波数据生成、1D 配对分类、SEMD 时频图分类和 Siamese 候选匹配。

## 目录

- `src/generation/`：PM、SIS 与 unlensed 波形生成。
- `src/classifier/`：1D 周期 ResNet 分类器的训练、消融和评估。
- `src/semd/`：CQT/Mel 预处理及 SEMD 图像分类。
- `src/matching/`：Siamese 表征学习、候选匹配与评估。
- `scripts/figures/`：保留的最终论文绘图脚本（历史复现用途）。
- `data/`：有效数据集；不纳入 Git。
- `artifacts/`：关键权重、历史运行和汇总结果；不纳入 Git。
- `docs/`：审计记录与数据说明。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认数据根目录是 `data/ligo_full`。若要切换数据集，统一使用环境变量：

```bash
export GW_DATA_ROOT=/absolute/path/to/dataset
export GW_TMPDIR=/absolute/path/to/tmp
```

## 入口

```bash
# 1D 分类器
python src/classifier/train.py
python src/classifier/train_ablation.py --dataset SIS --exp_name Baseline

# SEMD：先生成时频图，再训练
python src/semd/preprocess_offline.py
python src/semd/main.py

# Siamese matching（单卡调试或 torchrun）
python src/matching/main_ddp.py
torchrun --nproc_per_node=4 src/matching/main_ddp.py
```

训练前请先在对应目录的 `config*.py` 中确认 `MODEL_TYPE`、`DATA_MODE`、batch size 和 GPU 数量。第一轮清理只建立了清晰、可追踪的基线，没有启动新的训练。

更详细的保留/删除依据见 `docs/PROJECT_AUDIT.md`。
