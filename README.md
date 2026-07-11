# GW Lensing Pair Classification

强透镜引力波事件对研究代码，包含四条当前开发链路：数据生成、1D 配对分类、SEMD 时频图分类和 Siamese 候选匹配。

## 目录

- `src/generation/`：生成 PM、SIS 和 unlensed 波形数据。
- `src/classifier/`：训练与评估 1D 周期 ResNet 配对分类器。
- `src/semd/`：生成 CQT/Mel 时频图并训练 SEMD 分类器。
- `src/matching/`：训练 Siamese 表征并执行候选匹配。
- `data/`：本地数据集，不纳入 Git。
- `docs/DATASETS.md`：数据目录、规模和默认选择。

仓库不保留清理前的实验指标、旧 checkpoint 或论文结果图。后续实验结果必须由当前 Git 版本重新生成，避免混用不同数据规模、配置或代码版本的数字。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认读取 `data/ligo_full`。可通过环境变量切换数据根目录和临时目录：

```bash
export GW_DATA_ROOT=/absolute/path/to/dataset
export GW_TMPDIR=/absolute/path/to/tmp
```

## 主要入口

```bash
# 1D 分类器
python src/classifier/train.py
python src/classifier/train_ablation.py --dataset SIS --exp_name Baseline

# SEMD：先生成时频图，再训练
python src/semd/preprocess_offline.py
python src/semd/main.py

# Siamese matching
python src/matching/main_ddp.py
torchrun --nproc_per_node=4 src/matching/main_ddp.py
```

运行前应在对应的 `config*.py` 中确认 `MODEL_TYPE`、`DATA_MODE`、batch size 和 GPU 数量。新实验建议从空的 `runs/` 开始，并将 Git commit、数据根目录、随机种子和完整配置写入结果元数据。
