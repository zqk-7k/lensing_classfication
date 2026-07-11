import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from dataset import load_data, LensedDataset, get_train_transforms, get_val_transforms
from model import get_deit_tiny_distilled_enhanced
from train import train
from evaluate import evaluate
from utils import set_seed

# 引入你的配置文件
import config as cfg


def main():
    parser = argparse.ArgumentParser(description='Train GW Image Classifier')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of total epochs to run')
    # 注意：在 DataParallel 下，这里的 batch_size 是 4 张显卡的总 Batch Size
    # 如果你想每张卡跑 64，这里就传 256
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Total batch size for all GPUs')
    parser.add_argument('--lr', type=float, default=cfg.LR,
                        help='Initial learning rate')
    # parser.add_argument('--model_name', type=str, required=True,
    #                     help='Name for saving the best model')
    args = parser.parse_args()

    # --- 自动拼接模型名称 ---
    auto_model_name = f"{cfg.MODEL_TYPE}_{cfg.DATA_MODE}_{cfg.TRANSFORM_METHOD}_model"
    print(f"============================================================")
    print(f"🚀 Auto-generated model name: {auto_model_name}")
    print(f"============================================================")

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检测可用的 GPU 数量
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, detected {n_gpus} GPUs.")

    # 动态读取我们在 preprocess_offline.py 中生成的离线图片路径
    data_dirs = {
        'lensed': os.path.join(cfg.IMAGE_SAVE_DIR, 'lensed'),
        'unlensed': os.path.join(cfg.IMAGE_SAVE_DIR, 'unlensed')
    }

    if not os.path.exists(cfg.IMAGE_SAVE_DIR):
        raise FileNotFoundError(f"找不到数据集文件夹 {cfg.IMAGE_SAVE_DIR}，请先运行 preprocess_offline.py 生成图片！")

    print(f"Loading images from: {cfg.IMAGE_SAVE_DIR}")
    splits = load_data(data_dirs, split_ratios=(0.75, 0.125, 0.125), random_seed=42)
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        paths, labels = splits[phase]
        if phase == 'train':
            dataset = LensedDataset(paths, labels, transform=train_transform)
            shuffle = True
        else:
            dataset = LensedDataset(paths, labels, transform=val_transform)
            shuffle = False

        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            # 将 num_workers 调大，保证 CPU 读取图片的速度能喂饱 4 张 GPU
            num_workers=8 if n_gpus > 1 else 4,
            pin_memory=True,
            drop_last=(phase == 'train')
        )

    # 初始化模型
    model = get_deit_tiny_distilled_enhanced(
        num_classes=2,
        pretrained=True,
        hidden_dim=512,
        dropout_rate=0.5,
        freeze_backbone=False
    )

    # ================= 多显卡并行核心代码 =================
    if n_gpus > 1:
        print(f"🚀 Let's use {n_gpus} GPUs with DataParallel!")
        # 这一行就是魔法，自动把输入数据切分给所有 GPU 并汇总梯度
        model = nn.DataParallel(model)
    # =====================================================

    # 将模型（或多卡包装后的模型）送入主设备
    model = model.to(device)

    best_model_path = train(
        model,
        {'train': dataloaders['train'], 'val': dataloaders['val']},
        device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=os.path.join(cfg.OUT_DIR, 'checkpoints'),
        model_name=auto_model_name,
        clip_grad=1.0
    )
    print(f"\n✅ Best model saved at: {best_model_path}")

    # ===== Evaluation Phase =====
    print("\n--- Starting Evaluation on Test Set ---")

    # 重新加载最佳模型权重
    # DataParallel 保存的权重键名带有 'module.' 前缀，
    # 既然我们是用 DataParallel 包装的模型，直接 load 进去是完美匹配的
    model.load_state_dict(torch.load(best_model_path))

    evaluate(model, dataloaders['test'], device,
             output_dir=os.path.join(cfg.OUT_DIR, f'outputs_{auto_model_name}'))


if __name__ == '__main__':
    main()
