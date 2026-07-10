# config_classifier.py
import os



# ========================== 核心开关 ==========================
# 1. 任务模式
DATA_MODE = "noisy"
MODEL_TYPE = "PM" #SIS

# 2. 训练配置
USE_DDP     = True        # 是否开启分布式训练
N_GPUS      = 4           # 显卡数量
BATCH_SIZE  = 64          # 单卡 Batch Size (总 Batch = 64 * 4 = 256)
EPOCHS      = 50
LR          = 1e-5
WEIGHT_DECAY= 1e-4

# 3. 数据维度与预处理
RAW_LEN     = 98304
TARGET_LEN  = 8192        # 只截取最后 8192
STRIDE      = 2           # 下采样 2 -> 最终输入长度 4096
IN_CHANS    = 2           # 双通道输入 (Signal 1, Signal 2)

# 4. 数据增强
AUG_PROB    = 0.5         # 进行增强的概率
AUG_FLIP    = True        # 峰值翻转 (建议开启)
AUG_INDEPENDENT_ROLL = True # 【新功能】独立随机平移开关 (True=两通道分别平移, False=同步平移)
AUG_ROLL_MAX = 256        # 平移最大幅度

# 5. 负样本采样比例 (重要)
# 定义在训练每个 Batch 中，负样本的组成比例
# 格式: {'diff_event': 权重, 'noise': 权重}
# diff_event: 错配的透镜事件 (Hard Negative)
# noise:      纯背景噪声 (Easy Negative)
NEG_RATIO   = {'diff_event': 0.7, 'noise': 0.3}

# ========================== 路径配置 (自动构建) ==========================
#DATA_ROOT = "/root/autodl-tmp/qkzhang"
DATA_ROOT = "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/ET_vs_LIGO/LIGO_data"
if MODEL_TYPE == "SIS":
    SOURCE_DIR = os.path.join(DATA_ROOT, "SIS_data") #SIS_data_0222
    FILE_PREFIX = "SIS"
elif MODEL_TYPE == "PM":
    SOURCE_DIR = os.path.join(DATA_ROOT, "PM_data")  #PM_data_0222
    FILE_PREFIX = "PM"
else:
    raise ValueError("Unknown MODEL_TYPE")




# 根据模式选择文件名
if DATA_MODE == "pure":
    STRAIN_TAG = "h_strain"
    UNL_FILENAME = "unlensed_h_strain.npy"
else:
    STRAIN_TAG = "data_strain"
    UNL_FILENAME = "unlensed_data_strain.npy"

L1_PATH = os.path.join(SOURCE_DIR, f"{FILE_PREFIX}_{STRAIN_TAG}_1.npy")
L2_PATH = os.path.join(SOURCE_DIR, f"{FILE_PREFIX}_{STRAIN_TAG}_2.npy")
UNL_PATHS = [os.path.join(DATA_ROOT, "Unlensed_data", UNL_FILENAME)]  #Unlensed_data_0222

# 输出目录
OUT_DIR = f"./runs/classifier_{MODEL_TYPE}_{DATA_MODE}_v1"

# 数据集划分
SPLIT = dict(train=0.8, val=0.1, test=0.1)
SEED  = 42

# ========================== 图像预处理配置 (新增) ==========================
# 选择时频转换方法: 'cqt' (论文的恒定Q变换) 或 'mel' (极速Mel频谱平替)
TRANSFORM_METHOD = 'cqt'

# 根据当前模式动态生成图片保存路径 (例如: ./dataset_images_SIS_noisy_cqt )
IMAGE_SAVE_DIR = os.path.join(DATA_ROOT, f"dataset_images_{MODEL_TYPE}_{DATA_MODE}_{TRANSFORM_METHOD}")