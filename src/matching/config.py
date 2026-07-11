"""Configuration for the Siamese matching pipeline."""

import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get("GW_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "ligo_full"))

# Dataset selection
MODEL_TYPE = "SIS"  # SIS or PM
DATA_MODE = "pure"  # pure or noisy

if MODEL_TYPE not in {"SIS", "PM"}:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

SOURCE_DIR = os.path.join(DATA_ROOT, f"{MODEL_TYPE}_data")
FILE_PREFIX = MODEL_TYPE
STRAIN_TAG = "h_strain" if DATA_MODE == "pure" else "data_strain"
UNL_FILENAME = "unlensed_h_strain.npy" if DATA_MODE == "pure" else "unlensed_data_strain.npy"

L1_PATH = os.path.join(SOURCE_DIR, f"{FILE_PREFIX}_{STRAIN_TAG}_1.npy")
L2_PATH = os.path.join(SOURCE_DIR, f"{FILE_PREFIX}_{STRAIN_TAG}_2.npy")
TIME_PATH_1 = os.path.join(SOURCE_DIR, f"{FILE_PREFIX}_time_array_1.npy")
TIME_PATH_2 = os.path.join(SOURCE_DIR, f"{FILE_PREFIX}_time_array_2.npy")
UNL_DIR = os.path.join(DATA_ROOT, "Unlensed_data")
UNL_PATHS = [os.path.join(UNL_DIR, UNL_FILENAME)]
TIME_PATH_U = os.path.join(UNL_DIR, "unlensed_time_array.npy")

OUT_DIR = os.path.join(
    PROJECT_ROOT, "runs", "matching", f"siamese_{MODEL_TYPE.lower()}_{DATA_MODE}"
)

# Runtime and model settings
USE_HILBERT = False
USE_DDP = True
N_GPUS = 4
TARGET_LEN = 8192
STRIDE = 2
MAX_DT = 2.5 * 365 * 24 * 3600

LENSED_LIMIT = 2500
UNL_LIMIT = 2500
SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
TAU = 0.07
D_MODEL = 256
EMB_DIM = 128
N_LAYERS = 4
WIDTH_SCALE = 4.0

AUG_ROLL = 128
AUG_SCALE = 0.10
AUG_NOISE = 0.01
AUG_PHASE = False
AUG_FLIP = False

SPLIT = dict(train=0.8, val=0.1, test=0.1)
NEG_RATIO = {"diff_event": 0.7, "noise": 0.3}
USE_GCC = True
TOPK = 50
THRESHOLD = None
SAVE_XLSX = False
ROUND_XLSX = 6
