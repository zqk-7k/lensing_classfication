"""Default configuration for PI-ResNet pair classification."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MODE = "noisy"
MODEL_TYPE = "SIS"
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-4
RAW_LEN = 98_304
TARGET_LEN = 8_192
STRIDE = 2
AUG_PROB = 0.5
AUG_FLIP = True
AUG_INDEPENDENT_ROLL = True
AUG_ROLL_MAX = 1_024
NEG_RATIO = {"diff_event": 0.7, "noise": 0.3}
SEED = 42
DATA_ROOT = Path(os.environ.get("GW_DATA_ROOT", PROJECT_ROOT / "data"))
ARTIFACT_ROOT = Path(os.environ.get("GW_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts"))
