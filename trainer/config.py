"""Configuration for screen detector V3 trainer.

Single-stage CNN + FFT + DWT Branch architecture for 3-class classification:
- natural, screenshot, screen_photo

Frequency features:
- FFT: log(abs(fft)) magnitude spectrum
- DWT: Haar wavelet decomposition (LL, LH, HL, HH)
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"
TRAINER_ROOT = PROJECT_ROOT / "trainer"
CHECKPOINT_DIR = TRAINER_ROOT / "checkpoints"
LOG_DIR = TRAINER_ROOT / "logs"

# Three-class config
CLASS_NAMES_THREE_CLASS = ["natural", "screenshot", "screen_photo"]
THREE_CLASS_DATA_MAP = {
    "natural": ["natural_photo"],
    "screenshot": ["screenshot", "hard_negative"],
    "screen_photo": ["screen_photo"],
}

# Class weights for imbalanced dataset (total=2856)
# natural=947, screenshot=1034, screen_photo=282
# Optimized for screen_photo detection: alpha=[1,1,4]
CLASS_WEIGHTS_THREE_CLASS = [1.0, 1.0, 4.0]

# Model
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
INPUT_CHANNELS = 3
NUM_CLASSES = 3  # Three-class classification

# Training
BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 40  # Increased for better convergence
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Focal Loss (optimized for screen_photo: gamma=3, alpha=[1,1,4])
FOCAL_LOSS_GAMMA = 3.0
USE_FOCAL_LOSS = True

# Oversampling
USE_WEIGHTED_SAMPLER = True

# Hard Negative Mining
HARD_NEGATIVE_WEIGHT = 5  # Hard negative 重复采样权重
HARD_NEGATIVE_DIR = DATA_DIR / "hard_negative"

# Best metric weights (prioritize screen_photo recall)
# best_metric = 0.7 * screen_photo_recall + 0.3 * accuracy
BEST_METRIC_RECALL_WEIGHT = 0.7
BEST_METRIC_ACCURACY_WEIGHT = 0.3

# Augmentation
JPEG_QUALITY_RANGE = (50, 95)
BLUR_SIGMA_RANGE = (0.5, 2.0)
NOISE_STD_RANGE = (5, 25)
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
