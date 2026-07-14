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

# Class weights for imbalanced dataset (total=2823)
# natural=939, screenshot=1081, screen_photo=319, hard_negative=484
# Further optimized for F1 balance: alpha=[1.0, 1.0, 1.5]
CLASS_WEIGHTS_THREE_CLASS = [1.0, 1.0, 1.5]

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
EPOCHS_FINETUNE = 20  # Reduced for faster training
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Focal Loss (optimized for screen_photo: gamma=3, alpha=[1,1,4])
FOCAL_LOSS_GAMMA = 3.0
USE_FOCAL_LOSS = True

# Oversampling
USE_WEIGHTED_SAMPLER = True

# Hard Negative Mining
HARD_NEGATIVE_WEIGHT = 3  # Reduced to avoid screenshot boundary pollution
HARD_NEGATIVE_DIR = DATA_DIR / "hard_negative"

# Best metric weights (F1-oriented optimization)
# best_metric = 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1
BEST_METRIC_F1_WEIGHT = 0.5
BEST_METRIC_ACCURACY_WEIGHT = 0.3
BEST_METRIC_MACRO_F1_WEIGHT = 0.2

# Augmentation
JPEG_QUALITY_RANGE = (50, 95)
BLUR_SIGMA_RANGE = (0.5, 2.0)
NOISE_STD_RANGE = (5, 25)
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

# ============================================================
# Optimization Module Configuration
# ============================================================

# Center Loss - enhances intra-class compactness
USE_CENTER_LOSS = True
CENTER_LOSS_WEIGHT = 0.02  # Weight for center loss in combined loss

# OHEM (Online Hard Example Mining) - focuses on hard samples
USE_OHEM = True
OHEM_HARD_RATIO = 0.3  # Train on hardest 30% samples

# ArcFace - angular margin classifier for better class separation
USE_ARCFACE = True
ARCFACE_SCALE = 30.0  # Logits scale factor
ARCFACE_MARGIN = 0.50  # Angular margin (~28.6 degrees)

# FFT Attention - attention modules for frequency branches
USE_FFT_ATTENTION = True
FFT_ATTENTION_TYPE = "cbam"  # 'cbam' or 'coordinate'
FFT_ATTENTION_REDUCTION = 16  # Channel reduction ratio

# Adaptive Threshold - optimize thresholds on validation set
USE_ADAPTIVE_THRESHOLD = True
ADAPTIVE_THRESHOLD_TARGET = "screen_photo"  # Class to optimize
ADAPTIVE_THRESHOLD_METRIC = "f1"  # Metric to optimize

# ============================================================
# PAH-ViT Configuration (Legacy - not used in current optimization)
# ============================================================
PAH_VIT_MODEL_NAME = "efficientvit_b0"
PAH_VIT_EPOCHS_HEAD = 3       # Stage A: freeze backbone
PAH_VIT_EPOCHS_FINETUNE = 10  # Stage B: unfreeze last 4 stages
PAH_VIT_BATCH_SIZE = 16
PAH_VIT_LEARNING_RATE = 1e-3
PAH_VIT_WEIGHT_DECAY = 1e-4
PAH_VIT_LAMBDA_CONTRASTIVE = 0.3  # Weight for Patch Contrastive Loss
PAH_VIT_FOCAL_GAMMA = 3.0
PAH_VIT_NUM_WORKERS = 0
PAH_VIT_NUM_QUERIES = 4       # Number of learnable query tokens in Patch Branch
PAH_VIT_PATCH_DIM = 256       # Patch branch feature dimension
PAH_VIT_FUSED_DIM = 512       # Fusion output dimension
PAH_VIT_MIXER_LAYERS = 2      # Number of Fourier Token Mixer layers
