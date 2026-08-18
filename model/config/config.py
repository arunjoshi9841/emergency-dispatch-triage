"""Central configuration for the model training pipeline."""

from pathlib import Path

# -- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"
DATASET_CSV = DATA_ROOT / "dispatch_dataset.csv"
CATEGORIES_TXT = DATA_ROOT / "final_categories.txt"

TRANSCRIPT_DIRS = [
    DATA_ROOT / "transcripts",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

# -- Labels -------------------------------------------------------------
# Category mapping is computed dynamically from the dataset.
# Categories with fewer than MIN_CATEGORY_SAMPLES training records are
# excluded from the supervised label space at load time
# (see preprocessing/loader.py).
MIN_CATEGORY_SAMPLES = 100

SEVERITY_LABELS = [0, 1, 2, 3]
NUM_SEVERITY = len(SEVERITY_LABELS)

DISPATCH_LABELS = ["police", "emt", "fire"]
NUM_DISPATCH = len(DISPATCH_LABELS)

# -- Model --------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 300

# Cascade embedding dimensions (for feeding predictions downstream)
CAT_EMB_DIM = 32
SEV_EMB_DIM = 8
MLP_HIDDEN = 256

# -- Training -----------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 12
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

LOSS_WEIGHT_CATEGORY = 1.0
LOSS_WEIGHT_SEVERITY = 1.0
LOSS_WEIGHT_DISPATCH = 1.0

# Regularize the single-label tasks to reduce the train/validation gap.
CATEGORY_LABEL_SMOOTHING = 0.05
SEVERITY_LABEL_SMOOTHING = 0.05

EARLY_STOPPING_PATIENCE = 4

SEED = 42
