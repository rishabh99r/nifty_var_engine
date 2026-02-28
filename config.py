# config.py
import os
import torch
import numpy as np
import random

# --- 1. SEED LOCKING UTILITY ---
def set_seed(seed):
    """Call this explicitly inside execution loops. NEVER run it globally here."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# The specific seeds for the robustness audit
VALIDATION_SEEDS = [42, 123, 999, 2024, 777]

# --- 2. TIMELINE SETTINGS ---
START_DATE = "2007-01-01"
TEST_START_DATE = "2025-01-01"

# --- 3. TFT ARCHITECTURE (Baseline limits for HPO fallback) ---
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 64
HIDDEN_SIZE = 64
ATTENTION_HEADS = 4
DROPOUT = 0.3
EPOCHS = 150
LEARNING_RATE = 0.03
