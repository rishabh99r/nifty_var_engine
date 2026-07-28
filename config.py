# config.py
import os
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

VALIDATION_SEEDS = [42, 123, 999, 2024, 777]

START_DATE = "2007-01-01"
TEST_START_DATE = "2025-01-01"

# Structural Architecture Limits (Optimized for Financial Time Series)
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 64
HIDDEN_SIZE = 32         # Sized to 50:1 Observation-to-Parameter ratio
ATTENTION_HEADS = 4
DROPOUT = 0.3
EPOCHS = 100
LEARNING_RATE = 0.0018   # Fallback if HPO is bypassed
