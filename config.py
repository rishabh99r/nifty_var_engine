# config.py
# =============================================================================
# SINGLE SOURCE OF TRUTH for the entire Nifty VaR research pipeline.
# All modules import configuration constants from here. No hardcoded
# values in build_data.py / tft_model.py / main.py / generate_report_plots.py.
# =============================================================================
import os
import random

import numpy as np

try:
    import torch
except ImportError:
    # torch is only required for the deep-learning pipeline (Colab). Config
    # remains importable for pure-statistics modules (e.g. metrics.py) on
    # machines without torch installed.
    torch = None


def set_seed(seed):
    """Deterministic seeding across Python, NumPy, and PyTorch (CPU+CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
# NOTE: ^INDIAVIX has limited history on Yahoo Finance. If coverage is
# insufficient we fall back to an honestly-labelled realized-volatility proxy.
START_DATE = "2015-01-01"
END_DATE = "2026-08-01"

TICKERS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
}

# Domestic volatility: attempt to ingest the REAL India VIX first.
INDIA_VIX_SYMBOL = "^INDIAVIX"
# If the India VIX series is too short (< MIN_INDIA_VIX_OBS), the pipeline
# falls back to a lagged first-difference of a rolling realized-vol proxy and
# MUST label it Domestic_RV_Proxy (never "India VIX").
MIN_INDIA_VIX_OBS = 200

# ----------------------------------------------------------------------------
# ROLLING GJR-GARCH(1,1) SKEW-T PIT FILTER
# ----------------------------------------------------------------------------
LOOKBACK_DAYS = 1000      # estimation window for each parameter refit
REFIT_FREQ = 21           # monthly (21 trading days) parameter re-estimation
VAR_ALPHA = 0.01          # 99% VaR
GARCH_MIN_VARIANCE = 1e-6

# ----------------------------------------------------------------------------
# TFT ARCHITECTURE (Champion config -- kept as the auditable, committed spec)
# ----------------------------------------------------------------------------
HIDDEN_SIZE = 64
ATTENTION_HEADS = 4
DROPOUT = 0.30
LEARNING_RATE = 0.001552
HIDDEN_CONTINUOUS_SIZE = max(4, HIDDEN_SIZE // 2)
OUTPUT_SIZE = 3                     # quantiles
QUANTILES = [0.01, 0.5, 0.99]
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 8
REDUCE_ON_PLATEAU_PATIENCE = 4
GRADIENT_CLIP_VAL = 0.1
BATCH_SIZE = 64

# ----------------------------------------------------------------------------
# TEMPORAL SPLIT & WINDOW
# ----------------------------------------------------------------------------
ENCODER_LENGTH = 21                 # lookback window for the TFT encoder
PREDICTION_LENGTH = 1               # 1-step-ahead forecasts
BACKTEST_DAYS = 500                 # out-of-sample test horizon (days)
VAL_DAYS = 250                      # validation window (days)

# ----------------------------------------------------------------------------
# MULTI-SEED VALIDATION SUITE
# ----------------------------------------------------------------------------
VALIDATION_SEEDS = [42, 123, 777]

# ----------------------------------------------------------------------------
# DOWNSIDE QUANTILE & BASEL PARAMETERS
# ----------------------------------------------------------------------------
BASEL_ALPHA = 0.01                  # 99% VaR
BASEL_GREEN_CUM = 0.95
BASEL_YELLOW_CUM = 0.9999

# ----------------------------------------------------------------------------
# EXPECTED SHORTFALL BACKTEST (McNeil-Frey)
# ----------------------------------------------------------------------------
ES_ALPHA = 0.01                     # 99% ES
ES_MIN_BREACHES = 1                 # require >=1 exceedance to report descriptive ES
# Minimum exceedances required to run a MEANINGFUL statistical test on ES.
# Below this the t-stat/p-value are degenerate (tiny sample -> near-zero
# variance -> absurd t-stats). With a 500-day backtest at alpha=1% the expected
# exceedance count is ~5, so a threshold of 5 is the honest minimum.
ES_MIN_BREACHES_TESTABLE = 5

# ----------------------------------------------------------------------------
# OUTPUT PATHS
# ----------------------------------------------------------------------------
OUTPUT_DIR = "/content/drive/MyDrive/GARCH_TFT_Results"
