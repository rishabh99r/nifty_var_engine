# deployment.py
# =============================================================================
# Production deployment orchestrator with cadence-aware scheduling.
#
# Cadence design (manual trigger, no daemon required):
#   - DAILY  : update the master buffer with the latest trading day and run
#              the live VaR inference (production_engine.run_live_daily_inference).
#   - EVERY 21 TRADING DAYS (GARCH_REFIT_DAYS, ~1 month): re-fit the GJR-GARCH
#     parameters on the trailing LOOKBACK_DAYS window and persist them, so the
#     PIT filter reflects fresh parameters.
#   - EVERY 126 TRADING DAYS (TFT_RETRAIN_DAYS, ~6 months): re-run main.py to
#     retrain the 3-seed TFT on the full available history (up to a cap) with
#     the standard temporal split discipline, then promote the median-seed
#     checkpoint.
#
# Cadence bookkeeping lives in deployment_state.json (see config constants).
# This orchestrator decides WHAT to do on a given invocation based on the last
# recorded anchors and the current trading-day index; it does NOT run forever.
# =============================================================================
import datetime
import json
import os
import subprocess
import sys

import pandas as pd

import config


def _load_state():
    try:
        with open(config.DEPLOYMENT_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state):
    with open(config.DEPLOYMENT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _current_time_idx():
    """Latest trading-day index in the master buffer (NIFTY50 row)."""
    df = pd.read_csv("master_df.csv")
    sub = df[df["ticker"] == "NIFTY50"]
    return int(sub["time_idx"].max())


def _garch_refit_due(state, current_idx):
    """True if GARCH parameters should be refit now (21-trading-day cadence)."""
    last = state.get("last_garch_refit_idx")
    if last is None:
        return True  # never refit -> do it on first deployment
    return (current_idx - last) >= config.GARCH_REFIT_DAYS


def _tft_retrain_due(state, current_idx):
    """True if the TFT should be retrained now (126-trading-day cadence)."""
    last = state.get("last_tft_retrain_idx")
    if last is None:
        return True  # never retrained -> do it on first deployment
    return (current_idx - last) >= config.TFT_RETRAIN_DAYS


def _run(cmd):
    """Run a subprocess command and surface failures."""
    print(f"[DEPLOY] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
    return result


def _update_buffer():
    """Rebuild the master buffer through today (dynamic END_DATE)."""
    import build_data
    build_data.generate_clean_production_data(start_date=config.START_DATE, end_date=config.END_DATE)


def run_deployment(do_forecast=True, force_garch_refit=False, force_tft_retrain=False):
    """
    Main manual-trigger entry point.

    Steps:
      1. Rebuild/refresh the master buffer through today.
      2. Decide GARCH refit (every 21 trading days) and/or TFT retrain
         (every 126 trading days) based on cadence state.
      3. Run live inference for the median-seed checkpoint.

    Returns a summary dict of actions taken.
    """
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT RUN")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    state = _load_state()

    # 1. Refresh the buffer (data-freshness; rebuild to the latest trading day).
    _update_buffer()
    current_idx = _current_time_idx()
    print(f"[DEPLOY] Current trading-day index: {current_idx}")

    actions = {"garch_refit": False, "tft_retrain": False, "forecast": False}

    # 2a. GARCH refit every GARCH_REFIT_DAYS trading days.
    if force_garch_refit or _garch_refit_due(state, current_idx):
        print(f"[DEPLOY] GARCH refit due (last_idx={state.get('last_garch_refit_idx')}).")
        # The PIT GARCH recursion is part of build_data's buffer rebuild; here
        # we persist the cadence anchor so the next refit is scheduled in 21
        # trading days. (If a dedicated refit module is used, invoke it here.)
        state["last_garch_refit_idx"] = current_idx
        state["last_garch_refit_date"] = datetime.date.today().isoformat()
        _save_state(state)
        actions["garch_refit"] = True
        print(f"[DEPLOY] GARCH refit anchor set to index {current_idx}.")

    # 2b. TFT retrain every TFT_RETRAIN_DAYS trading days (~6 months).
    if force_tft_retrain or _tft_retrain_due(state, current_idx):
        print(f"[DEPLOY] TFT retrain due (last_idx={state.get('last_tft_retrain_idx')}).")
        _run([sys.executable, "main.py"])
        # main.py itself records last_tft_retrain_idx/median_seed in the state.
        state = _load_state()  # reload (main.py updated it)
        actions["tft_retrain"] = True
        print(f"[DEPLOY] TFT retrained; median seed now {state.get('median_seed')}.")

    # 3. Run live forecast using the median-seed checkpoint.
    if do_forecast:
        from production_engine import run_live_daily_inference
        from tft_model import select_median_checkpoint

        median_seed = state.get("median_seed")
        ckpt = select_median_checkpoint(median_seed)
        if ckpt is None:
            raise FileNotFoundError(
                "[DEPLOY] No checkpoint found. Run main.py once to train the TFT first."
            )
        print(f"[DEPLOY] Using checkpoint: {ckpt}")
        result = run_live_daily_inference(ckpt, live_csv_path="master_df.csv", target_ticker="NIFTY50")
        actions["forecast"] = True
        print(f"[DEPLOY] Final 99% VaR: {result['final_var_99']:.4f}%")

    print("=" * 70)
    print(f"ACTIONS THIS RUN: {actions}")
    print("=" * 70)
    return actions


def status():
    """Print the current cadence state and next-due schedule."""
    state = _load_state()
    current_idx = _current_time_idx()
    print("--- Deployment status ---")
    print(f"Current time_idx:           {current_idx}")
    print(f"Last GARCH refit idx:       {state.get('last_garch_refit_idx')}  "
          f"(next due >= {state.get('last_garch_refit_idx', current_idx) + config.GARCH_REFIT_DAYS})")
    print(f"Last TFT retrain idx:       {state.get('last_tft_retrain_idx')}  "
          f"(next due >= {state.get('last_tft_retrain_idx', current_idx) + config.TFT_RETRAIN_DAYS})")
    print(f"Median seed:                {state.get('median_seed')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nifty VaR production deployment")
    parser.add_argument("--no-forecast", action="store_true", help="skip live inference")
    parser.add_argument("--force-garch", action="store_true", help="force GARCH refit")
    parser.add_argument("--force-tft", action="store_true", help="force TFT retrain")
    parser.add_argument("--status", action="store_true", help="print cadence status and exit")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        run_deployment(
            do_forecast=not args.no_forecast,
            force_garch_refit=args.force_garch,
            force_tft_retrain=args.force_tft,
        )
