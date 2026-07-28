# build_data.py
import pandas as pd
from data_loader import fetch_and_clean_data
from garch_engine import run_rolling_garch

def main():
    print("===== PHASE 0: STANDALONE DATA & GARCH PIPELINE =====")
    print("[BUILD] Fetching raw market data and running stationarity audits...")

    # 1. Fetch, transform, and audit stationarity
    raw_df = fetch_and_clean_data()

    # 2. Run rolling GARCH and save directly to master_df.csv
    print("\n[BUILD] Executing Skew-T GJR-GARCH rolling engine...")
    master_df = run_rolling_garch(raw_df, csv_path="master_df.csv")

    print(f"\n[BUILD] SUCCESS! Master dataset saved to master_df.csv ({len(master_df)} rows).")
    print("[BUILD] You never need to run this script again unless your raw data changes.")

if __name__ == "__main__":
    main()
