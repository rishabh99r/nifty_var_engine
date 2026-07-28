# circuit_breaker.py
import numpy as np

def apply_regulatory_circuit_breaker(tft_var_99, parametric_floor_var, gjr_std_resid, crash_threshold=4.0):
    """
    Enforces deterministic regulatory boundaries on neural network risk forecasts.
    Ref: Bank of England Working Paper No. 525 (Volatility Floors in FHS Models).
    """
    status_flags = []
    final_var = tft_var_99

    # RULE 1: Black-Swan Innovation Override (OOD Detection)
    # If daily standardized market shock exceeds 4 standard deviations, neural weights are OOD.
    if abs(gjr_std_resid) > crash_threshold:
        status_flags.append(f"[CIRCUIT BREAKER] OOD Market Shock Detected (|z_t| = {abs(gjr_std_resid):.2f} > {crash_threshold}). Reverting to Stressed Parametric Floor.")
        final_var = min(tft_var_99, parametric_floor_var * 1.25) # Apply 25% stress multiplier to floor

    # RULE 2: Economic Hallucination / Positive VaR Floor
    # 99% VaR on a long equity portfolio cannot predict positive returns or sit above parametric floor during crises.
    if tft_var_99 > 0.0:
        status_flags.append("[CIRCUIT BREAKER] Neural Hallucination (Positive VaR Forecast). Clamped to Parametric Floor.")
        final_var = parametric_floor_var
    elif tft_var_99 > parametric_floor_var and abs(gjr_std_resid) > 2.0:
        status_flags.append("[CIRCUIT BREAKER] TFT Under-predicting active selloff regime. Clamped to GJR-GARCH Floor.")
        final_var = parametric_floor_var

    if not status_flags:
        status_flags.append("[OK] Neural Forecast within Normal Regulatory Operating Bounds.")

    return final_var, status_flags

if __name__ == "__main__":
    # Test Simulation of Flash Crash
    sim_tft_var = -1.85
    sim_gjr_floor = -4.12
    sim_shock = -5.40  # 5.4 sigma selloff

    exec_var, logs = apply_regulatory_circuit_breaker(sim_tft_var, sim_gjr_floor, sim_shock)
    for log in logs:
        print(log)
    print(f"Final Executable 99% VaR: {exec_var:.4f}%")
