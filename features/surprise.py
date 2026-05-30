"""
Temperature Surprise Signal
===========================
Markets price in expected weather from forecasts. What moves prices is
the SURPRISE — the unexpected change in heating/cooling demand.

Without access to archived ECMWF/GFS forecast runs (paid), the best
publicly computable proxy is the rate of change of HDD anomaly:

    surprise[t] = HDD_zscore[t] - HDD_zscore[t-7]

Interpretation:
  +2 → it's suddenly 2 std colder than normal vs 1 week ago (cold snap)
  -2 → sudden warming surprise

This captures regime shifts (the thing markets react to) rather than
the level (which is already priced into the forward curve).
"""

import pandas as pd
import numpy as np


def add_surprise_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rate of change of the HDD anomaly over 7 days
    # = "how much colder/warmer has it gotten, relative to seasonal normal"
    df["hdd_delta_7d"] = df["HDD_zscore"].diff(7)

    # Also compute a 5d momentum for shorter lead
    df["hdd_delta_5d"] = df["HDD_zscore"].diff(5)

    # Z-score the delta (normalize against rolling distribution)
    mu  = df["hdd_delta_7d"].rolling(252, min_periods=60).mean()
    sig = df["hdd_delta_7d"].rolling(252, min_periods=60).std()
    df["surprise_zscore"] = (df["hdd_delta_7d"] - mu) / sig.clip(lower=0.01)

    return df


def add_combined_signal(df: pd.DataFrame, w_level: float = 0.4, w_surprise: float = 0.6) -> pd.DataFrame:
    """
    Blend HDD level (where we are vs normal) with HDD surprise (rate of change).
    Level = persistent cold → sustained demand
    Surprise = cold snap arriving → price spike catalyst
    """
    df = df.copy()
    df["combined_zscore"] = (
        w_level   * df["HDD_zscore"] +
        w_surprise * df["surprise_zscore"]
    )
    return df
