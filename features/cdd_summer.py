"""
CDD Summer Signal (Cooling Degree Days)
=========================================
Mirror of the HDD surprise signal, applied to summer months (Apr–Sep).
CDD captures electricity demand for air conditioning → affects nat gas
consumed by gas-fired power plants (now ~40% of US electricity).

CDD_surprise = CDD_zscore[t] - CDD_zscore[t-7]   (rate of change)

Positive surprise = hotter than expected → more power demand → bullish NG
Negative surprise = cooler than expected → bearish NG

Combined seasonal signal:
  - Oct–Mar: HDD surprise dominant
  - Apr–Sep: CDD surprise dominant
  - Shoulder months: blend
"""

import pandas as pd
import numpy as np


def add_cdd_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Compute CDD_30d if not already in df (cache may only have CDD_7d)
    if "CDD_30d" not in df.columns:
        df["CDD_30d"] = df["CDD"].rolling(30).sum()

    # CDD z-score (analogous to HDD_zscore already computed)
    df["month"] = df.index.month
    df["day"]   = df.index.day

    stats = (
        df.groupby(["month", "day"])["CDD_30d"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "cdd_mean", "std": "cdd_std"})
    )
    df = df.join(stats, on=["month", "day"])
    df["CDD_zscore"] = (df["CDD_30d"] - df["cdd_mean"]) / df["cdd_std"].clip(lower=0.1)

    # CDD surprise = rate of change over 7 days
    df["cdd_delta_7d"] = df["CDD_zscore"].diff(7)

    mu  = df["cdd_delta_7d"].rolling(252, min_periods=60).mean()
    sig = df["cdd_delta_7d"].rolling(252, min_periods=60).std()
    df["cdd_surprise_z"] = (df["cdd_delta_7d"] - mu) / sig.clip(lower=0.01)

    df.drop(columns=["month", "day", "cdd_mean", "cdd_std"], inplace=True)
    return df


def add_seasonal_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend HDD (winter) and CDD (summer) surprises into one seasonal signal.
    Transition months: smooth blend over Apr and Oct.
    """
    df = df.copy()
    m = df.index.month

    # Heating weight: 1.0 in Nov-Mar, 0.0 in Jun-Aug, smooth transition
    w_heat = pd.Series(
        np.select(
            [m.isin([11, 12, 1, 2, 3]), m.isin([6, 7, 8])],
            [1.0, 0.0],
            default=0.4,   # Apr, May, Sep, Oct = shoulder
        ),
        index=df.index,
    )
    w_cool = 1.0 - w_heat

    hdd_sig = df.get("surprise_zscore", pd.Series(0, index=df.index))
    cdd_sig = df.get("cdd_surprise_z",  pd.Series(0, index=df.index))

    df["seasonal_signal"] = w_heat * hdd_sig + w_cool * cdd_sig
    df["w_heat"] = w_heat
    df["w_cool"] = w_cool
    return df
