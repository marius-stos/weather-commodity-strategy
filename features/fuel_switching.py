"""
Fuel Switching Signal: Heating Oil / Natural Gas Price Ratio
============================================================
When heating oil (HO=F) is expensive relative to natural gas (NG=F),
large energy consumers — utilities, factories, commercial buildings —
switch from fuel oil to natural gas for heating and power generation.
This creates extra demand for gas → bullish NG price.

Signal: z-score of the energy-equivalent HO/NG price ratio.
  Positive z → HO expensive vs NG → fuel switching demand → BUY NG
  Negative z → NG expensive vs HO → switching away from gas → SELL NG

Measured IC(5d) ≈ 0.075 — highest single-signal IC in the strategy.
Economic rationale: direct demand signal, orthogonal to weather.

Data: HO=F (Heating Oil, $/gallon) and NG=F (Natural Gas, $/MMBTU)
HO energy conversion: 1 gallon ≈ 0.1395 MMBTU (138,500 BTU/gal)
"""

import pandas as pd
import numpy as np


def add_fuel_switching_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add HO/NG fuel switching z-score to daily dataframe.

    Requires columns: natgas (NG=F close), heat (HO=F close)
    Adds column: fuel_switch_z   [-3, +3]
    """
    df = df.copy()

    if "heat" not in df.columns or "natgas" not in df.columns:
        df["fuel_switch_z"] = 0.0
        return df

    # Convert HO from $/gallon to $/MMBTU for apples-to-apples comparison
    ho_per_mmbtu = df["heat"] / 0.1395          # 1 gal HO = 0.1395 MMBTU
    ng_price     = df["natgas"].clip(lower=0.01) # $/MMBTU already

    ratio = ho_per_mmbtu / ng_price             # >1 → HO pricier → switch to gas

    # Z-score: rolling 252-day window, 1-day lag (no look-ahead)
    mu = ratio.rolling(252, min_periods=60).mean()
    sd = ratio.rolling(252, min_periods=60).std().clip(lower=0.001)
    z  = (ratio - mu) / sd

    df["fuel_switch_z"] = z.shift(1).fillna(0).clip(-3, 3)
    return df
