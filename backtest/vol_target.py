"""
Volatility-Targeted Position Sizing
=====================================
Instead of a fixed ±1 position, scale exposure so the strategy
always targets a constant annual volatility (default 15%).

    position[t] = signal_direction[t] × (vol_target / realized_vol[t-1])

realized_vol = rolling 20-day annualized std of NG daily returns
Clipped at [0, 2×] to avoid absurd leverage in low-vol periods.

Effect: in crisis periods (NG 2022: 200% vol), exposure drops automatically.
        In quiet periods, exposure can rise above 1.0 (leveraged).
"""

import pandas as pd
import numpy as np


VOL_TARGET = 0.15          # 15% annual target vol
VOL_LOOKBACK = 20          # days for realized vol estimate
MAX_LEVERAGE = 2.0         # cap on position size
MIN_VOL = 0.05             # floor to avoid division near zero


def apply_vol_targeting(df: pd.DataFrame,
                        vol_target: float = VOL_TARGET,
                        lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    """
    Add vol-targeted position column `position_vt` to df.
    df must have: position (raw), natgas_ret.
    """
    df = df.copy()

    # Realized vol of the underlying (annualized)
    rv = df["natgas_ret"].rolling(lookback).std() * np.sqrt(252)
    rv = rv.shift(1).clip(lower=MIN_VOL)   # lagged 1d, no look-ahead

    # Scale factor: how much to multiply raw position
    scale = (vol_target / rv).clip(upper=MAX_LEVERAGE)

    # Apply scaling to raw signal direction only (preserve sign and threshold logic)
    df["realized_vol"] = rv
    df["vol_scale"]    = scale
    df["position_vt"]  = df["position"] * scale

    # Vol-targeted strategy return
    df["strat_ret_vt"]     = df["position_vt"] * df["natgas_ret"]
    df["turnover_vt"]      = df["position_vt"].diff().abs()
    df["cost_vt"]          = df["turnover_vt"] * 0.001
    df["strat_ret_vt_net"] = df["strat_ret_vt"] - df["cost_vt"]

    return df
