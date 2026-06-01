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

Weekly rebalancing (default ON):
  Position is only updated on REBAL_WEEKDAY (Monday by default).
  Reduces round-trip costs from ~200/year to ~52/year.
  Same vol-targeting logic, but direction only changes once a week.
"""

import pandas as pd
import numpy as np


VOL_TARGET = 0.15          # 15% annual target vol
VOL_LOOKBACK = 20          # days for realized vol estimate
MAX_LEVERAGE = 2.0         # cap on position size
MIN_VOL = 0.05             # floor to avoid division near zero


def apply_vol_targeting(df: pd.DataFrame,
                        vol_target: float = VOL_TARGET,
                        lookback: int = VOL_LOOKBACK,
                        weekly: bool = False) -> pd.DataFrame:
    """
    Add vol-targeted position column `position_vt` to df.
    df must have: position (raw), natgas_ret.

    weekly=False : position updates every day (default, best for rule-based
                   signals with daily IC like HDD/CDD surprise).
    weekly=True  : position only updates on REBAL_WEEKDAY (default Monday).
                   Use for ML signals that target weekly forward returns.
    """
    from config import REBAL_WEEKDAY

    df = df.copy()

    # Realized vol of the underlying (annualized)
    rv = df["natgas_ret"].rolling(lookback).std() * np.sqrt(252)
    rv = rv.shift(1).clip(lower=MIN_VOL)   # lagged 1d, no look-ahead

    # Scale factor: how much to multiply raw position
    scale = (vol_target / rv).clip(upper=MAX_LEVERAGE)

    df["realized_vol"] = rv
    df["vol_scale"]    = scale
    df["position_vt"]  = df["position"] * scale

    # ── Weekly rebalancing: freeze position between rebal days ────────
    if weekly:
        rebal_mask = df.index.dayofweek == REBAL_WEEKDAY
        df["position_vt"] = (
            df["position_vt"]
            .where(rebal_mask, other=np.nan)
            .ffill()
        )

    # Vol-targeted strategy return
    df["strat_ret_vt"]     = df["position_vt"] * df["natgas_ret"]
    df["turnover_vt"]      = df["position_vt"].diff().abs()
    df["cost_vt"]          = df["turnover_vt"] * 0.001
    df["strat_ret_vt_net"] = df["strat_ret_vt"] - df["cost_vt"]

    return df
