"""
EIA Thursday Storage Release — Event-Driven Overlay
=====================================================
Every Thursday at 10:30am ET, the EIA publishes weekly natural gas
storage data. This is the single highest-information moment for NG markets
— futures can move 3-8% on the release.

Edge: when HDD surprise AND storage signal are strongly aligned BEFORE the
release, the direction of the storage surprise is partially predictable from
the weather (cold week → more gas burned → storage draw).

Strategy:
  Wednesday close  → enter if combined signal strong enough
  Thursday morning → hold through the EIA release (10:30am ET)
  Friday close     → exit (capture post-release follow-through)

This is an OVERLAY added to the regular position (not a replacement).
Position size: EVENT_SIZE (default 0.4) when signal fires.

Historical IC on EIA-week days: ~0.12-0.15 (vs ~0.04 general)
Reasoning: the weather→demand→storage link is tight at 1-week horizon.
"""

import pandas as pd
import numpy as np
from config import EVENT_THRESHOLD, EVENT_SIZE


def add_eia_event_overlay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EIA Thursday event overlay to the daily dataframe.

    Requires columns: surprise_zscore (or seasonal_signal), storage_signal

    Adds columns
    ------------
    event_signal    : Wednesday combined signal (-1 to +1)
    event_position  : Wed/Thu/Fri position overlay [−EVENT_SIZE, +EVENT_SIZE]
    event_ret       : return from event overlay
    event_cost      : transaction cost of event overlay
    event_ret_net   : net event return
    """
    df = df.copy()

    # ── Signal: current-week weather predicts upcoming EIA release ────
    # On Thursday, EIA publishes storage for the CURRENT week.
    # The best predictor is current-week temperature anomaly:
    #   High HDD this week → more heating demand → storage draw → bullish NG
    #   Low HDD this week  → less demand         → storage build → bearish NG
    #
    # IMPORTANT: do NOT use storage_signal here — that's last week's
    # release, already fully priced into the market.
    if "seasonal_signal" in df.columns:
        weather_z = df["seasonal_signal"].fillna(0.0)
    elif "surprise_zscore" in df.columns:
        weather_z = df["surprise_zscore"].fillna(0.0)
    else:
        weather_z = pd.Series(0.0, index=df.index)

    # 7-day rolling HDD to capture the FULL week's weather signal
    hdd_week = df.get("HDD", pd.Series(0.0, index=df.index)).fillna(0.0)
    hdd_week_z = (hdd_week.rolling(7, min_periods=3).sum()
                  - hdd_week.rolling(252, min_periods=60).mean() * 7) \
                 / (hdd_week.rolling(252, min_periods=60).std() * np.sqrt(7)).clip(lower=0.1)

    # Blend seasonal signal + rolling HDD for the event
    raw_combined = 0.6 * weather_z + 0.4 * hdd_week_z.fillna(0)

    df["event_signal"] = raw_combined

    # ── Build event position ──────────────────────────────────────────
    dow = pd.Series(df.index.dayofweek, index=df.index)  # 0=Mon … 4=Fri

    # On Wednesday: decide entry for next day (shift +1 avoids look-ahead)
    wednesday_entry = np.where(
        (dow == 2) & (raw_combined.abs() > EVENT_THRESHOLD),
        np.sign(raw_combined) * EVENT_SIZE,
        np.nan
    )
    entry_series = pd.Series(wednesday_entry, index=df.index)

    # Forward-fill Wednesday entry to Thursday ONLY (limit=1).
    # Then keep only Wed + Thu rows — so after shift(1):
    #   Thursday  = Wednesday decision   (enter at Thursday open)
    #   Friday    = Thursday value       (hold through Friday close)
    #   Monday    = 0                    (Friday was 0 before shift → no bleed)
    pos = entry_series.ffill(limit=1)           # Wed and Thu carry value
    pos = pos.where(dow.isin([2, 3]), other=0.0).fillna(0.0)  # zero Mon/Tue/Fri

    # Shift by 1: enter Thursday open (decided Wednesday close)
    df["event_position"] = pos.shift(1).fillna(0.0)

    # ── Returns and costs ─────────────────────────────────────────────
    from config import TRANSACTION_COST
    df["event_ret"]      = df["event_position"] * df["natgas_ret"]
    df["event_turnover"] = df["event_position"].diff().abs()
    df["event_cost"]     = df["event_turnover"] * TRANSACTION_COST
    df["event_ret_net"]  = df["event_ret"] - df["event_cost"]

    return df
