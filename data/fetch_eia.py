"""
EIA Natural Gas Data — Storage, Production, LNG Exports
=========================================================
All series via EIA v2 API (free, DEMO_KEY works for public series).

Series fetched:
  Storage    (weekly)  : NW2_EPG0_SWO_R48_BCF   — working underground storage
  Production (monthly) : N9070US2                 — dry gas production (MMcf)
  LNG exports(monthly) : N9133US2                 — LNG exports (MMcf)

Surprise signals:
  storage_surprise_z   : actual injection vs 5yr seasonal avg
  production_surprise_z: MoM production change vs trend
  lng_surprise_z       : MoM LNG export change vs trend (affects net supply)
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from config import CACHE_DIR


EIA_BASE   = "https://api.eia.gov/v2/seriesid"
EIA_KEY    = "DEMO_KEY"


def _eia_get(series_id: str, num: int = 900) -> pd.Series:
    url = f"{EIA_BASE}/{series_id}?api_key={EIA_KEY}&num={num}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()["response"]["data"]
            df = pd.DataFrame(data)[["period", "value"]].copy()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"]  = pd.to_datetime(df["period"])
            return df.set_index("date")["value"].sort_index()
        except Exception as e:
            print(f"    [WARN] EIA {series_id} attempt {attempt+1}: {e}")
            time.sleep(10)
    return pd.Series(dtype=float)


# ── Storage ────────────────────────────────────────────────────────────

def fetch_storage(cache: str = f"{CACHE_DIR}/eia_storage.parquet") -> pd.DataFrame:
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    print("  Fetching EIA storage...")
    raw = _eia_get("NG.NW2_EPG0_SWO_R48_BCF.W")
    df = raw.rename("storage").to_frame()
    df.index = pd.to_datetime(df.index)
    df["injection"]   = df["storage"].diff()
    df["week"]        = df.index.isocalendar().week.astype(int)
    df["expected"]    = df.groupby("week")["injection"].transform(
                            lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df["surprise_bcf"] = df["injection"] - df["expected"]
    mu  = df["surprise_bcf"].rolling(156, min_periods=52).mean()
    sig = df["surprise_bcf"].rolling(156, min_periods=52).std()
    df["storage_surprise_z"] = (df["surprise_bcf"] - mu) / sig.clip(lower=0.1)
    df["storage_signal"]     = -df["storage_surprise_z"]   # surplus → bearish
    df.to_parquet(cache)
    return df


def add_storage(daily_df: pd.DataFrame) -> pd.DataFrame:
    storage = fetch_storage()
    sig = storage[["storage_surprise_z", "storage_signal"]].shift(1)  # lag 1d
    sig_daily = sig.reindex(daily_df.index, method="ffill")
    df = daily_df.join(sig_daily, how="left")
    df[["storage_surprise_z", "storage_signal"]] = \
        df[["storage_surprise_z", "storage_signal"]].fillna(0)
    return df


# ── Production ─────────────────────────────────────────────────────────

def fetch_production(cache: str = f"{CACHE_DIR}/eia_production.parquet") -> pd.DataFrame:
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    print("  Fetching EIA production (monthly)...")
    raw = _eia_get("NG.N9070US2.M", num=300)
    if raw.empty:
        return pd.DataFrame()
    df = raw.rename("production_mmcf").to_frame()
    df["prod_yoy"]   = df["production_mmcf"].pct_change(12)  # YoY growth
    df["prod_trend"] = df["production_mmcf"].rolling(12).mean()
    df["prod_dev"]   = df["production_mmcf"] / df["prod_trend"] - 1   # deviation from trend
    mu  = df["prod_dev"].rolling(36, min_periods=12).mean()
    sig = df["prod_dev"].rolling(36, min_periods=12).std()
    df["production_surprise_z"] = (df["prod_dev"] - mu) / sig.clip(lower=0.01)
    # Bearish sign: above-trend production → more supply → short NG
    df["production_signal"] = -df["production_surprise_z"]
    df.to_parquet(cache)
    return df


def fetch_lng(cache: str = f"{CACHE_DIR}/eia_lng.parquet") -> pd.DataFrame:
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    print("  Fetching EIA LNG exports (monthly)...")
    raw = _eia_get("NG.N9133US2.M", num=300)
    if raw.empty:
        return pd.DataFrame()
    df = raw.rename("lng_mmcf").to_frame()
    df["lng_trend"]  = df["lng_mmcf"].rolling(12).mean()
    df["lng_dev"]    = df["lng_mmcf"] / df["lng_trend"].clip(lower=1) - 1
    mu  = df["lng_dev"].rolling(36, min_periods=12).mean()
    sig = df["lng_dev"].rolling(36, min_periods=12).std()
    df["lng_surprise_z"] = (df["lng_dev"] - mu) / sig.clip(lower=0.01)
    # Bullish: high LNG exports reduce domestic supply → upward price pressure
    df["lng_signal"] = df["lng_surprise_z"]
    df.to_parquet(cache)
    return df


def add_fundamental(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Join production and LNG signals onto daily df (forward-filled monthly → daily)."""
    df = daily_df.copy()

    prod = fetch_production()
    if not prod.empty:
        prod_daily = prod[["production_surprise_z", "production_signal"]].resample("D").ffill()
        df = df.join(prod_daily, how="left")
        df[["production_surprise_z","production_signal"]] = \
            df[["production_surprise_z","production_signal"]].ffill().fillna(0)
    else:
        df["production_surprise_z"] = 0.0
        df["production_signal"]     = 0.0

    lng = fetch_lng()
    if not lng.empty:
        lng_daily = lng[["lng_surprise_z", "lng_signal"]].resample("D").ffill()
        df = df.join(lng_daily, how="left")
        df[["lng_surprise_z","lng_signal"]] = \
            df[["lng_surprise_z","lng_signal"]].ffill().fillna(0)
    else:
        df["lng_surprise_z"] = 0.0
        df["lng_signal"]     = 0.0

    return df
