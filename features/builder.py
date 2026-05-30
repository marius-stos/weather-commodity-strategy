"""
Feature builder — assembles all data sources into one DataFrame
for ML training. Every feature is computed look-ahead free.

Groups:
  1. Temperature (HDD/CDD levels + surprises)
  2. Satellite (wind/solar renewable deficit)
  3. ENSO macro
  4. EIA fundamental (storage + production + LNG)
  5. Market (momentum, realized vol, vol-of-vol)
  6. Seasonality (sin/cos encoding)
"""

import pandas as pd
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────

def _rolling_zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mu  = s.rolling(window, min_periods=60).mean()
    sig = s.rolling(window, min_periods=60).std()
    return (s - mu) / sig.clip(lower=1e-6)


def _sin_cos_doy(index: pd.DatetimeIndex) -> pd.DataFrame:
    doy = index.dayofyear
    return pd.DataFrame({
        "sin_doy":   np.sin(2 * np.pi * doy   / 365.25),
        "cos_doy":   np.cos(2 * np.pi * doy   / 365.25),
        "sin_month": np.sin(2 * np.pi * index.month / 12),
        "cos_month": np.cos(2 * np.pi * index.month / 12),
    }, index=index)


# ── Feature groups ──────────────────────────────────────────────────────

def temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["hdd_zscore"]     = df.get("HDD_zscore",     0)
    f["hdd_30d"]        = df.get("HDD_30d",         0)
    f["cdd_zscore"]     = df.get("CDD_zscore",      0)
    f["avg_temp_f"]     = df.get("avg_temp_f",       0)
    f["hdd_delta_5d"]   = df["HDD_zscore"].diff(5)  if "HDD_zscore" in df else 0
    f["hdd_delta_10d"]  = df["HDD_zscore"].diff(10) if "HDD_zscore" in df else 0
    f["hdd_delta_21d"]  = df["HDD_zscore"].diff(21) if "HDD_zscore" in df else 0
    f["cdd_delta_7d"]   = df.get("cdd_surprise_z", 0)
    f["seasonal_sig"]   = df.get("seasonal_signal", 0)
    # Non-linear: cold AND getting colder → amplified demand
    f["hdd_accel"]      = f["hdd_zscore"] * f["hdd_delta_5d"]
    return f


def satellite_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["wind_ms"]            = df.get("wind_ms",            np.nan)
    f["solar_kwh"]          = df.get("solar_kwh",          np.nan)
    f["wind_deficit_z"]     = df.get("wind_deficit_7d_z",  0)
    f["solar_deficit_z"]    = df.get("solar_deficit_7d_z", 0)
    f["renewable_deficit_z"]= df.get("renewable_deficit_z",0)
    # Interaction: cold snap + renewable deficit = maximum demand
    f["power_demand_x_hdd"] = f["renewable_deficit_z"] * df.get("HDD_zscore", 0)
    return f


def enso_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["oni"]           = df.get("oni",           0)
    f["enso_regime"]   = df.get("enso_regime",   0)
    f["enso_x_season"] = df.get("enso_regime", 0) * df.get("w_heat", 0.5)
    # PDO — if available
    f["pdo"]           = df.get("pdo", 0)
    return f


def fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["storage_surprise_z"]    = df.get("storage_surprise_z",    0)
    f["storage_signal"]        = df.get("storage_signal",        0)
    f["production_surprise_z"] = df.get("production_surprise_z", 0)
    f["lng_surprise_z"]        = df.get("lng_surprise_z",        0)
    # Net supply signal: production ↑ + LNG ↑ = bearish
    f["net_supply_signal"] = (
        -df.get("production_surprise_z", 0) * 0.6
        + df.get("lng_surprise_z",        0) * 0.4
    )
    return f


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["natgas_ret"]
    f = pd.DataFrame(index=df.index)
    f["ret_5d"]          = ret.rolling(5).sum()
    f["ret_10d"]         = ret.rolling(10).sum()
    f["ret_21d"]         = ret.rolling(21).sum()
    f["ret_63d"]         = ret.rolling(63).sum()
    f["realized_vol_10"] = ret.rolling(10).std() * np.sqrt(252)
    f["realized_vol_20"] = ret.rolling(20).std() * np.sqrt(252)
    f["realized_vol_60"] = ret.rolling(60).std() * np.sqrt(252)
    f["vol_ratio"]       = f["realized_vol_20"] / (f["realized_vol_60"] + 1e-8)
    f["vol_of_vol"]      = f["realized_vol_20"].rolling(20).std()
    # Crude oil spread (NG vs CL correlation regime)
    if "crude" in df.columns:
        f["crude_ret_10d"] = df["crude"].pct_change(10)
    return f


def seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    f = _sin_cos_doy(df.index)
    f["heating_season"] = df.index.month.isin([10,11,12,1,2,3]).astype(float)
    f["cooling_season"] = df.index.month.isin([6,7,8]).astype(float)
    return f


# ── Master builder ──────────────────────────────────────────────────────

ALL_GROUPS = [
    ("temp",        temperature_features),
    ("satellite",   satellite_features),
    ("enso",        enso_features),
    ("fundamental", fundamental_features),
    ("market",      market_features),
    ("seasonality", seasonality_features),
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete feature matrix from the combined signal DataFrame.
    All NaN filled with 0 (missing data treated as no signal).
    """
    parts = []
    for name, fn in ALL_GROUPS:
        try:
            feat = fn(df)
            parts.append(feat)
        except Exception as e:
            print(f"  [WARN] feature group '{name}': {e}")

    X = pd.concat(parts, axis=1)
    X = X.fillna(0).astype(float)
    return X


FEATURE_COLS = None   # populated lazily after first call to build_feature_matrix
