"""
Rule-based signal pipeline (v4 + Vol-Targeting) — the production baseline.

Pipeline:
  1. HDD/CDD surprise (rate of change vs seasonality)
  2. Forecast vs actual surprise (persistence+seasonal proxy)
  3. ENSO + AO + PDO macro regime multiplier
  4. EIA storage + production + LNG
  5. Satellite renewable deficit (wind + solar)
  6. Blend → vol-targeted position
  7. EIA Thursday event overlay
"""

import pandas as pd
import numpy as np

from features.surprise          import add_surprise_signal
from features.cdd_summer        import add_cdd_signal, add_seasonal_signal
from features.enso              import add_enso_signal
from features.forecast_surprise import add_forecast_surprise  # computed, not blended
from data.fetch_eia             import add_storage, add_fundamental
from data.fetch_satellite       import compute_renewable_features
from data.fetch_ao_pdo          import add_ao_pdo_signal
from signals.event_signal       import add_eia_event_overlay
from backtest.vol_target        import apply_vol_targeting
from config                     import (W_WEATHER, W_AO, W_STORAGE, W_STORAGE_TREND,
                                        W_PRODUCTION, W_PRODUCTION_ST, W_SATELLITE,
                                        THRESHOLD, TRANSACTION_COST)


def _zscore(s: pd.Series, w: int = 252) -> pd.Series:
    mu = s.rolling(w, min_periods=60).mean()
    sd = s.rolling(w, min_periods=60).std()
    return (s - mu) / sd.clip(lower=1e-6)


def build_pipeline(weather: pd.DataFrame,
                   prices:  pd.DataFrame,
                   satellite: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assemble the full rule-based signal.
    Returns DataFrame ready for backtest (has strat_ret_net column).
    """
    df = prices.join(weather, how="inner").sort_index()

    # ── 1. Weather features ───────────────────────────────────────────
    df = add_surprise_signal(df)
    df = add_cdd_signal(df)
    df = add_seasonal_signal(df)

    # ── 2. Forecast vs actual surprise (NEW) ─────────────────────────
    df = add_forecast_surprise(df)

    # ── 3. Macro: ENSO + AO + PDO (NEW) ──────────────────────────────
    df = add_enso_signal(df)
    df = add_ao_pdo_signal(df)

    # ── 4. Fundamental ────────────────────────────────────────────────
    df = add_storage(df)
    df = add_fundamental(df)

    # ── 5. Satellite ──────────────────────────────────────────────────
    if satellite is not None:
        sat = compute_renewable_features(satellite)
        sat.index = sat.index.normalize()
        df = df.join(sat[["wind_deficit_7d_z", "solar_deficit_7d_z",
                           "renewable_deficit_z"]], how="left")
        for col in ["wind_deficit_7d_z", "solar_deficit_7d_z", "renewable_deficit_z"]:
            df[col] = df[col].ffill().fillna(0)
    else:
        for col in ["wind_deficit_7d_z", "solar_deficit_7d_z", "renewable_deficit_z"]:
            df[col] = 0.0

    # ── 6. Blend signals ──────────────────────────────────────────────
    df["storage_z"]    = _zscore(df["storage_signal"].fillna(0))
    df["production_z"] = _zscore(df["production_signal"].fillna(0))
    df["satellite_z"]  = _zscore(df["renewable_deficit_z"].fillna(0))
    ao_z               = df.get("ao_z", pd.Series(0.0, index=df.index)).fillna(0)

    # ── NEW: 4-week rolling storage trend ──────────────────────────────
    # Captures multi-week storage build/draw trajectory (IC(10d)≈0.026, orthogonal to weekly snap)
    storage_4w = df["storage_signal"].fillna(0).rolling(28, min_periods=7).mean()
    df["storage_4w_z"] = _zscore(storage_4w)

    # ── NEW: Short-term production z-score (21-day window) ─────────────
    # Captures recent production changes vs past 3 weeks (IC(5d)≈0.053, best single signal!)
    # Sign convention: positive = production deficit (bullish NG) — same as production_signal
    prod = df["production_signal"].fillna(0)
    mu_21 = prod.rolling(21, min_periods=7).mean()
    sd_21 = prod.rolling(21, min_periods=7).std().clip(lower=1e-6)
    df["production_st_z"] = ((prod - mu_21) / sd_21).clip(-3, 3)   # 3σ cap prevents outlier distortion

    # Weights: all components sum to 1.0 (satellite gets residual)
    w_sat = max(0, 1.0 - W_WEATHER - W_AO - W_STORAGE - W_STORAGE_TREND
                - W_PRODUCTION - W_PRODUCTION_ST)
    df["blended_z"] = (
        W_WEATHER       * df["seasonal_signal"].fillna(0) +
        W_AO            * ao_z +
        W_STORAGE       * df["storage_z"].fillna(0) +
        W_STORAGE_TREND * df["storage_4w_z"].fillna(0) +
        W_PRODUCTION    * df["production_z"].fillna(0) +
        W_PRODUCTION_ST * df["production_st_z"].fillna(0) +
        w_sat           * df["satellite_z"].fillna(0)
    )

    # Macro multiplier: ENSO (regime) × PDO boost (subtle)
    enso_mult = df["enso_multiplier"].fillna(1.0)
    pdo_boost = df.get("pdo_boost", pd.Series(1.0, index=df.index)).fillna(1.0)
    df["macro_multiplier"] = enso_mult * pdo_boost

    df["final_signal"] = df["blended_z"] * df["macro_multiplier"]

    # ── 7. Base position ──────────────────────────────────────────────
    raw_pos = np.where(
        df["final_signal"].abs() > THRESHOLD,
        np.sign(df["final_signal"]) * np.minimum(df["final_signal"].abs() / 1.5, 1.0),
        0.0,
    )
    df["position"] = pd.Series(raw_pos, index=df.index).shift(1)

    # Returns (fixed sizing)
    df["strat_ret"]     = df["position"] * df["natgas_ret"]
    df["turnover"]      = df["position"].diff().abs()
    df["cost"]          = df["turnover"] * TRANSACTION_COST
    df["strat_ret_net"] = df["strat_ret"] - df["cost"]

    # ── 8. EIA Thursday event overlay (NEW) ──────────────────────────
    df = add_eia_event_overlay(df)

    return df.dropna(subset=["strat_ret_net", "final_signal"])
