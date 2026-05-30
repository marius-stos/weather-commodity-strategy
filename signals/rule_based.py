"""
Rule-based signal pipeline (v3 + Vol-Targeting) — the production baseline.

Pipeline:
  1. HDD/CDD surprise (rate of change vs seasonality)
  2. ENSO macro multiplier
  3. EIA storage + production + LNG
  4. Satellite renewable deficit (wind + solar)
  5. Blend → vol-targeted position
"""

import pandas as pd
import numpy as np

from features.surprise     import add_surprise_signal
from features.cdd_summer   import add_cdd_signal, add_seasonal_signal
from features.enso         import add_enso_signal
from data.fetch_eia        import add_storage, add_fundamental
from data.fetch_satellite  import compute_renewable_features
from backtest.vol_target   import apply_vol_targeting
from config                import (W_WEATHER, W_STORAGE, W_PRODUCTION,
                                   W_SATELLITE, THRESHOLD, TRANSACTION_COST)


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

    # ── Weather features ──────────────────────────────────────────────
    df = add_surprise_signal(df)
    df = add_cdd_signal(df)
    df = add_seasonal_signal(df)

    # ── Macro ─────────────────────────────────────────────────────────
    df = add_enso_signal(df)

    # ── Fundamental ───────────────────────────────────────────────────
    df = add_storage(df)
    df = add_fundamental(df)

    # ── Satellite ─────────────────────────────────────────────────────
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

    # ── Blend signals ─────────────────────────────────────────────────
    df["storage_z"]    = _zscore(df["storage_signal"].fillna(0))
    df["production_z"] = _zscore(df["production_signal"].fillna(0))
    df["satellite_z"]  = _zscore(df["renewable_deficit_z"].fillna(0))

    # Weighted blend (satellite gets remaining weight after main 3)
    w_sat = max(0, 1.0 - W_WEATHER - W_STORAGE - W_PRODUCTION)
    df["blended_z"] = (
        W_WEATHER    * df["seasonal_signal"].fillna(0) +
        W_STORAGE    * df["storage_z"].fillna(0) +
        W_PRODUCTION * df["production_z"].fillna(0) +
        w_sat        * df["satellite_z"].fillna(0)
    )

    # ENSO multiplier (regime-aware scaling)
    df["final_signal"] = df["blended_z"] * df["enso_multiplier"].fillna(1.0)

    # Position
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

    return df.dropna(subset=["strat_ret_net", "final_signal"])
