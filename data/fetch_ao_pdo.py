"""
Arctic Oscillation (AO) and Pacific Decadal Oscillation (PDO)
==============================================================
AO  — daily/monthly index from NOAA CPC
  Negative AO → polar vortex weakens → cold air spills south into US
  AO < -1 is a strong cold signal, especially Nov–Mar
  Source: https://www.cpc.ncep.noaa.gov/

PDO — monthly index from NOAA ERDDAP (PFEG)
  Negative PDO + La Niña (ENSO < 0) → cold US winters (compound signal)
  Source: https://oceanview.pfeg.noaa.gov/erddap/

Combined climate_multiplier:
  1.0  (neutral baseline)
  + ao_contribution  (negative AO in winter → adds 0.0–0.30)
  + pdo_enso_contribution (both negative in winter → adds 0.0–0.20)
"""

import pandas as pd
import numpy as np
import requests
import os
from config import CACHE_DIR

AO_CACHE  = f"{CACHE_DIR}/ao_index.parquet"
PDO_CACHE = f"{CACHE_DIR}/pdo_index.parquet"

AO_URL  = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink"
           "/daily_ao_index/monthly.ao.index.b50.current.ascii")
PDO_URL = ("https://oceanview.pfeg.noaa.gov/erddap/tabledap/cciea_OC_PDO.csv"
           "?time,PDO&time>=2009-01-01")


# ── Arctic Oscillation ─────────────────────────────────────────────────

def fetch_ao(cache: str = AO_CACHE) -> pd.Series:
    if os.path.exists(cache):
        return pd.read_parquet(cache).squeeze()
    print("  Fetching AO index (NOAA CPC)...")
    try:
        r = requests.get(AO_URL, timeout=20)
        r.raise_for_status()
        records = []
        for line in r.text.splitlines():
            parts = line.split()
            if len(parts) == 3:
                try:
                    year, month, val = int(parts[0]), int(parts[1]), float(parts[2])
                    records.append({"date": pd.Timestamp(year=year, month=month, day=1),
                                    "ao": val})
                except ValueError:
                    continue
        s = pd.DataFrame(records).set_index("date")["ao"].sort_index()
        s.to_frame("ao").to_parquet(cache)
        return s
    except Exception as e:
        print(f"  [WARN] AO fetch failed: {e} — AO signal disabled")
        return pd.Series(dtype=float)


# ── Pacific Decadal Oscillation ────────────────────────────────────────

def fetch_pdo(cache: str = PDO_CACHE) -> pd.Series:
    if os.path.exists(cache):
        return pd.read_parquet(cache).squeeze()
    print("  Fetching PDO index (NOAA ERDDAP)...")
    try:
        r = requests.get(PDO_URL, timeout=20)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        # Skip 2-line header: "time,PDO" and "UTC,Normalized"
        records = []
        for line in lines[2:]:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    date = pd.Timestamp(parts[0])
                    val  = float(parts[1])
                    records.append({"date": date, "pdo": val})
                except ValueError:
                    continue
        s = pd.DataFrame(records).set_index("date")["pdo"].sort_index()
        s.index = s.index.normalize().tz_localize(None)   # strip UTC tz
        s.to_frame("pdo").to_parquet(cache)
        return s
    except Exception as e:
        print(f"  [WARN] PDO fetch failed: {e} — PDO signal disabled")
        return pd.Series(dtype=float)


# ── Combined climate multiplier ────────────────────────────────────────

def add_ao_pdo_signal(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AO/PDO climate signals to daily_df — ADDITIVE components, not multipliers.

    Design rationale:
      Multiplicative approach (AO × existing_signal) was wrong: it would amplify
      bearish signals during polar vortex events (AO negative = cold = NG bullish,
      but a negative blended_z × 1.3 becomes MORE bearish — opposite of intent).

      Additive approach: AO contributes a DIRECTIONAL z-score to blended_z.
      Negative AO in winter → polar vortex → bullish NG → positive ao_z.

    Columns added:
      ao_index   : monthly AO (forward-filled, lagged 1 day)
      pdo_index  : monthly PDO (forward-filled, lagged 1 day)
      ao_z       : standardized bullish signal when AO strongly negative in winter
      pdo_boost  : small additional ENSO multiplier boost when PDO + ENSO both negative
    """
    df = daily_df.copy()

    ao  = fetch_ao()
    pdo = fetch_pdo()

    # ── Forward-fill monthly → daily (lag 1 day: no look-ahead) ──────
    if not ao.empty:
        ao_daily = ao.reindex(df.index, method="ffill").ffill().bfill().fillna(0.0)
    else:
        ao_daily = pd.Series(0.0, index=df.index)

    if not pdo.empty:
        pdo_daily = pdo.reindex(df.index, method="ffill").ffill().bfill().fillna(0.0)
    else:
        pdo_daily = pd.Series(0.0, index=df.index)

    df["ao_index"]  = ao_daily.shift(1).fillna(0.0)
    df["pdo_index"] = pdo_daily.shift(1).fillna(0.0)

    # ── Winter mask (Oct–Mar) ─────────────────────────────────────────
    month     = pd.Series(df.index.month, index=df.index)
    in_winter = month.isin([10, 11, 12, 1, 2, 3]).astype(float)

    # ── AO directional signal ─────────────────────────────────────────
    # Negative AO in winter → polar vortex breakdown → NG bullish
    # raw = (-AO) × in_winter, clipped at 0 (only negative AO contributes)
    ao_raw = (-df["ao_index"]).clip(lower=0.0) * in_winter

    # Normalize to z-score (rolling 3yr to capture typical AO range)
    mu  = ao_raw.rolling(756, min_periods=90).mean()
    sig = ao_raw.rolling(756, min_periods=90).std().clip(lower=0.1)
    df["ao_z"] = (ao_raw - mu) / sig

    # ── PDO boost: subtle positive bias on ENSO multiplier ───────────
    enso      = df.get("oni", pd.Series(0.0, index=df.index))
    pdo_neg   = (df["pdo_index"] < -0.5).astype(float)
    enso_neg  = (enso < -0.3).astype(float)
    # +5% to ENSO multiplier when BOTH PDO and ENSO are negative in winter
    df["pdo_boost"] = 1.0 + pdo_neg * enso_neg * 0.05 * in_winter

    return df
