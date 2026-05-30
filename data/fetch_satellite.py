"""
NASA POWER API — Satellite-derived climate data
================================================
Free, no API key, global coverage, daily from 1981.
Source: NASA Langley Research Center POWER Project
        https://power.larc.nasa.gov/

Variables fetched:
  T2M            — 2m air temperature (°C), satellite-corrected
  WS10M          — wind speed at 10m (m/s)
  ALLSKY_SFC_SW_DWN — downward solar irradiance (kWh/m²/day)
  PRECTOTCORR    — precipitation (mm/day)

Why this matters for nat gas:
  ~40% of US natural gas demand is for power generation.
  When wind + solar output is LOW, gas turbines must compensate.
  A cold, cloudy, windless day = maximum possible gas demand.
  This "renewable energy deficit" is orthogonal to HDD/CDD.
"""

import requests
import pandas as pd
import numpy as np
import time
import os


POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

VARIABLES = "T2M,ALLSKY_SFC_SW_DWN,WS10M,PRECTOTCORR"

# US regions weighted by natural gas power generation capacity
# (gas MW in region / total US gas MW)
POWER_CITIES = {
    # Texas (ERCOT) — largest gas power state ~25% of US
    "Texas_W":    (31.97, -99.90, 0.15),
    "Texas_E":    (30.27, -97.74, 0.10),
    # Southeast — 20% of US gas power
    "Georgia":    (33.45, -84.39, 0.07),
    "Florida":    (27.99, -81.73, 0.10),
    # Mid-Atlantic / NE — 18% of US gas power
    "PJM_Mid":   (39.95, -75.17, 0.08),
    "NewEngland": (42.36, -71.06, 0.07),
    # Midwest — 10%
    "Midwest":   (41.85, -87.65, 0.06),
    # CAISO (California) — 12%
    "California": (36.78, -119.42, 0.10),
    # Southwest — 7%
    "Arizona":   (33.45, -112.07, 0.07),
    # Northwest — 5%
    "Northwest": (47.61, -122.33, 0.05),  # hydro-heavy, gas backup
}
TOTAL_WEIGHT = sum(w for _, _, w in POWER_CITIES.values())


def _fetch_one(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch NASA POWER data for one location, one time range."""
    r = requests.get(
        POWER_URL,
        params={
            "parameters": VARIABLES,
            "community":  "RE",
            "latitude":   lat,
            "longitude":  lon,
            "start":      start,
            "end":        end,
            "format":     "JSON",
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.columns.name = None
    return df


def fetch_satellite_data(
    start: str = "20100101",
    end:   str = "20251231",
    cache_path: str = "data/cache/satellite.parquet",
) -> pd.DataFrame:
    """
    Fetch population-weighted satellite data across US power regions.
    Returns daily DataFrame with weighted-mean:
      T2M_sat, WS10M, SW_DWN, PRECIP
    """
    if os.path.exists(cache_path):
        print("  Loading satellite data from cache...")
        return pd.read_parquet(cache_path)

    frames = {}
    weights = {}
    for name, (lat, lon, w) in POWER_CITIES.items():
        print(f"  Fetching NASA POWER: {name} ({lat:.1f}, {lon:.1f})...")
        try:
            df = _fetch_one(lat, lon, start, end)
            frames[name] = df
            weights[name] = w
            time.sleep(0.5)   # polite rate limiting
        except Exception as e:
            print(f"    [WARN] {name}: {e}")

    # Weighted average across locations
    out = pd.DataFrame(index=next(iter(frames.values())).index)
    for var in ["T2M", "WS10M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]:
        total_w = sum(weights[n] for n in frames)
        out[var] = sum(
            frames[n][var] * weights[n] / total_w
            for n in frames if var in frames[n].columns
        )

    out = out.rename(columns={
        "T2M":               "T2M_sat",
        "WS10M":             "wind_ms",
        "ALLSKY_SFC_SW_DWN": "solar_kwh",
        "PRECTOTCORR":       "precip_mm",
    })

    # Replace fill values (-999) with NaN
    out = out.replace(-999.0, np.nan).ffill()

    out.to_parquet(cache_path)
    print(f"  Saved {len(out)} days of satellite data → {cache_path}")
    return out


def compute_renewable_features(sat: pd.DataFrame) -> pd.DataFrame:
    """
    Build 'renewable energy deficit' signal:
      - wind below seasonal normal → gas turbines must compensate
      - solar below seasonal normal → gas turbines must compensate
      Combined: cold + low wind + low sun = maximum gas demand day
    """
    df = sat.copy()
    df["month"] = df.index.month
    df["doy"]   = df.index.dayofyear

    # Seasonal normals (10yr climatology per day of year)
    for col in ["wind_ms", "solar_kwh"]:
        normal = df.groupby("doy")[col].transform("mean")
        df[f"{col}_norm"] = normal
        df[f"{col}_anom"] = df[col] - normal  # positive = more than normal

    # Wind deficit (rolling 7d): below-normal wind = gas demand ↑
    df["wind_deficit_7d"] = (-df["wind_ms_anom"]).rolling(7).mean()

    # Solar deficit (rolling 7d): below-normal sun = gas demand ↑
    df["solar_deficit_7d"] = (-df["solar_kwh_anom"]).rolling(7).mean()

    # Z-scores (rolling 252d for normalization)
    for col in ["wind_deficit_7d", "solar_deficit_7d"]:
        mu  = df[col].rolling(252, min_periods=60).mean()
        sig = df[col].rolling(252, min_periods=60).std()
        df[f"{col}_z"] = (df[col] - mu) / sig.clip(lower=0.01)

    # Combined renewable deficit signal (equal weight wind + solar)
    df["renewable_deficit_z"] = 0.5 * df["wind_deficit_7d_z"] + 0.5 * df["solar_deficit_7d_z"]

    df.drop(columns=["month", "doy"], inplace=True)
    return df
