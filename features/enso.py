"""
ENSO Macro Signal (El Niño / La Niña)
======================================
Source: NOAA Oceanic Niño Index (ONI) — embedded as static data.
Historical ONI does not change for past periods.
To update: fetch from https://www.cpc.noaa.gov/data/indices/oni.ascii.txt

ONI > +0.5 = El Niño → tends toward warmer US winters → bearish NG
ONI < -0.5 = La Niña → tends toward colder US winters → bullish NG

Lead time: 3–6 months (ENSO peaks Dec-Feb, signal visible by Jun-Sep).
Used as a POSITION MULTIPLIER (regime filter).
"""

import pandas as pd
import numpy as np
import io


# Historical ONI (Oceanic Niño Index) — 3-month SST anomaly, Niño 3.4 region
# Format: year, jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
# Source: NOAA CPC (public domain)
_ONI_RAW = """
year,jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec
2009,-0.8,-0.9,-0.8,-0.6,-0.4,-0.1,0.2,0.4,0.6,0.9,1.1,1.3
2010,1.4,1.3,1.0,0.6,0.1,-0.4,-0.7,-1.0,-1.3,-1.5,-1.6,-1.6
2011,-1.4,-1.2,-0.9,-0.7,-0.6,-0.6,-0.7,-0.9,-1.0,-1.1,-1.1,-1.0
2012,-0.9,-0.7,-0.6,-0.4,-0.2,0.0,0.1,0.3,0.3,0.2,0.0,-0.2
2013,-0.4,-0.4,-0.2,-0.1,-0.2,-0.4,-0.5,-0.5,-0.3,-0.2,0.0,0.2
2014,0.4,0.5,0.3,0.3,0.4,0.3,0.2,0.1,0.2,0.4,0.6,0.8
2015,0.6,0.7,0.8,0.9,1.1,1.4,1.6,1.8,2.0,2.2,2.4,2.6
2016,2.5,2.2,1.7,1.0,0.5,0.0,-0.3,-0.5,-0.6,-0.6,-0.7,-0.7
2017,-0.7,-0.6,-0.3,-0.1,0.1,0.2,0.2,0.1,-0.1,-0.4,-0.7,-0.9
2018,-0.9,-0.8,-0.6,-0.4,-0.2,0.0,0.1,0.2,0.5,0.8,0.9,0.9
2019,0.8,0.8,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.3,0.4,0.5
2020,0.5,0.4,0.3,0.1,-0.2,-0.4,-0.6,-0.8,-1.0,-1.2,-1.3,-1.3
2021,-1.2,-1.1,-0.9,-0.7,-0.5,-0.4,-0.4,-0.5,-0.7,-0.9,-1.0,-1.0
2022,-1.0,-0.9,-0.9,-0.8,-0.7,-0.6,-0.7,-0.9,-1.0,-1.0,-0.9,-0.8
2023,-0.5,-0.2,0.1,0.5,0.8,1.0,1.2,1.5,1.8,1.9,2.0,2.0
2024,1.9,1.6,1.2,0.8,0.5,0.2,-0.1,-0.3,-0.4,-0.5,-0.6,-0.6
2025,-0.5,-0.4,-0.2,-0.1,0.0,0.1,0.1,0.1,0.2,0.3,0.3,0.3
"""

WINTER_SEASONS = {"SON", "OND", "NDJ", "DJF", "JFM", "FMA"}


def _build_oni_series() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_ONI_RAW.strip()))
    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

    rows = []
    for _, row in df.iterrows():
        yr = int(row["year"])
        for i, m in enumerate(months, 1):
            rows.append({
                "date": pd.Timestamp(f"{yr}-{i:02d}-01"),
                "oni": float(row[m]),
            })

    out = pd.DataFrame(rows).set_index("date").sort_index()

    # Assign 3-month season label
    season_map = {1:"DJF",2:"JFM",3:"FMA",4:"MAM",5:"AMJ",6:"MJJ",
                  7:"JJA",8:"JAS",9:"ASO",10:"SON",11:"OND",12:"NDJ"}
    out["season"] = out.index.month.map(season_map)
    return out


def fetch_oni(cache_path: str = "data/cache/oni.parquet") -> pd.DataFrame:
    import os
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = _build_oni_series()
    df.to_parquet(cache_path)
    return df


def add_enso_signal(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join ONI signal onto daily dataframe.
    Adds: oni, enso_regime, enso_multiplier.
    """
    oni = fetch_oni()

    # Forward-fill ONI to daily
    oni_daily = oni[["oni", "season"]].resample("D").ffill()

    df = daily_df.copy()
    df = df.join(oni_daily, how="left")
    df["oni"]    = df["oni"].ffill().fillna(0)
    df["season"] = df["season"].ffill().fillna("DJF")

    threshold = 0.5
    df["enso_regime"] = np.where(
        df["oni"] <= -threshold,  1.0,   # La Niña → cold → bullish NG
        np.where(df["oni"] >= threshold, -1.0, 0.0)   # El Niño → warm → bearish
    )

    df["in_winter_season"] = df["season"].isin(WINTER_SEASONS).astype(float)

    # Multiplier: ±20% position size based on ENSO regime in winter
    df["enso_multiplier"] = 1.0 + (df["enso_regime"] * df["in_winter_season"] * 0.2)

    return df
