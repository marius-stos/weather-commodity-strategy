"""
Fetch daily temperature data for key US gas-consumption cities via Open-Meteo.
Computes population-weighted HDD (Heating Degree Days) and anomaly z-score.

HDD = max(0, 65°F - avg_temp_F)  ← heating demand proxy
"""

import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime


CITIES = {
    "Chicago":      (41.85, -87.65),
    "New_York":     (40.71, -74.01),
    "Detroit":      (42.33, -83.05),
    "Minneapolis":  (44.98, -93.27),
    "Pittsburgh":   (40.44, -79.99),
    "Cleveland":    (41.50, -81.69),
    "Boston":       (42.36, -71.06),
    "Philadelphia": (39.95, -75.17),
}

WEIGHTS = {
    "Chicago": 9.5, "New_York": 20.1, "Detroit": 4.4, "Minneapolis": 3.6,
    "Pittsburgh": 2.4, "Cleveland": 2.1, "Boston": 4.9, "Philadelphia": 6.1,
}
TOTAL_WEIGHT = sum(WEIGHTS.values())


def _get_client():
    session = requests_cache.CachedSession("data/cache/.openmeteo_cache", expire_after=3600)
    session = retry(session, retries=3, backoff_factor=0.5)
    return openmeteo_requests.Client(session=session)


def fetch_city_temps(start: datetime, end: datetime) -> pd.DataFrame:
    client = _get_client()
    frames = {}

    for city, (lat, lon) in CITIES.items():
        resp = client.weather_api(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":        lat,
                "longitude":       lon,
                "daily":           "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "start_date":      start.strftime("%Y-%m-%d"),
                "end_date":        end.strftime("%Y-%m-%d"),
                "timezone":        "America/New_York",
            }
        )
        r    = resp[0]
        daily = r.Daily()
        dates = pd.date_range(
            start=pd.Timestamp(daily.Time(), unit="s", tz="UTC").tz_localize(None),
            periods=daily.VariablesLength(),
            freq="D",
        )
        # Reindex dates to match actual data length
        n = len(daily.Variables(0).ValuesAsNumpy())
        dates = pd.date_range(
            start=pd.Timestamp(daily.Time(), unit="s").tz_localize(None),
            periods=n,
            freq="D",
        )
        temps = pd.Series(daily.Variables(0).ValuesAsNumpy(), index=dates, name=city)
        frames[city] = temps
        print(f"  {city}: {len(temps)} days")

    combined = pd.DataFrame(frames)
    combined.index = combined.index.normalize()  # strip time component → midnight
    combined["avg_temp_f"] = (combined * pd.Series(WEIGHTS)).sum(axis=1) / TOTAL_WEIGHT
    return combined[["avg_temp_f"]].dropna()


def compute_hdd_cdd(temps: pd.DataFrame, base_f: float = 65.0) -> pd.DataFrame:
    df = temps.copy()
    df["HDD"] = np.maximum(0, base_f - df["avg_temp_f"])
    df["CDD"] = np.maximum(0, df["avg_temp_f"] - base_f)
    df["HDD_7d"]  = df["HDD"].rolling(7).sum()
    df["HDD_30d"] = df["HDD"].rolling(30).sum()
    df["CDD_7d"]  = df["CDD"].rolling(7).sum()
    return df


def compute_hdd_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df.index.month
    df["day"]   = df.index.day

    stats = (
        df.groupby(["month", "day"])["HDD_30d"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "hdd_mean", "std": "hdd_std"})
    )

    df = df.join(stats, on=["month", "day"])
    df["HDD_zscore"] = (df["HDD_30d"] - df["hdd_mean"]) / df["hdd_std"].clip(lower=0.1)
    df.drop(columns=["month", "day", "hdd_mean", "hdd_std"], inplace=True)
    return df


if __name__ == "__main__":
    start = datetime(2010, 1, 1)
    end   = datetime(2025, 12, 31)
    print("Fetching temperature data via Open-Meteo...")
    temps = fetch_city_temps(start, end)
    df    = compute_hdd_cdd(temps)
    df    = compute_hdd_anomaly(df)
    df.to_parquet("data/cache/weather.parquet")
    print(f"\nSaved {len(df)} rows → data/cache/weather.parquet")
    print(df[["avg_temp_f", "HDD", "HDD_30d", "HDD_zscore"]].tail())
