"""
Fetch commodity futures prices via yfinance.
Natural Gas front-month (NG=F) is the primary target.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime


TICKERS = {
    "natgas": "NG=F",    # Natural Gas front-month
    "crude":  "CL=F",    # WTI Crude (correlated benchmark)
    "heat":   "HO=F",    # Heating Oil
}


def fetch_prices(start: str = "2010-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    frames = {}
    for name, ticker in TICKERS.items():
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  [WARN] No data for {ticker}")
            continue
        close = raw["Close"].squeeze()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        frames[name] = close
        print(f"  {name} ({ticker}): {len(close)} days, {close.index[0].date()} → {close.index[-1].date()}")

    df = pd.DataFrame(frames)
    df["natgas_ret"] = df["natgas"].pct_change()
    df["natgas_fwd1d"] = df["natgas_ret"].shift(-1)   # 1-day ahead return (target)
    df["natgas_fwd5d"] = df["natgas"].pct_change(5).shift(-5)  # 5-day ahead
    return df


if __name__ == "__main__":
    print("Fetching commodity prices...")
    prices = fetch_prices()
    prices.to_parquet("data/cache/prices.parquet")
    print(f"\nSaved {len(prices)} rows → data/cache/prices.parquet")
    print(prices.tail())
