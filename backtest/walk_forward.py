"""
Walk-Forward Validation
========================
Simulates live trading: calibrate signal parameters on past data,
trade on the next out-of-sample period, roll forward.

Setup:
  - Train window : 3 years (756 trading days)
  - Test window  : 1 year  (252 trading days)
  - Step         : 1 year  (roll forward annually)

What gets optimized per window:
  - threshold  : z-score entry threshold (grid: 0.3 to 1.2)
  - w_weather  : weight on weather surprise (grid: 0.3 to 0.8)
  - w_storage  : weight on storage signal (remainder split with enso)

Optimization criterion: in-sample Sharpe ratio.

Output: concatenated out-of-sample periods → true out-of-sample equity curve.
"""

import pandas as pd
import numpy as np
from itertools import product


TRAIN_DAYS = 756   # ~3 years
TEST_DAYS  = 252   # ~1 year

# Param grid (small to stay fast)
THRESHOLDS  = [0.3, 0.5, 0.75, 1.0]
W_WEATHERS  = [0.3, 0.5, 0.7]
W_STORAGES  = [0.1, 0.2, 0.3]


def _sharpe(returns: pd.Series) -> float:
    if returns.std() < 1e-8 or len(returns) < 20:
        return -99.0
    return returns.mean() / returns.std() * np.sqrt(252)


def _simulate(df: pd.DataFrame,
              threshold: float,
              w_weather: float,
              w_storage: float) -> pd.Series:
    """Run one parameter set, return net daily returns."""
    w_enso = max(0, 1.0 - w_weather - w_storage)

    blended = (
        w_weather * df["seasonal_signal"].fillna(0) +
        w_storage * df["storage_z"].fillna(0)
    )
    final = blended * df["enso_multiplier"].fillna(1.0)

    raw_pos = np.where(
        final.abs() > threshold,
        np.sign(final) * np.minimum(final.abs() / 1.5, 1.0),
        0.0,
    )
    pos = pd.Series(raw_pos, index=df.index).shift(1)

    # Vol targeting inline
    rv = df["natgas_ret"].rolling(20).std() * np.sqrt(252)
    rv = rv.shift(1).clip(lower=0.05)
    pos_vt = (pos * (0.15 / rv)).clip(-2, 2)

    ret      = pos_vt * df["natgas_ret"]
    turnover = pos_vt.diff().abs()
    net      = ret - turnover * 0.001
    return net


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      oos_ret     : out-of-sample net return (vol-targeted)
      best_thresh : threshold used in this period
      best_ww     : weather weight used
      best_ws     : storage weight used
      in_sharpe   : best in-sample Sharpe achieved
    """
    df = df.sort_index().copy()
    n  = len(df)

    results = []
    start   = TRAIN_DAYS

    while start + TEST_DAYS <= n:
        train = df.iloc[start - TRAIN_DAYS : start]
        test  = df.iloc[start : start + TEST_DAYS]

        # Grid search on training window
        best_sharpe = -99.0
        best_params = (0.75, 0.5, 0.2)

        for thr, ww, ws in product(THRESHOLDS, W_WEATHERS, W_STORAGES):
            if ww + ws > 0.95:
                continue
            ret = _simulate(train, thr, ww, ws)
            sh  = _sharpe(ret.dropna())
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = (thr, ww, ws)

        # Apply best params to test period
        thr, ww, ws = best_params
        oos_ret = _simulate(test, thr, ww, ws)

        for date, r in oos_ret.items():
            results.append({
                "date":        date,
                "oos_ret":     r,
                "best_thresh": thr,
                "best_ww":     ww,
                "best_ws":     ws,
                "in_sharpe":   round(best_sharpe, 3),
            })

        start += TEST_DAYS
        period = f"{test.index[0].date()} → {test.index[-1].date()}"
        print(f"    [{period}] best params: thr={thr}, ww={ww}, ws={ws}  IS-Sharpe={best_sharpe:.2f}")

    return pd.DataFrame(results).set_index("date").sort_index()
