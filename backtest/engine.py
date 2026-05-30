"""
Backtest engine: compute equity curve, drawdowns, and performance metrics.
"""

import pandas as pd
import numpy as np


def compute_metrics(returns: pd.Series, label: str = "Strategy") -> dict:
    ann = 252
    total   = (1 + returns).prod() - 1
    cagr    = (1 + total) ** (ann / len(returns)) - 1
    vol     = returns.std() * np.sqrt(ann)
    sharpe  = (returns.mean() * ann) / (returns.std() * np.sqrt(ann) + 1e-9)
    cum     = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd      = (cum - roll_max) / roll_max
    max_dd  = dd.min()
    calmar  = cagr / (-max_dd + 1e-9)
    win_rate = (returns > 0).mean()

    return {
        "label":    label,
        "total":    round(total * 100, 2),
        "cagr":     round(cagr * 100, 2),
        "vol":      round(vol * 100, 2),
        "sharpe":   round(sharpe, 3),
        "max_dd":   round(max_dd * 100, 2),
        "calmar":   round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
        "n_days":   len(returns),
    }


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (equity_df, strategy_metrics, buyhold_metrics)."""
    strat  = df["strat_ret_net"].fillna(0)
    bh     = df["natgas_ret"].fillna(0)

    equity = pd.DataFrame({
        "strategy":  (1 + strat).cumprod(),
        "buy_hold":  (1 + bh).cumprod(),
        "cash":      1.0,
    }, index=df.index)

    m_strat = compute_metrics(strat, "HDD Strategy")
    m_bh    = compute_metrics(bh,    "Buy & Hold NG")

    return equity, m_strat, m_bh


def print_report(m_strat: dict, m_bh: dict):
    labels = ["CAGR %", "Vol %", "Sharpe", "Max DD %", "Calmar", "Win Rate %"]
    keys   = ["cagr",   "vol",   "sharpe", "max_dd",   "calmar", "win_rate"]

    print("\n" + "="*52)
    print(f"{'Metric':<18} {'HDD Strategy':>15} {'Buy&Hold NG':>15}")
    print("="*52)
    for label, key in zip(labels, keys):
        print(f"{label:<18} {str(m_strat[key]):>15} {str(m_bh[key]):>15}")
    print("="*52)
    print(f"  Period: {m_strat['n_days']} trading days")
