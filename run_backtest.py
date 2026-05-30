"""
Weather Commodity Strategy — Main Runner
=========================================
Usage:
  python run_backtest.py              # full run + save results
  python run_backtest.py --fast       # skip ML, rule-based only
  python run_backtest.py --dashboard  # run dashboard after backtest
"""

import sys, os, argparse, pickle
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from config                   import DATA_START, DATA_END, VOL_TARGET
from data.fetch_weather       import fetch_city_temps, compute_hdd_cdd, compute_hdd_anomaly
from data.fetch_prices        import fetch_prices
from data.fetch_satellite     import fetch_satellite_data
from data.fetch_eia           import add_storage, add_fundamental
from signals.rule_based       import build_pipeline
from signals.ml_signal        import run_walkforward
from backtest.vol_target      import apply_vol_targeting
from backtest.engine          import compute_metrics, print_report

CACHE = {
    "weather":   "data/cache/weather.parquet",
    "prices":    "data/cache/prices.parquet",
    "satellite": "data/cache/satellite.parquet",
}


def load(key, fetch_fn, *args):
    if os.path.exists(CACHE[key]):
        print(f"  Loading {key} from cache...")
        return pd.read_parquet(CACHE[key])
    print(f"  Fetching {key}...")
    df = fetch_fn(*args)
    df.to_parquet(CACHE[key])
    return df


def main(fast=False, launch_dashboard=False):
    print("\n" + "="*60)
    print("  Weather Commodity Strategy  |  Natural Gas (NG=F)")
    print("="*60 + "\n")

    # ── 1. Load data ───────────────────────────────────────────────
    if not os.path.exists(CACHE["weather"]):
        print("  Fetching weather...")
        temps   = fetch_city_temps(DATA_START, DATA_END)
        weather = compute_hdd_anomaly(compute_hdd_cdd(temps))
        weather.to_parquet(CACHE["weather"])
    else:
        print("  Loading weather from cache...")
        weather = pd.read_parquet(CACHE["weather"])

    if not os.path.exists(CACHE["prices"]):
        print("  Fetching prices...")
        prices = fetch_prices(str(DATA_START.date()), str(DATA_END.date()))
        prices.to_parquet(CACHE["prices"])
    else:
        print("  Loading prices from cache...")
        prices = pd.read_parquet(CACHE["prices"])

    satellite = fetch_satellite_data()   # uses cache automatically

    print(f"\n  Weather : {len(weather):,} days")
    print(f"  Prices  : {len(prices):,} days")
    print(f"  Satellite: {len(satellite):,} days | {satellite.columns.tolist()}")

    # ── 2. Rule-based strategy ────────────────────────────────────
    print("\n[1/3] Building rule-based signal...")
    df_rule    = build_pipeline(weather, prices, satellite=satellite)
    df_rule_vt = apply_vol_targeting(df_rule)

    # ── 3. ML strategy ────────────────────────────────────────────
    fold_stats, feat_imp = [], pd.Series(dtype=float)
    if not fast:
        print("\n[2/3] Running ML walk-forward (LightGBM + XGBoost + Ridge)...")
        df_ml, fold_stats, feat_imp = run_walkforward(df_rule)
    else:
        print("\n[2/3] Skipping ML (--fast mode)")
        df_ml = df_rule.copy()
        df_ml["ml_ret"] = df_ml["strat_ret_net"]   # use rule as placeholder

    # ── 4. Metrics ────────────────────────────────────────────────
    oos = "2014-01-01"
    def oos_ret(df, col):
        return df.loc[df.index >= oos, col].dropna()

    bh_ret   = df_rule["natgas_ret"]
    rule_ret  = df_rule_vt["strat_ret_vt_net"]
    ml_ret    = df_ml.get("strat_ret_net", rule_ret)

    m_bh   = compute_metrics(oos_ret(df_rule, "natgas_ret"),         "Buy & Hold NG")
    m_rule = compute_metrics(rule_ret.loc[rule_ret.index >= oos].dropna(), "Rule-Based + VolTgt")
    m_ml   = compute_metrics(ml_ret.loc[ml_ret.index >= oos].dropna(), "ML Ensemble")

    print("\n" + "="*60)
    print(f"{'Strategy':<25} {'CAGR%':>7} {'Vol%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7}")
    print("="*60)
    for m in [m_bh, m_rule, m_ml]:
        print(f"{m['label']:<25} {m['cagr']:>7.2f} {m['vol']:>7.2f} "
              f"{m['sharpe']:>7.3f} {m['max_dd']:>8.2f} {m['calmar']:>7.3f}")
    print("="*60)

    if feat_imp is not None and len(feat_imp) > 0:
        print("\nTop 10 features:")
        for f, v in feat_imp.head(10).items():
            bar = "█" * int(v * 300)
            print(f"  {f:<28} {v:.4f}  {bar}")

    # ── 5. Save for dashboard ─────────────────────────────────────
    print("\n[3/3] Saving results for dashboard...")
    os.makedirs("data/cache", exist_ok=True)

    dash_df = df_rule_vt.copy()
    dash_df["rule_ret"]      = rule_ret
    dash_df["ml_ret"]        = ml_ret
    dash_df["bh_ret"]        = bh_ret
    dash_df["rule_position"] = df_rule_vt.get("position_vt", df_rule["position"])
    if "ml_pred" in df_ml.columns:
        dash_df["ml_pred"] = df_ml["ml_pred"]
    if "ml_position_vt" in df_ml.columns:
        dash_df["ml_position"] = df_ml["ml_position_vt"]

    # Add satellite signals for dashboard display
    from data.fetch_satellite import compute_renewable_features
    sat_feat = compute_renewable_features(satellite)
    sat_feat.index = sat_feat.index.normalize()
    for col in ["wind_deficit_7d_z", "solar_deficit_7d_z", "renewable_deficit_z"]:
        if col in sat_feat.columns:
            dash_df[col] = sat_feat[col].reindex(dash_df.index).ffill()

    dash_df.to_parquet("data/cache/dashboard_data.parquet")
    pickle.dump(fold_stats, open("data/cache/fold_stats.pkl", "wb"))
    if feat_imp is not None and len(feat_imp) > 0:
        feat_imp.to_frame("importance").to_parquet("data/cache/feat_imp.parquet")

    print("  Saved → data/cache/dashboard_data.parquet")
    print("\n✓ Done. Run `python dashboard/app.py` to launch the dashboard.")

    if launch_dashboard:
        from dashboard.app import run_dashboard
        run_dashboard(dash_df, fold_stats, feat_imp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",      action="store_true", help="Skip ML, rule-based only")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard after run")
    parser.add_argument("--save-dashboard", action="store_true", help="(internal) same as default")
    args = parser.parse_args()
    main(fast=args.fast, launch_dashboard=args.dashboard)
