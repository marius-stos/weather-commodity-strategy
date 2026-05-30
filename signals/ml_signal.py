"""
LightGBM + XGBoost + Ridge Ensemble — Walk-Forward Signal
===========================================================
Three-model ensemble optimized for financial time-series with low IC:

  1. LightGBM DART  — gradient boosting, dropout regularization
  2. XGBoost        — gradient boosting, different bias/variance tradeoff
  3. Ridge           — linear baseline, always stable

Ensemble weights per fold: proportional to in-sample IC (not Sharpe,
to avoid overfitting to a particular risk profile).

Walk-forward: 4yr train → 1yr test, rolling annually.
Target: 10-day forward return (less noise than 5d, more signal per obs).
Position: z-scored ensemble prediction → vol-targeted size.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model  import Ridge
from sklearn.preprocessing import RobustScaler
from scipy.stats           import spearmanr

from features.builder  import build_feature_matrix
from config            import ML_TRAIN_DAYS, ML_TEST_DAYS, ML_VAL_FRAC, \
                              ML_HORIZON, LGBM_PARAMS, XGB_PARAMS, \
                              VOL_TARGET, TRANSACTION_COST


def _target(df: pd.DataFrame, horizon: int = ML_HORIZON) -> pd.Series:
    return df["natgas_ret"].rolling(horizon).sum().shift(-horizon)


def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = ~np.isnan(actual)
    if mask.sum() < 10:
        return 0.0
    return spearmanr(pred[mask], actual[mask]).statistic


def _train_lgbm(X_tr, y_tr, X_vl, y_vl):
    params = {**LGBM_PARAMS, "boosting_type": "dart",
              "drop_rate": 0.1, "skip_drop": 0.5}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    return model


def _train_xgb(X_tr, y_tr, X_vl, y_vl):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return model


def _train_ridge(X_tr, y_tr):
    model = Ridge(alpha=10.0)
    model.fit(X_tr, y_tr)
    return model


def run_walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns:
      result_df  : original df + ml_pred, ml_position_vt, strat_ret_net
      fold_stats : list of per-fold dicts with IC, weights, feature importances
    """
    X_full = build_feature_matrix(df)
    y_full = _target(df)
    feat_cols = X_full.columns.tolist()

    combined = X_full.join(y_full.rename("y"), how="inner").dropna()
    n        = len(combined)

    all_preds  = {}
    fold_stats = []

    start = ML_TRAIN_DAYS
    fold  = 0
    while start + ML_TEST_DAYS <= n:
        fold += 1
        train_raw = combined.iloc[start - ML_TRAIN_DAYS : start]
        test_raw  = combined.iloc[start : start + ML_TEST_DAYS]

        val_cut = int(len(train_raw) * (1 - ML_VAL_FRAC))
        tr, vl  = train_raw.iloc[:val_cut], train_raw.iloc[val_cut:]

        X_tr, y_tr = tr[feat_cols].values,  tr["y"].values
        X_vl, y_vl = vl[feat_cols].values,  vl["y"].values
        X_te        = test_raw[feat_cols].values

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)
        X_te_s = scaler.transform(X_te)

        # Train three models
        lgbm_m  = _train_lgbm(X_tr_s, y_tr, X_vl_s, y_vl)
        xgb_m   = _train_xgb (X_tr_s, y_tr, X_vl_s, y_vl)
        ridge_m = _train_ridge(X_tr_s, y_tr)

        p_lgbm  = lgbm_m.predict(X_te_s)
        p_xgb   = xgb_m.predict(X_te_s)
        p_ridge = ridge_m.predict(X_te_s)

        # In-sample IC on validation set (for ensemble weighting)
        ic_lgbm  = max(_ic(lgbm_m.predict(X_vl_s),  y_vl), 0)
        ic_xgb   = max(_ic(xgb_m.predict(X_vl_s),   y_vl), 0)
        ic_ridge = max(_ic(ridge_m.predict(X_vl_s),  y_vl), 0)
        total_ic = ic_lgbm + ic_xgb + ic_ridge + 1e-9

        w_lgbm  = ic_lgbm  / total_ic
        w_xgb   = ic_xgb   / total_ic
        w_ridge = ic_ridge / total_ic

        # Minimum weight guarantee so no model is zeroed out
        w_lgbm, w_xgb, w_ridge = (
            max(w_lgbm, 0.15), max(w_xgb, 0.15), max(w_ridge, 0.15))
        total = w_lgbm + w_xgb + w_ridge
        w_lgbm /= total; w_xgb /= total; w_ridge /= total

        ensemble = w_lgbm * p_lgbm + w_xgb * p_xgb + w_ridge * p_ridge
        oos_ic   = _ic(ensemble, test_raw["y"].values)

        period = f"{test_raw.index[0].date()} → {test_raw.index[-1].date()}"
        print(f"  Fold {fold:02d} [{period}]  "
              f"w=({w_lgbm:.2f}/{w_xgb:.2f}/{w_ridge:.2f})  "
              f"val_IC=({ic_lgbm:.3f}/{ic_xgb:.3f}/{ic_ridge:.3f})  "
              f"OOS_IC={oos_ic:.4f}")

        for date, pred in zip(test_raw.index, ensemble):
            all_preds[date] = pred

        # Feature importances (lgbm as reference)
        try:
            fi = dict(zip(feat_cols, lgbm_m.feature_importances_))
        except Exception:
            fi = {}

        fold_stats.append({
            "fold":      fold,
            "period":    period,
            "w_lgbm":    round(w_lgbm, 3),
            "w_xgb":     round(w_xgb, 3),
            "w_ridge":   round(w_ridge, 3),
            "val_ic_lgbm": round(ic_lgbm, 4),
            "oos_ic":    round(oos_ic, 4),
            "feat_imp":  fi,
            "n_train":   len(train_raw),
        })

        start += ML_TEST_DAYS

    # ── Assemble output ────────────────────────────────────────────────
    pred_series = pd.Series(all_preds, name="ml_pred").sort_index()
    out = df.copy()
    out["ml_pred"] = pred_series

    # z-score prediction (rolling 252d)
    pred_std = out["ml_pred"].rolling(252, min_periods=60).std().shift(1).clip(lower=1e-8)
    raw_sig  = out["ml_pred"] / pred_std

    # Position: threshold at 0.4 (lower → more trades, relies on vol-targeting to control risk)
    threshold = 0.4
    raw_pos = np.where(
        raw_sig.abs() > threshold,
        np.sign(raw_sig) * np.minimum(raw_sig.abs() / 1.5, 1.0),
        0.0
    )
    out["ml_position"]     = pd.Series(raw_pos, index=out.index).shift(1)

    # Vol targeting
    rv = out["natgas_ret"].rolling(20).std() * np.sqrt(252)
    rv = rv.shift(1).clip(lower=0.05)
    out["ml_position_vt"]  = (out["ml_position"] * (VOL_TARGET / rv)).clip(-2, 2)

    # Returns
    out["ml_ret"]         = out["ml_position_vt"] * out["natgas_ret"]
    out["ml_turnover"]    = out["ml_position_vt"].diff().abs()
    out["ml_cost"]        = out["ml_turnover"] * TRANSACTION_COST
    out["strat_ret_net"]  = out["ml_ret"] - out["ml_cost"]

    # Aggregate feature importances across folds
    all_fi = {}
    for fs in fold_stats:
        for feat, imp in fs["feat_imp"].items():
            all_fi[feat] = all_fi.get(feat, 0) + imp
    avg_fi = pd.Series(all_fi) / len(fold_stats) if fold_stats else pd.Series()
    avg_fi = avg_fi.sort_values(ascending=False)

    return out, fold_stats, avg_fi
