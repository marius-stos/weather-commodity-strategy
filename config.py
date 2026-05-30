"""
Central configuration — all tuneable parameters in one place.
"""
from datetime import datetime

# ── Data ──────────────────────────────────────────────────────────────
DATA_START      = datetime(2010, 1, 1)
DATA_END        = datetime(2025, 12, 31)
CACHE_DIR       = "data/cache"

# US cities for temperature (population-weighted HDD/CDD)
TEMP_CITIES = {
    "Chicago":      (41.85, -87.65, 9.5),
    "New_York":     (40.71, -74.01, 20.1),
    "Detroit":      (42.33, -83.05, 4.4),
    "Minneapolis":  (44.98, -93.27, 3.6),
    "Pittsburgh":   (40.44, -79.99, 2.4),
    "Cleveland":    (41.50, -81.69, 2.1),
    "Boston":       (42.36, -71.06, 4.9),
    "Philadelphia": (39.95, -75.17, 6.1),
}

# US regions for satellite wind/solar (gas power generation weighted)
POWER_REGIONS = {
    "Texas_W":    (31.97, -99.90, 0.15),
    "Texas_E":    (30.27, -97.74, 0.10),
    "Georgia":    (33.45, -84.39, 0.07),
    "Florida":    (27.99, -81.73, 0.10),
    "PJM_Mid":    (39.95, -75.17, 0.08),
    "NewEngland": (42.36, -71.06, 0.07),
    "Midwest":    (41.85, -87.65, 0.06),
    "California": (36.78, -119.42, 0.10),
    "Arizona":    (33.45, -112.07, 0.07),
    "Northwest":  (47.61, -122.33, 0.05),
    "Heartland":  (39.10, -94.58,  0.08),  # Kansas City / wind belt
    "Gulf_Coast": (29.76, -95.37,  0.07),  # Houston gas hub
}

# ── Futures ───────────────────────────────────────────────────────────
PRICE_TICKERS = {
    "natgas": "NG=F",   # Natural Gas front-month
    "crude":  "CL=F",   # WTI Crude (benchmark correlation)
    "heat":   "HO=F",   # Heating Oil
}
TARGET_TICKER = "natgas"

# ── Signal ────────────────────────────────────────────────────────────
HDD_BASE_F      = 65.0
THRESHOLD       = 0.5       # z-score entry threshold (rule-based)
W_WEATHER       = 0.45      # weight: weather surprise
W_STORAGE       = 0.25      # weight: EIA storage surprise
W_PRODUCTION    = 0.15      # weight: EIA production surprise
W_SATELLITE     = 0.15      # weight: renewable deficit (wind+solar)

# ── Risk ──────────────────────────────────────────────────────────────
VOL_TARGET      = 0.15      # 15% annual target volatility
VOL_LOOKBACK    = 20        # days for realized vol estimate
MAX_LEVERAGE    = 2.0       # max position size (>1 = leveraged)
TRANSACTION_COST = 0.001    # 0.1% per unit position change

# ── ML ────────────────────────────────────────────────────────────────
ML_TRAIN_DAYS   = 1008      # 4 years training window
ML_TEST_DAYS    = 252       # 1 year test window
ML_VAL_FRAC     = 0.20      # validation fraction within train
ML_HORIZON      = 10        # forward return horizon (days)

LGBM_PARAMS = dict(
    n_estimators      = 400,
    learning_rate     = 0.03,
    max_depth         = 4,
    num_leaves        = 15,
    subsample         = 0.7,
    colsample_bytree  = 0.6,
    min_child_samples = 30,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

XGB_PARAMS = dict(
    n_estimators         = 300,
    learning_rate        = 0.03,
    max_depth            = 3,
    subsample            = 0.7,
    colsample_bytree     = 0.6,
    min_child_weight     = 10,
    reg_alpha            = 0.5,
    reg_lambda           = 2.0,
    gamma                = 0.1,
    random_state         = 42,
    n_jobs               = -1,
    early_stopping_rounds = 40,
    eval_metric          = "rmse",
)

# ── Dashboard ─────────────────────────────────────────────────────────
DASH_PORT       = 8050
DASH_THEME      = "DARKLY"   # Bootstrap theme
