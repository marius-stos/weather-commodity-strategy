# ⛅ Weather Commodity Strategy

> A systematic trading strategy for Natural Gas futures driven by meteorological data, satellite imagery, macro climate indices, and ML.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Thesis

**40% of US natural gas demand is for electricity generation.**  
When it's cold AND wind/solar output is below normal, gas demand spikes — and prices follow.  
This strategy captures that signal before it's fully reflected in futures prices.

---

## Results (2014–2025, out-of-sample)

| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| Buy & Hold NG | −0.52% | 62.4% | 0.30 | −83.7% | −0.01 |
| **Rule-Based + Vol Target** | **+1.27%** | **4.2%** | **0.32** | **−9.8%** | **0.13** |
| Rule + EIA Event Overlay | +0.25% | 8.5% | 0.07 | −15.6% | 0.02 |
| ML Ensemble (LightGBM + XGBoost + Ridge) | −0.10% | 8.6% | 0.03 | −30.5% | −0.003 |

*All results are walk-forward validated (4yr train → 1yr test, rolling 11 folds).*  
*ML uses IC gate (wtd val_IC ≥ 0.04) + weekly rebalancing: 4 folds flat, 7 folds active.*  
*COVID 2020 and energy-crisis 2021 folds create regime breaks that hurt ML; rule-based is the production signal.*  
*EIA event overlay IC ≈ 0.026 (weather proxy) — needs real Bloomberg/Reuters survey consensus for 0.12–0.15 theoretical IC.*

---

## Data Sources

| Source | Data | Frequency | Access |
|---|---|---|---|
| [Open-Meteo](https://open-meteo.com) | Temperature (8 US cities) | Daily | Free, no key |
| [NASA POWER](https://power.larc.nasa.gov) | Wind speed, solar irradiance (12 US regions) | Daily | Free, no key |
| [EIA](https://www.eia.gov/opendata) | Storage, Production, LNG exports | Weekly/Monthly | Free, DEMO_KEY |
| [NOAA CPC](https://www.cpc.noaa.gov) | ENSO ONI + AO indices | Monthly | Free (embedded) |
| [NOAA ERDDAP](https://oceanview.pfeg.noaa.gov/erddap) | PDO index | Monthly | Free, no key |
| [Yahoo Finance](https://finance.yahoo.com) | NG=F, CL=F, HO=F futures | Daily | Free (yfinance) |

---

## Signal Architecture

```
Temperature HDD/CDD z-score    W=0.30  IC(1d)≈0.022
Arctic Oscillation (AO)        W=0.10  IC(1d)≈0.019  (additive, polar vortex)
EIA Storage level z-score      W=0.20  IC(5d)≈0.037
EIA Storage 4-week trend z     W=0.05  IC(10d)≈0.026  ← new
EIA Production 252d z-score    W=0.10  IC(5d)≈0.032
EIA Production 21d z-score ★  W=0.10  IC(5d)≈0.053  ← new, best signal
Satellite renewable deficit    W=0.15  (residual)
                                   │
                        × ENSO × PDO multiplier
                                   │
                            Blended z-score
                                   │
                       Volatility targeting (15% ann.)
                                   │
                        BASE POSITION [-2, +2]
                                   │
                     EIA Thursday event overlay
                     (enter Wed close, exit Fri)
                                   │
                            FINAL POSITION
```

---

## Quickstart

```bash
git clone https://github.com/marius-stos/weather-commodity-strategy
cd weather-commodity-strategy
pip install -r requirements.txt

# Full run (includes ML walk-forward, ~5 min)
python run_backtest.py

# Fast run (rule-based only, ~30 sec)
python run_backtest.py --fast

# Launch interactive dashboard
python dashboard/app.py
# → open http://localhost:8050
```

---

## Project Structure

```
weather-commodity-strategy/
├── config.py                  # All parameters
├── run_backtest.py            # Main entry point
│
├── data/
│   ├── fetch_weather.py       # Open-Meteo temperature
│   ├── fetch_prices.py        # yfinance futures
│   ├── fetch_satellite.py     # NASA POWER wind/solar
│   ├── fetch_eia.py           # EIA storage/production/LNG
│   └── fetch_ao_pdo.py        # NOAA AO + PDO climate indices (NEW)
│
├── features/
│   ├── surprise.py            # HDD/CDD rate-of-change signals
│   ├── cdd_summer.py          # Cooling season features
│   ├── enso.py                # ENSO macro regime
│   ├── forecast_surprise.py   # Persistence+seasonal proxy (diagnostic)
│   └── builder.py             # Full ML feature matrix (~32 features)
│
├── signals/
│   ├── rule_based.py          # Weighted z-score + vol targeting (v4)
│   ├── event_signal.py        # EIA Thursday event overlay (NEW)
│   └── ml_signal.py           # LightGBM + XGBoost + Ridge ensemble
│
├── backtest/
│   ├── engine.py              # Metrics, equity curves
│   ├── vol_target.py          # Volatility targeting (15% ann.)
│   └── walk_forward.py        # Parameter stability analysis
│
└── dashboard/
    └── app.py                 # Dash interactive dashboard (4 tabs)
```

---

## Dashboard Preview

4 interactive tabs:

| Tab | Content |
|---|---|
| 📈 Overview | Equity curves, annual returns, metrics table |
| 🌡 Signals | HDD/CDD surprise, satellite wind/solar, ENSO, EIA storage |
| 🤖 ML | Feature importances, IC by fold, ensemble weights |
| ⚠️ Risk | Drawdowns, rolling Sharpe, vol regime, monthly returns |

---

## Key Insights

1. **Temperature DELTA > temperature LEVEL**: markets price in absolute cold; the edge is in unexpected cold snaps (HDD rate-of-change, IC ≈ 0.04)
2. **Volatility targeting is critical**: NG realized vol swings from 20% to 200% (2022). Fixed position sizing destroys risk-adjusted returns; vol targeting cuts max DD from −51% to −9%
3. **Short-term production z-score (21d) is the best single signal**: IC(5d)=0.053 vs 0.032 for 252-day window (+65%). Captures week-over-week production changes vs recent 3-week trend.
4. **Satellite wind/solar**: when renewable deficit is high (low wind + clouds), gas demand for power rises — new alpha orthogonal to temperature
5. **ENSO + AO matters in winter**: La Niña → colder US winters → +20% position scaling; negative AO (polar vortex) adds directional bullish signal (additive, not a multiplier)
6. **High-vol periods are the BEST periods**: strategy earns +13% CAGR when RV > 2.5× mean (polar vortex events, supply crises). Vol targeting naturally scales position; don't cap it.
7. **ML needs regime-stable folds**: IC gate + weekly rebalancing rescued ML from −1.6% to −0.1%; COVID 2020 and energy-crisis 2021 are genuine regime breaks that no in-sample IC gate can filter

---

## Extensions

- **Other commodities**: Corn/Soy + ENSO (La Niña → drought → rally), OJ + Florida frost risk
- **Better ML signal**: ECMWF TIGGE forecast archives (academic access), actual weather vs forecast
- **Options strategy**: long NG straddles around EIA Thursday releases when HDD surprise is large
- **Live trading**: connect to Interactive Brokers API, refresh EIA data Thursday 10:30am ET

---

## License

MIT — use freely, no warranty. Not financial advice.
