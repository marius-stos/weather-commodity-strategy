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

| Strategy | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Buy & Hold NG | −0.52% | 62.4% | 0.30 | −83.7% |
| **Rule-Based + Vol Target** | **+1.17%** | **4.7%** | **0.27** | **−12.1%** |
| Rule + EIA Event Overlay | +0.15% | 8.9% | 0.06 | −15.7% |
| ML Ensemble (LightGBM + XGBoost + Ridge) | −0.10% | 8.6% | 0.03 | −30.5% |

*All results are walk-forward validated (4yr train → 1yr test, rolling 11 folds).*  
*ML uses IC gate (wtd val_IC ≥ 0.04) + weekly rebalancing: 4 folds flat (no signal), 7 folds active.*  
*EIA event overlay IC ≈ 0.026 — needs real Bloomberg/Reuters survey consensus to realise 0.12–0.15 theoretical IC.*  
*COVID 2020 and energy-crisis 2021 folds hurt ML (regime breaks); rule-based is the production signal.*

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
Temperature (HDD/CDD)      ──┐
  └ Surprise z-score (W=0.35)│
                              │
Arctic Oscillation (AO)    ──┤  (additive directional z, W=0.10)
  └ Polar vortex signal       │   negative AO in winter → bullish NG
                              ├── Weighted blend
Satellite (wind + solar)    ──┤   × ENSO × PDO multiplier
  └ Renewable deficit z       │
                              ├──► Blended z-score
EIA Storage surprise     (0.25)│
EIA Production surprise  (0.15)│
                              ┘
                              │
                              ▼
                    Volatility targeting (15% ann.)
                              │
                              ▼
                     BASE POSITION [-2, +2]
                              │
                    EIA Thursday event overlay
                    (enter Wed close, exit Fri)
                              │
                              ▼
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
3. **Satellite wind/solar**: when renewable deficit is high (low wind + clouds), gas demand for power rises — new alpha orthogonal to temperature
4. **ENSO + AO matters in winter**: La Niña → colder US winters → +20% position scaling; negative AO (polar vortex) adds directional bullish signal (NOT a multiplier — AO is additive in the blend)
5. **EIA event: need real survey data**: the theoretical IC of 0.12–0.15 (EIA Thursday release) requires Bloomberg/Reuters consensus estimates; our weather proxy achieves IC ≈ 0.03
6. **ML needs IC > 0.05 to beat costs**: IC gate + weekly rebalancing rescued ML from −1.6% to −0.1%; production/LNG data + actual forecast archives would close the gap

---

## Extensions

- **Other commodities**: Corn/Soy + ENSO (La Niña → drought → rally), OJ + Florida frost risk
- **Better ML signal**: ECMWF TIGGE forecast archives (academic access), actual weather vs forecast
- **Options strategy**: long NG straddles around EIA Thursday releases when HDD surprise is large
- **Live trading**: connect to Interactive Brokers API, refresh EIA data Thursday 10:30am ET

---

## License

MIT — use freely, no warranty. Not financial advice.
