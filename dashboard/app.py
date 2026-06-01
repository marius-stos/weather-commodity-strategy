"""
Weather Commodity Strategy — Interactive Dashboard
===================================================
Run:  python dashboard/app.py
Open: http://localhost:8050

Tabs:
  1. Overview   — equity curves, KPI table, current signal state
  2. Signals    — HDD/CDD, satellite wind/solar, ENSO, EIA storage
  3. ML         — feature importances, IC by fold, ensemble weights
  4. Risk       — drawdowns, rolling Sharpe, position history, vol regime
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc


# ── Load pre-computed results ──────────────────────────────────────────
def load_results():
    cache = "data/cache/dashboard_data.parquet"
    if not os.path.exists(cache):
        print("Running strategy pipeline to generate data...")
        import subprocess
        subprocess.run([sys.executable, "run_backtest.py", "--save-dashboard"], check=True)
    return pd.read_parquet(cache)


# ── Colour palette ─────────────────────────────────────────────────────
PAL = {
    "rule":    "#4CAF50",
    "ml":      "#2196F3",
    "bh":      "#FF5722",
    "hdd":     "#FF9800",
    "cdd":     "#03A9F4",
    "wind":    "#9C27B0",
    "solar":   "#FFC107",
    "storage": "#E91E63",
    "enso":    "#00BCD4",
    "bg":      "#1e1e2e",
    "card":    "#2a2a3e",
    "text":    "#e0e0e0",
}

METRIC_LABELS = {
    "cagr":     "CAGR %",
    "vol":      "Vol %",
    "sharpe":   "Sharpe",
    "max_dd":   "Max DD %",
    "calmar":   "Calmar",
    "win_rate": "Win Rate %",
}


# ── App init ───────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Weather Trading Strategy",
    suppress_callback_exceptions=True,
)


# ── Layout ─────────────────────────────────────────────────────────────

def kpi_card(title, value, color="#4CAF50"):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, style={"color": color, "fontWeight": "bold"}),
        ])
    ], style={"background": PAL["card"], "border": "none"})


def make_layout(df):
    # Compute metrics
    def equity(ret): return (1 + ret.fillna(0)).cumprod()

    eq_rule = equity(df["rule_ret"])
    eq_ml   = equity(df["ml_ret"])
    eq_bh   = equity(df["bh_ret"])

    # Current signal
    last = df.iloc[-1]
    cur_sig  = last.get("final_signal", 0)
    cur_pos  = last.get("rule_position", 0)
    sig_dir  = "🔺 LONG" if cur_pos > 0.1 else ("🔻 SHORT" if cur_pos < -0.1 else "⬛ FLAT")

    return dbc.Container([

        # ── Header ────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.H2("⛅ Weather Commodity Strategy — Natural Gas",
                            style={"color": PAL["text"]})),
            dbc.Col(html.P(f"Updated: {df.index[-1].date()}  |  "
                           f"Current position: {sig_dir}  ({cur_pos:+.2f})",
                           className="text-muted text-end mt-2")),
        ], className="my-3"),

        # ── Date range ────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Div(id="date-range-label",
                         style={"color": "#aaa", "fontSize": "0.85rem",
                                "marginBottom": "4px", "textAlign": "center"}),
                dcc.RangeSlider(
                    id="date-range",
                    min=0, max=len(df)-1,
                    value=[0, len(df)-1],
                    marks={
                        i: {"label": str(df.index[i].year),
                            "style": {"color": "#aaa", "fontSize": "0.8rem"}}
                        for i in range(0, len(df), max(1, len(df)//8))
                    },
                    tooltip={"always_visible": False},   # hide raw-index tooltip
                )
            ], width=12)
        ], className="mb-3"),

        # ── KPI cards ─────────────────────────────────────────────────
        html.Div(id="kpi-row"),

        # ── Tabs ──────────────────────────────────────────────────────
        dbc.Tabs([
            dbc.Tab(label="📈 Overview",  tab_id="tab-overview"),
            dbc.Tab(label="🌡 Signals",   tab_id="tab-signals"),
            dbc.Tab(label="🤖 ML",        tab_id="tab-ml"),
            dbc.Tab(label="⚠️ Risk",      tab_id="tab-risk"),
        ], id="tabs", active_tab="tab-overview", className="mb-3"),

        html.Div(id="tab-content"),

    ], fluid=True, style={"background": PAL["bg"], "minHeight": "100vh",
                          "padding": "20px"})


# ── Callbacks ─────────────────────────────────────────────────────────

def register_callbacks(app, df, fold_stats, feat_imp):

    @app.callback(
        Output("date-range-label", "children"),
        Input("date-range", "value")
    )
    def update_date_label(date_range):
        i0, i1 = date_range
        d0 = df.index[i0].strftime("%b %Y")
        d1 = df.index[i1].strftime("%b %Y")
        n_days = i1 - i0 + 1
        n_years = n_days / 252
        return f"📅  {d0}  →  {d1}  ({n_years:.1f} years)"

    @app.callback(
        [Output("kpi-row", "children"),
         Output("tab-content", "children")],
        [Input("tabs", "active_tab"),
         Input("date-range", "value")]
    )
    def update_content(active_tab, date_range):
        i0, i1 = date_range
        sub = df.iloc[i0:i1+1]

        # KPIs
        from backtest.engine import compute_metrics
        m_rule = compute_metrics(sub["rule_ret"].dropna(), "Rule-based")
        m_ml   = compute_metrics(sub["ml_ret"].dropna(),   "ML Ensemble")

        kpis = dbc.Row([
            dbc.Col(kpi_card("Rule-Based CAGR", f"{m_rule['cagr']:+.1f}%", PAL["rule"]), width=2),
            dbc.Col(kpi_card("Rule-Based Sharpe", f"{m_rule['sharpe']:.3f}", PAL["rule"]), width=2),
            dbc.Col(kpi_card("Rule Max DD", f"{m_rule['max_dd']:.1f}%", "#FF5722"), width=2),
            dbc.Col(kpi_card("ML CAGR", f"{m_ml['cagr']:+.1f}%", PAL["ml"]), width=2),
            dbc.Col(kpi_card("ML Sharpe", f"{m_ml['sharpe']:.3f}", PAL["ml"]), width=2),
            dbc.Col(kpi_card("ML Max DD", f"{m_ml['max_dd']:.1f}%", "#FF5722"), width=2),
        ], className="mb-3")

        content = build_tab(active_tab, sub, fold_stats, feat_imp)
        return kpis, content

    return app


def build_tab(tab_id, df, fold_stats, feat_imp):
    if tab_id == "tab-overview":
        return tab_overview(df)
    elif tab_id == "tab-signals":
        return tab_signals(df)
    elif tab_id == "tab-ml":
        return tab_ml(df, fold_stats, feat_imp)
    elif tab_id == "tab-risk":
        return tab_risk(df)
    return html.Div("Select a tab")


# ── Tab 1: Overview ────────────────────────────────────────────────────

def tab_overview(df):
    eq_rule = (1 + df["rule_ret"].fillna(0)).cumprod()
    eq_ml   = (1 + df["ml_ret"].fillna(0)).cumprod()
    eq_bh   = (1 + df["bh_ret"].fillna(0)).cumprod()

    fig = go.Figure()
    for name, eq, col, dash in [
        ("Rule-Based + VolTarget", eq_rule, PAL["rule"], "solid"),
        ("ML Ensemble",            eq_ml,   PAL["ml"],   "solid"),
        ("Buy & Hold NG",          eq_bh,   PAL["bh"],   "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=name,
            line=dict(color=col, width=2 if "Hold" not in name else 1,
                      dash=dash),
            opacity=0.4 if "Hold" in name else 1.0,
        ))
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(**_dark_layout("Equity Curves (starting $1)"))

    # Annual returns bar chart
    annual = pd.DataFrame({
        "Rule":    df["rule_ret"].resample("YE").apply(lambda x: (1+x).prod()-1) * 100,
        "ML":      df["ml_ret"].resample("YE").apply(lambda x: (1+x).prod()-1) * 100,
        "BuyHold": df["bh_ret"].resample("YE").apply(lambda x: (1+x).prod()-1) * 100,
    }).dropna(how="all")

    fig2 = go.Figure()
    for col, color in [("Rule", PAL["rule"]), ("ML", PAL["ml"]), ("BuyHold", PAL["bh"])]:
        if col in annual:
            fig2.add_trace(go.Bar(
                x=[str(y.year) for y in annual.index],
                y=annual[col], name=col, marker_color=color, opacity=0.85,
            ))
    fig2.update_layout(**_dark_layout("Annual Returns %"), barmode="group")

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig, style={"height": "420px"}))]),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig2, style={"height": "320px"}))]),
        _metrics_table(df),
    ])


def _metrics_table(df):
    from backtest.engine import compute_metrics
    rows = []
    for label, ret in [("Rule-Based", df["rule_ret"]),
                       ("ML Ensemble", df["ml_ret"]),
                       ("Buy & Hold NG", df["bh_ret"])]:
        m = compute_metrics(ret.dropna(), label)
        rows.append({
            "Strategy": label,
            "CAGR %":   f"{m['cagr']:+.2f}",
            "Vol %":    f"{m['vol']:.2f}",
            "Sharpe":   f"{m['sharpe']:.3f}",
            "Max DD %": f"{m['max_dd']:.2f}",
            "Calmar":   f"{m['calmar']:.3f}",
            "Win %":    f"{m['win_rate']:.1f}",
        })
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"marginTop": "20px"},
        style_header={"backgroundColor": PAL["card"], "color": PAL["text"],
                      "fontWeight": "bold"},
        style_data={"backgroundColor": PAL["bg"], "color": PAL["text"]},
        style_data_conditional=[
            {"if": {"row_index": 0}, "color": PAL["rule"]},
            {"if": {"row_index": 1}, "color": PAL["ml"]},
        ],
    )


# ── Tab 2: Signals ─────────────────────────────────────────────────────

def tab_signals(df):
    figs = []

    # HDD/CDD surprise
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=["HDD Surprise (cold snap signal)",
                                          "CDD Surprise (heat wave signal)"],
                          vertical_spacing=0.08)
    if "HDD_zscore" in df.columns:
        hdd_delta = df["HDD_zscore"].diff(7)
        fig1.add_trace(go.Scatter(x=df.index, y=hdd_delta,
            fill="tozeroy", name="HDD delta 7d",
            line=dict(color=PAL["hdd"], width=0.8)), row=1, col=1)
    if "CDD_zscore" in df.columns:
        cdd_delta = df["CDD_zscore"].diff(7)
        fig1.add_trace(go.Scatter(x=df.index, y=cdd_delta,
            fill="tozeroy", name="CDD delta 7d",
            line=dict(color=PAL["cdd"], width=0.8)), row=2, col=1)
    fig1.update_layout(**_dark_layout("Temperature Surprise Signals", height=380))
    figs.append(dcc.Graph(figure=fig1))

    # Satellite wind + solar
    fig2 = go.Figure()
    for col, name, color in [
        ("wind_deficit_7d_z",  "Wind deficit z", PAL["wind"]),
        ("solar_deficit_7d_z", "Solar deficit z", PAL["solar"]),
        ("renewable_deficit_z","Combined",        "#ffffff"),
    ]:
        if col in df.columns:
            fig2.add_trace(go.Scatter(x=df.index, y=df[col],
                name=name, line=dict(color=color,
                width=1.5 if "Combined" in name else 0.8)))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig2.update_layout(**_dark_layout("🛰 Satellite: Renewable Energy Deficit (wind+solar)", height=280))
    figs.append(dcc.Graph(figure=fig2))

    # ENSO + Storage
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=["ENSO ONI Index", "EIA Storage Surprise"],
                          vertical_spacing=0.08)
    if "oni" in df.columns:
        oni = df["oni"]
        fig3.add_trace(go.Scatter(x=df.index, y=oni, name="ONI",
            line=dict(color=PAL["enso"], width=1.2)), row=1, col=1)
        fig3.add_hrect(y0=0.5,  y1=oni.max()+0.1, fillcolor="red",    opacity=0.05, row=1, col=1)
        fig3.add_hrect(y0=oni.min()-0.1, y1=-0.5, fillcolor="blue",   opacity=0.05, row=1, col=1)
    if "storage_surprise_z" in df.columns:
        ss = df["storage_surprise_z"]
        fig3.add_trace(go.Bar(x=df.index, y=ss,
            marker_color=[PAL["storage"] if v < 0 else "#888" for v in ss],
            name="Storage surprise z", opacity=0.8), row=2, col=1)
    fig3.update_layout(**_dark_layout("Macro + Fundamental Signals", height=380))
    figs.append(dcc.Graph(figure=fig3))

    # Final blended signal + position
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=["Blended Signal z-score", "Position"],
                          vertical_spacing=0.06)
    if "final_signal" in df.columns:
        fig4.add_trace(go.Scatter(x=df.index, y=df["final_signal"],
            name="Signal", line=dict(color=PAL["rule"], width=1)), row=1, col=1)
    if "rule_position" in df.columns:
        pos = df["rule_position"]
        fig4.add_trace(go.Bar(x=df.index, y=pos,
            marker_color=[PAL["rule"] if v > 0 else "#F44336" for v in pos],
            name="Position", opacity=0.8), row=2, col=1)
    fig4.update_layout(**_dark_layout("Rule-Based Signal and Position", height=350))
    figs.append(dcc.Graph(figure=fig4))

    return html.Div(figs)


# ── Tab 3: ML ─────────────────────────────────────────────────────────

def tab_ml(df, fold_stats, feat_imp):
    figs = []

    # OOS IC by fold
    if fold_stats:
        folds_df = pd.DataFrame(fold_stats)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=folds_df["period"], y=folds_df["oos_ic"],
            marker_color=[PAL["ml"] if v > 0 else "#F44336" for v in folds_df["oos_ic"]],
            name="OOS IC", text=[f"{v:.3f}" for v in folds_df["oos_ic"]],
            textposition="outside",
        ))
        fig1.add_hline(y=0, line_color="gray", opacity=0.5)
        fig1.update_layout(**_dark_layout("Out-of-Sample IC by Fold", height=300))
        figs.append(dcc.Graph(figure=fig1))

        # Ensemble weights over time
        fig2 = go.Figure()
        for col, name, color in [
            ("w_lgbm", "LightGBM", PAL["ml"]),
            ("w_xgb",  "XGBoost",  PAL["rule"]),
            ("w_ridge","Ridge",    PAL["solar"]),
        ]:
            if col in folds_df.columns:
                fig2.add_trace(go.Scatter(
                    x=folds_df["period"], y=folds_df[col],
                    name=name, line=dict(color=color, width=2),
                    stackgroup="one", mode="lines",
                ))
        fig2.update_layout(**_dark_layout("Ensemble Weights by Fold (IC-weighted)", height=280))
        figs.append(dcc.Graph(figure=fig2))

    # Feature importances
    if feat_imp is not None and len(feat_imp) > 0:
        top = feat_imp.head(20).sort_values()
        sat_feats = {"wind_ms","solar_kwh","wind_deficit_z","solar_deficit_z",
                     "renewable_deficit_z","power_demand_x_hdd"}
        colors = [PAL["solar"] if f in sat_feats else PAL["ml"] for f in top.index]
        fig3 = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color=colors, opacity=0.85,
        ))
        fig3.update_layout(**_dark_layout("Feature Importances (LightGBM, avg across folds)",
                                           height=500))
        fig3.update_layout(
            annotations=[dict(
                text="🟡 = satellite features",
                x=0.98, y=0.02, xref="paper", yref="paper",
                showarrow=False, font=dict(color=PAL["solar"], size=11),
            )]
        )
        figs.append(dcc.Graph(figure=fig3))

    # ML prediction vs actual scatter
    if "ml_pred" in df.columns and "natgas_fwd5d" in df.columns:
        mask = df["ml_pred"].abs() > 0
        pred = df.loc[mask, "ml_pred"].dropna()
        act  = df.loc[mask, "natgas_fwd5d"].dropna()
        common = pred.index.intersection(act.index)
        if len(common) > 20:
            p, a = pred.loc[common], act.loc[common]
            ic = p.corr(a)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=p, y=a, mode="markers",
                marker=dict(color=PAL["ml"], size=3, opacity=0.3),
                name="Prediction vs Actual",
            ))
            # Regression line
            z = np.polyfit(p, a, 1)
            xr = np.linspace(p.quantile(0.02), p.quantile(0.98), 100)
            fig4.add_trace(go.Scatter(
                x=xr, y=np.polyval(z, xr),
                line=dict(color="red", width=2), name=f"fit  IC={ic:.4f}",
            ))
            fig4.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
            fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
            fig4.update_layout(**_dark_layout(f"ML Predicted vs Actual 10d Return  (IC={ic:.4f})",
                                               height=380))
            figs.append(dcc.Graph(figure=fig4))

    return html.Div(figs)


# ── Tab 4: Risk ────────────────────────────────────────────────────────

def tab_risk(df):
    figs = []

    # Drawdowns
    fig1 = go.Figure()
    for name, ret, color in [
        ("Rule-Based", df["rule_ret"], PAL["rule"]),
        ("ML",         df["ml_ret"],   PAL["ml"]),
        ("Buy & Hold", df["bh_ret"],   PAL["bh"]),
    ]:
        eq = (1 + ret.fillna(0)).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        fig1.add_trace(go.Scatter(
            x=dd.index, y=dd, name=name,
            line=dict(color=color, width=1.5 if "Hold" not in name else 0.8),
            fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba") if color.startswith("rgb") else color+"22",
            opacity=0.7 if "Hold" not in name else 0.3,
        ))
    fig1.update_layout(**_dark_layout("Drawdown %", height=320))
    figs.append(dcc.Graph(figure=fig1))

    # Rolling 90d Sharpe
    fig2 = go.Figure()
    for name, ret, color in [
        ("Rule-Based", df["rule_ret"], PAL["rule"]),
        ("ML",         df["ml_ret"],   PAL["ml"]),
    ]:
        sh = ret.rolling(90).mean() / ret.rolling(90).std() * np.sqrt(252)
        fig2.add_trace(go.Scatter(x=sh.index, y=sh, name=name,
            line=dict(color=color, width=1.5)))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig2.update_layout(**_dark_layout("Rolling 90-day Sharpe", height=280))
    figs.append(dcc.Graph(figure=fig2))

    # Realized vol + vol-targeting effect
    fig3 = go.Figure()
    rv = df["bh_ret"].rolling(20).std() * np.sqrt(252) * 100
    fig3.add_trace(go.Scatter(x=rv.index, y=rv,
        name="NG realized vol 20d (ann.)",
        line=dict(color="#FF9800", width=1.2), fill="tozeroy",
        fillcolor="#FF980022"))
    fig3.add_hline(y=15, line_dash="dash", line_color=PAL["rule"],
                   annotation_text="Vol target 15%")
    fig3.update_layout(**_dark_layout("Natural Gas Realized Volatility — Vol Targeting Context",
                                       height=280))
    figs.append(dcc.Graph(figure=fig3))

    # Monthly returns heatmap
    for name, ret, color in [
        ("Rule-Based", df["rule_ret"], PAL["rule"]),
        ("ML",         df["ml_ret"],   PAL["ml"]),
    ]:
        monthly = ret.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
        fig = go.Figure(go.Bar(
            x=[d.strftime("%Y-%m") for d in monthly.index],
            y=monthly.values,
            marker_color=[color if v > 0 else "#F44336" for v in monthly],
            opacity=0.85, name=name,
        ))
        fig.add_hline(y=0, line_color="gray", opacity=0.4)
        fig.update_layout(**_dark_layout(f"{name} — Monthly Returns %", height=250))
        figs.append(dcc.Graph(figure=fig))

    return html.Div(figs)


# ── Dark layout helper ─────────────────────────────────────────────────

def _dark_layout(title: str, height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=PAL["text"], size=14)),
        height=height,
        paper_bgcolor=PAL["card"],
        plot_bgcolor=PAL["bg"],
        font=dict(color=PAL["text"], size=11),
        legend=dict(bgcolor=PAL["card"], bordercolor="#444"),
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#333", showgrid=True),
        yaxis=dict(gridcolor="#333", showgrid=True),
    )


# ── Main ───────────────────────────────────────────────────────────────

def run_dashboard(df: pd.DataFrame, fold_stats: list, feat_imp: pd.Series):
    app.layout = make_layout(df)
    register_callbacks(app, df, fold_stats, feat_imp)
    from config import DASH_PORT
    print(f"\n🚀  Dashboard running → http://localhost:{DASH_PORT}")
    app.run(debug=False, host="0.0.0.0", port=DASH_PORT)


if __name__ == "__main__":
    import pickle
    dash_cache = "data/cache/dashboard_data.parquet"
    fold_cache  = "data/cache/fold_stats.pkl"
    fi_cache    = "data/cache/feat_imp.parquet"

    if not os.path.exists(dash_cache):
        print("Run `python run_backtest.py` first to generate data.")
        sys.exit(1)

    df        = pd.read_parquet(dash_cache)
    fold_stats = pickle.load(open(fold_cache, "rb")) if os.path.exists(fold_cache) else []
    feat_imp   = pd.read_parquet(fi_cache).squeeze() if os.path.exists(fi_cache) else pd.Series()

    run_dashboard(df, fold_stats, feat_imp)
