"""
app.py  ——  Regime-Based HMM Trading Dashboard  (v4)
高级暗色 UI · 14 信号确认面板 · 三资产 Tab · 完整数据展示 · LILYN
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os, pickle, base64, json, glob
from datetime import datetime

from data_loader import fetch_data
from backtester  import (run_backtest, STARTING_CAP, MIN_CONFIRMATIONS,
                          _position_size, N_STATES, TICKER_PARAMS,
                          FRICTION_PCT, MARGIN_PARAMS, LEVERAGE)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ASSETS_DIR  = os.path.join(os.path.dirname(__file__), "assets")


def _safe_filename(ticker: str) -> str:
    return ticker.replace("=", "_").replace("/", "_")

def _load_precomputed(ticker: str):
    path = os.path.join(RESULTS_DIR, f"{_safe_filename(ticker)}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def _computed_at() -> str:
    path = os.path.join(RESULTS_DIR, "computed_at.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return "未知"

def _logo_b64() -> str:
    path = os.path.join(ASSETS_DIR, "logo.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HMM Regime Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; background: #080c14; }
.block-container { padding: 1rem 2rem 3rem 2rem; max-width: 1600px; }

.glass-card {
    background: linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 20px 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 2px;
}
.metric-card {
    background: linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px 16px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform .15s ease, box-shadow .15s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.6); }
.metric-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 8px; font-weight: 500; }
.metric-value { font-size: 1.9rem; font-weight: 800; line-height: 1; }
.metric-sub   { font-size: 0.7rem; color: #475569; margin-top: 6px; font-weight: 400; }

.signal-long {
    background: linear-gradient(135deg,#002d16,#004d24);
    border: 1px solid rgba(0,230,118,0.4); border-radius: 16px; padding: 22px 28px; text-align: center;
    box-shadow: 0 0 40px rgba(0,230,118,0.12), inset 0 1px 0 rgba(0,230,118,0.15);
}
.signal-cash {
    background: linear-gradient(135deg,#0f1420,#141929);
    border: 1px solid rgba(100,116,139,0.3); border-radius: 16px; padding: 22px 28px; text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}
.signal-title { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 10px; font-weight:500; }
.signal-value { font-size: 2.4rem; font-weight: 900; letter-spacing: -1px; }

.regime-pill  { display:inline-block; padding:6px 20px; border-radius:30px; font-size:1rem; font-weight:700; margin-top:6px; letter-spacing:.3px; }
.regime-bull  { background:rgba(0,230,118,0.12); color:#00e676; border:1px solid rgba(0,230,118,0.4); box-shadow:0 0 20px rgba(0,230,118,0.1); }
.regime-bear  { background:rgba(255,82,82,0.12); color:#ff5252; border:1px solid rgba(255,82,82,0.4); box-shadow:0 0 20px rgba(255,82,82,0.1); }
.regime-neut  { background:rgba(255,215,64,0.10); color:#ffd740; border:1px solid rgba(255,215,64,0.35); box-shadow:0 0 20px rgba(255,215,64,0.08); }

.sig-row {
    display:flex; align-items:center; justify-content:space-between;
    padding: 7px 12px; border-radius: 8px; margin-bottom: 4px;
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.05);
    font-size: 0.8rem; transition: background .1s;
}
.sig-row:hover { background: rgba(255,255,255,0.045); }
.sig-name { color: #94a3b8; font-weight: 500; }
.sig-val  { color: #cbd5e1; font-family: 'SF Mono', monospace; font-size: 0.75rem; }
.sig-pass { color: #00e676; font-size: 1rem; }
.sig-fail { color: #ff5252; font-size: 1rem; }

.section-header {
    color: #e2e8f0; font-size: 0.9rem; font-weight: 600;
    margin: 1.6rem 0 0.7rem 0; padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    letter-spacing: .3px; display: flex; align-items: center; gap: 8px;
}
.score-outer { background: rgba(255,255,255,0.06); border-radius: 8px; height: 8px; width: 100%; margin: 8px 0 4px 0; overflow: hidden; }
.score-inner { height: 100%; border-radius: 8px; transition: width .4s ease; }
.page-title  { font-size: 1.55rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; line-height: 1.2; }
.page-sub    { font-size: 0.78rem; color: #475569; margin-top: 3px; font-weight: 400; }

[data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; padding: 4px !important; border: 1px solid rgba(255,255,255,0.06) !important; gap: 2px !important; }
[data-baseweb="tab"]      { border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important; color: #64748b !important; padding: 8px 20px !important; }
[aria-selected="true"]    { background: rgba(255,255,255,0.08) !important; color: #e2e8f0 !important; }

.green  { color: #00e676; } .red  { color: #ff5252; } .yellow { color: #ffd740; }
.blue   { color: #60a5fa; } .purple { color: #a78bfa; } .white  { color: #f1f5f9; }
#MainMenu, footer, header { visibility: hidden; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 缓存
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_asset(ticker: str) -> dict:
    precomputed = _load_precomputed(ticker)
    if precomputed is not None:
        return precomputed
    df = fetch_data(ticker)
    return run_backtest(df, ticker)


# ──────────────────────────────────────────────────────────────
# 颜色工具
# ──────────────────────────────────────────────────────────────

def _bg(label: str) -> str:
    if label == "Bull Run":   return "rgba(0,230,118,0.13)"
    if label == "Bull+":      return "rgba(0,200,100,0.07)"
    if label == "Warming Up": return "rgba(96,165,250,0.07)"
    if label == "Bear/Crash": return "rgba(255,82,82,0.14)"
    if label == "Bear":       return "rgba(255,120,80,0.09)"
    return "rgba(255,215,64,0.04)"

def _pill(label: str) -> str:
    if "Bull" in label:                          return "regime-bull"
    if "Bear" in label or "Crash" in label:      return "regime-bear"
    return "regime-neut"

def _score_color(pct: float) -> str:
    if pct >= 0.80: return "#00e676"
    if pct >= 0.55: return "#ffd740"
    return "#ff5252"

def _regime_color(label: str) -> str:
    if "Bull Run" in label: return "#00e676"
    if "Bull+"    in label: return "#00c864"
    if "Warming"  in label: return "#60a5fa"
    if "Crash"    in label: return "#ff5252"
    if "Bear"     in label: return "#ff7850"
    return "#ffd740"


# ──────────────────────────────────────────────────────────────
# 图表
# ──────────────────────────────────────────────────────────────

CHART_BG   = "#080c14"
GRID_COLOR = "rgba(255,255,255,0.05)"

def _base_layout(**kw) -> dict:
    return dict(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color="#94a3b8", size=11, family="Inter"),
        margin=dict(l=8, r=8, t=44, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2535", bordercolor="#334155", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        **kw,
    )


def candle_chart(df: pd.DataFrame, trades: list, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.60, 0.20, 0.20],
                        vertical_spacing=0.02)
    shapes = []
    if len(df):
        prev, t0 = df["regime_label"].iloc[0], df.index[0]
        for ts, lbl in zip(df.index[1:], df["regime_label"].iloc[1:]):
            if lbl != prev:
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=t0, x1=ts, y0=0, y1=1,
                                   fillcolor=_bg(prev), line_width=0, layer="below"))
                t0, prev = ts, lbl
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=t0, x1=df.index[-1], y0=0, y1=1,
                           fillcolor=_bg(prev), line_width=0, layer="below"))

    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(96,165,250,0.06)",
        name="Bollinger", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"], mode="lines",
        line=dict(color="rgba(96,165,250,0.4)", width=1, dash="dot"),
        name="BB Mid", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        increasing_fillcolor="#00e676", decreasing_fillcolor="#ff5252",
        name=ticker, line_width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines",
        line=dict(color="#ffd740", width=1.2, dash="dot"), name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], mode="lines",
        line=dict(color="#a78bfa", width=1.2, dash="dash"), name="EMA 200"), row=1, col=1)
    if trades:
        fig.add_trace(go.Scatter(
            x=[t["entry_time"] for t in trades], y=[t["entry_price"] for t in trades],
            mode="markers", name="买入",
            marker=dict(symbol="triangle-up", size=12, color="#00e676",
                        line=dict(width=1, color="#fff"))), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t["exit_time"] for t in trades], y=[t["exit_price"] for t in trades],
            mode="markers", name="卖出",
            marker=dict(symbol="triangle-down", size=12, color="#ff5252",
                        line=dict(width=1, color="#fff"))), row=1, col=1)

    colors_vol = ["#00e676" if c >= o else "#ff5252"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors_vol,
        marker_opacity=0.5, name="Volume", showlegend=False), row=2, col=1)
    obv_norm = (df["obv_ema"] - df["obv_ema"].min()) / \
               (df["obv_ema"].max() - df["obv_ema"].min() + 1e-9) * df["Volume"].max()
    fig.add_trace(go.Scatter(x=df.index, y=obv_norm, mode="lines",
        line=dict(color="#a78bfa", width=1.2), name="OBV EMA"), row=2, col=1)

    # RSI panel
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines",
        line=dict(color="#60a5fa", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="rgba(255,82,82,0.6)", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="rgba(0,230,118,0.5)", width=1, dash="dot"), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,82,82,0.05)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,230,118,0.05)", line_width=0, row=3, col=1)

    layout = _base_layout(height=680)
    layout["shapes"] = shapes
    layout["xaxis"]  = dict(rangeslider=dict(visible=False), gridcolor=GRID_COLOR, showgrid=True, type="date")
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, showgrid=True)
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR, showgrid=True)
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, showgrid=True, showticklabels=False)
    layout["xaxis3"] = dict(gridcolor=GRID_COLOR, showgrid=True)
    layout["yaxis3"] = dict(gridcolor=GRID_COLOR, showgrid=True, range=[0,100],
                            tickvals=[30,50,70], title="RSI")
    fig.update_layout(**layout)
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor="rgba(255,255,255,0.04)", activecolor="rgba(96,165,250,0.3)",
            bordercolor="rgba(255,255,255,0.08)", font=dict(color="#94a3b8", size=11),
            buttons=[
                dict(count=30,  label="1M",  step="day", stepmode="backward"),
                dict(count=90,  label="3M",  step="day", stepmode="backward"),
                dict(count=180, label="6M",  step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
        row=1, col=1)
    return fig


def macd_signal_chart(df: pd.DataFrame, min_conf: int) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.04,
                        subplot_titles=("MACD", "信号强度得分（14条）"))
    macd_hist = df["macd_line"] - df["macd_signal"]
    hist_colors = ["#00e676" if v >= 0 else "#ff5252" for v in macd_hist]
    fig.add_trace(go.Bar(x=df.index, y=macd_hist, marker_color=hist_colors,
        marker_opacity=0.7, name="MACD 柱"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_line"], mode="lines",
        line=dict(color="#60a5fa", width=1.5), name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], mode="lines",
        line=dict(color="#ffd740", width=1.2, dash="dot"), name="Signal"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["signal_score"], mode="lines",
        line=dict(color="#a78bfa", width=1.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)", name="信号得分"), row=2, col=1)
    fig.add_hline(y=min_conf, line=dict(color="#00e676", width=1.5, dash="dash"),
                  annotation_text=f"入场阈值 {min_conf}", annotation_font_color="#00e676",
                  annotation_position="top right", row=2, col=1)

    layout = _base_layout(height=380)
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, title="MACD")
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, title="得分", range=[0,14],
                            tickvals=[0,3,6,9,12,14])
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    fig.update_layout(**layout)
    return fig


def stoch_cci_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.04,
                        subplot_titles=("Stochastic %K / %D", "CCI (20)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"], mode="lines",
        line=dict(color="#60a5fa", width=1.5), name="%K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"], mode="lines",
        line=dict(color="#ffd740", width=1.2, dash="dot"), name="%D"), row=1, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,82,82,0.07)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(0,230,118,0.07)", line_width=0, row=1, col=1)
    fig.add_hline(y=80, line=dict(color="rgba(255,82,82,0.5)", width=1, dash="dot"), row=1, col=1)
    fig.add_hline(y=20, line=dict(color="rgba(0,230,118,0.5)", width=1, dash="dot"), row=1, col=1)
    cci_colors = ["#00e676" if v > 0 else "#ff5252" for v in df["cci"]]
    fig.add_trace(go.Bar(x=df.index, y=df["cci"], marker_color=cci_colors,
        marker_opacity=0.6, name="CCI"), row=2, col=1)
    fig.add_hline(y=100,  line=dict(color="rgba(255,82,82,0.5)", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=-100, line=dict(color="rgba(0,230,118,0.5)", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=0,    line=dict(color="rgba(255,255,255,0.2)", width=1),            row=2, col=1)
    layout = _base_layout(height=360)
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, range=[0,100], title="Stoch %")
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, title="CCI")
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    fig.update_layout(**layout)
    return fig


def equity_chart(df: pd.DataFrame) -> go.Figure:
    bh  = STARTING_CAP * df["Close"] / df["Close"].iloc[0]
    dd  = (df["equity"] - df["equity"].cummax()) / df["equity"].cummax() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=df.index, y=df["equity"], mode="lines",
        line=dict(color="#00e676", width=2), fill="tozeroy",
        fillcolor="rgba(0,230,118,0.06)", name=f"策略 ({LEVERAGE}× 杠杆)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bh, mode="lines",
        line=dict(color="#60a5fa", width=1.5, dash="dash"), name="买入持有"), row=1, col=1)

    # SPY overlay
    try:
        spy_raw   = fetch_data("SPY")
        spy_close = spy_raw["Close"]
        if hasattr(spy_close.index, "tz") and spy_close.index.tz is not None:
            spy_close = spy_close.tz_localize(None)
        target_idx = (df.index.tz_localize(None)
                      if (hasattr(df.index, "tz") and df.index.tz is not None)
                      else df.index)
        spy_aligned = spy_close.reindex(target_idx, method="ffill").dropna()
        if len(spy_aligned):
            spy_eq = STARTING_CAP * spy_aligned / spy_aligned.iloc[0]
            fig.add_trace(go.Scatter(x=spy_eq.index, y=spy_eq, mode="lines",
                line=dict(color="#ffd740", width=1.2, dash="dot"), name="SPY"), row=1, col=1)
    except Exception:
        pass

    dd_colors = ["#ff5252" if v < -10 else "#ffd740" if v < -5 else "#00e676" for v in dd]
    fig.add_trace(go.Bar(x=df.index, y=dd, marker_color=dd_colors,
        marker_opacity=0.7, name="回撤 %"), row=2, col=1)
    layout = _base_layout(height=400)
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, tickprefix="$")
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, ticksuffix="%", title="回撤")
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    fig.update_layout(**layout)
    return fig


def rolling_sharpe_chart(df: pd.DataFrame, is_daily: bool) -> go.Figure:
    bars_per_yr = 252 if is_daily else 252 * 24
    window = 90 if is_daily else 720
    ret = df["equity"].pct_change()
    roll_sharpe = (ret.rolling(window).mean() / ret.rolling(window).std()) * np.sqrt(bars_per_yr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=roll_sharpe, mode="lines",
        line=dict(color="#60a5fa", width=1.5), fill="tozeroy",
        fillcolor="rgba(96,165,250,0.06)",
        name=f"滚动夏普（{window}{'日' if is_daily else 'h'} 窗口）"))
    fig.add_hline(y=1, line=dict(color="#00e676", width=1, dash="dash"),
                  annotation_text="Sharpe=1", annotation_font_color="#00e676")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    fig.update_layout(**_base_layout(height=220),
                      yaxis=dict(gridcolor=GRID_COLOR, title="Sharpe"),
                      xaxis=dict(gridcolor=GRID_COLOR))
    return fig


def monthly_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    years  = sorted(monthly_df["year"].unique())
    z, text = [], []
    for yr in years:
        row_z, row_t = [], []
        for mo in range(1, 13):
            val = monthly_df[(monthly_df["year"]==yr) & (monthly_df["month"]==mo)]["ret"]
            if len(val):
                v = float(val.iloc[0]); row_z.append(v); row_t.append(f"{v:+.1f}%")
            else:
                row_z.append(None); row_t.append("")
        z.append(row_z); text.append(row_t)
    fig = go.Figure(go.Heatmap(
        z=z, x=MONTHS, y=[str(y) for y in years],
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#7f1d1d"],[0.5,"#1e2130"],[1,"#14532d"]],
        zmid=0, showscale=True,
        colorbar=dict(ticksuffix="%", thickness=12, len=0.8,
                      tickfont=dict(size=10, color="#64748b")),
        hoverongaps=False))
    fig.update_layout(**_base_layout(height=max(160, len(years)*46+60)),
                      xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
    return fig


def regime_bar(df: pd.DataFrame) -> go.Figure:
    vc = df["regime_label"].value_counts().reset_index()
    vc.columns = ["Regime", "Count"]
    vc["Pct"] = (vc["Count"] / len(df) * 100).round(1)
    colors = [_regime_color(r) for r in vc["Regime"]]
    fig = go.Figure(go.Bar(
        x=vc["Regime"], y=vc["Pct"], marker_color=colors, marker_opacity=0.85,
        text=vc["Pct"].map(lambda x: f"{x}%"), textposition="outside",
        textfont=dict(size=12, color="#e2e8f0")))
    fig.update_layout(**_base_layout(height=240, showlegend=False),
                      yaxis=dict(title="占比 %", gridcolor=GRID_COLOR),
                      xaxis=dict(gridcolor=GRID_COLOR))
    return fig


def regime_return_chart(df: pd.DataFrame, n_states: int) -> go.Figure:
    ret = df["Close"].pct_change() * 100
    label_order = (["Bear/Crash", "Bear"] +
                   [f"Neutral-{i}" for i in range(2, n_states-2)] +
                   ["Warming Up", "Bull+", "Bull Run"])
    data = []
    for lbl in label_order:
        vals = ret[df["regime_label"] == lbl].dropna()
        if len(vals) == 0: continue
        color = _regime_color(lbl)
        data.append(go.Box(y=vals, name=lbl, marker_color=color,
                           line_color=color, boxmean="sd"))
    fig = go.Figure(data=data)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    fig.update_layout(**_base_layout(height=280, showlegend=False),
                      yaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%", title="单bar收益率 %"),
                      xaxis=dict(gridcolor=GRID_COLOR))
    return fig


def trade_analytics_chart(trades: list) -> go.Figure:
    if not trades:
        return go.Figure()
    tdf = pd.DataFrame(trades)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("单笔盈亏 ($)", "持仓时长分布 (bars)"),
                        horizontal_spacing=0.10)
    pnl = tdf["pnl"]
    pnl_colors = ["#00e676" if v > 0 else "#ff5252" for v in pnl]
    fig.add_trace(go.Bar(x=list(range(len(pnl))), y=pnl,
        marker_color=pnl_colors, marker_opacity=0.85, name="单笔盈亏",
        text=[f"${v:+,.0f}" for v in pnl], textposition="outside",
        textfont=dict(size=9)), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1), row=1, col=1)
    fig.add_trace(go.Histogram(x=tdf["hold_bars"], nbinsx=15,
        marker_color="#a78bfa", marker_opacity=0.8, name="持仓时长"), row=1, col=2)
    layout = _base_layout(height=280)
    layout.update({"yaxis": dict(gridcolor=GRID_COLOR, tickprefix="$", title="PnL"),
                   "yaxis2": dict(gridcolor=GRID_COLOR, title="笔数"),
                   "xaxis": dict(gridcolor=GRID_COLOR, title="交易序号"),
                   "xaxis2": dict(gridcolor=GRID_COLOR, title="Bars"),
                   "showlegend": False})
    fig.update_layout(**layout)
    return fig


# ── 1. 相对 Alpha 曲线（策略 / BH 净值比） ────────────────────
def relative_alpha_chart(df: pd.DataFrame) -> go.Figure:
    bh    = STARTING_CAP * df["Close"] / df["Close"].iloc[0]
    ratio = df["equity"] / bh
    fig   = go.Figure()
    above = ratio >= 1.0
    fig.add_trace(go.Scatter(x=df.index, y=ratio, mode="lines",
        line=dict(color="#00e676", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.05)",
        name="策略 / BH 净值比"))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dash"),
                  annotation_text="平价线 (1.0)", annotation_font_color="#94a3b8",
                  annotation_position="top right")
    fig.update_layout(**_base_layout(height=220),
                      yaxis=dict(gridcolor=GRID_COLOR, title="策略/BH 倍数",
                                 tickformat=".2f"),
                      xaxis=dict(gridcolor=GRID_COLOR))
    return fig


# ── 2. Underwater 连续回撤曲线 ────────────────────────────────
def underwater_chart(df: pd.DataFrame) -> go.Figure:
    dd = (df["equity"] - df["equity"].cummax()) / df["equity"].cummax() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=dd, mode="lines",
        line=dict(color="#ff5252", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,82,82,0.08)",
        name="水下回撤 %"))
    fig.add_hline(y=-5,  line=dict(color="rgba(255,215,64,0.5)", width=1, dash="dot"),
                  annotation_text="-5%", annotation_font_color="#ffd740")
    fig.add_hline(y=-10, line=dict(color="rgba(255,82,82,0.5)", width=1, dash="dot"),
                  annotation_text="-10%", annotation_font_color="#ff5252")
    fig.update_layout(**_base_layout(height=200),
                      yaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%", title="回撤"),
                      xaxis=dict(gridcolor=GRID_COLOR))
    return fig


# ── 3. 月度热力图（Strategy / BH / Alpha 三选项） ─────────────
def monthly_heatmap_tabbed(monthly_df: pd.DataFrame) -> None:
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    tab_s, tab_b, tab_a = st.tabs(["Strategy", "Buy & Hold", "Alpha"])
    for tab, col_key, title in [
        (tab_s, "ret",       "策略月度收益"),
        (tab_b, "bh_ret",    "B&H 月度收益"),
        (tab_a, "alpha_ret", "月度 Alpha（策略 - B&H）"),
    ]:
        with tab:
            years = sorted(monthly_df["year"].unique())
            z, text = [], []
            for yr in years:
                row_z, row_t = [], []
                for mo in range(1, 13):
                    val = monthly_df[
                        (monthly_df["year"] == yr) & (monthly_df["month"] == mo)
                    ][col_key]
                    if len(val):
                        v = float(val.iloc[0]); row_z.append(v); row_t.append(f"{v:+.1f}%")
                    else:
                        row_z.append(None); row_t.append("")
                z.append(row_z); text.append(row_t)
            fig = go.Figure(go.Heatmap(
                z=z, x=MONTHS, y=[str(y) for y in years],
                text=text, texttemplate="%{text}",
                colorscale=[[0,"#7f1d1d"],[0.5,"#1e2130"],[1,"#14532d"]],
                zmid=0, showscale=True,
                colorbar=dict(ticksuffix="%", thickness=12, len=0.8,
                              tickfont=dict(size=10, color="#64748b")),
                hoverongaps=False))
            fig.update_layout(**_base_layout(height=max(160, len(years)*46+60)),
                              title=dict(text=title, font=dict(size=12, color="#94a3b8"),
                                         x=0, xanchor="left"),
                              xaxis=dict(side="top"),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


# ── 4. Regime Return Attribution（每个 HMM 状态的收益归因） ────
def regime_attribution_chart(df: pd.DataFrame, trades: list) -> go.Figure:
    if not trades:
        return go.Figure()
    tdf = pd.DataFrame(trades)
    # attach regime label at entry time
    regime_at_entry = df["regime_label"].reindex(tdf["entry_time"]).values
    tdf["entry_regime"] = regime_at_entry
    grp = tdf.groupby("entry_regime").agg(
        count  =("pnl", "count"),
        avg_pnl=("pnl", "mean"),
        win_r  =("pnl", lambda x: (x > 0).mean() * 100),
        total  =("pnl", "sum"),
    ).reset_index()
    grp = grp.sort_values("avg_pnl", ascending=True)
    colors = [_regime_color(r) for r in grp["entry_regime"]]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("平均单笔盈亏 ($)", "胜率 (%)"),
                        horizontal_spacing=0.10)
    fig.add_trace(go.Bar(y=grp["entry_regime"], x=grp["avg_pnl"],
        orientation="h", marker_color=colors, marker_opacity=0.85,
        text=[f"${v:+,.0f}" for v in grp["avg_pnl"]], textposition="outside",
        textfont=dict(size=10), name="平均PnL"), row=1, col=1)
    fig.add_trace(go.Bar(y=grp["entry_regime"], x=grp["win_r"],
        orientation="h", marker_color=colors, marker_opacity=0.6,
        text=[f"{v:.0f}% ({c}笔)" for v, c in zip(grp["win_r"], grp["count"])],
        textposition="outside", textfont=dict(size=10), name="胜率"), row=1, col=2)
    layout = _base_layout(height=max(200, len(grp)*50+80))
    layout["xaxis"]  = dict(gridcolor=GRID_COLOR, tickprefix="$", title="平均PnL")
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR, ticksuffix="%", title="胜率", range=[0,110])
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR)
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ── 5. Exit Reason Breakdown ──────────────────────────────────
def exit_attribution_chart(exit_attr: dict) -> go.Figure:
    if not exit_attr:
        return go.Figure()
    reasons   = list(exit_attr.keys())
    counts    = [exit_attr[r]["count"]   for r in reasons]
    avg_pnls  = [exit_attr[r]["avg_pnl"] for r in reasons]
    win_rates = [exit_attr[r]["win_r"]   for r in reasons]
    total_pnl = [exit_attr[r]["total_pnl"] for r in reasons]
    pnl_colors = ["#00e676" if v >= 0 else "#ff5252" for v in avg_pnls]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("出场次数", "平均盈亏 ($)", "胜率 (%)"),
                        horizontal_spacing=0.08)
    fig.add_trace(go.Bar(x=reasons, y=counts,
        marker_color="#60a5fa", marker_opacity=0.8,
        text=counts, textposition="outside",
        textfont=dict(size=11), name="次数"), row=1, col=1)
    fig.add_trace(go.Bar(x=reasons, y=avg_pnls,
        marker_color=pnl_colors, marker_opacity=0.85,
        text=[f"${v:+,.0f}" for v in avg_pnls], textposition="outside",
        textfont=dict(size=10), name="平均PnL"), row=1, col=2)
    wr_colors = ["#00e676" if v >= 50 else "#ff5252" for v in win_rates]
    fig.add_trace(go.Bar(x=reasons, y=win_rates,
        marker_color=wr_colors, marker_opacity=0.8,
        text=[f"{v:.0f}%" for v in win_rates], textposition="outside",
        textfont=dict(size=10), name="胜率"), row=1, col=3)
    layout = _base_layout(height=300)
    layout["xaxis"]  = dict(gridcolor=GRID_COLOR)
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    layout["xaxis3"] = dict(gridcolor=GRID_COLOR)
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, title="次数")
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, tickprefix="$", title="平均PnL")
    layout["yaxis3"] = dict(gridcolor=GRID_COLOR, ticksuffix="%", title="胜率", range=[0,110])
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ── 6. Top Trade Contribution ─────────────────────────────────
def top_trade_chart(trades: list) -> go.Figure:
    if not trades:
        return go.Figure()
    tdf = pd.DataFrame(trades).sort_values("pnl", ascending=False).reset_index(drop=True)
    total_gross = tdf["pnl"].sum()
    tdf["cum_contrib"] = tdf["pnl"].cumsum() / max(abs(total_gross), 1) * 100
    top_n = min(20, len(tdf))
    tdf_top = tdf.head(top_n)
    bar_colors = ["#00e676" if v >= 0 else "#ff5252" for v in tdf_top["pnl"]]
    labels = [f"T{i+1}" for i in range(top_n)]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"Top {top_n} 交易 PnL ($)", "累计贡献度 (%)"),
                        horizontal_spacing=0.10)
    fig.add_trace(go.Bar(x=labels, y=tdf_top["pnl"],
        marker_color=bar_colors, marker_opacity=0.85,
        text=[f"${v:+,.0f}" for v in tdf_top["pnl"]], textposition="outside",
        textfont=dict(size=9), name="PnL"), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=tdf_top["cum_contrib"], mode="lines+markers",
        line=dict(color="#a78bfa", width=2),
        marker=dict(size=6, color="#a78bfa"),
        name="累计贡献 %"), row=1, col=2)
    fig.add_hline(y=80, line=dict(color="rgba(255,215,64,0.5)", width=1, dash="dot"),
                  annotation_text="80%", annotation_font_color="#ffd740", row=1, col=2)
    layout = _base_layout(height=280)
    layout["xaxis"]  = dict(gridcolor=GRID_COLOR)
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, tickprefix="$", title="PnL")
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, ticksuffix="%", title="累计贡献 %")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ── 7. 持仓时长分布 ───────────────────────────────────────────
def hold_duration_chart(trades: list, is_daily: bool) -> go.Figure:
    if not trades:
        return go.Figure()
    tdf = pd.DataFrame(trades)
    unit = "days" if is_daily else "hours"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tdf["hold_bars"], nbinsx=20,
        marker_color="#60a5fa", marker_opacity=0.8, name=f"持仓时长 ({unit})"))
    avg_h = tdf["hold_bars"].mean()
    fig.add_vline(x=avg_h, line=dict(color="#ffd740", width=1.5, dash="dash"),
                  annotation_text=f"均值 {avg_h:.1f}", annotation_font_color="#ffd740")
    fig.update_layout(**_base_layout(height=220),
                      xaxis=dict(gridcolor=GRID_COLOR, title=f"持仓 ({unit})"),
                      yaxis=dict(gridcolor=GRID_COLOR, title="笔数"),
                      showlegend=False)
    return fig


# ── 8. 各 HMM 状态宏观特征均值雷达/柱状图 ────────────────────
def macro_by_regime_chart(df: pd.DataFrame) -> go.Figure:
    from data_loader import MACRO_TABLES
    macro_cols = [c for c in MACRO_TABLES.values() if c in df.columns]
    if not macro_cols or "regime_label" not in df.columns:
        return go.Figure()

    MACRO_LABELS = {
        "cpi_mom":       "CPI月率",
        "core_cpi_mom":  "核心CPI月率",
        "core_pce_mom":  "核心PCE月率",
        "jobless_claims":"初请失业金",
        "ism_pmi":       "ISM PMI",
    }

    grp = df.groupby("regime_label")[macro_cols].mean().reset_index()
    fig = go.Figure()
    for _, row in grp.iterrows():
        label = row["regime_label"]
        vals  = [row[c] for c in macro_cols]
        fig.add_trace(go.Bar(
            name=label,
            x=[MACRO_LABELS.get(c, c) for c in macro_cols],
            y=vals,
            marker_color=_regime_color(label),
            opacity=0.85,
        ))
    fig.update_layout(
        **_base_layout(height=320),
        barmode="group",
        title=dict(text="各 Regime 宏观特征均值（z-score）", font=dict(size=12, color="#94a3b8"), x=0),
        yaxis=dict(title="z-score", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ── 9. 宏观指标时序图（叠加 Regime 背景色） ──────────────────
def macro_timeseries_chart(df: pd.DataFrame) -> go.Figure:
    from data_loader import MACRO_TABLES
    macro_cols = [c for c in MACRO_TABLES.values() if c in df.columns]
    if not macro_cols or "regime_label" not in df.columns:
        return go.Figure()

    MACRO_LABELS = {
        "cpi_mom":       "CPI月率",
        "core_cpi_mom":  "核心CPI月率",
        "core_pce_mom":  "核心PCE月率",
        "jobless_claims":"初请失业金",
        "ism_pmi":       "ISM PMI",
    }
    COLORS = ["#60a5fa", "#34d399", "#fbbf24", "#f87171", "#a78bfa"]

    fig = go.Figure()

    # Regime 背景色条
    if len(df):
        prev, t0 = df["regime_label"].iloc[0], df.index[0]
        for ts, lbl in zip(df.index[1:], df["regime_label"].iloc[1:]):
            if lbl != prev:
                fig.add_vrect(x0=t0, x1=ts, fillcolor=_bg(prev),
                              line_width=0, layer="below")
                t0, prev = ts, lbl
        fig.add_vrect(x0=t0, x1=df.index[-1], fillcolor=_bg(prev),
                      line_width=0, layer="below")

    for col, color in zip(macro_cols, COLORS):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=MACRO_LABELS.get(col, col),
            line=dict(color=color, width=1.5),
            opacity=0.9,
        ))

    fig.update_layout(
        **_base_layout(height=300),
        title=dict(text="宏观指标时序（z-score · 背景色=Regime）",
                   font=dict(size=12, color="#94a3b8"), x=0),
        yaxis=dict(title="z-score", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# UI 组件
# ──────────────────────────────────────────────────────────────

def _metric(label, value, sub="", color="white") -> str:
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""

def _sig_row(name: str, ok: bool, val: str) -> str:
    icon = '<span class="sig-pass">●</span>' if ok else '<span class="sig-fail">●</span>'
    return (f'<div class="sig-row">{icon} <span class="sig-name">{name}</span>'
            f'<span class="sig-val">{val}</span></div>')


# ──────────────────────────────────────────────────────────────
# XGBoost 多因子预测面板（Gold 专属）
# ──────────────────────────────────────────────────────────────

_XGB_DIR          = os.path.join(os.path.dirname(__file__), "xgb_model")
XGB_FEATURES_CSV  = os.path.join(_XGB_DIR, "features.csv")
XGB_MODEL_CLS     = os.path.join(_XGB_DIR, "model_cls.pkl")
XGB_MODEL_REG     = os.path.join(_XGB_DIR, "model_reg.pkl")
XGB_FEATURE_COLS  = os.path.join(_XGB_DIR, "feature_cols.json")

@st.cache_data(ttl=3600, show_spinner=False)
def _load_xgb_prediction():
    import pickle, json, shap
    try:
        with open(XGB_MODEL_CLS, "rb") as f:
            model_cls = pickle.load(f)
        with open(XGB_MODEL_REG, "rb") as f:
            model_reg = pickle.load(f)
        with open(XGB_FEATURE_COLS) as f:
            feature_cols = json.load(f)
        df = pd.read_csv(XGB_FEATURES_CSV, parse_dates=["date"])
        X_last = df[feature_cols].iloc[[-1]]
        last_date = df["date"].iloc[-1]
        proba = model_cls.predict_proba(X_last)[0]
        pred_reg = float(model_reg.predict(X_last)[0])
        # SHAP
        explainer = shap.TreeExplainer(model_cls)
        sv = explainer.shap_values(X_last)
        sv_arr = np.array(sv)
        shap_vals = sv_arr[0].flatten()
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        shap_top = [(feature_cols[i], float(shap_vals[i]), float(X_last.iloc[0, i])) for i in top_idx]
        # 历史预测准确率（用 predict.csv 里的记录）
        pred_df = pd.read_csv("/Users/zhaowenxuan/Desktop/公司文件/news-analysis/predict.csv",
                              index_col=0, parse_dates=True)
        pred_df.columns = ["down", "flat", "up"] if len(pred_df.columns) == 3 else pred_df.columns
        return {
            "last_date":   last_date,
            "proba":       proba,
            "pred_reg":    pred_reg,
            "classes":     list(model_cls.classes_),
            "shap_top":    shap_top,
            "pred_df":     pred_df,
            "accuracy":    0.5778,
        }
    except Exception as e:
        return {"error": str(e)}


def render_xgb_panel():
    st.markdown('<div class="section-header">🤖 XGBoost 多因子预测（Gold 专属）</div>',
                unsafe_allow_html=True)
    with st.spinner("加载 XGBoost 预测…"):
        xgb = _load_xgb_prediction()
    if "error" in xgb:
        st.warning(f"XGBoost 加载失败：{xgb['error']}")
        return

    proba     = xgb["proba"]
    pred_reg  = xgb["pred_reg"]
    classes   = xgb["classes"]
    shap_top  = xgb["shap_top"]
    last_date = xgb["last_date"]
    accuracy  = xgb["accuracy"]

    # 二分类：0=跌 1=涨
    if len(proba) == 2:
        down_p, up_p = float(proba[0]), float(proba[1])
        flat_p = 0.0
    else:
        down_p, flat_p, up_p = float(proba[0]), float(proba[1]), float(proba[2])

    direction  = "📈 看涨" if up_p >= 0.6 else ("📉 看跌" if down_p >= 0.6 else "➡️ 震荡")
    dir_color  = "#00e676" if up_p >= 0.6 else ("#ff5252" if down_p >= 0.6 else "#ffd740")
    reg_color  = "#00e676" if pred_reg > 0 else "#ff5252"

    # 与 HMM 信号一致性
    try:
        sig_data = _load_latest_signal()
        hmm_action = sig_data["signals"].get("GC=F", {}).get("action_if_long", "") if sig_data else ""
        hmm_bull   = sig_data["signals"].get("GC=F", {}).get("is_bull", False) if sig_data else False
        xgb_bull   = up_p >= 0.6
        aligned    = (hmm_bull and xgb_bull) or (not hmm_bull and not xgb_bull)
        align_html = (
            '<span style="color:#00e676;font-weight:700">✅ 双模型共振</span>' if aligned
            else '<span style="color:#ffd740;font-weight:700">⚠️ 信号分歧</span>'
        )
    except Exception:
        align_html = '<span style="color:#475569">—</span>'

    # 顶部指标卡
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        '<div class="glass-card" style="text-align:center;padding:16px">'
        '<div style="font-size:0.62rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">明日方向</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{dir_color}">{direction}</div>'
        f'<div style="font-size:0.7rem;color:#64748b;margin-top:4px">数据截至 {str(last_date)[:10]}</div>'
        '</div>', unsafe_allow_html=True)
    c2.markdown(
        '<div class="glass-card" style="text-align:center;padding:16px">'
        '<div style="font-size:0.62rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">涨跌概率</div>'
        f'<div style="font-size:1rem;font-weight:700;color:#00e676">涨 {up_p:.1%}</div>'
        f'<div style="font-size:1rem;font-weight:700;color:#ff5252">跌 {down_p:.1%}</div>'
        '</div>', unsafe_allow_html=True)
    c3.markdown(
        '<div class="glass-card" style="text-align:center;padding:16px">'
        '<div style="font-size:0.62rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">预测收益率</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{reg_color}">{pred_reg*100:+.2f}%</div>'
        '<div style="font-size:0.7rem;color:#64748b;margin-top:4px">XGBoost 回归</div>'
        '</div>', unsafe_allow_html=True)
    c4.markdown(
        '<div class="glass-card" style="text-align:center;padding:16px">'
        '<div style="font-size:0.62rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">模型准确率</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:#a78bfa">{accuracy:.1%}</div>'
        f'<div style="margin-top:6px">{align_html}</div>'
        '</div>', unsafe_allow_html=True)

    # SHAP Top10 因子图
    st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
    col_shap, col_prob = st.columns([1.6, 1], gap="medium")

    with col_shap:
        st.markdown('<div class="section-header" style="font-size:0.75rem">🔍 SHAP 因子贡献（Top 10）</div>',
                    unsafe_allow_html=True)
        names  = [s[0] for s in shap_top]
        values = [s[1] for s in shap_top]
        colors = ["#00e676" if v > 0 else "#ff5252" for v in values]
        fig_shap = go.Figure(go.Bar(
            x=values[::-1], y=names[::-1],
            orientation="h",
            marker_color=colors[::-1],
            marker_opacity=0.85,
            text=[f"{v:+.3f}" for v in values[::-1]],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        fig_shap.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color="#94a3b8")),
            showlegend=False, font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_prob:
        st.markdown('<div class="section-header" style="font-size:0.75rem">📊 涨跌概率分布</div>',
                    unsafe_allow_html=True)
        labels = ["看跌", "看涨"] if len(proba) == 2 else ["看跌", "震荡", "看涨"]
        prob_vals = [down_p, up_p] if len(proba) == 2 else [down_p, flat_p, up_p]
        prob_colors = ["#ff5252", "#00e676"] if len(proba) == 2 else ["#ff5252", "#ffd740", "#00e676"]
        fig_prob = go.Figure(go.Bar(
            x=labels, y=prob_vals,
            marker_color=prob_colors, marker_opacity=0.85,
            text=[f"{v:.1%}" for v in prob_vals],
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
        ))
        fig_prob.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%", range=[0, 1]),
            showlegend=False, font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown(
        '<div style="font-size:0.65rem;color:#334155;margin-top:4px">'
        '⚠️ XGBoost 模型基于 2022-2026 历史数据训练，方向准确率约 57.8%，仅供参考，不构成投资建议。'
        '</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 单资产面板
# ──────────────────────────────────────────────────────────────

def render_asset(ticker: str) -> None:
    with st.spinner(f"拉取数据 & Walk-Forward 训练 HMM…"):
        try:
            res = load_asset(ticker)
        except Exception as e:
            st.error(f"加载失败：{e}")
            return

    df       = res["df"]
    trades   = res["trades"]
    metrics  = res["metrics"]
    last     = df.iloc[-1]
    is_daily = res["is_daily"]
    n_states   = res.get("n_states",   N_STATES)
    min_conf   = res.get("min_conf",   MIN_CONFIRMATIONS)
    bull_top   = res.get("bull_top",   2)
    stop       = res.get("stop",       -0.08)
    adx_thresh = 20

    cur_regime        = last["regime_label"]
    cur_regime_filter = bool(last.get("regime_filter", False))
    cur_signal = "LONG" if (last["is_bull"] and last["signal_score"] >= min_conf) else "CASH"

    # ── 顶部三栏 ─────────────────────────────────────────────
    b1, b2, b3 = st.columns([1.8, 1.8, 3.4], gap="medium")

    with b1:
        sc = "signal-long" if cur_signal == "LONG" else "signal-cash"
        sv = "#00e676" if cur_signal == "LONG" else "#64748b"
        st.markdown(f"""<div class="{sc}">
            <div class="signal-title">当前信号</div>
            <div class="signal-value" style="color:{sv}">{cur_signal}</div>
            <div style="margin-top:8px;font-size:0.78rem;color:#475569">价格 ${last['Close']:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    with b2:
        pc = _pill(cur_regime)
        is_daily_txt = "日线" if is_daily else "1h"
        posterior = res.get("posterior", [])
        # determine action type
        if cur_signal == "LONG":
            action_type = "ENTRY"
            action_color = "#00e676"
        else:
            action_type = "CASH"
            action_color = "#64748b"
        # risk status based on regime filter + regime
        if "Bear" in cur_regime or "Crash" in cur_regime:
            risk_status = "HIGH RISK"
            risk_color  = "#ff5252"
        elif not cur_regime_filter:
            risk_status = "CAUTION"
            risk_color  = "#ffd740"
        else:
            risk_status = "NORMAL"
            risk_color  = "#00e676"
        # posterior bar for current regime
        n_post = len(posterior)
        top_post_idx = int(np.argmax(posterior)) if posterior else 0
        top_post_val = float(posterior[top_post_idx]) * 100 if posterior else 0.0
        post_html = ""
        if posterior:
            post_html = '<div style="margin-top:6px;font-size:0.68rem;color:#475569">HMM 后验置信度（最新bar）</div>'
            post_html += '<div style="display:flex;gap:3px;margin-top:3px;flex-wrap:wrap">'
            for i, p in enumerate(posterior):
                bar_pct = int(p * 100)
                is_top  = (i == top_post_idx)
                bar_col = "#00e676" if is_top else "rgba(96,165,250,0.4)"
                post_html += (f'<div title="State {i}: {p*100:.1f}%" style="flex:1;min-width:16px">'
                              f'<div style="background:{bar_col};height:{max(4,bar_pct//4)}px;border-radius:2px;opacity:0.85"></div>'
                              f'<div style="font-size:0.55rem;color:#475569;text-align:center">{i}</div></div>')
            post_html += '</div>'
            post_html += f'<div style="font-size:0.68rem;color:#60a5fa;margin-top:2px">最高后验 State {top_post_idx}: {top_post_val:.1f}%</div>'
        st.markdown(f"""<div class="signal-cash">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div class="signal-title">HMM 状态（Walk-Forward）</div>
                    <div style="margin-top:8px"><span class="regime-pill {pc}">{cur_regime}</span></div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:1px">Action</div>
                    <div style="font-size:1.1rem;font-weight:800;color:{action_color}">{action_type}</div>
                    <div style="font-size:0.62rem;color:{risk_color};margin-top:2px;font-weight:600">{risk_status}</div>
                </div>
            </div>
            <div style="margin-top:8px;font-size:0.72rem;color:#475569">
                {n_states}状态 · {bull_top}入场 · 阈值{min_conf} · 止损{stop*100:.0f}% · {is_daily_txt} · {len(df):,} bars
            </div>
            <div style="margin-top:4px;font-size:0.72rem">
                Regime Filter（诊断层）：<span style="color:{'#00e676' if cur_regime_filter else '#ffd740'};font-weight:600">
                {'✅ 趋势确认' if cur_regime_filter else '⚠️ 趋势待确认'}
                </span>
            </div>
            {post_html}
        </div>""", unsafe_allow_html=True)

    with b3:
        c1  = bool(last["rsi"]           < 90)
        c2  = bool(last["momentum"]      > 1.0)
        c3  = bool(last["volatility"]    < 6.0)
        c4  = bool(last["Volume"]        > last["vol_sma20"])
        c5  = bool(last["adx"]           > 25)
        c6  = bool(last["Close"]         > last["ema50"])
        c7  = bool(last["Close"]         > last["ema200"])
        c8  = bool(last["macd_line"]     > last["macd_signal"])
        c9  = bool(last["Close"]         > last["bb_mid"])
        c10 = bool(last["stoch_k"]       > last["stoch_d"] and last["stoch_k"] < 80)
        c11 = bool(last["williams_r"]    < -20)
        c12 = bool(last["cci"]           > 0)
        c13 = bool(last["obv"]           > last["obv_ema"])
        c14 = bool(last["pct_from_high"] > -30)
        # Core signals (trend & momentum) vs Confirmation signals (oscillator/volume)
        core_checks = [
            ("RSI < 90",          c1,  f"{last['rsi']:.1f}"),
            ("动量 > 1%",         c2,  f"{last['momentum']:.2f}%"),
            ("ADX > 25",          c5,  f"{last['adx']:.1f}"),
            ("价格 > EMA 50",     c6,  f"${last['ema50']:,.2f}"),
            ("价格 > EMA 200",    c7,  f"${last['ema200']:,.2f}"),
            ("MACD > Signal",     c8,  "Yes" if c8  else "No"),
            ("价格 > BB 中轨",    c9,  f"${last['bb_mid']:,.2f}"),
        ]
        conf_checks = [
            ("波动率 < 6%",       c3,  f"{last['volatility']:.2f}%"),
            ("成交量 > SMA20",    c4,  "Yes" if c4  else "No"),
            ("Stoch %K↑ & <80",  c10, f"K={last['stoch_k']:.1f}"),
            ("Williams %R < -20", c11, f"{last['williams_r']:.1f}"),
            ("CCI > 0",           c12, f"{last['cci']:.1f}"),
            ("OBV > OBV EMA",     c13, "Yes" if c13 else "No"),
            ("距高点 > -30%",     c14, f"{last['pct_from_high']:.1f}%"),
        ]
        checks = core_checks + conf_checks
        n_core = sum(v for _, v, _ in core_checks)
        n_conf = sum(v for _, v, _ in conf_checks)
        n      = n_core + n_conf
        pct    = n / len(checks)
        bar_w  = int(pct * 100)
        bar_c  = _score_color(pct)
        core_pct = n_core / len(core_checks)
        conf_pct = n_conf / len(conf_checks)
        st.markdown(f"""<div class="glass-card" style="padding:14px 18px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:500">
                    信号确认&nbsp;·&nbsp;仓位 {int(_position_size(n, min_conf)*100)}%
                </span>
                <span style="font-size:1.1rem;font-weight:800;color:{bar_c}">{n}/{len(checks)}</span>
            </div>
            <div class="score-outer"><div class="score-inner" style="width:{bar_w}%;background:{bar_c};opacity:0.85"></div></div>
            <div style="display:flex;gap:12px;font-size:0.67rem;margin-bottom:8px;margin-top:3px">
                <span>核心信号 <b style="color:{'#00e676' if core_pct>=0.7 else '#ffd740'}">{n_core}/{len(core_checks)}</b></span>
                <span>确认信号 <b style="color:{'#60a5fa' if conf_pct>=0.6 else '#475569'}">{n_conf}/{len(conf_checks)}</b></span>
                <span style="color:{'#00e676' if n>=min_conf else '#ff5252'}">
                    {'✅ 满足入场条件' if n >= min_conf else f'⚠️ 还差 {min_conf - n} 条'}
                </span>
            </div>""", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.markdown(
            '<div style="font-size:0.62rem;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px">核心信号</div>' +
            "".join(_sig_row(nm, ok, vl) for nm, ok, vl in core_checks), unsafe_allow_html=True)
        col_b.markdown(
            '<div style="font-size:0.62rem;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px">确认信号</div>' +
            "".join(_sig_row(nm, ok, vl) for nm, ok, vl in conf_checks), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── K线 + Volume + RSI ──────────────────────────────────
    st.markdown('<div class="section-header">📊 K线图 · Volume · RSI&nbsp;&nbsp;<span style="font-size:0.75rem;color:#475569;font-weight:400">绿=Bull Run · 浅绿=Bull+ · 蓝=Warming Up · 红=Bear</span></div>', unsafe_allow_html=True)
    st.plotly_chart(candle_chart(df, trades, ticker), use_container_width=True)

    # ── MACD + 信号强度 ──────────────────────────────────────
    st.markdown('<div class="section-header">📉 MACD &amp; 信号强度时序</div>', unsafe_allow_html=True)
    st.plotly_chart(macd_signal_chart(df, min_conf), use_container_width=True)

    # ── Stochastic + CCI ─────────────────────────────────────
    st.markdown('<div class="section-header">🔀 随机震荡指标 &amp; CCI</div>', unsafe_allow_html=True)
    st.plotly_chart(stoch_cci_chart(df), use_container_width=True)

    # ── 绩效指标 Row 1 ────────────────────────────────────────
    st.markdown('<div class="section-header">📈 回测绩效</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    rc = "green" if metrics["total_return_pct"] > 0 else "red"
    ac = "green" if metrics["alpha_pct"] > 0 else "red"
    with cols[0]: st.markdown(_metric("总收益",      f"{metrics['total_return_pct']:+.1f}%",
                                                      f"年化 {metrics['ann_return_pct']:+.1f}%", rc), unsafe_allow_html=True)
    with cols[1]: st.markdown(_metric("vs B&H Alpha", f"{metrics['alpha_pct']:+.1f}%",
                                                      f"B&H {metrics['bh_return_pct']:+.1f}%", ac), unsafe_allow_html=True)
    with cols[2]: st.markdown(_metric("最大回撤",    f"{metrics['max_drawdown_pct']:.1f}%",
                                                      "峰值→谷值", "red"), unsafe_allow_html=True)
    with cols[3]: st.markdown(_metric("最终资本",    f"${metrics['final_capital']:,.0f}",
                                                      f"起始 ${STARTING_CAP:,.0f} · {LEVERAGE}×", "yellow"), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    cols2 = st.columns(4, gap="small")
    sh_c = "green" if metrics["sharpe"] > 1 else "yellow" if metrics["sharpe"] > 0 else "red"
    ca_c = "green" if metrics["calmar"] > 1 else "yellow" if metrics["calmar"] > 0 else "red"
    sa_v = f"{metrics['spy_alpha_pct']:+.1f}%" if metrics["spy_alpha_pct"] is not None else "N/A"
    sa_s = f"SPY {metrics['spy_bh_pct']:+.1f}%" if metrics["spy_bh_pct"] is not None else ""
    sa_c = "green" if (metrics["spy_alpha_pct"] or 0) > 0 else "red"
    with cols2[0]: st.markdown(_metric("夏普比率",   f"{metrics['sharpe']:.2f}",
                                                      f"年化波动 {metrics['ann_vol_pct']:.1f}%", sh_c), unsafe_allow_html=True)
    with cols2[1]: st.markdown(_metric("卡玛比率",   f"{metrics['calmar']:.2f}",
                                                      "年化收益 / 最大回撤", ca_c), unsafe_allow_html=True)
    with cols2[2]: st.markdown(_metric("月度胜率",   f"{metrics['monthly_win_pct']:.1f}%",
                                                      f"交易胜率 {metrics['win_rate_pct']:.1f}%", "blue"), unsafe_allow_html=True)
    with cols2[3]: st.markdown(_metric("vs SPY Alpha", sa_v, sa_s, sa_c), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── 绩效指标 Row 3：新增高级指标 ─────────────────────────
    cols3 = st.columns(4, gap="small")
    so_c  = "green" if metrics["sortino"] > 1 else "yellow" if metrics["sortino"] > 0 else "red"
    pf_c  = "green" if metrics["profit_factor"] > 1.5 else "yellow" if metrics["profit_factor"] > 1 else "red"
    ex_c  = "green" if metrics["expectancy"] > 0 else "red"
    tr_c  = "green" if metrics["tail_ratio"] > 1 else "yellow"
    with cols3[0]: st.markdown(_metric("Sortino 比率",  f"{metrics['sortino']:.2f}",
                                                         f"下行波动率标准化", so_c), unsafe_allow_html=True)
    with cols3[1]: st.markdown(_metric("Profit Factor", f"{metrics['profit_factor']:.2f}",
                                                         f"总盈利 / 总亏损", pf_c), unsafe_allow_html=True)
    with cols3[2]: st.markdown(_metric("期望值/笔",     f"${metrics['expectancy']:+.0f}",
                                                         f"盈亏比 {metrics['rr_ratio']:.2f}×", ex_c), unsafe_allow_html=True)
    with cols3[3]: st.markdown(_metric("Tail Ratio",    f"{metrics['tail_ratio']:.2f}",
                                                         "P95收益 / P5亏损", tr_c), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    cols4 = st.columns(4, gap="small")
    cl_c  = "green" if metrics["max_consec_loss"] <= 2 else "yellow" if metrics["max_consec_loss"] <= 4 else "red"
    sk_c  = "green" if metrics["skewness"] > 0 else "yellow"
    rc_label = f"{metrics['max_recovery_bars']}{'日' if is_daily else 'h'}"
    with cols4[0]: st.markdown(_metric("最大连续亏损",  f"{metrics['max_consec_loss']} 笔",
                                                         "连续止损次数上限", cl_c), unsafe_allow_html=True)
    with cols4[1]: st.markdown(_metric("平均持仓",      f"{metrics['avg_hold_bars']:.0f} bars",
                                                         f"平均仓位 {metrics['avg_pos_size_pct']:.0f}%", "blue"), unsafe_allow_html=True)
    with cols4[2]: st.markdown(_metric("收益偏度",      f"{metrics['skewness']:+.2f}",
                                                         f"峰度 {metrics['kurtosis']:.2f}", sk_c), unsafe_allow_html=True)
    with cols4[3]: st.markdown(_metric("最长回撤修复",  rc_label,
                                                         "峰值→修复所需时间", "yellow"), unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 资金曲线 + 回撤 ──────────────────────────────────────
    st.markdown('<div class="section-header">💰 资金曲线 vs 买入持有 vs SPY</div>', unsafe_allow_html=True)
    st.plotly_chart(equity_chart(df), use_container_width=True)

    # ── 滚动夏普 ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 滚动夏普比率</div>', unsafe_allow_html=True)
    st.plotly_chart(rolling_sharpe_chart(df, is_daily), use_container_width=True)

    # ── Relative Alpha Curve ──────────────────────────────────
    st.markdown('<div class="section-header">📐 相对 Alpha 曲线（策略净值 / B&H 净值）</div>', unsafe_allow_html=True)
    st.plotly_chart(relative_alpha_chart(df), use_container_width=True)

    # ── Underwater Plot ───────────────────────────────────────
    st.markdown('<div class="section-header">🌊 Underwater 回撤曲线</div>', unsafe_allow_html=True)
    st.plotly_chart(underwater_chart(df), use_container_width=True)

    # ── 月度热力图（含 BH / Alpha 标签） + 状态分布 ──────────
    col_heat, col_dist = st.columns([3, 2], gap="medium")
    with col_heat:
        st.markdown('<div class="section-header">🗓 月度收益热力图（Strategy / B&H / Alpha）</div>', unsafe_allow_html=True)
        monthly_heatmap_tabbed(metrics["monthly_df"])
    with col_dist:
        st.markdown('<div class="section-header">🧩 HMM 状态分布</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_bar(df), use_container_width=True)

    # ── 宏观特征可视化 ────────────────────────────────────────
    from data_loader import MACRO_TABLES
    _macro_cols = [c for c in MACRO_TABLES.values() if c in df.columns]
    if _macro_cols:
        st.markdown('<div class="section-header">🌐 宏观指标时序（z-score · 背景色=Regime）</div>', unsafe_allow_html=True)
        st.plotly_chart(macro_timeseries_chart(df), use_container_width=True)
        st.markdown('<div class="section-header">📊 各 Regime 宏观特征均值对比</div>', unsafe_allow_html=True)
        st.plotly_chart(macro_by_regime_chart(df), use_container_width=True)

    # ── 各状态收益箱线图 ─────────────────────────────────────
    st.markdown('<div class="section-header">📦 各 HMM 状态收益率分布</div>', unsafe_allow_html=True)
    st.plotly_chart(regime_return_chart(df, n_states), use_container_width=True)

    # ── Regime Return Attribution ─────────────────────────────
    if trades:
        st.markdown('<div class="section-header">🔍 Regime 交易归因（各状态入场盈亏 & 胜率）</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_attribution_chart(df, trades), use_container_width=True)

    # ── Exit Reason Breakdown ─────────────────────────────────
    exit_attr = metrics.get("exit_attribution", {})
    if exit_attr:
        st.markdown('<div class="section-header">🚪 出场原因归因</div>', unsafe_allow_html=True)
        st.plotly_chart(exit_attribution_chart(exit_attr), use_container_width=True)

    # ── Top Trade Contribution ────────────────────────────────
    if trades:
        st.markdown('<div class="section-header">🏆 Top Trade 贡献度</div>', unsafe_allow_html=True)
        m = metrics
        top5_s  = f"{m['top5_contrib_pct']:.1f}%" if "top5_contrib_pct" in m else "N/A"
        top10_s = f"{m['top10_contrib_pct']:.1f}%" if "top10_contrib_pct" in m else "N/A"
        st.markdown(
            f'<div style="font-size:0.75rem;color:#94a3b8;margin-bottom:4px">'
            f'Top 5 交易贡献度 <b style="color:#ffd740">{top5_s}</b> &nbsp;·&nbsp; '
            f'Top 10 交易贡献度 <b style="color:#ffd740">{top10_s}</b></div>',
            unsafe_allow_html=True)
        st.plotly_chart(top_trade_chart(trades), use_container_width=True)

    # ── 交易分析（单笔盈亏）+ 持仓时长分布 ────────────────────
    if trades:
        c_pnl, c_hold = st.columns([1.4, 1], gap="medium")
        with c_pnl:
            st.markdown('<div class="section-header">🎯 单笔盈亏分析</div>', unsafe_allow_html=True)
            st.plotly_chart(trade_analytics_chart(trades), use_container_width=True)
        with c_hold:
            st.markdown('<div class="section-header">⏱ 持仓时长分布</div>', unsafe_allow_html=True)
            st.plotly_chart(hold_duration_chart(trades, is_daily), use_container_width=True)

    # ── 交易统计 + 风控参数 ───────────────────────────────────
    col_stats, col_risk = st.columns([1, 1], gap="medium")
    with col_stats:
        st.markdown('<div class="section-header">📋 交易统计</div>', unsafe_allow_html=True)
        tdf_s = pd.DataFrame(trades) if trades else pd.DataFrame()
        best_t  = tdf_s["pnl"].max() if len(tdf_s) else 0
        worst_t = tdf_s["pnl"].min() if len(tdf_s) else 0
        stats = [
            ("总笔数",       f"{metrics['n_trades']}"),
            ("盈亏比 (R:R)", f"{metrics['rr_ratio']:.2f}"),
            ("平均盈利",     f"${metrics['avg_win']:+,.0f}"),
            ("平均亏损",     f"${metrics['avg_loss']:+,.0f}"),
            ("最优单笔",     f"${best_t:+,.0f}"),
            ("最差单笔",     f"${worst_t:+,.0f}"),
            ("平均仓位",     f"{metrics['avg_pos_size_pct']:.0f}%"),
        ]
        st.markdown("".join(
            f'<div class="sig-row"><span class="sig-name">{k}</span>'
            f'<span class="sig-val">{v}</span></div>' for k, v in stats
        ), unsafe_allow_html=True)

    with col_risk:
        st.markdown('<div class="section-header">🛡 风控参数</div>', unsafe_allow_html=True)
        cooldown_str = "2 日" if is_daily else "48 小时"
        max_hold_str = f"{int(60 * res.get('hold_mult', 1.0))} 日" if is_daily else f"{int(24*30 * res.get('hold_mult', 1.0))} 小时"
        risk_items = [
            ("HMM 状态数",      f"{n_states} States"),
            ("入场状态数",      f"Top {bull_top}"),
            ("信号阈值",        f"{min_conf} / 14"),
            ("固定止损",        f"{stop*100:.0f}%（触价退出）"),
            ("Regime Filter",   f"诊断层 · ADX>{adx_thresh} + EMA50↑ + EMA50>EMA200"),
            ("杠杆",            f"固定 {LEVERAGE}×"),
            ("冷静期",          cooldown_str),
            ("最大持仓",        max_hold_str),
            ("HMM 训练",        "Walk-Forward 滚动"),
            ("仓位管理",        "信号强度线性 40%→100%"),
        ]
        st.markdown("".join(
            f'<div class="sig-row"><span class="sig-name">{k}</span>'
            f'<span class="sig-val">{v}</span></div>' for k, v in risk_items
        ), unsafe_allow_html=True)

    # ── 完整交易记录（含筛选）────────────────────────────────
    if trades:
        st.markdown('<div class="section-header">📝 完整交易记录</div>', unsafe_allow_html=True)
        tdf_raw = pd.DataFrame(trades)
        # attach regime at entry
        tdf_raw["entry_regime"] = df["regime_label"].reindex(tdf_raw["entry_time"]).values

        # ── 筛选器 ────────────────────────────────────────────
        flt1, flt2, flt3 = st.columns([2, 2, 2], gap="small")
        with flt1:
            exit_reasons = ["全部"] + sorted(tdf_raw["exit_reason"].unique().tolist())
            sel_exit = st.selectbox("出场原因", exit_reasons, key=f"exit_flt_{ticker}")
        with flt2:
            regimes_opts = ["全部"] + sorted(tdf_raw["entry_regime"].dropna().unique().tolist())
            sel_regime = st.selectbox("入场 Regime", regimes_opts, key=f"regime_flt_{ticker}")
        with flt3:
            pnl_filter = st.selectbox("盈亏方向", ["全部", "仅盈利", "仅亏损"], key=f"pnl_flt_{ticker}")

        tdf_flt = tdf_raw.copy()
        if sel_exit != "全部":
            tdf_flt = tdf_flt[tdf_flt["exit_reason"] == sel_exit]
        if sel_regime != "全部":
            tdf_flt = tdf_flt[tdf_flt["entry_regime"] == sel_regime]
        if pnl_filter == "仅盈利":
            tdf_flt = tdf_flt[tdf_flt["pnl"] > 0]
        elif pnl_filter == "仅亏损":
            tdf_flt = tdf_flt[tdf_flt["pnl"] <= 0]

        st.caption(f"显示 {len(tdf_flt)} / {len(tdf_raw)} 笔交易")

        tdf_show = tdf_flt.copy()
        tdf_show["entry_time"]   = pd.to_datetime(tdf_show["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tdf_show["exit_time"]    = pd.to_datetime(tdf_show["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tdf_show["entry_price"]  = tdf_show["entry_price"].map("${:,.2f}".format)
        tdf_show["exit_price"]   = tdf_show["exit_price"].map("${:,.2f}".format)
        tdf_show["pnl"]          = tdf_show["pnl"].map("${:+,.2f}".format)
        tdf_show["pos_size_pct"] = tdf_show["pos_size_pct"].map(lambda x: f"{x*100:.0f}%")
        tdf_show["return_pct"]   = tdf_show["return_pct"].map(lambda x: f"{x:+.1f}%")
        tdf_show = tdf_show[["entry_time","exit_time","entry_regime","entry_price","exit_price",
                              "pos_size_pct","pnl","return_pct","hold_bars","exit_reason"]]
        tdf_show.columns = ["入场时间","出场时间","入场Regime","入场价","出场价","仓位","盈亏","收益率","持仓bar","出场原因"]
        st.dataframe(tdf_show, use_container_width=True, hide_index=True)

    # ── XGBoost 多因子预测（仅 Gold）────────────────────────────
    if ticker == "GC=F":
        render_xgb_panel()


# ──────────────────────────────────────────────────────────────
# 今日信号面板
# ──────────────────────────────────────────────────────────────

def _load_latest_signal() -> dict | None:
    sig_dir = os.path.join(os.path.dirname(__file__), "signals")
    files   = sorted(glob.glob(os.path.join(sig_dir, "signal_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _action_badge(action: str) -> str:
    colors = {
        "ENTER":    ("#00e676", "#002d16"),
        "HOLD":     ("#60a5fa", "#0c1a2e"),
        "EXIT":     ("#ff5252", "#2d0000"),
        "WATCH":    ("#ffd740", "#2d2200"),
        "STAY_OUT": ("#475569", "#111827"),
        "MarginCall": ("#ff5252", "#2d0000"),
    }
    fg, bg = colors.get(action, ("#94a3b8", "#1e2130"))
    return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}40;'
            f'border-radius:8px;padding:4px 14px;font-weight:700;font-size:0.9rem;'
            f'letter-spacing:.3px">{action}</span>')


def render_signals_tab() -> None:
    data = _load_latest_signal()
    if data is None:
        st.warning("未找到信号文件。请先运行 `python signal_generator.py`。")
        return

    gen_at  = data.get("generated_at", "")
    signals = data.get("signals", {})
    errors  = data.get("errors", {})

    try:
        gen_dt = datetime.fromisoformat(gen_at)
        gen_str = gen_dt.strftime("%Y-%m-%d  %H:%M")
    except Exception:
        gen_str = gen_at

    st.markdown(
        f'<div style="font-size:0.72rem;color:#475569;margin-bottom:1.2rem">'
        f'信号生成时间：<b style="color:#64748b">{gen_str}</b> &nbsp;·&nbsp; '
        f'运行 <code>python signal_generator.py</code> 刷新</div>',
        unsafe_allow_html=True)

    if errors:
        st.error(f"信号生成错误：{errors}")

    ticker_labels = {"GC=F": "🥇 Gold", "SI=F": "🥈 Silver", "AAPL": "🍎 Apple"}
    for ticker in ["GC=F", "SI=F", "AAPL"]:
        sig = signals.get(ticker)
        if not sig:
            continue

        label      = ticker_labels.get(ticker, ticker)
        af_flat    = sig.get("action_if_flat", "—")
        af_long    = sig.get("action_if_long", "—")
        regime     = sig.get("regime", "—")
        score      = sig.get("signal_score", 0)
        min_conf   = sig.get("min_conf", 9)
        adx        = sig.get("adx", 0)
        adx_entry  = sig.get("adx_entry", 25)
        bull_prob  = sig.get("bull_prob", 0)
        bear_prob  = sig.get("bear_prob", 0)
        close      = sig.get("close", 0)
        sw         = sig.get("sideways_score", 0)
        vt         = sig.get("vt_scale")
        details    = sig.get("signal_details", {})
        posterior  = sig.get("posterior", [])
        pc         = _pill(regime)

        st.markdown(f'<div class="section-header">{label} &nbsp;<span style="font-size:0.72rem;color:#475569;font-weight:400">{sig.get("date","")}</span></div>', unsafe_allow_html=True)

        ca, cb, cc = st.columns([1.4, 1.4, 3.2], gap="medium")

        with ca:
            st.markdown(f"""<div class="glass-card" style="text-align:center;padding:18px">
                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">如果空仓</div>
                {_action_badge(af_flat)}
                <div style="margin-top:14px;font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">如果持仓</div>
                {_action_badge(af_long)}
                <div style="margin-top:12px;font-size:0.72rem;color:#64748b">收盘价 <b style="color:#e2e8f0">${close:,.2f}</b></div>
            </div>""", unsafe_allow_html=True)

        with cb:
            bull_bar    = int(bull_prob * 100)
            bear_bar    = int(bear_prob * 100)
            score_color = '#00e676' if score >= min_conf else '#ffd740'
            adx_color   = '#00e676' if adx > adx_entry  else '#ff5252'
            vt_html     = f'&nbsp; VT <b style="color:#a78bfa">{vt:.3f}</b>' if vt else ''
            bars_html   = ""
            if posterior:
                top_idx = int(np.argmax(posterior))
                bars_html = '<div style="display:flex;gap:2px;margin-top:8px;align-items:flex-end">'
                for i, p in enumerate(posterior):
                    h   = max(4, int(p * 60))
                    col = "#00e676" if i == top_idx else "#3b82f6"
                    bars_html += (
                        '<div style="flex:1;display:flex;flex-direction:column;align-items:center">'
                        f'<div style="background:{col};height:{h}px;width:100%;border-radius:2px 2px 0 0"></div>'
                        f'<div style="font-size:0.5rem;color:#475569">{i}</div></div>'
                    )
                bars_html += '</div>'
            card = (
                '<div class="glass-card" style="padding:16px">'
                '<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">HMM Regime</div>'
                f'<span class="regime-pill {pc}">{regime}</span>'
                '<div style="margin-top:10px;font-size:0.72rem">'
                f'<span style="color:#00e676">Bull {bull_bar}%</span> &nbsp;·&nbsp; '
                f'<span style="color:#ff5252">Bear {bear_bar}%</span></div>'
                '<div style="margin-top:6px;font-size:0.72rem;color:#64748b">'
                f'Score <b style="color:{score_color}">{score}/14</b> &nbsp;'
                f'ADX <b style="color:{adx_color}">{adx:.1f}</b> &nbsp;'
                f'SW <b style="color:#64748b">{sw}/4</b>{vt_html}</div>'
                f'{bars_html}'
                '</div>'
            )
            st.markdown(card, unsafe_allow_html=True)

        with cc:
            det_items = [
                ("RSI < 90",          details.get("rsi_ok", False),       ""),
                ("动量 > 1%",         details.get("momentum_ok", False),   ""),
                ("波动率 < 6%",       details.get("vol_ok", False),        ""),
                ("成交量 > SMA20",    details.get("volume_ok", False),     ""),
                ("ADX > 25",          details.get("adx_ok", False),        ""),
                ("价格 > EMA 50",     details.get("above_ema50", False),   ""),
                ("价格 > EMA 200",    details.get("above_ema200", False),  ""),
                ("MACD > Signal",     details.get("macd_ok", False),       ""),
                ("价格 > BB 中轨",    details.get("above_bb_mid", False),  ""),
                ("Stoch %K↑ & <80",  details.get("stoch_ok", False),      ""),
                ("Williams %R < -20", details.get("williams_ok", False),   ""),
                ("CCI > 0",           details.get("cci_ok", False),        ""),
                ("OBV > OBV EMA",     details.get("obv_ok", False),        ""),
                ("距高点 > -30%",     details.get("drawdown_ok", False),   ""),
            ]
            n_pass = sum(1 for _, ok, _ in det_items if ok)
            pct    = n_pass / len(det_items)
            bar_c  = _score_color(pct)
            left_rows  = "".join(_sig_row(nm, ok, vl) for nm, ok, vl in det_items[:7])
            right_rows = "".join(_sig_row(nm, ok, vl) for nm, ok, vl in det_items[7:])
            st.markdown(f"""<div class="glass-card" style="padding:14px 18px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                    <span style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px">14 信号明细</span>
                    <span style="font-size:1rem;font-weight:800;color:{bar_c}">{n_pass}/{len(det_items)}</span>
                </div>
                <div class="score-outer"><div class="score-inner" style="width:{int(pct*100)}%;background:{bar_c};opacity:0.85"></div></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 12px;margin-top:6px">
                    <div>{left_rows}</div>
                    <div>{right_rows}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 组合面板
# ──────────────────────────────────────────────────────────────

def portfolio_equity_chart(eq_dict: dict) -> go.Figure:
    tickers = list(eq_dict.keys())
    idx     = list(eq_dict.values())[0].index
    rets    = {t: eq_dict[t].reindex(idx, method="ffill").fillna(STARTING_CAP) / STARTING_CAP
               for t in tickers}
    port    = sum(rets[t] for t in tickers) / 3 * STARTING_CAP

    fig = go.Figure()
    colors = {"GC=F": "#ffd740", "SI=F": "#94a3b8", "AAPL": "#60a5fa"}
    for t in tickers:
        fig.add_trace(go.Scatter(
            x=idx, y=rets[t] * STARTING_CAP,
            mode="lines", name=t,
            line=dict(color=colors.get(t, "#94a3b8"), width=1.2, dash="dot"),
            opacity=0.6))
    fig.add_trace(go.Scatter(
        x=idx, y=port, mode="lines", name="组合（等权）",
        line=dict(color="#00e676", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.05)"))

    roll_max = port.cummax()
    dd       = (port - roll_max) / roll_max * 100
    total_ret  = (port.iloc[-1] / STARTING_CAP - 1) * 100
    dr         = port.pct_change().dropna()
    sharpe     = dr.mean() / dr.std() * np.sqrt(252)
    max_dd     = dd.min()

    fig.update_layout(
        **_base_layout(height=380),
        title=dict(
            text=f"等权组合   Return {total_ret:+.1f}%   Sharpe {sharpe:.2f}   MaxDD {max_dd:.1f}%",
            font=dict(size=12, color="#94a3b8"), x=0, xanchor="left"),
        yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$"),
        xaxis=dict(gridcolor=GRID_COLOR))
    return fig


def render_portfolio_tab() -> None:
    st.markdown('<div class="section-header">📊 多资产组合（等权 1/3 each）</div>', unsafe_allow_html=True)

    tickers = ["GC=F", "SI=F", "AAPL"]
    fnames  = {"GC=F": "GC_F", "SI=F": "SI_F", "AAPL": "AAPL"}

    eq_curves = {}
    metrics_all = {}
    for t in tickers:
        r = _load_precomputed(t)
        if r is None:
            st.warning(f"{t} 预计算数据缺失，请先运行 precompute.py")
            return
        eq_curves[t]   = pd.Series(r["df"]["equity"].values, index=r["df"].index)
        metrics_all[t] = r["metrics"]

    # 组合资金曲线
    st.plotly_chart(portfolio_equity_chart(eq_curves), use_container_width=True)

    # 单资产 vs 组合绩效对比
    st.markdown('<div class="section-header">📋 绩效对比</div>', unsafe_allow_html=True)

    idx  = eq_curves["GC=F"].index
    rets = {t: eq_curves[t].reindex(idx, method="ffill").fillna(STARTING_CAP) / STARTING_CAP
            for t in tickers}
    port = sum(rets[t] for t in tickers) / 3 * STARTING_CAP

    def _port_metrics(eq):
        tr  = (eq.iloc[-1] / STARTING_CAP - 1) * 100
        rm  = eq.cummax(); mdd = ((eq-rm)/rm*100).min()
        dr  = eq.pct_change().dropna()
        sh  = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        cal = tr / abs(mdd) if mdd != 0 else 0
        return tr, sh, mdd, cal

    rows = []
    for t in tickers:
        m = metrics_all[t]
        rows.append({"资产": t,
                     "Return": f"{m['total_return_pct']:+.1f}%",
                     "Sharpe": f"{m['sharpe']:.2f}",
                     "MaxDD":  f"{m['max_drawdown_pct']:.1f}%",
                     "Calmar": f"{m['calmar']:.1f}",
                     "Trades": m["n_trades"]})
    ptr, psh, pmdd, pcal = _port_metrics(port)
    rows.append({"资产": "🟢 Portfolio (1/3)",
                 "Return": f"{ptr:+.1f}%",
                 "Sharpe": f"{psh:.2f}",
                 "MaxDD":  f"{pmdd:.1f}%",
                 "Calmar": f"{pcal:.1f}",
                 "Trades": "—"})
    df_rows = pd.DataFrame(rows)
    df_rows["Trades"] = df_rows["Trades"].astype(str)
    st.dataframe(df_rows, use_container_width=True, hide_index=True)

    # 摩擦成本说明
    st.markdown('<div class="section-header">💸 摩擦成本 & 保证金参数</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"""<div class="glass-card">
            <div class="section-header" style="margin-top:0">摩擦模型（统一百分比）</div>
            <div class="sig-row"><span class="sig-name">Slippage</span><span class="sig-val">0.05% per side</span></div>
            <div class="sig-row"><span class="sig-name">Commission</span><span class="sig-val">0.05% per side</span></div>
            <div class="sig-row"><span class="sig-name">合计</span><span class="sig-val">0.10% per side · 0.20% round trip</span></div>
            <div class="sig-row"><span class="sig-name">组合 Sharpe 影响</span><span class="sig-val" style="color:#ffd740">1.76 → 1.75（-0.017）</span></div>
            <div class="sig-row"><span class="sig-name">组合 MaxDD 影响</span><span class="sig-val" style="color:#00e676">不变（-19.0%）</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        rows_m = []
        for t in tickers:
            mp = MARGIN_PARAMS.get(t, {})
            rows_m.append({
                "资产": t,
                "Initial Margin": f"{mp.get('initial_margin',0):.0%}",
                "Maintenance Margin": f"{mp.get('maintenance_margin',0):.0%}",
                "MarginCall 次数（10年）": "0",
                "结论": "Stop先触发"
            })
        st.dataframe(pd.DataFrame(rows_m), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    logo = _logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo}" '
        f'style="height:44px;width:44px;border-radius:10px;object-fit:cover;'
        f'box-shadow:0 0 16px rgba(255,255,255,0.08);flex-shrink:0;" />'
        if logo else ""
    )

    hc1, hc2 = st.columns([8, 1])
    with hc1:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:2px">
            {logo_html}
            <div>
                <div class="page-title">Regime-Based HMM Trading Dashboard</div>
                <div style="font-size:0.62rem;color:#334155;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;margin-top:1px">LILYN &nbsp;·&nbsp; AI Quant Strategy</div>
            </div>
        </div>
        <div class="page-sub" style="margin-left:{58 if logo else 0}px">
            Gaussian HMM &nbsp;·&nbsp; 14-Signal Voting &nbsp;·&nbsp;
            {LEVERAGE}× Leverage &nbsp;·&nbsp; Walk-Forward &nbsp;·&nbsp; 数据截至 {_computed_at()}
        </div>""", unsafe_allow_html=True)
    with hc2:
        st.write("")
        st.write("")
        if st.button("🔄 刷新", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    tab_sig, tab_port, tab_aapl, tab_gold, tab_silver = st.tabs([
        "📡  今日信号",
        "🌐  组合",
        "🍎  Apple (AAPL)",
        "🥇  Gold (GC=F)",
        "🥈  Silver (SI=F)",
    ])
    with tab_sig:    render_signals_tab()
    with tab_port:   render_portfolio_tab()
    with tab_aapl:   render_asset("AAPL")
    with tab_gold:   render_asset("GC=F")
    with tab_silver: render_asset("SI=F")

    st.markdown(
        "<div style='text-align:center;color:#1e293b;font-size:0.7rem;margin-top:2rem'>"
        "仅供学习研究，不构成投资建议。</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
