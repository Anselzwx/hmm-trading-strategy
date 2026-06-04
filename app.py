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
                          FRICTION_PCT, MARGIN_PARAMS, LEVERAGE,
                          ATR_TRAIL_MULT, ENABLE_SHORT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ASSETS_DIR  = os.path.join(os.path.dirname(__file__), "assets")


def _safe_filename(ticker: str) -> str:
    return ticker.replace("=", "_").replace("/", "_")

@st.cache_data(show_spinner=False)
def _load_precomputed(ticker: str, _mtime: float = 0):
    path = os.path.join(RESULTS_DIR, f"{_safe_filename(ticker)}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _load_result(ticker: str):
    path = os.path.join(RESULTS_DIR, f"{_safe_filename(ticker)}.pkl")
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    return _load_precomputed(ticker, _mtime=mtime)

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

def _get_theme():
    return st.session_state.get("_theme", "dark")

def _theme_vars():
    t = _get_theme()
    if t == "light":
        return {
            "bg": "#f8fafc", "card_bg": "rgba(0,0,0,0.03)", "card_border": "rgba(0,0,0,0.08)",
            "text": "#0f172a", "sub": "#64748b", "grid": "rgba(0,0,0,0.06)",
            "chart_bg": "#f8fafc", "nav_bg": "rgba(0,0,0,0.04)", "nav_sel": "rgba(0,0,0,0.10)",
        }
    return {
        "bg": "#080c14", "card_bg": "rgba(255,255,255,0.04)", "card_border": "rgba(255,255,255,0.08)",
        "text": "#f1f5f9", "sub": "#475569", "grid": "rgba(255,255,255,0.05)",
        "chart_bg": "#080c14", "nav_bg": "rgba(255,255,255,0.03)", "nav_sel": "rgba(255,255,255,0.08)",
    }

def _inject_css():
    t = _get_theme()
    v = _theme_vars()
    is_light = (t == "light")
    metric_label_c  = "#475569" if is_light else "#64748b"
    metric_sub_c    = "#64748b" if is_light else "#475569"
    sig_name_c      = "#475569" if is_light else "#94a3b8"
    sig_val_c       = "#1e293b" if is_light else "#cbd5e1"
    sig_row_bg      = "rgba(0,0,0,0.03)" if is_light else "rgba(255,255,255,0.025)"
    sig_row_bor     = "rgba(0,0,0,0.07)" if is_light else "rgba(255,255,255,0.05)"
    sig_row_hov     = "rgba(0,0,0,0.06)" if is_light else "rgba(255,255,255,0.045)"
    section_c       = "#1e293b" if is_light else "#e2e8f0"
    section_bor     = "rgba(0,0,0,0.10)" if is_light else "rgba(255,255,255,0.07)"
    page_title_c    = "#0f172a" if is_light else "#f1f5f9"
    page_sub_c      = "#64748b" if is_light else "#475569"
    tab_c           = "#475569" if is_light else "#64748b"
    tab_sel_c       = "#0f172a" if is_light else "#e2e8f0"
    signal_long_bg  = "linear-gradient(135deg,#d1fae5,#a7f3d0)" if is_light else "linear-gradient(135deg,#002d16,#004d24)"
    signal_long_bor = "rgba(0,150,80,0.4)" if is_light else "rgba(0,230,118,0.4)"
    signal_long_sh  = "0 0 20px rgba(0,150,80,0.08)" if is_light else "0 0 40px rgba(0,230,118,0.12),inset 0 1px 0 rgba(0,230,118,0.15)"
    signal_cash_bg  = "linear-gradient(135deg,#f1f5f9,#e2e8f0)" if is_light else "linear-gradient(135deg,#0f1420,#141929)"
    signal_cash_bor = "rgba(100,116,139,0.3)"
    signal_cash_sh  = "0 2px 8px rgba(0,0,0,0.06)" if is_light else "0 4px 24px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04)"
    signal_title_c  = "#475569" if is_light else "#64748b"
    regime_bull_bg  = "rgba(0,150,60,0.10)"  if is_light else "rgba(0,230,118,0.12)"
    regime_bull_c   = "#006633"              if is_light else "#00c864"
    regime_bear_bg  = "rgba(200,30,30,0.10)" if is_light else "rgba(255,82,82,0.12)"
    regime_bear_c   = "#b91c1c"              if is_light else "#ff5252"
    regime_neut_bg  = "rgba(146,100,0,0.10)" if is_light else "rgba(255,215,64,0.10)"
    regime_neut_c   = "#92400e"              if is_light else "#d97706"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; background: {v['bg']}; color: {v['text']}; }}
.block-container {{ padding: 1rem 2rem 3rem 2rem; max-width: 1600px; }}

.glass-card {{
    background: linear-gradient(135deg,{v['card_bg']},{v['card_bg']});
    border: 1px solid {v['card_border']}; border-radius: 16px; padding: 20px 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,{'0.1' if is_light else '0.4'}), inset 0 1px 0 {v['card_border']};
    margin-bottom: 2px;
}}
.metric-card {{
    background: linear-gradient(135deg,{v['card_bg']},{v['card_bg']});
    border: 1px solid {v['card_border']}; border-radius: 14px; padding: 18px 16px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,{'0.06' if is_light else '0.5'}), inset 0 1px 0 {v['card_border']};
    transition: transform .15s ease, box-shadow .15s ease;
}}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,{'0.12' if is_light else '0.6'}); }}
.metric-label {{ font-size: 0.68rem; color: {metric_label_c}; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 8px; font-weight: 500; }}
.metric-value {{ font-size: 1.9rem; font-weight: 800; line-height: 1; }}
.metric-sub   {{ font-size: 0.7rem; color: {metric_sub_c}; margin-top: 6px; font-weight: 400; }}

.signal-long {{
    background: {signal_long_bg};
    border: 1px solid {signal_long_bor}; border-radius: 16px; padding: 22px 28px; text-align: center;
    box-shadow: {signal_long_sh};
}}
.signal-cash {{
    background: {signal_cash_bg};
    border: 1px solid {signal_cash_bor}; border-radius: 16px; padding: 22px 28px; text-align: center;
    box-shadow: {signal_cash_sh};
}}
.signal-title {{ font-size: 0.7rem; color: {signal_title_c}; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 10px; font-weight:500; }}
.signal-value {{ font-size: 2.4rem; font-weight: 900; letter-spacing: -1px; }}

.regime-pill  {{ display:inline-block; padding:6px 20px; border-radius:30px; font-size:1rem; font-weight:700; margin-top:6px; letter-spacing:.3px; }}
.regime-bull  {{ background:{regime_bull_bg}; color:{regime_bull_c}; border:1px solid {regime_bull_c}66; box-shadow:0 0 20px {regime_bull_bg}; }}
.regime-bear  {{ background:{regime_bear_bg}; color:{regime_bear_c}; border:1px solid {regime_bear_c}66; box-shadow:0 0 20px {regime_bear_bg}; }}
.regime-neut  {{ background:{regime_neut_bg}; color:{regime_neut_c}; border:1px solid {regime_neut_c}66; box-shadow:0 0 20px {regime_neut_bg}; }}

.sig-row {{
    display:flex; align-items:center; justify-content:space-between;
    padding: 7px 12px; border-radius: 8px; margin-bottom: 4px;
    background: {sig_row_bg}; border: 1px solid {sig_row_bor};
    font-size: 0.8rem; transition: background .1s;
}}
.sig-row:hover {{ background: {sig_row_hov}; }}
.sig-name {{ color: {sig_name_c}; font-weight: 500; }}
.sig-val  {{ color: {sig_val_c}; font-family: 'SF Mono', monospace; font-size: 0.75rem; }}
.sig-pass {{ color: #00c864; font-size: 1rem; }}
.sig-fail {{ color: #ff5252; font-size: 1rem; }}

.section-header {{
    color: {section_c}; font-size: 0.9rem; font-weight: 600;
    margin: 1.6rem 0 0.7rem 0; padding-bottom: 8px;
    border-bottom: 1px solid {section_bor};
    letter-spacing: .3px; display: flex; align-items: center; gap: 8px;
}}
.score-outer {{ background: {'rgba(0,0,0,0.06)' if is_light else 'rgba(255,255,255,0.06)'}; border-radius: 8px; height: 8px; width: 100%; margin: 8px 0 4px 0; overflow: hidden; }}
.score-inner {{ height: 100%; border-radius: 8px; transition: width .4s ease; }}
.page-title  {{ font-size: 1.55rem; font-weight: 800; color: {page_title_c}; letter-spacing: -0.5px; line-height: 1.2; }}
.page-sub    {{ font-size: 0.78rem; color: {page_sub_c}; margin-top: 3px; font-weight: 400; }}

[data-baseweb="tab-list"] {{ background: {v['nav_bg']} !important; border-radius: 12px !important; padding: 4px !important; border: 1px solid {v['card_border']} !important; gap: 2px !important; }}
[data-baseweb="tab"]      {{ border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important; color: {tab_c} !important; padding: 8px 20px !important; }}
[aria-selected="true"]    {{ background: {v['nav_sel']} !important; color: {tab_sel_c} !important; }}

.green  {{ color: #00c864; }} .red  {{ color: #ff5252; }} .yellow {{ color: #d97706; }}
.blue   {{ color: #2563eb; }} .purple {{ color: #7c3aed; }} .white  {{ color: {page_title_c}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

/* Mobile responsive */
@media (max-width: 768px) {{
    .block-container {{ padding: 0.5rem 0.8rem 2rem 0.8rem; }}
    .metric-value {{ font-size: 1.3rem; }}
    .signal-value {{ font-size: 1.8rem; }}
    .page-title {{ font-size: 1.1rem; }}
    .glass-card {{ padding: 12px 14px; }}
    .metric-card {{ padding: 12px 10px; }}
    div[data-testid="column"] {{ min-width: 0 !important; }}
}}
@media (max-width: 480px) {{
    .metric-label {{ font-size: 0.58rem; letter-spacing: 0.8px; }}
    .metric-value {{ font-size: 1.1rem; }}
    .sig-row {{ padding: 5px 8px; font-size: 0.72rem; }}
    .section-header {{ font-size: 0.8rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 缓存
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_asset(ticker: str) -> dict:
    precomputed = _load_result(ticker)
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

def _chart_colors():
    t = st.session_state.get("_theme", "dark")
    if t == "light":
        return "#ffffff", "rgba(0,0,0,0.06)", "#475569", "#f1f5f9", "#1e293b"
    return "#080c14", "rgba(255,255,255,0.05)", "#94a3b8", "#1e2535", "#334155"

CHART_BG   = "#080c14"
GRID_COLOR = "rgba(255,255,255,0.05)"

def _gc() -> str:
    _, grid, *_ = _chart_colors()
    return grid

def _base_layout(**kw) -> dict:
    bg, grid, font_c, hover_bg, hover_bor = _chart_colors()
    return dict(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=font_c, size=11, family="Inter"),
        margin=dict(l=8, r=8, t=44, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=hover_bg, bordercolor=hover_bor, font_size=12),
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
    layout["xaxis"]  = dict(rangeslider=dict(visible=False), gridcolor=_gc(), showgrid=True, type="date")
    layout["yaxis"]  = dict(gridcolor=_gc(), showgrid=True)
    layout["xaxis2"] = dict(gridcolor=_gc(), showgrid=True)
    layout["yaxis2"] = dict(gridcolor=_gc(), showgrid=True, showticklabels=False)
    layout["xaxis3"] = dict(gridcolor=_gc(), showgrid=True)
    layout["yaxis3"] = dict(gridcolor=_gc(), showgrid=True, range=[0,100],
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
    layout["yaxis"]  = dict(gridcolor=_gc(), title="MACD")
    layout["yaxis2"] = dict(gridcolor=_gc(), title="得分", range=[0,14],
                            tickvals=[0,3,6,9,12,14])
    layout["xaxis2"] = dict(gridcolor=_gc())
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
    layout["yaxis"]  = dict(gridcolor=_gc(), range=[0,100], title="Stoch %")
    layout["yaxis2"] = dict(gridcolor=_gc(), title="CCI")
    layout["xaxis2"] = dict(gridcolor=_gc())
    fig.update_layout(**layout)
    return fig


def equity_chart(df: pd.DataFrame, res: dict = None, best_key: str = "equity") -> go.Figure:
    bh  = STARTING_CAP * df["Close"] / df["Close"].iloc[0]
    best_eq = df["equity"] / df["equity"].iloc[0] * STARTING_CAP
    dd  = (best_eq - best_eq.cummax()) / best_eq.cummax() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.03)

    _key_map = {
        "A · HMM信号投票":   "equity",
        "B · Trailing Stop": "equity_b",
        "C · EMA趋势跟踪":   "equity_c",
        "D · HMM+布林带":    "equity_d",
    }

    # ── 4条策略线 ──────────────────────────────────────────────
    strategies = [
        ("equity",   "策略A · HMM信号投票",   "#ffd740", 2.0, None),
        ("equity_b", "策略B · Trailing Stop",  "#fb923c", 1.5, "dot"),
        ("equity_c", "策略C · EMA趋势跟踪",    "#4ade80", 1.5, "dashdot"),
        ("equity_d", "策略D · HMM+布林带",     "#f87171", 1.5, "longdash"),
    ]
    for key, name, color, width, dash in strategies:
        _is_best = (key == best_key)
        name = ("⭐ " if _is_best else "") + name
        width = 3.5 if _is_best else width
        eq_data = None
        if key == "equity":
            eq_data = df["equity"] if "equity" in df.columns else None
        elif res is not None:
            eq_data = res.get(key)
        if eq_data is not None and len(eq_data) > 0 and eq_data.iloc[0] != 0:
            eq_data = eq_data / eq_data.iloc[0] * STARTING_CAP
            fig.add_trace(go.Scatter(
                x=eq_data.index, y=eq_data, mode="lines", name=name,
                line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width),
            ), row=1, col=1)

    # 买入持有
    fig.add_trace(go.Scatter(x=df.index, y=bh, mode="lines",
        line=dict(color="#60a5fa", width=1.5, dash="dash"), name="📈 买入持有"), row=1, col=1)

    # SPY
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
                line=dict(color="#94a3b8", width=1.2, dash="dot"), name="⚪ SPY"), row=1, col=1)
    except Exception:
        pass

    dd_colors = ["#ff5252" if v < -10 else "#ffd740" if v < -5 else "#00e676" for v in dd]
    fig.add_trace(go.Bar(x=df.index, y=dd, marker_color=dd_colors,
        marker_opacity=0.7, name="回撤 %"), row=2, col=1)
    layout = _base_layout(height=460)
    layout["yaxis"]  = dict(gridcolor=_gc(), tickprefix="$")
    layout["yaxis2"] = dict(gridcolor=_gc(), ticksuffix="%", title="回撤")
    layout["xaxis2"] = dict(gridcolor=_gc())
    layout["legend"] = dict(orientation="h", y=1.08, x=0, font=dict(size=11))
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
                      yaxis=dict(gridcolor=_gc(), title="Sharpe"),
                      xaxis=dict(gridcolor=_gc()))
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
                      yaxis=dict(title="占比 %", gridcolor=_gc()),
                      xaxis=dict(gridcolor=_gc()))
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
                      yaxis=dict(gridcolor=_gc(), ticksuffix="%", title="单bar收益率 %"),
                      xaxis=dict(gridcolor=_gc()))
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
    layout.update({"yaxis": dict(gridcolor=_gc(), tickprefix="$", title="PnL"),
                   "yaxis2": dict(gridcolor=_gc(), title="笔数"),
                   "xaxis": dict(gridcolor=_gc(), title="交易序号"),
                   "xaxis2": dict(gridcolor=_gc(), title="Bars"),
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
                      yaxis=dict(gridcolor=_gc(), title="策略/BH 倍数",
                                 tickformat=".2f"),
                      xaxis=dict(gridcolor=_gc()))
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
                      yaxis=dict(gridcolor=_gc(), ticksuffix="%", title="回撤"),
                      xaxis=dict(gridcolor=_gc()))
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
    layout["xaxis"]  = dict(gridcolor=_gc(), tickprefix="$", title="平均PnL")
    layout["xaxis2"] = dict(gridcolor=_gc(), ticksuffix="%", title="胜率", range=[0,110])
    layout["yaxis"]  = dict(gridcolor=_gc())
    layout["yaxis2"] = dict(gridcolor=_gc())
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
    layout["xaxis"]  = dict(gridcolor=_gc())
    layout["xaxis2"] = dict(gridcolor=_gc())
    layout["xaxis3"] = dict(gridcolor=_gc())
    layout["yaxis"]  = dict(gridcolor=_gc(), title="次数")
    layout["yaxis2"] = dict(gridcolor=_gc(), tickprefix="$", title="平均PnL")
    layout["yaxis3"] = dict(gridcolor=_gc(), ticksuffix="%", title="胜率", range=[0,110])
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
    layout["xaxis"]  = dict(gridcolor=_gc())
    layout["xaxis2"] = dict(gridcolor=_gc())
    layout["yaxis"]  = dict(gridcolor=_gc(), tickprefix="$", title="PnL")
    layout["yaxis2"] = dict(gridcolor=_gc(), ticksuffix="%", title="累计贡献 %")
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
                      xaxis=dict(gridcolor=_gc(), title=f"持仓 ({unit})"),
                      yaxis=dict(gridcolor=_gc(), title="笔数"),
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
        yaxis=dict(title="z-score", gridcolor=_gc()),
        xaxis=dict(gridcolor=_gc()),
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
        yaxis=dict(title="z-score", gridcolor=_gc()),
        xaxis=dict(gridcolor=_gc()),
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
        pred_df = pd.read_csv(os.path.join(_XGB_DIR, "predict.csv"),
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
            xaxis=dict(gridcolor=_gc(), zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor=_gc(), tickfont=dict(size=10, color="#94a3b8")),
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
            xaxis=dict(gridcolor=_gc()),
            yaxis=dict(gridcolor=_gc(), tickformat=".0%", range=[0, 1]),
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

def _build_equity_from_trades(trades: list, df: pd.DataFrame) -> pd.Series:
    """Reconstruct equity curve from trade list (growth strategy)."""
    # strategy_growth stores equity in res; this is a fallback if missing
    cap = float(STARTING_CAP)
    equity = []
    sorted_trades = sorted(trades, key=lambda t: t["entry_time"])
    t_idx = 0
    for i, ts in enumerate(df.index):
        # advance past closed trades to update cap
        while t_idx < len(sorted_trades) and sorted_trades[t_idx]["exit_time"] < ts:
            cap += sorted_trades[t_idx]["pnl"]
            t_idx += 1
        # check if in an open trade
        price = df["Close"].iloc[i]
        in_t = (t_idx < len(sorted_trades) and
                sorted_trades[t_idx]["entry_time"] <= ts <= sorted_trades[t_idx]["exit_time"])
        if in_t:
            ep = sorted_trades[t_idx]["entry_price"]
            shares = cap / ep if ep > 0 else 0
            equity.append(shares * price + (cap - shares * ep))
        else:
            equity.append(cap)
    return pd.Series(equity, index=df.index)


def render_asset(ticker: str) -> None:
    with st.spinner(f"拉取数据 & Walk-Forward 训练 HMM…"):
        try:
            res = load_asset(ticker)
        except Exception as e:
            st.error(f"加载失败：{e}")
            return

    df       = res["df"]
    is_daily = res.get("is_daily", True)
    is_growth = res.get("is_growth", False)

    if is_growth:
        # ── 成长股：使用专属最优策略，不跑 HMM ──────────────────
        metrics = res["metrics"]
        trades  = res["trades"]
        growth_label = res.get("growth_strategy_label", "动量趋势")
        best_name = f"成长股专属 · {growth_label}"
        _best_n = metrics.get("n_trades") or len(trades)
        st.caption(f"📊 策略模式：**{best_name}**  Calmar {metrics.get('calmar',0):.2f}  Sharpe {metrics.get('sharpe',0):.2f}  交易 {_best_n} 笔  |  非 HMM 策略，基于价格动量/趋势信号")
        _all_strats = {best_name: (metrics, trades)}

        # 成长股没有 HMM 字段，构造 fallback
        last = df.iloc[-1]
        # 计算技术指标（如果 df 还没有）
        from backtester import compute_indicators, _ema
        if "ema200" not in df.columns:
            df = compute_indicators(df, ticker)
            last = df.iloc[-1]
        cur_regime        = "N/A (非HMM)"
        cur_regime_filter = bool(last.get("Close", 0) > last.get("ema200", 0))
        # 当前信号：依据 growth_strategy_type 实时判断
        _gtype = res.get("growth_strategy_type", "ema200")
        _c = last["Close"]
        if _gtype == "ema200":
            _sig_now = _c > last.get("ema200", 0)
        elif _gtype == "ema21_50":
            _sig_now = last.get("ema21", last.get("ema50", _c)) > last.get("ema50", 0) if "ema21" in last else _c > last.get("ema200", 0)
        elif _gtype == "ema50_200":
            _sig_now = last.get("ema50", _c) > last.get("ema200", 0)
        elif _gtype in ("52wh80", "52wh75"):
            _pct = 0.80 if _gtype == "52wh80" else 0.75
            _h52 = df["Close"].rolling(252, min_periods=50).max().iloc[-1]
            _sig_now = _c > _h52 * _pct
        else:
            _sig_now = True  # buyhold
        cur_signal = "LONG" if _sig_now else "CASH"
        n_states = 0; min_conf = 0; bull_top = 0; stop = -0.20; adx_thresh = 20
        posterior = []
        # 补充 candle_chart / macd_signal_chart 需要的 HMM 伪列
        if "regime_label" not in df.columns:
            df["regime_label"]  = "N/A"
        if "is_bull" not in df.columns:
            df["is_bull"]       = _sig_now
        if "signal_score" not in df.columns:
            df["signal_score"]  = 0
    else:
        # ── 原有 HMM 路径 ─────────────────────────────────────────
        # 综合得分选最优策略：Calmar × log(n_trades+1) 权重，要求至少20笔且总收益为正
        _all_strats = {
            "A · HMM信号投票":   (res["metrics"],              res["trades"]),
            "B · Trailing Stop": (res.get("metrics_b") or {}, res.get("trades_b") or []),
            "C · EMA趋势跟踪":   (res.get("metrics_c") or {}, res.get("trades_c") or []),
            "D · HMM+布林带":    (res.get("metrics_d") or {}, res.get("trades_d") or []),
        }
        def _score(m):
            if not m or m.get("total_return_pct", 0) <= 0:
                return -999
            n = m.get("n_trades") or 0
            if n < 20:
                return -999
            calmar = m.get("calmar", 0)
            return calmar * np.log(n + 1)
        best_name = max(_all_strats, key=lambda k: _score(_all_strats[k][0]))
        metrics, trades = _all_strats[best_name]
        if not metrics:
            metrics, trades = res["metrics"], res["trades"]
            best_name = "A · HMM信号投票"
        _best_n = metrics.get("n_trades") or len(trades)
        _best_composite = metrics.get("calmar", 0) * np.log(_best_n + 1) if _best_n >= 20 else 0
        st.caption(f"📊 最优策略（综合得分 = Calmar × log(笔数+1)，≥20笔）：**策略{best_name}**  综合得分 {_best_composite:.2f}  Calmar {metrics.get('calmar',0):.2f}  交易 {_best_n} 笔")

        last     = df.iloc[-1]
        n_states   = res.get("n_states",   N_STATES)
        min_conf   = res.get("min_conf",   MIN_CONFIRMATIONS)
        bull_top   = res.get("bull_top",   2)
        stop       = res.get("stop",       -0.08)
        adx_thresh = 20
        posterior  = res.get("posterior", [])

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
        is_daily_txt = "日线" if is_daily else "1h"
        if cur_signal == "LONG":
            action_type = "ENTRY"
            action_color = "#00e676"
        else:
            action_type = "CASH"
            action_color = "#64748b"

        if is_growth:
            # 成长股：显示策略信号卡片
            _trend_ok = cur_regime_filter  # 价格 > EMA200
            risk_status = "NORMAL" if _trend_ok else "CAUTION"
            risk_color  = "#00e676" if _trend_ok else "#ffd740"
            _gtype_disp = res.get("growth_strategy_label", "动量趋势")
            _e200_v = last.get("ema200", 0)
            _e50_v  = last.get("ema50", 0)
            _e21_v  = last.get("ema21", last.get("ema50", 0))
            st.markdown(f"""<div class="signal-cash">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <div class="signal-title">策略信号（{_gtype_disp}）</div>
                        <div style="margin-top:8px;font-size:0.78rem;color:#94a3b8">
                            EMA200: <b style="color:#60a5fa">${_e200_v:,.1f}</b>&nbsp;&nbsp;
                            EMA50: <b style="color:#a78bfa">${_e50_v:,.1f}</b>
                        </div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:1px">Action</div>
                        <div style="font-size:1.1rem;font-weight:800;color:{action_color}">{action_type}</div>
                        <div style="font-size:0.62rem;color:{risk_color};margin-top:2px;font-weight:600">{risk_status}</div>
                    </div>
                </div>
                <div style="margin-top:8px;font-size:0.72rem;color:#475569">
                    止损 -20% · 日线 · {len(df):,} bars · 非HMM策略
                </div>
                <div style="margin-top:4px;font-size:0.72rem">
                    趋势状态：<span style="color:{'#00e676' if _trend_ok else '#ffd740'};font-weight:600">
                    {'✅ 价格站上 EMA200' if _trend_ok else '⚠️ 价格跌破 EMA200'}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            # HMM 路径
            pc = _pill(cur_regime)
            if "Bear" in cur_regime or "Crash" in cur_regime:
                risk_status = "HIGH RISK"
                risk_color  = "#ff5252"
            elif not cur_regime_filter:
                risk_status = "CAUTION"
                risk_color  = "#ffd740"
            else:
                risk_status = "NORMAL"
                risk_color  = "#00e676"
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

    # ── 时间段选择器（预设 + 自定义）────────────────────────────
    st.markdown('<div class="section-header">📅 时间区间筛选</div>', unsafe_allow_html=True)

    _date_min = df.index.min().date()
    _date_max = df.index.max().date()

    PRESETS = {
        "全区间":       (_date_min,                            _date_max),
        "2008金融危机": (max(_date_min, __import__('datetime').date(2008,1,1)),  __import__('datetime').date(2009,6,30)),
        "2010-2015":   (max(_date_min, __import__('datetime').date(2010,1,1)),  __import__('datetime').date(2015,12,31)),
        "2018熊市":    (max(_date_min, __import__('datetime').date(2018,1,1)),  __import__('datetime').date(2018,12,31)),
        "2020疫情":    (max(_date_min, __import__('datetime').date(2020,1,1)),  __import__('datetime').date(2020,12,31)),
        "2022加息":    (max(_date_min, __import__('datetime').date(2022,1,1)),  __import__('datetime').date(2022,12,31)),
        "近3年":       (max(_date_min, (_date_max.replace(year=_date_max.year-3))), _date_max),
        "近1年":       (max(_date_min, (_date_max.replace(year=_date_max.year-1))), _date_max),
    }

    _preset_cols = st.columns(len(PRESETS), gap="small")
    _safe_t      = ticker.replace("=", "_").replace("/", "_")
    _preset_key  = f"preset_{_safe_t}"
    _start_key   = f"eq_start_{_safe_t}"
    _end_key     = f"eq_end_{_safe_t}"

    # 初始化 session_state
    if _preset_key not in st.session_state:
        st.session_state[_preset_key] = "全区间"
    if _start_key not in st.session_state:
        st.session_state[_start_key] = _date_min
    if _end_key not in st.session_state:
        st.session_state[_end_key] = _date_max

    for i, (label, (ps, pe)) in enumerate(PRESETS.items()):
        with _preset_cols[i]:
            _active = st.session_state[_preset_key] == label
            if st.button(label, key=f"preset_{ticker}_{label}",
                         type="primary" if _active else "secondary",
                         use_container_width=True):
                st.session_state[_preset_key] = label
                st.session_state[_start_key]  = ps
                st.session_state[_end_key]    = pe
                st.rerun()

    _pcol_l, _pcol_r = st.columns(2, gap="small")
    with _pcol_l:
        _start = st.date_input("起始日期",
                               min_value=_date_min, max_value=_date_max,
                               key=_start_key)
    with _pcol_r:
        _end = st.date_input("结束日期",
                             min_value=_date_min, max_value=_date_max,
                             key=_end_key)

    if _start >= _end:
        st.warning("起始日期必须早于结束日期")
        _df_slice = df
    else:
        _s, _e = str(_start), str(_end)
        _df_slice = df.loc[_s:_e].copy()

    # 写入 equity 列（成长股从 res["equity_growth"] 或重建；HMM 已经在 df 里）
    if is_growth:
        _eq_series = res.get("equity_growth")
        if _eq_series is None:
            _eq_series = _build_equity_from_trades(trades, df)
        df["equity"] = _eq_series.reindex(df.index).fillna(method="ffill").fillna(STARTING_CAP)
        _df_slice["equity"] = df["equity"].reindex(_df_slice.index)

    # 用切片区间重算指标
    if not is_growth:
        from backtester import _compute_metrics as _cm
        _trades_slice = [t for t in trades
                         if t["entry_time"] >= _df_slice.index[0]
                         and t["exit_time"]  <= _df_slice.index[-1]]
        try:
            metrics = _cm(_df_slice, _trades_slice, ticker, is_daily)
        except Exception:
            pass  # 切片太短时保留全区间指标
    else:
        _trades_slice = [t for t in trades
                         if t["entry_time"] >= _df_slice.index[0]
                         and t["exit_time"]  <= _df_slice.index[-1]]

    # ── K线 + Volume + RSI ──────────────────────────────────
    st.markdown('<div class="section-header">📊 K线图 · Volume · RSI&nbsp;&nbsp;<span style="font-size:0.75rem;color:#475569;font-weight:400">绿=Bull Run · 浅绿=Bull+ · 蓝=Warming Up · 红=Bear</span></div>', unsafe_allow_html=True)
    st.plotly_chart(candle_chart(_df_slice, _trades_slice, ticker), use_container_width=True)

    # ── MACD + 信号强度 ──────────────────────────────────────
    st.markdown('<div class="section-header">📉 MACD &amp; 信号强度时序</div>', unsafe_allow_html=True)
    st.plotly_chart(macd_signal_chart(_df_slice, min_conf), use_container_width=True)

    # ── Stochastic + CCI ─────────────────────────────────────
    st.markdown('<div class="section-header">🔀 随机震荡指标 &amp; CCI</div>', unsafe_allow_html=True)
    st.plotly_chart(stoch_cci_chart(_df_slice), use_container_width=True)

    # ── 绩效指标 Row 1 ────────────────────────────────────────
    st.markdown('<div class="section-header">📈 回测绩效</div>', unsafe_allow_html=True)

    # 区间不足1年时，年化换算失真——显示警告并隐藏年化指标
    _slice_bars   = len(_df_slice)
    _slice_years  = _slice_bars / 252
    _short_window = _slice_bars < 252

    if _short_window:
        st.warning(f"⚠️ 当前区间仅 {_slice_bars} 个交易日（不足1年），夏普/卡玛等年化指标统计意义有限，仅供参考。")

    def _fmt_ratio(v: float, cap: float = 50.0) -> str:
        """对年化比率做上限截断，避免短区间虚假数字。"""
        if _short_window:
            return "—" if abs(v) > cap else f"{v:.2f}"
        return f"{v:.2f}"

    def _fmt_ann(v: float) -> str:
        if _short_window:
            return "—"
        return f"{v:+.1f}%"

    cols = st.columns(4, gap="small")
    rc = "green" if metrics["total_return_pct"] > 0 else "red"
    _bh_ret  = metrics.get("bh_return_pct") or (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
    _alpha   = metrics.get("alpha_pct") or (metrics["total_return_pct"] - _bh_ret)
    ac = "green" if _alpha > 0 else "red"
    _final_cap = metrics.get("final_capital") or (STARTING_CAP * (1 + metrics["total_return_pct"]/100))
    with cols[0]: st.markdown(_metric("总收益",      f"{metrics['total_return_pct']:+.1f}%",
                                                      f"年化 {_fmt_ann(metrics['ann_return_pct'])}", rc), unsafe_allow_html=True)
    with cols[1]: st.markdown(_metric("vs B&H Alpha", f"{_alpha:+.1f}%",
                                                      f"B&H {_bh_ret:+.1f}%", ac), unsafe_allow_html=True)
    with cols[2]: st.markdown(_metric("最大回撤",    f"{metrics['max_drawdown_pct']:.1f}%",
                                                      "峰值→谷值", "red"), unsafe_allow_html=True)
    with cols[3]: st.markdown(_metric("最终资本",    f"${_final_cap:,.0f}",
                                                      f"起始 ${STARTING_CAP:,.0f} · {LEVERAGE}×", "yellow"), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    cols2 = st.columns(4, gap="small")
    _sh = metrics["sharpe"]; _ca = metrics["calmar"]
    sh_c = "green" if _sh > 1 else "yellow" if _sh > 0 else "red"
    ca_c = "green" if _ca > 1 else "yellow" if _ca > 0 else "red"
    _ann_vol = metrics.get("ann_vol_pct") or 0
    _mwin    = metrics.get("monthly_win_pct") or 0
    _wrate   = metrics.get("win_rate_pct") or 0
    _spy_a   = metrics.get("spy_alpha_pct")
    _spy_bh  = metrics.get("spy_bh_pct")
    sa_v = f"{_spy_a:+.1f}%" if _spy_a is not None else "N/A"
    sa_s = f"SPY {_spy_bh:+.1f}%" if _spy_bh is not None else ""
    sa_c = "green" if (_spy_a or 0) > 0 else "red"
    with cols2[0]: st.markdown(_metric("夏普比率",   _fmt_ratio(_sh),
                                                      f"年化波动 {_ann_vol:.1f}%", sh_c), unsafe_allow_html=True)
    with cols2[1]: st.markdown(_metric("卡玛比率",   _fmt_ratio(_ca),
                                                      "年化收益 / 最大回撤", ca_c), unsafe_allow_html=True)
    with cols2[2]: st.markdown(_metric("月度胜率",   f"{_mwin:.1f}%" if _mwin else "—",
                                                      f"交易胜率 {_wrate:.1f}%", "blue"), unsafe_allow_html=True)
    with cols2[3]: st.markdown(_metric("vs SPY Alpha", sa_v, sa_s, sa_c), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── 绩效指标 Row 3：高级指标（HMM 专属字段用 .get 安全取值）─
    cols3 = st.columns(4, gap="small")
    _so  = metrics.get("sortino") or 0
    _pf  = metrics.get("profit_factor") or 0
    _ex  = metrics.get("expectancy") or 0
    _rr  = metrics.get("rr_ratio") or 0
    _tr  = metrics.get("tail_ratio") or 0
    so_c  = "green" if _so > 1 else "yellow" if _so > 0 else "red"
    pf_c  = "green" if _pf > 1.5 else "yellow" if _pf > 1 else "red"
    ex_c  = "green" if _ex > 0 else "red"
    tr_c  = "green" if _tr > 1 else "yellow"
    with cols3[0]: st.markdown(_metric("Sortino 比率",  _fmt_ratio(_so),
                                                         f"下行波动率标准化", so_c), unsafe_allow_html=True)
    with cols3[1]: st.markdown(_metric("Profit Factor", f"{_pf:.2f}" if _pf else "—",
                                                         f"总盈利 / 总亏损", pf_c), unsafe_allow_html=True)
    with cols3[2]: st.markdown(_metric("期望值/笔",     f"${_ex:+.0f}" if _ex else "—",
                                                         f"盈亏比 {_rr:.2f}×" if _rr else "", ex_c), unsafe_allow_html=True)
    with cols3[3]: st.markdown(_metric("Tail Ratio",    f"{_tr:.2f}" if _tr else "—",
                                                         "P95收益 / P5亏损", tr_c), unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    cols4 = st.columns(4, gap="small")
    _mcl  = metrics.get("max_consec_loss") or 0
    _ahb  = metrics.get("avg_hold_bars") or 0
    _aps  = metrics.get("avg_pos_size_pct") or 0
    _sk   = metrics.get("skewness") or 0
    _ku   = metrics.get("kurtosis") or 0
    _mrb  = metrics.get("max_recovery_bars") or 0
    cl_c  = "green" if _mcl <= 2 else "yellow" if _mcl <= 4 else "red"
    sk_c  = "green" if _sk > 0 else "yellow"
    rc_label = f"{_mrb}{'日' if is_daily else 'h'}"
    with cols4[0]: st.markdown(_metric("最大连续亏损",  f"{_mcl} 笔" if _mcl else "—",
                                                         "连续止损次数上限", cl_c), unsafe_allow_html=True)
    with cols4[1]: st.markdown(_metric("平均持仓",      f"{_ahb:.0f} bars" if _ahb else "—",
                                                         f"平均仓位 {_aps:.0f}%" if _aps else "", "blue"), unsafe_allow_html=True)
    with cols4[2]: st.markdown(_metric("收益偏度",      f"{_sk:+.2f}" if _sk else "—",
                                                         f"峰度 {_ku:.2f}" if _ku else "", sk_c), unsafe_allow_html=True)
    with cols4[3]: st.markdown(_metric("最长回撤修复",  rc_label,
                                                         "峰值→修复所需时间", "yellow"), unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 四策略对比表 ─────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 四策略对比</div>', unsafe_allow_html=True)
    _strat_rows = []
    for _sname, (_sm, _st) in _all_strats.items():
        if not _sm:
            continue
        _is_best = (_sname == best_name)
        _prefix = "⭐ " if _is_best else "　 "
        _n_trades = _sm.get('n_trades') or len(_st)
        _calmar = _sm.get('calmar')
        if _calmar is None and _sm.get('ann_return_pct') and _sm.get('max_drawdown_pct'):
            _calmar = abs(_sm['ann_return_pct'] / _sm['max_drawdown_pct']) if _sm['max_drawdown_pct'] != 0 else 0
        _composite = (_calmar or 0) * np.log(_n_trades + 1) if _n_trades >= 20 else 0
        _strat_rows.append({
            "策略": _prefix + _sname,
            "总收益":   f"{_sm.get('total_return_pct',0):+.1f}%",
            "年化":     f"{_sm.get('ann_return_pct',0):+.1f}%",
            "Sharpe":   f"{_sm.get('sharpe',0):.2f}",
            "Calmar":   f"{(_calmar or 0):.2f}",
            "综合得分": f"{_composite:.2f}" + ("" if _n_trades >= 20 else " ⚠️<20笔"),
            "MaxDD":    f"{_sm.get('max_drawdown_pct',0):.1f}%",
            "胜率":     f"{_sm.get('win_rate_pct',0):.1f}%",
            "交易笔数": str(_n_trades),
        })
    if _strat_rows:
        st.dataframe(pd.DataFrame(_strat_rows), use_container_width=True, hide_index=True)

    # ── 资金曲线 + 回撤（使用已切片的 _df_slice）────────────────
    st.markdown('<div class="section-header">💰 资金曲线 vs 买入持有 vs SPY</div>', unsafe_allow_html=True)
    _res_eq = dict(res)
    _s, _e = str(_df_slice.index[0].date()), str(_df_slice.index[-1].date())
    if not is_growth:
        for _k in ["equity_b", "equity_c", "equity_d"]:
            _v = res.get(_k)
            if _v is not None and isinstance(_v, pd.Series):
                _res_eq[_k] = _v.loc[_s:_e]
        _key_map = {"A · HMM信号投票": "equity", "B · Trailing Stop": "equity_b",
                    "C · EMA趋势跟踪": "equity_c", "D · HMM+布林带": "equity_d"}
        _best_eq_key = _key_map.get(best_name, "equity")
    else:
        _best_eq_key = "equity"
    st.plotly_chart(equity_chart(_df_slice, _res_eq, best_key=_best_eq_key), use_container_width=True)

    # ── 滚动夏普 ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 滚动夏普比率</div>', unsafe_allow_html=True)
    st.plotly_chart(rolling_sharpe_chart(_df_slice, is_daily), use_container_width=True)

    # ── Relative Alpha Curve ──────────────────────────────────
    st.markdown('<div class="section-header">📐 相对 Alpha 曲线（策略净值 / B&H 净值）</div>', unsafe_allow_html=True)
    st.plotly_chart(relative_alpha_chart(_df_slice), use_container_width=True)

    # ── Underwater Plot ───────────────────────────────────────
    st.markdown('<div class="section-header">🌊 Underwater 回撤曲线</div>', unsafe_allow_html=True)
    st.plotly_chart(underwater_chart(_df_slice), use_container_width=True)

    # ── 月度热力图 + 状态分布（成长股无 HMM 状态分布）─────────
    if not is_growth:
        col_heat, col_dist = st.columns([3, 2], gap="medium")
        with col_heat:
            st.markdown('<div class="section-header">🗓 月度收益热力图（Strategy / B&H / Alpha）</div>', unsafe_allow_html=True)
            monthly_heatmap_tabbed(metrics["monthly_df"])
        with col_dist:
            st.markdown('<div class="section-header">🧩 HMM 状态分布</div>', unsafe_allow_html=True)
            st.plotly_chart(regime_bar(_df_slice), use_container_width=True)
    else:
        st.markdown('<div class="section-header">🗓 月度收益热力图</div>', unsafe_allow_html=True)
        # Build monthly_df for growth strategy
        _eq_m = df["equity"].resample("ME").last()
        _monthly_ret = _eq_m.pct_change().dropna() * 100
        _mdf = pd.DataFrame({"strategy": _monthly_ret})
        _mdf["bh"] = (df["Close"].resample("ME").last().pct_change().dropna() * 100).reindex(_mdf.index)
        _mdf["alpha"] = _mdf["strategy"] - _mdf["bh"]
        _mdf["year"] = _mdf.index.year; _mdf["month"] = _mdf.index.month
        try:
            monthly_heatmap_tabbed(_mdf)
        except Exception:
            st.caption("月度热力图数据不足")

    # ── 宏观特征可视化（仅 HMM 资产）────────────────────────────
    if not is_growth:
        from data_loader import MACRO_TABLES
        _macro_cols = [c for c in MACRO_TABLES.values() if c in df.columns]
        if _macro_cols:
            st.markdown('<div class="section-header">🌐 宏观指标时序（z-score · 背景色=Regime）</div>', unsafe_allow_html=True)
            st.plotly_chart(macro_timeseries_chart(df), use_container_width=True)
            st.markdown('<div class="section-header">📊 各 Regime 宏观特征均值对比</div>', unsafe_allow_html=True)
            st.plotly_chart(macro_by_regime_chart(df), use_container_width=True)

    # ── 各状态收益箱线图（仅 HMM）────────────────────────────────
    if not is_growth:
        st.markdown('<div class="section-header">📦 各 HMM 状态收益率分布</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_return_chart(_df_slice, n_states), use_container_width=True)

    # ── Regime Return Attribution（仅 HMM）───────────────────────
    if not is_growth and _trades_slice:
        st.markdown('<div class="section-header">🔍 Regime 交易归因（各状态入场盈亏 & 胜率）</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_attribution_chart(_df_slice, _trades_slice), use_container_width=True)

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
            ("盈亏比 (R:R)", f"{metrics.get('rr_ratio', 0):.2f}"),
            ("平均盈利",     f"${metrics.get('avg_win', 0):+,.0f}"),
            ("平均亏损",     f"${metrics.get('avg_loss', 0):+,.0f}"),
            ("最优单笔",     f"${best_t:+,.0f}"),
            ("最差单笔",     f"${worst_t:+,.0f}"),
            ("平均仓位",     f"{metrics.get('avg_pos_size_pct', 0):.0f}%"),
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
    _strat_base  = ["A · HMM信号投票", "B · Trailing Stop", "C · EMA趋势跟踪", "D · HMM+布林带"]
    _strat_mets  = [res["metrics"], res.get("metrics_b") or {}, res.get("metrics_c") or {}, res.get("metrics_d") or {}]
    _strat_data  = [res["trades"], res.get("trades_b") or [], res.get("trades_c") or [], res.get("trades_d") or []]
    _trade_opts = [("⭐ " if s == best_name else "") + s for s in _strat_base]
    _default_idx = next((i for i, s in enumerate(_strat_base) if s == best_name), 0)
    _safe_t2 = ticker.replace("=","_").replace("/","_")
    _sel_strat = st.selectbox("查看策略交易记录", _trade_opts,
                               index=_default_idx,
                               key=f"trade_strat_{_safe_t2}")
    _sel_idx   = _trade_opts.index(_sel_strat)
    trades_view = _strat_data[_sel_idx]

    if trades_view:
        st.markdown('<div class="section-header">📝 完整交易记录</div>', unsafe_allow_html=True)
        tdf_raw = pd.DataFrame(trades_view)
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
        if "pos_size_pct" not in tdf_show.columns:
            tdf_show["pos_size_pct"] = 1.0
        if "entry_capital" not in tdf_show.columns:
            tdf_show["entry_capital"] = float("nan")
        if "exit_capital" not in tdf_show.columns:
            tdf_show["exit_capital"] = float("nan")
        tdf_show["entry_time"]    = pd.to_datetime(tdf_show["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tdf_show["exit_time"]     = pd.to_datetime(tdf_show["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tdf_show["entry_price"]   = tdf_show["entry_price"].map("${:,.2f}".format)
        tdf_show["exit_price"]    = tdf_show["exit_price"].map("${:,.2f}".format)
        tdf_show["pnl"]           = tdf_show["pnl"].map("${:+,.2f}".format)
        tdf_show["pos_size_pct"]  = tdf_show["pos_size_pct"].map(lambda x: f"{x*100:.0f}%")
        tdf_show["return_pct"]    = tdf_show["return_pct"].map(lambda x: f"{x:+.1f}%")
        tdf_show["entry_capital"] = tdf_show["entry_capital"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
        tdf_show["exit_capital"]  = tdf_show["exit_capital"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
        tdf_show = tdf_show[["entry_time","exit_time","entry_regime","entry_price","exit_price",
                              "pos_size_pct","entry_capital","exit_capital","pnl","return_pct","hold_bars","exit_reason"]]
        tdf_show.columns = ["入场时间","出场时间","入场Regime","入场价","出场价","仓位","入场本金","交易后金额","盈亏","收益率","持仓bar","出场原因"]
        st.dataframe(tdf_show, use_container_width=True, hide_index=True)

    # ── XGBoost 多因子预测（仅 Gold）────────────────────────────
    if ticker == "GC=F":
        render_xgb_panel()

    # ── 参数敏感性分析 ────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 参数敏感性分析</div>', unsafe_allow_html=True)
    st.caption("调整以下参数，实时查看对回测结果的影响（不影响主回测）")
    _safe_t3 = ticker.replace("=","_").replace("/","_")
    with st.expander("展开参数调整面板", expanded=False):
        _sc1, _sc2, _sc3 = st.columns(3, gap="medium")
        with _sc1:
            _s_stop = st.slider("止损比例 (%)", -20, -3,
                                int(res.get("stop", -8) * 100) if res.get("stop") else -8,
                                step=1, key=f"sens_stop_{_safe_t3}") / 100
            _s_atr = st.slider("ATR 止损倍数", 1.0, 6.0, float(ATR_TRAIL_MULT),
                               step=0.5, key=f"sens_atr_{_safe_t3}")
        with _sc2:
            _s_conf = st.slider("最小信号分 (min_conf)", 1, 4,
                                int(res.get("min_conf", min_conf)),
                                step=1, key=f"sens_conf_{_safe_t3}")
            _s_adx = st.slider("ADX 门槛", 10, 40, 20,
                               step=5, key=f"sens_adx_{_safe_t3}")
        with _sc3:
            _s_cb = st.slider("熔断阈值 (%)", -30, -5, -15,
                              step=5, key=f"sens_cb_{_safe_t3}") / 100
            _s_short = st.checkbox("启用做空", value=ENABLE_SHORT,
                                   key=f"sens_short_{_safe_t3}")

        if st.button("运行敏感性回测", key=f"sens_run_{_safe_t3}", type="primary"):
            with st.spinner("回测中..."):
                try:
                    import backtester as _bt
                    _orig_atr   = _bt.ATR_TRAIL_MULT
                    _orig_cb    = _bt.CIRCUIT_BREAKER_THRESH
                    _orig_short = _bt.ENABLE_SHORT
                    _bt.ATR_TRAIL_MULT          = _s_atr
                    _bt.CIRCUIT_BREAKER_THRESH  = _s_cb
                    _bt.ENABLE_SHORT            = _s_short
                    _tp_orig = _bt.TICKER_PARAMS.get(ticker, {}).copy()
                    _bt.TICKER_PARAMS[ticker] = {**_tp_orig, "stop": _s_stop,
                                                 "min_conf": _s_conf, "adx_entry": _s_adx}
                    _s_res = _bt.run_backtest(df, ticker)
                    _bt.ATR_TRAIL_MULT         = _orig_atr
                    _bt.CIRCUIT_BREAKER_THRESH = _orig_cb
                    _bt.ENABLE_SHORT           = _orig_short
                    _bt.TICKER_PARAMS[ticker]  = _tp_orig

                    _sm = _s_res["metrics"]
                    _d1, _d2, _d3, _d4 = st.columns(4, gap="small")
                    _rc = "green" if _sm["total_return_pct"] > metrics["total_return_pct"] else "red"
                    _d1.markdown(_metric("新总收益",  f"{_sm['total_return_pct']:+.1f}%",
                                         f"原 {metrics['total_return_pct']:+.1f}%", _rc), unsafe_allow_html=True)
                    _rc2 = "green" if _sm["sharpe"] > metrics["sharpe"] else "red"
                    _d2.markdown(_metric("新 Sharpe", f"{_sm['sharpe']:.2f}",
                                         f"原 {metrics['sharpe']:.2f}", _rc2), unsafe_allow_html=True)
                    _rc3 = "green" if _sm["max_drawdown_pct"] > metrics["max_drawdown_pct"] else "red"
                    _d3.markdown(_metric("新 MaxDD",  f"{_sm['max_drawdown_pct']:.1f}%",
                                         f"原 {metrics['max_drawdown_pct']:.1f}%", _rc3), unsafe_allow_html=True)
                    _rc4 = "green" if _sm["calmar"] > metrics["calmar"] else "red"
                    _d4.markdown(_metric("新 Calmar", f"{_sm['calmar']:.2f}",
                                         f"原 {metrics['calmar']:.2f}", _rc4), unsafe_allow_html=True)
                except Exception as _e:
                    st.error(f"回测失败: {_e}")


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
        with st.expander(f"⚠️ {len(errors)} 个资产信号生成失败（点击展开）", expanded=False):
            for _t, _e in errors.items():
                st.caption(f"**{_t}**: {_e[:120]}…")

    ticker_labels = {
        "AAPL": "🍎 Apple",   "GC=F": "🥇 Gold",   "SI=F": "🥈 Silver",
        "CL=F": "🛢 Oil",     "NVDA": "🟩 NVDA",    "META": "🔵 META",
        "AMZN": "📦 AMZN",   "GOOG": "🔍 GOOG",    "MSFT": "🪟 MSFT",
        "TSLA": "⚡ TSLA",    "HOOD": "🪶 HOOD",    "SPY":  "📊 SPY",
        "FXI":  "🇨🇳 FXI",    "PLTR": "🛡 PLTR",
    }
    all_signal_tickers = ["AAPL","GC=F","SI=F","CL=F","NVDA","META","AMZN","GOOG","MSFT","TSLA","HOOD","SPY","FXI","PLTR"]

    # Show quick summary grid first — two rows of 7
    sig_available = [t for t in all_signal_tickers if signals.get(t)]
    if sig_available:
        st.markdown('<div class="section-header">⚡ 全资产信号速览</div>', unsafe_allow_html=True)
        _ACTION_COLOR = {"ENTER": "#00c864", "HOLD": "#60a5fa", "EXIT": "#ff5252",
                         "WATCH": "#d97706", "STAY_OUT": "#475569"}
        # Split into rows of 7
        _COLS_PER_ROW = 7
        for _row_start in range(0, len(sig_available), _COLS_PER_ROW):
            _row_tickers = sig_available[_row_start:_row_start + _COLS_PER_ROW]
            _grid_cols = st.columns(len(_row_tickers), gap="small")
            for _i, _t in enumerate(_row_tickers):
                _s  = signals[_t]
                _af = _s.get("action_if_flat", "—")
                _cl = _s.get("close", 0)
                _bl = _s.get("is_bull", False)
                _af_c  = _ACTION_COLOR.get(_af, "#94a3b8")
                _reg_c = "#00c864" if _bl else "#ff5252"
                with _grid_cols[_i]:
                    st.markdown(
                        f'<div class="metric-card" style="padding:10px 8px">'
                        f'<div style="font-size:0.65rem;color:#475569;margin-bottom:4px">{ticker_labels.get(_t, _t)}</div>'
                        f'<div style="font-size:0.9rem;font-weight:800;color:{_af_c}">{_af}</div>'
                        f'<div style="font-size:0.58rem;color:{_reg_c};margin-top:2px">{"Bull" if _bl else "Bear"}</div>'
                        f'<div style="font-size:0.6rem;color:#475569;margin-top:1px">${_cl:,.1f}</div>'
                        f'</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    for ticker in all_signal_tickers:
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
                f'Score <b style="color:{score_color}">{score}/4</b> &nbsp;'
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


def render_portfolio_tab() -> None:
    ALL_TICKERS = ["AAPL","GC=F","SI=F","CL=F","NVDA","META","AMZN","GOOG","MSFT","TSLA","HOOD","SPY","FXI","PLTR"]

    # 加载所有品种数据
    eq_curves   = {}
    metrics_all = {}
    bull_ratios = {}
    for t in ALL_TICKERS:
        r = _load_result(t)
        if r is None:
            continue
        if r.get("is_growth") and "equity_growth" in r:
            eq_curves[t] = r["equity_growth"]
        else:
            eq_curves[t] = pd.Series(r["df"]["equity"].values, index=r["df"].index)
        metrics_all[t] = r["metrics"]
        # bull ratio：该品种历史上bull状态占比（用于动态权重）
        bull_ratios[t] = float(r["df"]["is_bull"].mean()) if "is_bull" in r["df"].columns else 0.5

    if not eq_curves:
        st.warning("无预计算数据，请先运行 precompute.py")
        return

    loaded = list(eq_curves.keys())

    # 对齐时间轴（取所有品种的共同日期区间）
    all_idx = sorted(set.intersection(*[set(eq_curves[t].index) for t in loaded]))
    all_idx = pd.DatetimeIndex(all_idx)

    def _align(t):
        return eq_curves[t].reindex(all_idx, method="ffill").fillna(STARTING_CAP) / STARTING_CAP

    rets = {t: _align(t) for t in loaded}

    # ── 等权组合 ─────────────────────────────────────────────
    n   = len(loaded)
    port_eq = sum(rets[t] for t in loaded) / n * STARTING_CAP

    # ── 动态权重组合（bull_ratio归一化）─────────────────────
    total_bull = sum(bull_ratios[t] for t in loaded)
    dyn_w = {t: bull_ratios[t] / total_bull for t in loaded}
    port_dyn = sum(rets[t] * dyn_w[t] for t in loaded) * STARTING_CAP

    # ── 波动率平价组合（1/vol 归一化）────────────────────────
    vols = {}
    for t in loaded:
        eq = eq_curves[t].reindex(all_idx, method="ffill").fillna(STARTING_CAP)
        daily_ret = eq.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)
        vols[t] = ann_vol if ann_vol > 0 else 1e-6
    inv_vol = {t: 1.0 / vols[t] for t in loaded}
    total_inv = sum(inv_vol.values())
    vol_w = {t: inv_vol[t] / total_inv for t in loaded}
    port_vol = sum(rets[t] * vol_w[t] for t in loaded) * STARTING_CAP

    def _port_metrics(eq):
        tr  = (eq.iloc[-1] / STARTING_CAP - 1) * 100
        rm  = eq.cummax(); mdd = ((eq - rm) / rm * 100).min()
        dr  = eq.pct_change().dropna()
        sh  = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        cal = abs(tr / mdd) if mdd != 0 else 0
        ann = ((eq.iloc[-1] / STARTING_CAP) ** (252 / max(len(eq), 1)) - 1) * 100
        return tr, sh, mdd, cal, ann

    # ── 顶部指标卡 ───────────────────────────────────────────
    st.markdown('<div class="section-header">📊 全品种组合（14资产）</div>', unsafe_allow_html=True)
    ptr_e, psh_e, pmdd_e, pcal_e, pann_e = _port_metrics(port_eq)
    ptr_d, psh_d, pmdd_d, pcal_d, pann_d = _port_metrics(port_dyn)
    ptr_v, psh_v, pmdd_v, pcal_v, pann_v = _port_metrics(port_vol)

    cc1, cc2, cc3 = st.columns(3, gap="small")
    cc1.markdown(_metric("等权组合收益",    f"{ptr_e:+.1f}%", f"Sharpe {psh_e:.2f}  MaxDD {pmdd_e:.1f}%", "green" if ptr_e>0 else "red"), unsafe_allow_html=True)
    cc2.markdown(_metric("动态权重收益",    f"{ptr_d:+.1f}%", f"Sharpe {psh_d:.2f}  MaxDD {pmdd_d:.1f}%", "green" if ptr_d>0 else "red"), unsafe_allow_html=True)
    cc3.markdown(_metric("波动率平价收益",  f"{ptr_v:+.1f}%", f"Sharpe {psh_v:.2f}  MaxDD {pmdd_v:.1f}%", "green" if ptr_v>0 else "red"), unsafe_allow_html=True)

    # ── 自定义权重组合 ────────────────────────────────────────
    st.markdown('<div class="section-header">⚖️ 自定义权重组合</div>', unsafe_allow_html=True)
    with st.expander("调整各资产权重（拖动滑块）", expanded=False):
        _tick_emoji = {
            "AAPL": "🍎", "GC=F": "🥇", "SI=F": "🥈", "CL=F": "🛢",
            "NVDA": "🟩", "META": "🔵", "AMZN": "📦", "GOOG": "🔍",
            "MSFT": "🪟", "TSLA": "⚡", "HOOD": "🪶", "SPY":  "📊",
            "FXI":  "🇨🇳", "PLTR": "🛡",
        }
        _slider_cols = st.columns(2, gap="medium")
        _raw_w = {}
        for _i, _t in enumerate(loaded):
            with _slider_cols[_i % 2]:
                _default_w = int(round(100.0 / len(loaded)))
                _raw_w[_t] = st.slider(
                    f"{_tick_emoji.get(_t,'')}{_t}",
                    min_value=0, max_value=100,
                    value=st.session_state.get(f"_pw_{_t}", _default_w),
                    step=5, key=f"_pw_{_t}",
                    format="%d%%",
                )
        _total_w = sum(_raw_w.values())
        if _total_w > 0:
            _cust_w = {_t: _raw_w[_t] / _total_w for _t in loaded}
            port_cust = sum(rets[_t] * _cust_w[_t] for _t in loaded) * STARTING_CAP
            ptr_c, psh_c, pmdd_c, pcal_c2, pann_c = _port_metrics(port_cust)
            _wc1, _wc2, _wc3, _wc4 = st.columns(4, gap="small")
            _wc1.markdown(_metric("自定义收益", f"{ptr_c:+.1f}%", f"年化 {pann_c:+.1f}%", "green" if ptr_c>0 else "red"), unsafe_allow_html=True)
            _wc2.markdown(_metric("Sharpe", f"{psh_c:.2f}", "", "green" if psh_c>1 else "yellow"), unsafe_allow_html=True)
            _wc3.markdown(_metric("MaxDD", f"{pmdd_c:.1f}%", "", "red"), unsafe_allow_html=True)
            _wc4.markdown(_metric("Calmar", f"{pcal_c2:.2f}", "", "green" if pcal_c2>1 else "yellow"), unsafe_allow_html=True)
            # show weights summary
            _w_items = sorted(_cust_w.items(), key=lambda x: -x[1])
            st.markdown(
                '<div style="font-size:0.72rem;color:#475569;margin-top:8px">'
                + "  ".join(f'<b style="color:#94a3b8">{_t}</b> {_v*100:.1f}%' for _t, _v in _w_items if _v > 0.001)
                + f'  <span style="color:{"#00c864" if abs(_total_w-100)<0.1 else "#d97706"}">总计 {_total_w}%</span>'
                + '</div>', unsafe_allow_html=True)
        else:
            st.warning("请至少为一个资产设置非零权重")

    # ── 组合资金曲线 ─────────────────────────────────────────
    fig_port = go.Figure()
    COLORS = ["#ffd740","#94a3b8","#60a5fa","#a78bfa","#34d399","#fb923c",
              "#f87171","#38bdf8","#e879f9","#4ade80","#facc15","#f472b6","#818cf8"]
    for i, t in enumerate(loaded):
        fig_port.add_trace(go.Scatter(
            x=all_idx, y=rets[t] * STARTING_CAP,
            mode="lines", name=t,
            line=dict(color=COLORS[i % len(COLORS)], width=1.0, dash="dot"),
            opacity=0.45))
    fig_port.add_trace(go.Scatter(
        x=all_idx, y=port_eq, mode="lines", name="🟡 等权组合",
        line=dict(color="#ffd740", width=2.5)))
    fig_port.add_trace(go.Scatter(
        x=all_idx, y=port_dyn, mode="lines", name="🟢 动态权重组合",
        line=dict(color="#00e676", width=2.5)))
    fig_port.add_trace(go.Scatter(
        x=all_idx, y=port_vol, mode="lines", name="🔵 波动率平价组合",
        line=dict(color="#60a5fa", width=2.5),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.04)"))
    _port_layout = _base_layout(height=420)
    _port_layout["legend"] = dict(orientation="h", y=1.06, x=0, font=dict(size=10))
    _port_layout["yaxis"] = dict(gridcolor=_gc(), tickprefix="$")
    _port_layout["xaxis"] = dict(gridcolor=_gc())
    fig_port.update_layout(**_port_layout)
    st.plotly_chart(fig_port, use_container_width=True)

    # ── 各品种绩效表 ─────────────────────────────────────────
    st.markdown('<div class="section-header">📋 各品种绩效 vs 组合</div>', unsafe_allow_html=True)
    rows = []
    for t in loaded:
        m = metrics_all[t]
        bh = m.get("bh_return_pct", 0)
        beat = "✅" if m["total_return_pct"] > bh else "❌"
        rows.append({
            "资产":     t,
            "策略收益":  f"{m['total_return_pct']:+.1f}%",
            "B&H":      f"{bh:+.1f}%",
            "跑赢":     beat,
            "Sharpe":   f"{m['sharpe']:.2f}",
            "Calmar":   f"{m['calmar']:.2f}",
            "MaxDD":    f"{m['max_drawdown_pct']:.1f}%",
            "年化波动":  f"{vols[t]*100:.1f}%",
            "动态权重":  f"{dyn_w[t]*100:.1f}%",
            "波动率平价": f"{vol_w[t]*100:.1f}%",
        })
    rows.append({
        "资产": "🟡 等权组合", "策略收益": f"{ptr_e:+.1f}%", "B&H": "—", "跑赢": "—",
        "Sharpe": f"{psh_e:.2f}", "Calmar": f"{pcal_e:.2f}",
        "MaxDD": f"{pmdd_e:.1f}%", "年化波动": "—", "动态权重": "—", "波动率平价": "—",
    })
    rows.append({
        "资产": "🟢 动态权重组合", "策略收益": f"{ptr_d:+.1f}%", "B&H": "—", "跑赢": "—",
        "Sharpe": f"{psh_d:.2f}", "Calmar": f"{pcal_d:.2f}",
        "MaxDD": f"{pmdd_d:.1f}%", "年化波动": "—", "动态权重": "100%", "波动率平价": "—",
    })
    rows.append({
        "资产": "🔵 波动率平价组合", "策略收益": f"{ptr_v:+.1f}%", "B&H": "—", "跑赢": "—",
        "Sharpe": f"{psh_v:.2f}", "Calmar": f"{pcal_v:.2f}",
        "MaxDD": f"{pmdd_v:.1f}%", "年化波动": "—", "动态权重": "—", "波动率平价": "100%",
    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
        for t in loaded:
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

def _asset_last_date(ticker: str) -> str:
    path = os.path.join(RESULTS_DIR, f"{_safe_filename(ticker)}.pkl")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            r = pickle.load(f)
        last_dt = r["df"].index[-1]
        return pd.Timestamp(last_dt).strftime("%m/%d")
    except Exception:
        return ""


def main() -> None:
    # Theme must be initialized before CSS injection
    if "_theme" not in st.session_state:
        st.session_state["_theme"] = "dark"

    _inject_css()

    logo = _logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo}" '
        f'style="height:44px;width:44px;border-radius:10px;object-fit:cover;'
        f'box-shadow:0 0 16px rgba(255,255,255,0.08);flex-shrink:0;" />'
        if logo else ""
    )

    hc1, hc2 = st.columns([7, 2])
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
        _btn_c1, _btn_c2 = st.columns(2, gap="small")
        with _btn_c1:
            if st.button("🔄 刷新", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with _btn_c2:
            _cur_theme = st.session_state.get("_theme", "dark")
            _theme_label = "☀️ 浅色" if _cur_theme == "dark" else "🌙 深色"
            if st.button(_theme_label, use_container_width=True):
                st.session_state["_theme"] = "light" if _cur_theme == "dark" else "dark"
                st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Build nav labels with data timestamps
    _TICKER_MAP_BASE = [
        ("🍎  AAPL", "AAPL"), ("🥇  Gold", "GC=F"), ("🥈  Silver", "SI=F"),
        ("🛢  Oil",  "CL=F"), ("🟩  NVDA", "NVDA"), ("🔵  META",  "META"),
        ("📦  AMZN", "AMZN"), ("🔍  GOOG", "GOOG"), ("🪟  MSFT",  "MSFT"),
        ("⚡  TSLA", "TSLA"), ("🪶  HOOD", "HOOD"), ("📊  SPY",   "SPY"),
        ("🇨🇳  FXI", "FXI"),  ("🛡  PLTR", "PLTR"),
    ]
    NAV_TICKER = {label: tick for label, tick in _TICKER_MAP_BASE}

    NAV_OPTIONS = ["📡  今日信号", "🌐  组合"] + [label for label, _ in _TICKER_MAP_BASE]

    _active = st.radio(
        "导航", NAV_OPTIONS,
        index=st.session_state.get("_nav_idx", 0),
        horizontal=True,
        label_visibility="collapsed",
        key="_nav_radio",
    )
    st.session_state["_nav_idx"] = NAV_OPTIONS.index(_active)

    # Show data freshness bar for asset pages
    if _active in NAV_TICKER:
        _tick = NAV_TICKER[_active]
        _last = _asset_last_date(_tick)
        if _last:
            st.markdown(
                f'<div style="font-size:0.65rem;color:#475569;margin-bottom:0.2rem">'
                f'📅 数据更新至 <b style="color:#64748b">{_last}</b></div>',
                unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    if _active == "📡  今日信号":
        render_signals_tab()
    elif _active == "🌐  组合":
        render_portfolio_tab()
    elif _active in NAV_TICKER:
        render_asset(NAV_TICKER[_active])

    st.markdown(
        "<div style='text-align:center;color:#1e293b;font-size:0.7rem;margin-top:2rem'>"
        "仅供学习研究，不构成投资建议。</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
