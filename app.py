"""
app.py  ——  Regime-Based HMM Trading Dashboard  (v2)
高级暗色 UI · 14 信号确认面板 · 三资产 Tab
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import os
import pickle

from data_loader import fetch_data
from backtester  import run_backtest, STARTING_CAP, MIN_CONFIRMATIONS, _position_size

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


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

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #080c14;
}
.block-container { padding: 1rem 2rem 3rem 2rem; max-width: 1600px; }

/* ── Glassmorphism cards ───────────────────────── */
.glass-card {
    background: linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 2px;
}
.metric-card {
    background: linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform .15s ease, box-shadow .15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
.metric-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 8px; font-weight: 500; }
.metric-value { font-size: 1.9rem; font-weight: 800; line-height: 1; }
.metric-sub   { font-size: 0.7rem; color: #475569; margin-top: 6px; font-weight: 400; }

/* ── Signal banners ─────────────────────────────── */
.signal-long {
    background: linear-gradient(135deg,#002d16,#004d24);
    border: 1px solid rgba(0,230,118,0.4);
    border-radius: 16px; padding: 22px 28px; text-align: center;
    box-shadow: 0 0 40px rgba(0,230,118,0.12), inset 0 1px 0 rgba(0,230,118,0.15);
}
.signal-cash {
    background: linear-gradient(135deg,#0f1420,#141929);
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 16px; padding: 22px 28px; text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}
.signal-title { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 10px; font-weight:500; }
.signal-value { font-size: 2.4rem; font-weight: 900; letter-spacing: -1px; }

/* ── Regime pill ─────────────────────────────────── */
.regime-pill  { display:inline-block; padding:6px 20px; border-radius:30px; font-size:1rem; font-weight:700; margin-top:6px; letter-spacing:.3px; }
.regime-bull  { background:rgba(0,230,118,0.12); color:#00e676; border:1px solid rgba(0,230,118,0.4);  box-shadow:0 0 20px rgba(0,230,118,0.1); }
.regime-bear  { background:rgba(255,82,82,0.12);  color:#ff5252; border:1px solid rgba(255,82,82,0.4);  box-shadow:0 0 20px rgba(255,82,82,0.1); }
.regime-neut  { background:rgba(255,215,64,0.10); color:#ffd740; border:1px solid rgba(255,215,64,0.35); box-shadow:0 0 20px rgba(255,215,64,0.08); }

/* ── Signal check row ───────────────────────────── */
.sig-row {
    display:flex; align-items:center; justify-content:space-between;
    padding: 7px 12px; border-radius: 8px; margin-bottom: 4px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    font-size: 0.8rem;
    transition: background .1s;
}
.sig-row:hover { background: rgba(255,255,255,0.045); }
.sig-name { color: #94a3b8; font-weight: 500; }
.sig-val  { color: #cbd5e1; font-family: 'SF Mono', monospace; font-size: 0.75rem; }
.sig-pass { color: #00e676; font-size: 1rem; }
.sig-fail { color: #ff5252; font-size: 1rem; }

/* ── Section header ─────────────────────────────── */
.section-header {
    color: #e2e8f0; font-size: 0.9rem; font-weight: 600;
    margin: 1.6rem 0 0.7rem 0; padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    letter-spacing: .3px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Score bar ──────────────────────────────────── */
.score-outer {
    background: rgba(255,255,255,0.06); border-radius: 8px;
    height: 8px; width: 100%; margin: 8px 0 4px 0;
    overflow: hidden;
}
.score-inner {
    height: 100%; border-radius: 8px;
    transition: width .4s ease;
}

/* ── Page title ─────────────────────────────────── */
.page-title {
    font-size: 1.55rem; font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.5px; line-height: 1.2;
}
.page-sub { font-size: 0.78rem; color: #475569; margin-top: 3px; font-weight: 400; }

/* ── Tab override ───────────────────────────────── */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    gap: 2px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    padding: 8px 20px !important;
}
[aria-selected="true"] {
    background: rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
}

/* ── Color utilities ────────────────────────────── */
.green  { color: #00e676; }
.red    { color: #ff5252; }
.yellow { color: #ffd740; }
.blue   { color: #60a5fa; }
.purple { color: #a78bfa; }
.white  { color: #f1f5f9; }

/* ── Streamlit overrides ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrameResizable"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 数据缓存
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_asset(ticker: str) -> dict:
    # 优先读预计算结果（秒开）
    precomputed = _load_precomputed(ticker)
    if precomputed is not None:
        return precomputed
    # 没有预计算文件则实时计算（Streamlit Cloud 可能超时）
    df = fetch_data(ticker)
    return run_backtest(df, ticker)


# ──────────────────────────────────────────────────────────────
# 颜色工具
# ──────────────────────────────────────────────────────────────

def _bg(label: str) -> str:
    if label == "Bull Run":  return "rgba(0,230,118,0.11)"
    if label == "Bull+":     return "rgba(0,230,118,0.06)"
    if "Bear" in label or "Crash" in label: return "rgba(255,82,82,0.11)"
    return "rgba(255,215,64,0.05)"

def _pill(label: str) -> str:
    if "Bull"  in label: return "regime-bull"
    if "Bear"  in label or "Crash" in label: return "regime-bear"
    return "regime-neut"

def _score_color(pct: float) -> str:
    if pct >= 0.80: return "#00e676"
    if pct >= 0.55: return "#ffd740"
    return "#ff5252"


# ──────────────────────────────────────────────────────────────
# 图表构建
# ──────────────────────────────────────────────────────────────

CHART_BG   = "#080c14"
GRID_COLOR = "rgba(255,255,255,0.05)"

def _base_layout(**kw) -> dict:
    return dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color="#94a3b8", size=11, family="Inter"),
        margin=dict(l=8, r=8, t=44, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2535", bordercolor="#334155", font_size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        **kw,
    )


def candle_chart(df: pd.DataFrame, trades: list, ticker: str) -> go.Figure:
    fig    = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )
    shapes = []

    # regime bands
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

    # Bollinger Bands (filled)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(96,165,250,0.06)",
        name="Bollinger", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_mid"], mode="lines",
        line=dict(color="rgba(96,165,250,0.4)", width=1, dash="dot"),
        name="BB Mid", hoverinfo="skip",
    ), row=1, col=1)

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        increasing_fillcolor="#00e676",  decreasing_fillcolor="#ff5252",
        name=ticker, line_width=1,
    ), row=1, col=1)

    # EMA overlays
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"],  mode="lines",
        line=dict(color="#ffd740", width=1.2, dash="dot"),  name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], mode="lines",
        line=dict(color="#a78bfa", width=1.2, dash="dash"), name="EMA 200"), row=1, col=1)

    # Trade markers
    if trades:
        fig.add_trace(go.Scatter(
            x=[t["entry_time"] for t in trades], y=[t["entry_price"] for t in trades],
            mode="markers", name="买入",
            marker=dict(symbol="triangle-up", size=12, color="#00e676",
                        line=dict(width=1, color="#fff")),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t["exit_time"] for t in trades], y=[t["exit_price"] for t in trades],
            mode="markers", name="卖出",
            marker=dict(symbol="triangle-down", size=12, color="#ff5252",
                        line=dict(width=1, color="#fff")),
        ), row=1, col=1)

    # Volume bars (lower panel)
    colors_vol = ["#00e676" if c >= o else "#ff5252"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors_vol, marker_opacity=0.5,
        name="Volume", showlegend=False,
    ), row=2, col=1)
    # OBV EMA line on volume panel (normalised)
    obv_norm = (df["obv_ema"] - df["obv_ema"].min()) / \
               (df["obv_ema"].max() - df["obv_ema"].min() + 1e-9) * df["Volume"].max()
    fig.add_trace(go.Scatter(
        x=df.index, y=obv_norm, mode="lines",
        line=dict(color="#a78bfa", width=1.2),
        name="OBV EMA", showlegend=True,
    ), row=2, col=1)

    layout = _base_layout(height=560)
    layout["shapes"] = shapes
    layout["xaxis"]  = dict(rangeslider=dict(visible=False), gridcolor=GRID_COLOR, showgrid=True, type="date")
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, showgrid=True)
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR, showgrid=True)
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, showgrid=True, showticklabels=False)
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
            ],
        ),
        row=1, col=1,
    )
    return fig


def equity_chart(df: pd.DataFrame) -> go.Figure:
    bh   = STARTING_CAP * df["Close"] / df["Close"].iloc[0]
    dd   = (df["equity"] - df["equity"].cummax()) / df["equity"].cummax() * 100

    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.65, 0.35], vertical_spacing=0.03)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["equity"], mode="lines",
        line=dict(color="#00e676", width=2),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.06)",
        name="策略 (2.5× 杠杆)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bh, mode="lines",
        line=dict(color="#60a5fa", width=1.5, dash="dash"),
        name="买入持有",
    ), row=1, col=1)

    dd_colors = ["#ff5252" if v < -10 else "#ffd740" if v < -5 else "#00e676"
                 for v in dd]
    fig.add_trace(go.Bar(
        x=df.index, y=dd, marker_color=dd_colors, marker_opacity=0.7,
        name="回撤 %", showlegend=True,
    ), row=2, col=1)

    layout = _base_layout(height=380)
    layout["yaxis"]  = dict(gridcolor=GRID_COLOR, tickprefix="$")
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, ticksuffix="%", title="回撤")
    layout["xaxis2"] = dict(gridcolor=GRID_COLOR)
    fig.update_layout(**layout)
    return fig


def regime_bar(df: pd.DataFrame) -> go.Figure:
    vc = df["regime_label"].value_counts().reset_index()
    vc.columns = ["Regime", "Count"]
    vc["Pct"] = (vc["Count"] / len(df) * 100).round(1)
    colors = ["#00e676" if "Bull" in r else "#ff5252" if ("Bear" in r or "Crash" in r) else "#ffd740"
              for r in vc["Regime"]]
    fig = go.Figure(go.Bar(
        x=vc["Regime"], y=vc["Pct"],
        marker_color=colors, marker_opacity=0.85,
        text=vc["Pct"].map(lambda x: f"{x}%"), textposition="outside",
        textfont=dict(size=12, color="#e2e8f0"),
    ))
    fig.update_layout(**_base_layout(height=220, showlegend=False),
                      yaxis=dict(title="占比 %", gridcolor=GRID_COLOR),
                      xaxis=dict(gridcolor=GRID_COLOR))
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
    icon = f'<span class="sig-pass">●</span>' if ok else f'<span class="sig-fail">●</span>'
    return (f'<div class="sig-row">'
            f'{icon} <span class="sig-name">{name}</span>'
            f'<span class="sig-val">{val}</span>'
            f'</div>')


# ──────────────────────────────────────────────────────────────
# 单资产面板
# ──────────────────────────────────────────────────────────────

def monthly_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    """月度收益热力图。"""
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    years  = sorted(monthly_df["year"].unique())
    z, text = [], []
    for yr in years:
        row_z, row_t = [], []
        for mo in range(1, 13):
            val = monthly_df[(monthly_df["year"]==yr) & (monthly_df["month"]==mo)]["ret"]
            if len(val):
                v = float(val.iloc[0])
                row_z.append(v)
                row_t.append(f"{v:+.1f}%")
            else:
                row_z.append(None)
                row_t.append("")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=MONTHS, y=[str(y) for y in years],
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#7f1d1d"],[0.5,"#1e2130"],[1,"#14532d"]],
        zmid=0, showscale=True,
        colorbar=dict(ticksuffix="%", thickness=12, len=0.8,
                      tickfont=dict(size=10, color="#64748b")),
        hoverongaps=False,
    ))
    fig.update_layout(
        **_base_layout(height=max(160, len(years)*46 + 60)),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_asset(ticker: str) -> None:
    with st.spinner(f"拉取数据 & Walk-Forward 训练 HMM（首次约 60s）…"):
        try:
            res = load_asset(ticker)
        except Exception as e:
            st.error(f"加载失败：{e}")
            return

    df      = res["df"]
    trades  = res["trades"]
    metrics = res["metrics"]
    last    = df.iloc[-1]

    cur_regime  = last["regime_label"]
    cur_signal  = "LONG" if (last["is_bull"] and last["tech_signal"]) else "CASH"

    # ── 顶部三栏 ─────────────────────────────────────────────
    b1, b2, b3 = st.columns([1.8, 1.8, 3.4], gap="medium")

    with b1:
        sc = "signal-long" if cur_signal == "LONG" else "signal-cash"
        sv = "#00e676" if cur_signal == "LONG" else "#64748b"
        price_now = last["Close"]
        st.markdown(f"""<div class="{sc}">
            <div class="signal-title">当前信号</div>
            <div class="signal-value" style="color:{sv}">{cur_signal}</div>
            <div style="margin-top:8px;font-size:0.78rem;color:#475569">
                价格 ${price_now:,.2f}
            </div>
        </div>""", unsafe_allow_html=True)

    with b2:
        pc = _pill(cur_regime)
        is_daily_txt = "日线" if res["is_daily"] else "1h"
        st.markdown(f"""<div class="signal-cash">
            <div class="signal-title">HMM 状态（Walk-Forward）</div>
            <div style="margin-top:8px"><span class="regime-pill {pc}">{cur_regime}</span></div>
            <div style="margin-top:8px;font-size:0.72rem;color:#475569">
                9 状态 · {is_daily_txt} · {len(df):,} bars · 止损 -8%
            </div>
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

        checks = [
            ("RSI < 90",          c1,  f"{last['rsi']:.1f}"),
            ("动量 > 1%",         c2,  f"{last['momentum']:.2f}%"),
            ("波动率 < 6%",       c3,  f"{last['volatility']:.2f}%"),
            ("成交量 > SMA20",    c4,  "Yes" if c4  else "No"),
            ("ADX > 25",          c5,  f"{last['adx']:.1f}"),
            ("价格 > EMA 50",     c6,  f"${last['ema50']:,.2f}"),
            ("价格 > EMA 200",    c7,  f"${last['ema200']:,.2f}"),
            ("MACD > Signal",     c8,  "Yes" if c8  else "No"),
            ("价格 > BB 中轨",    c9,  f"${last['bb_mid']:,.2f}"),
            ("Stoch %K↑ & <80",  c10, f"K={last['stoch_k']:.1f}"),
            ("Williams %R < -20", c11, f"{last['williams_r']:.1f}"),
            ("CCI > 0",           c12, f"{last['cci']:.1f}"),
            ("OBV > OBV EMA",     c13, "Yes" if c13 else "No"),
            ("距高点 > -30%",     c14, f"{last['pct_from_high']:.1f}%"),
        ]
        n     = sum(v for _, v, _ in checks)
        pct   = n / len(checks)
        bar_w = int(pct * 100)
        bar_c = _score_color(pct)

        st.markdown(f"""<div class="glass-card" style="padding:14px 18px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:500">
                    信号确认&nbsp;·&nbsp;仓位 {int(_position_size(n)*100)}%
                </span>
                <span style="font-size:1.1rem;font-weight:800;color:{bar_c}">{n}/{len(checks)}</span>
            </div>
            <div class="score-outer">
                <div class="score-inner" style="width:{bar_w}%;background:{bar_c};opacity:0.85"></div>
            </div>
            <div style="font-size:0.68rem;color:#475569;margin-bottom:10px">
                {'✅ 满足入场条件' if n >= MIN_CONFIRMATIONS else f'⚠️ 还差 {MIN_CONFIRMATIONS - n} 条'}
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        half = len(checks) // 2
        col_a.markdown("".join(_sig_row(nm, ok, vl) for nm, ok, vl in checks[:half]), unsafe_allow_html=True)
        col_b.markdown("".join(_sig_row(nm, ok, vl) for nm, ok, vl in checks[half:]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── K 线图 ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 K 线图&nbsp;&nbsp;<span style="font-size:0.75rem;color:#475569;font-weight:400">绿=Bull Run · 浅绿=Bull+ · 红=Bear/Crash</span></div>', unsafe_allow_html=True)
    st.plotly_chart(candle_chart(df, trades, ticker), use_container_width=True)

    # ── 绩效指标 Row 1：核心四项 ─────────────────────────────
    st.markdown('<div class="section-header">📈 回测绩效</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    rc = "green" if metrics["total_return_pct"] > 0 else "red"
    ac = "green" if metrics["alpha_pct"] > 0 else "red"
    with cols[0]: st.markdown(_metric("总收益",   f"{metrics['total_return_pct']:+.1f}%",
                                                   f"年化 {metrics['ann_return_pct']:+.1f}%", rc), unsafe_allow_html=True)
    with cols[1]: st.markdown(_metric("vs B&H Alpha", f"{metrics['alpha_pct']:+.1f}%",
                                                   f"B&H {metrics['bh_return_pct']:+.1f}%", ac), unsafe_allow_html=True)
    with cols[2]: st.markdown(_metric("最大回撤", f"{metrics['max_drawdown_pct']:.1f}%",
                                                   "峰值→谷值", "red"), unsafe_allow_html=True)
    with cols[3]: st.markdown(_metric("最终资本", f"${metrics['final_capital']:,.0f}",
                                                   f"起始 ${STARTING_CAP:,.0f} · 2.5×", "yellow"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── 绩效指标 Row 2：风险调整后 ───────────────────────────
    cols2 = st.columns(4, gap="small")
    sh_c  = "green" if metrics["sharpe"] > 1 else "yellow" if metrics["sharpe"] > 0 else "red"
    ca_c  = "green" if metrics["calmar"] > 1 else "yellow" if metrics["calmar"] > 0 else "red"
    sa_v  = f"{metrics['spy_alpha_pct']:+.1f}%" if metrics["spy_alpha_pct"] is not None else "N/A"
    sa_s  = f"SPY {metrics['spy_bh_pct']:+.1f}%" if metrics["spy_bh_pct"] is not None else ""
    sa_c  = "green" if (metrics["spy_alpha_pct"] or 0) > 0 else "red"
    with cols2[0]: st.markdown(_metric("夏普比率",  f"{metrics['sharpe']:.2f}",
                                                    f"年化波动 {metrics['ann_vol_pct']:.1f}%", sh_c), unsafe_allow_html=True)
    with cols2[1]: st.markdown(_metric("卡玛比率",  f"{metrics['calmar']:.2f}",
                                                    "年化收益 / 最大回撤", ca_c), unsafe_allow_html=True)
    with cols2[2]: st.markdown(_metric("月度胜率",  f"{metrics['monthly_win_pct']:.1f}%",
                                                    f"交易胜率 {metrics['win_rate_pct']:.1f}%", "blue"), unsafe_allow_html=True)
    with cols2[3]: st.markdown(_metric("vs SPY Alpha", sa_v, sa_s, sa_c), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 资金曲线 + 回撤 ──────────────────────────────────────
    st.markdown('<div class="section-header">💰 资金曲线 vs 买入持有</div>', unsafe_allow_html=True)
    st.plotly_chart(equity_chart(df), use_container_width=True)

    # ── 月度热力图 + 状态分布 ────────────────────────────────
    col_heat, col_dist = st.columns([3, 2], gap="medium")
    with col_heat:
        st.markdown('<div class="section-header">🗓 月度收益热力图&nbsp;&nbsp;<span style="font-size:0.75rem;color:#475569;font-weight:400">绿=盈利月 · 红=亏损月</span></div>', unsafe_allow_html=True)
        st.plotly_chart(monthly_heatmap(metrics["monthly_df"]), use_container_width=True)
    with col_dist:
        st.markdown('<div class="section-header">🧩 HMM 状态分布</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_bar(df), use_container_width=True)

    # ── 交易统计 + 详情 ──────────────────────────────────────
    col_stats, col_risk = st.columns([1, 1], gap="medium")
    with col_stats:
        st.markdown('<div class="section-header">📋 交易统计</div>', unsafe_allow_html=True)
        stats = [
            ("总笔数",         f"{metrics['n_trades']}"),
            ("盈亏比 (R:R)",   f"{metrics['rr_ratio']:.2f}"),
            ("平均盈利",       f"${metrics['avg_win']:+,.0f}"),
            ("平均亏损",       f"${metrics['avg_loss']:+,.0f}"),
            ("平均仓位",       f"{metrics['avg_pos_size_pct']:.0f}%"),
        ]
        st.markdown("".join(
            f'<div class="sig-row"><span class="sig-name">{k}</span>'
            f'<span class="sig-val">{v}</span></div>' for k, v in stats
        ), unsafe_allow_html=True)

    with col_risk:
        st.markdown('<div class="section-header">🛡 风控参数</div>', unsafe_allow_html=True)
        risk_items = [
            ("固定止损",   "-8% / 笔"),
            ("杠杆",       "2.5×"),
            ("最小仓位",   "40%（9/14 信号）"),
            ("最大仓位",   "100%（14/14 信号）"),
            ("冷静期",     "48h / 2 日"),
            ("最大持仓",   "30天 / 60日"),
            ("HMM 训练",   "Walk-Forward 滚动"),
        ]
        st.markdown("".join(
            f'<div class="sig-row"><span class="sig-name">{k}</span>'
            f'<span class="sig-val">{v}</span></div>' for k, v in risk_items
        ), unsafe_allow_html=True)

    # ── 完整交易记录 ─────────────────────────────────────────
    if trades:
        st.markdown('<div class="section-header">📝 完整交易记录</div>', unsafe_allow_html=True)
        tdf = pd.DataFrame(trades)
        tdf["entry_time"]    = pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tdf["exit_time"]     = pd.to_datetime(tdf["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tdf["entry_price"]   = tdf["entry_price"].map("${:,.2f}".format)
        tdf["exit_price"]    = tdf["exit_price"].map("${:,.2f}".format)
        tdf["pnl"]           = tdf["pnl"].map("${:+,.2f}".format)
        tdf["pos_size_pct"]  = tdf["pos_size_pct"].map(lambda x: f"{x*100:.0f}%")
        tdf["return_pct"]    = tdf["return_pct"].map(lambda x: f"{x:+.1f}%")
        tdf = tdf[["entry_time","exit_time","entry_price","exit_price",
                   "pos_size_pct","pnl","return_pct","hold_bars","exit_reason"]]
        tdf.columns = ["入场时间","出场时间","入场价","出场价","仓位","盈亏","收益率","持仓bar","出场原因"]
        st.dataframe(tdf, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    # 顶部标题栏
    hc1, hc2 = st.columns([8, 1])
    with hc1:
        st.markdown('<div class="page-title">📈 Regime-Based HMM Trading Dashboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-sub">9-State Gaussian HMM &nbsp;·&nbsp; 14-Signal Voting &nbsp;·&nbsp; 2.5× Leverage &nbsp;·&nbsp; Walk-Forward &nbsp;·&nbsp; 数据截至 {_computed_at()}</div>', unsafe_allow_html=True)
    with hc2:
        st.write("")
        st.write("")
        if st.button("🔄 刷新", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    tab_aapl, tab_gold, tab_silver = st.tabs([
        "🍎  Apple (AAPL)",
        "🥇  Gold (GC=F)",
        "🥈  Silver (SI=F)",
    ])
    with tab_aapl:   render_asset("AAPL")
    with tab_gold:   render_asset("GC=F")
    with tab_silver: render_asset("SI=F")

    st.markdown(
        "<div style='text-align:center;color:#1e293b;font-size:0.7rem;margin-top:2rem'>"
        "仅供学习研究，不构成投资建议。</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
