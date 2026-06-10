"""
strategy_growth.py — Per-asset optimal momentum/trend strategies for growth stocks.

Each ticker uses the strategy with highest Calmar ratio from sensitivity analysis:
  AAPL  → Price > EMA200                         (Calmar 0.61, Return +3093%, MaxDD -34%)
  NVDA  → 52-Week High >80%                       (Calmar 0.94, Return +48446%, MaxDD -42%)
  META  → EMA21 > EMA50                           (Calmar 1.19, Return +2277%, MaxDD -21%, 近3年+112%)
          Entry gate: RSI14>58 AND 20日波动率<3%    (filters low-momentum & high-vol whipsaws)
  AMZN  → 近52周高点 >70%                           (Calmar 0.59, Return +5406%, MaxDD -41%, 年化+24.4% ≈ 买入持有)
  GOOG  → 52-Week High >80%                        (Calmar 0.54, Return +2323%, MaxDD -35%)
  MSFT  → 价格>EMA200 + RSI 30-65                   (Calmar 0.23, Return +?%, MaxDD -44%, 胜率64%)
  TSLA  → EMA7 > EMA21                            (Calmar 0.79, Return +20240%, MaxDD -51%, Sharpe 0.95)
          No entry gate — gates block TSLA bull-run entries, hurting returns
  HOOD  → 52-Week High >75%                        (Calmar 1.32, Return +503%, MaxDD -34%)
  PLTR  → Price > EMA200                           (Calmar 1.33, Return +2303%, MaxDD -57%)
  SOXL  → EMA21 > EMA50                           (Calmar 0.55, Return +18860%, MaxDD -70%, Sharpe 0.64)
          No entry gate — 3x leveraged ETF, gates block bull-run entries
  MU    → EMA21 > EMA50 + 成交量>20日均量            (Calmar 0.65, Return +5400%, MaxDD -38%, Sharpe 0.87, 胜率64%)
  MRVL  → EMA21 > EMA50                           (Calmar 0.42, Return +2293%, MaxDD -44%, Sharpe 0.54)
  AMD   → EMA10 > EMA30                           (Calmar 0.37, Return +4408%, MaxDD -62%, Sharpe 0.54)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from backtester import STARTING_CAP, FRICTION_PCT, LEVERAGE, _ema, compute_indicators


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)

# ── 每个成长股对应的策略类型 ──────────────────────────────────
GROWTH_STRATEGY: Dict[str, str] = {
    "AAPL": "ema200",
    "NVDA": "52wh80",
    "META": "ema21_50_vol",
    "AMZN": "52wh70",
    "GOOG": "52wh80",
    "MSFT": "ema200_rsi3065",
    "TSLA": "ema7_21",
    "HOOD": "52wh75",
    "PLTR": "ema200",
    "SOXL": "ema21_50",
    "MU":   "ema21_50_vol20",
    "MRVL": "ema21_50",
    "AMD":  "ema10_30",
}

STRATEGY_LABELS: Dict[str, str] = {
    "ema200":              "价格 > EMA200",
    "52wh80":              "近52周高点 >80%",
    "ema21_50":            "EMA21 > EMA50",
    "ema21_50_slope":      "EMA21>EMA50 + EMA200趋势向上",
    "ema21_50_slope_rsi":  "EMA21>EMA50 + EMA200趋势向上 + RSI动量确认",
    "ema21_50_vol":        "EMA21>EMA50（入场:RSI>58且低波动）",
    "ema50_200":           "EMA50 > EMA200",
    "52wh70":              "近52周高点 >70%",
    "ema21_50_vol20":      "EMA21>EMA50（入场:量>20日均量）",
    "ema200_rsi3065":      "价格>EMA200 且 RSI 30-65",
    "buyhold":             "买入持有",
    "52wh75":              "近52周高点 >75%",
    "ema7_21":             "EMA7 > EMA21",
    "ema10_30":            "EMA10 > EMA30",
}


def _compute_metrics(eq: pd.Series, trades: list) -> Dict:
    total_ret = (eq.iloc[-1] / STARTING_CAP - 1) * 100
    dd_series = (eq - eq.cummax()) / eq.cummax() * 100
    max_dd    = dd_series.min()
    pct_ret   = eq.pct_change().dropna()
    ann_ret   = ((eq.iloc[-1] / STARTING_CAP) ** (252 / max(len(eq), 1)) - 1) * 100
    ann_vol   = pct_ret.std() * np.sqrt(252) * 100
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    wins      = [t for t in trades if t["pnl"] > 0]
    win_rate  = len(wins) / len(trades) * 100 if trades else 0.0
    bh_ret    = (eq.index.map(lambda _: None))  # placeholder
    return {
        "total_return_pct": total_ret,
        "ann_return_pct":   ann_ret,
        "sharpe":           sharpe,
        "max_drawdown_pct": max_dd,
        "calmar":           calmar,
        "win_rate_pct":     win_rate,
        "n_trades":         len(trades),
    }


def _simulate_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    stop: float = -0.20,
    entry_gate: pd.Series | None = None,
    trail: float | None = None,
) -> Dict:
    """Simulate long-only strategy from a binary signal series (1=hold, 0=flat).
    Entry: next open after signal flips to 1.
    Exit:  next open after signal flips to 0, stop-loss, or trailing stop intrabar.

    entry_gate: optional boolean series; when provided, an entry is only taken on
    a fresh signal crossover (0→1) if the gate was also 1 at that bar.
    Gate is NOT used to exit — it only blocks low-quality entries.
    trail: trailing stop from peak (e.g. -0.12 means exit if price drops 12% from high).
    """
    cap       = float(STARTING_CAP)
    in_trade  = False
    entry_p   = 0.0
    shares    = 0.0
    peak_p    = 0.0
    equity    = []
    trades    = []
    entry_ts  = None
    hold_bars = 0

    sig_vals  = signal.reindex(df.index).fillna(0).values
    gate_vals = (entry_gate.reindex(df.index).fillna(0).values
                 if entry_gate is not None
                 else None)
    opens    = df["Open"].values
    closes   = df["Close"].values
    idx      = df.index
    entry_capital = 0.0

    for i in range(len(df)):
        price = closes[i]

        if in_trade:
            hold_bars += 1
            peak_p = max(peak_p, price)
            ret = (price - entry_p) / entry_p
            exit_reason = None
            if ret <= stop:
                exit_reason = f"StopLoss ({stop*100:.0f}%)"
            elif trail is not None and (price - peak_p) / peak_p <= trail:
                exit_reason = f"TrailingStop ({trail*100:.0f}%)"
            elif i > 0 and sig_vals[i - 1] == 0:
                # signal flipped to 0 yesterday → exit at today's open
                exit_price  = opens[i]
                pnl         = shares * (exit_price * (1 - FRICTION_PCT) - entry_p)
                cap        += pnl
                trades.append({
                    "entry_time":     entry_ts,  "exit_time": idx[i],
                    "entry_price":    entry_p,   "exit_price": exit_price,
                    "pnl":            pnl,        "hold_bars": hold_bars,
                    "return_pct":     (exit_price / entry_p - 1) * 100,
                    "exit_reason":    STRATEGY_LABELS.get(
                        GROWTH_STRATEGY.get("", ""), "Signal Exit"),
                    "entry_capital":  entry_capital,
                    "exit_capital":   cap,
                })
                in_trade = False; shares = 0.0; hold_bars = 0
                equity.append(cap)
                continue

            if exit_reason:
                pnl   = shares * (price * (1 - FRICTION_PCT) - entry_p)
                cap  += pnl
                trades.append({
                    "entry_time":    entry_ts, "exit_time": idx[i],
                    "entry_price":   entry_p,  "exit_price": price,
                    "pnl":           pnl,       "hold_bars": hold_bars,
                    "return_pct":    (price / entry_p - 1) * 100,
                    "exit_reason":   exit_reason,
                    "entry_capital": entry_capital,
                    "exit_capital":  cap,
                })
                in_trade = False; shares = 0.0; hold_bars = 0
                equity.append(cap)
                continue

        # Entry: signal was 1 yesterday, not yet in trade
        if not in_trade and i > 0 and sig_vals[i - 1] == 1:
            # If entry_gate provided, only enter on fresh crossover (sig flipped 0→1)
            # when the gate is also satisfied at crossover bar
            is_fresh = (i < 2) or (sig_vals[i - 2] == 0)
            gate_ok  = (gate_vals is None) or (gate_vals[i - 1] == 1)
            if is_fresh and gate_ok:
                entry_capital = cap
                entry_p  = opens[i] * (1 + FRICTION_PCT)
                peak_p   = entry_p
                shares   = cap / entry_p
                in_trade = True
                entry_ts = idx[i]
                hold_bars = 0
            elif not is_fresh and in_trade is False:
                # already in signal zone — enter only if gate was never blocking
                pass

        mtm = shares * price + (cap - shares * entry_p) if in_trade else cap
        equity.append(mtm)

    # Close open position at last close
    if in_trade and shares > 0:
        exit_price = closes[-1]
        pnl = shares * (exit_price * (1 - FRICTION_PCT) - entry_p)
        cap += pnl
        trades.append({
            "entry_time":    entry_ts, "exit_time": idx[-1],
            "entry_price":   entry_p,  "exit_price": exit_price,
            "pnl":           pnl,       "hold_bars": hold_bars,
            "return_pct":    (exit_price / entry_p - 1) * 100,
            "exit_reason":   "持仓中",
            "entry_capital": entry_capital,
            "exit_capital":  cap,
        })
        equity[-1] = cap

    eq = pd.Series(equity, index=df.index)
    metrics = _compute_metrics(eq, trades)
    return {"equity": eq, "trades": trades, "metrics": metrics}


def run_strategy_growth(df: pd.DataFrame, ticker: str) -> Dict:
    """Run the per-asset optimal growth strategy. Returns same schema as other strategies."""
    df   = df.copy()
    # Compute all technical indicators so df is complete for app.py charts
    if "ema200" not in df.columns:
        df = compute_indicators(df, ticker)
    c    = df["Close"]
    strat = GROWTH_STRATEGY.get(ticker)

    if strat is None:
        raise ValueError(f"run_strategy_growth: {ticker} is not a growth ticker")

    # ── Build signal ───────────────────────────────────────────
    if strat == "ema200_rsi3065":
        # MSFT 专用 — 价格>EMA200 且 RSI 30-65，追踪止损-12%
        # 胜率64%，Calmar 0.23，MaxDD -44%
        e200  = _ema(c, 200)
        r14   = _rsi(c, 14)
        signal = ((c > e200) & (r14 > 30) & (r14 < 65)).shift(1).fillna(False).astype(int)
        result = _simulate_signal(df, signal, stop=-0.10, trail=-0.12)
        result["strategy_type"]  = strat
        result["strategy_label"] = STRATEGY_LABELS[strat]
        return result

    elif strat == "ema200":
        e200 = _ema(c, 200)
        signal = (c > e200).shift(1).fillna(False).astype(int)

    elif strat == "ema21_50":
        e21 = _ema(c, 21)
        e50 = _ema(c, 50)
        signal = (e21 > e50).shift(1).fillna(False).astype(int)

    elif strat == "ema21_50_vol20":
        # MU 专用策略 — 入场门控：成交量 > 20日均量（过滤低量假突破）
        # Calmar 0.65, MaxDD -38%, 胜率64%, 止损-10%
        e21    = _ema(c, 21)
        e50    = _ema(c, 50)
        vol_ma = df["Volume"].rolling(20).mean()
        signal     = (e21 > e50).shift(1).fillna(False).astype(int)
        entry_gate = (df["Volume"] > vol_ma).shift(1).fillna(False).astype(int)
        result = _simulate_signal(df, signal, stop=-0.10, entry_gate=entry_gate)
        result["strategy_type"]  = strat
        result["strategy_label"] = STRATEGY_LABELS[strat]
        return result

    elif strat == "ema21_50_slope":
        # EMA21>EMA50 且 EMA200 20日斜率>0（长期趋势未转头，过滤震荡假信号）
        e21  = _ema(c, 21)
        e50  = _ema(c, 50)
        e200 = _ema(c, 200)
        e200_slope = e200.pct_change(20) * 100
        signal = ((e21 > e50) & (e200_slope > 0)).shift(1).fillna(False).astype(int)

    elif strat == "ema21_50_slope_rsi":
        # Legacy — kept for reference, superseded by ema21_50_vol
        e21  = _ema(c, 21)
        e50  = _ema(c, 50)
        e200 = _ema(c, 200)
        e200_slope  = e200.pct_change(20) * 100
        e21_slope5  = e21.pct_change(5) * 100
        rsi14       = _rsi(c, 14)
        signal      = ((e21 > e50) & (e200_slope > 0)).shift(1).fillna(False).astype(int)
        entry_gate  = ((rsi14 > 58) & (e21_slope5 > 1.0)).shift(1).fillna(False).astype(int)
        result = _simulate_signal(df, signal, stop=-0.20, entry_gate=entry_gate)
        result["strategy_type"]  = strat
        result["strategy_label"] = STRATEGY_LABELS[strat]
        return result

    elif strat == "ema21_50_vol":
        # META 专用策略 — 全参数扫描最优 (Calmar 1.19, Sharpe 1.04, 胜率 62.5%, MaxDD -21.4%)
        # 持仓信号: EMA21 > EMA50 (趋势跟踪，不添加额外过滤以保留大牛市持仓)
        # 入场门控: RSI14>58 (有动量) AND 20日波动率<3% (非高波震荡期)
        # 门控只在信号从0→1的穿越时刻生效，不影响已有持仓的继续持有和退出
        e21   = _ema(c, 21)
        e50   = _ema(c, 50)
        rsi14 = _rsi(c, 14)
        vol20 = c.pct_change().rolling(20).std() * 100
        signal     = (e21 > e50).shift(1).fillna(False).astype(int)
        entry_gate = ((rsi14 > 58) & (vol20 < 3.0)).shift(1).fillna(False).astype(int)
        result = _simulate_signal(df, signal, stop=-0.20, entry_gate=entry_gate)
        result["strategy_type"]  = strat
        result["strategy_label"] = STRATEGY_LABELS[strat]
        return result

    elif strat == "ema21_50":
        e21 = _ema(c, 21)
        e50 = _ema(c, 50)
        signal = (e21 > e50).shift(1).fillna(False).astype(int)

    elif strat == "ema50_200":
        e50  = _ema(c, 50)
        e200 = _ema(c, 200)
        signal = (e50 > e200).shift(1).fillna(False).astype(int)

    elif strat == "52wh70":
        high52 = c.rolling(252, min_periods=50).max()
        signal = (c > high52 * 0.70).shift(1).fillna(False).astype(int)

    elif strat == "52wh80":
        high52 = c.rolling(252, min_periods=50).max()
        signal = (c > high52 * 0.80).shift(1).fillna(False).astype(int)

    elif strat == "52wh75":
        high52 = c.rolling(252, min_periods=50).max()
        signal = (c > high52 * 0.75).shift(1).fillna(False).astype(int)

    elif strat == "ema7_21":
        # TSLA 专用策略 — 全参数扫描最优 (Calmar 0.79, Sharpe 0.95, Return +20240%, MaxDD -51%, 92笔)
        # 无入场门控：门控会阻挡TSLA大牛市启动信号，得不偿失
        e7  = _ema(c, 7)
        e21 = _ema(c, 21)
        signal = (e7 > e21).shift(1).fillna(False).astype(int)

    elif strat == "ema10_30":
        # AMD 专用策略 — 全参数扫描最优 (Calmar 0.37, Return +4408%, MaxDD -62%, 81笔)
        e10 = _ema(c, 10)
        e30 = _ema(c, 30)
        signal = (e10 > e30).shift(1).fillna(False).astype(int)

    elif strat == "buyhold":
        signal = pd.Series(1, index=df.index)

    else:
        raise ValueError(f"Unknown growth strategy: {strat}")

    TICKER_STOP = {
        "MU": -0.10,
    }
    stop = TICKER_STOP.get(ticker, -0.20)

    result = _simulate_signal(df, signal, stop=stop)
    result["strategy_type"]  = strat
    result["strategy_label"] = STRATEGY_LABELS[strat]
    return result
