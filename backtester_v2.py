"""
backtester_v2.py  —  Trailing Stop + 多信号入场
=================================================
核心改动 vs v1：
  1. 出场：Trailing Stop（激活阈值后从最高点回撤触发）替代 RegimeReduce + MaxHold
  2. 入场：HMM bull state 作为大方向过滤 + EMA金叉/MACD金叉触发，提高频率
  3. 不再依赖 14 信号投票，改为 3 个核心触发条件
  4. 保留硬止损 + HMM bear state 强制退出作为兜底
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

from data_loader import get_hmm_features, fetch_data
from backtester import (
    fit_hmm, identify_states, _walk_forward_states,
    RANDOM_SEED, STARTING_CAP, LEVERAGE, FRICTION_PCT,
    MARGIN_PARAMS, WF_TRAIN_RATIO, WF_STEP_RATIO,
    _ema, _rsi, _adx, _macd, _compute_metrics,
)

warnings.filterwarnings("ignore")


# ============================================================
# V2 参数
# ============================================================

TICKER_PARAMS_V2: Dict[str, Dict] = {
    "AAPL": {
        "n_states":    5,
        "bull_top":    3,
        "stop":       -0.06,
        "trail_act":   0.03,
        "trail_pct":   0.06,
        "adx_entry":  20,
        "ema_fast":   10,
        "ema_slow":   30,
        "bear_exit":  True,
        "cooldown":   2,
    },
    "GC=F": {
        "n_states":    7,
        "bull_top":    2,
        "stop":       -0.08,
        "trail_act":   0.03,
        "trail_pct":   0.05,
        "adx_entry":  20,
        "ema_fast":   10,
        "ema_slow":   30,
        "bear_exit":  True,
        "cooldown":   2,
    },
    "SI=F": {
        "n_states":    7,
        "bull_top":    2,
        "stop":       -0.08,
        "trail_act":   0.03,
        "trail_pct":   0.07,
        "adx_entry":  20,
        "ema_fast":   10,
        "ema_slow":   30,
        "bear_exit":  True,
        "cooldown":   2,
    },
}

BEAR_CONFIRM_V2: Dict[str, int] = {
    "AAPL": 1,
    "GC=F": 2,
    "SI=F": 1,
}


# ============================================================
# 指标计算
# ============================================================

def compute_indicators_v2(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    p  = TICKER_PARAMS_V2.get(ticker, TICKER_PARAMS_V2["GC=F"])
    df = df.copy()

    # EMA 金叉
    df["ema_fast"] = _ema(df["Close"], p["ema_fast"])
    df["ema_slow"] = _ema(df["Close"], p["ema_slow"])
    df["ema_cross"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df["ema_cross_up"] = (df["ema_cross"] == 1) & (df["ema_cross"].shift(1) == 0)

    # MACD
    macd_line, macd_sig = _macd(df["Close"])
    df["macd_line"]   = macd_line
    df["macd_signal"] = macd_sig
    df["macd_cross_up"] = (macd_line > macd_sig) & (macd_line.shift(1) <= macd_sig.shift(1))

    # ADX
    df["adx"] = _adx(df)

    # RSI（避免超买入场）
    df["rsi"] = _rsi(df["Close"])

    # EMA 200（大趋势过滤）
    df["ema200"] = _ema(df["Close"], 200)

    return df


def _entry_signal(row, adx_entry: float) -> Tuple[bool, int]:
    """
    入场条件（三选二）：
    1. EMA 金叉
    2. MACD 金叉
    3. ADX > 阈值
    + 必须：RSI < 80，价格 > EMA200
    """
    cond_ema   = bool(row.get("ema_cross_up", False))
    cond_macd  = bool(row.get("macd_cross_up", False))
    cond_adx   = float(row.get("adx", 0)) > adx_entry
    cond_rsi   = float(row.get("rsi", 50)) < 80
    cond_trend = float(row.get("Close", 0)) > float(row.get("ema200", 0))

    trigger_count = sum([cond_ema, cond_macd, cond_adx])
    return trigger_count >= 2 and cond_rsi and cond_trend, trigger_count


def _dynamic_position_size(trigger_count: int, adx: float, rsi: float, max_pos: float) -> float:
    """
    动态仓位：根据信号强度线性调整
    - 2个信号触发：基础仓位 55%
    - 3个信号触发：满仓 85%
    - ADX 越强，仓位越大（上限 max_pos）
    - RSI 越接近 80（超买），仓位打折
    """
    base = 0.55 if trigger_count == 2 else 0.80

    # ADX 强度加成：ADX>40 加5%，ADX>60 再加5%
    adx_bonus = 0.05 if adx > 40 else 0.0
    adx_bonus += 0.05 if adx > 60 else 0.0

    # RSI 超买折扣：RSI 60-80 线性折扣到 0.85
    rsi_factor = 1.0 if rsi < 60 else max(0.85, 1.0 - (rsi - 60) / 20 * 0.15)

    size = (base + adx_bonus) * rsi_factor
    return min(size, max_pos, 0.90)  # 硬上限 90%


# ============================================================
# 核心模拟
# ============================================================

def _simulate_v2(df: pd.DataFrame, ticker: str) -> Tuple[List, List]:
    p            = TICKER_PARAMS_V2.get(ticker, TICKER_PARAMS_V2["GC=F"])
    stop_loss    = p["stop"]
    trail_act    = p["trail_act"]
    trail_pct    = p["trail_pct"]
    adx_entry    = p["adx_entry"]
    bear_exit    = p["bear_exit"]
    bear_confirm = BEAR_CONFIRM_V2.get(ticker, 1)
    friction     = FRICTION_PCT
    mp           = MARGIN_PARAMS.get(ticker, {"initial_margin": 0.40, "maintenance_margin": 0.25})
    init_margin  = mp["initial_margin"]
    maint_margin = mp["maintenance_margin"]

    capital      = float(STARTING_CAP)
    position     = 0.0
    in_trade     = False
    entry_price  = 0.0
    entry_time   = None
    hold_bars    = 0
    peak_price   = 0.0          # highest price since entry（用于 trailing stop）
    trail_active = False        # trailing stop 是否已激活
    bear_consec  = 0
    cooldown     = 0
    pos_size_pct = 0.0

    equity_curve = []
    trades       = []

    for ts, row in df.iterrows():
        price = float(row["Close"])

        if in_trade:
            hold_bars  += 1
            current_ret = (price - entry_price) / entry_price

            # 更新最高价
            if price > peak_price:
                peak_price = price

            # 激活 trailing stop
            if not trail_active and current_ret >= trail_act:
                trail_active = True

            # bear 连续确认计数
            if row.get("is_bear", False):
                bear_consec += 1
            else:
                bear_consec = 0

            exit_reason = None

            # 1. Margin Call（最优先）
            notional       = position * price * LEVERAGE
            unrealised_pnl = (price - entry_price) * position * LEVERAGE
            acct_equity    = capital + unrealised_pnl
            if notional > 0 and acct_equity < maint_margin * notional:
                exit_reason = "MarginCall"

            # 2. 硬止损
            elif current_ret <= stop_loss:
                exit_reason = f"StopLoss ({stop_loss*100:.0f}%)"

            # 3. Trailing Stop（激活后从最高点回撤超过 trail_pct）
            elif trail_active and (peak_price - price) / peak_price >= trail_pct:
                exit_reason = f"TrailingStop (-{trail_pct*100:.0f}% from peak)"

            # 4. HMM bear state 强制出场
            elif bear_exit and bear_consec >= bear_confirm:
                exit_reason = "Regime → Bear"

            if exit_reason:
                exit_price_net = price * (1 - friction)
                pnl            = (exit_price_net - entry_price) * position * LEVERAGE
                capital       += pnl
                trades.append({
                    "entry_time":   entry_time,
                    "exit_time":    ts,
                    "entry_price":  entry_price,
                    "exit_price":   price,
                    "pnl":          pnl,
                    "pos_size_pct": pos_size_pct,
                    "exit_reason":  exit_reason,
                    "hold_bars":    hold_bars,
                    "return_pct":   (exit_price_net / entry_price - 1) * 100 * LEVERAGE,
                    "peak_price":   peak_price,
                    "trail_active": trail_active,
                })
                position     = 0.0
                in_trade     = False
                hold_bars    = 0
                peak_price   = 0.0
                trail_active = False
                bear_consec  = 0
                cooldown     = p.get("cooldown", 2)

        if cooldown > 0:
            cooldown -= 1

        # 开仓：HMM bull + 入场信号
        entry_ok, trigger_count = _entry_signal(row, adx_entry)
        if (not in_trade and cooldown == 0
                and row.get("is_bull", False)
                and entry_ok):

            max_pos = 1.0 / (LEVERAGE * init_margin)
            adx_val = float(row.get("adx", 0))
            rsi_val = float(row.get("rsi", 50))
            pos_size_pct = _dynamic_position_size(trigger_count, adx_val, rsi_val, max_pos)

            position    = capital * pos_size_pct / price
            entry_price = price * (1 + friction)
            entry_time  = ts
            in_trade    = True
            hold_bars   = 0
            peak_price  = price
            trail_active = False
            bear_consec = 0

        mtm = capital + (price - entry_price) * position * LEVERAGE if in_trade else capital
        equity_curve.append(mtm)

    # 回测结束时强制平仓
    if in_trade:
        last_price     = float(df["Close"].iloc[-1])
        last_price_net = last_price * (1 - friction)
        pnl            = (last_price_net - entry_price) * position * LEVERAGE
        capital       += pnl
        trades.append({
            "entry_time":   entry_time,
            "exit_time":    df.index[-1],
            "entry_price":  entry_price,
            "exit_price":   last_price,
            "pnl":          pnl,
            "pos_size_pct": pos_size_pct,
            "exit_reason":  "End of data",
            "hold_bars":    hold_bars,
            "return_pct":   (last_price - entry_price) / entry_price * 100 * LEVERAGE,
            "peak_price":   peak_price,
            "trail_active": trail_active,
        })

    return equity_curve, trades


# ============================================================
# 主入口
# ============================================================

def run_backtest_v2(df: pd.DataFrame, ticker: str = "GC=F") -> Dict:
    p        = TICKER_PARAMS_V2.get(ticker, TICKER_PARAMS_V2["GC=F"])
    n_states = p["n_states"]
    bull_top = p["bull_top"]

    features  = get_hmm_features(df)
    n         = len(features)
    wf_states, is_bull_arr, is_bear_arr = _walk_forward_states(
        features, n, n_states, bull_top=bull_top)

    df = df.copy()
    df["state"]   = wf_states
    df["is_bull"] = is_bull_arr
    df["is_bear"] = is_bear_arr

    df = compute_indicators_v2(df, ticker)

    equity_curve, trades = _simulate_v2(df, ticker)

    df["equity"] = equity_curve
    eq = df["equity"]
    metrics = _compute_metrics(df, trades, ticker, True)

    return {
        "df":      df,
        "equity":  pd.Series(equity_curve, index=df.index),
        "trades":  trades,
        "metrics": metrics,
        "n_states": n_states,
    }


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    for ticker in ["GC=F", "SI=F", "AAPL"]:
        df  = fetch_data(ticker)
        res = run_backtest_v2(df, ticker)
        m   = res["metrics"]
        t   = len(res["trades"])
        print(f"{ticker:6s}  Return={m['total_return_pct']:+8.1f}%  "
              f"Sharpe={m['sharpe']:.2f}  MaxDD={m['max_drawdown_pct']:.1f}%  "
              f"Trades={t}  WinRate={m['win_rate_pct']:.1f}%")
