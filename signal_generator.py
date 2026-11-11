"""
signal_generator.py — 每日实时信号生成
========================================
用法：
    python signal_generator.py

输出：
    signals/signal_YYYYMMDD.json   ← 主输出（文件）
    terminal 打印摘要               ← 辅助输出
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import backtester as bt
from backtester import (
    TICKER_PARAMS, BEAR_CONFIRM, STARTING_CAP, LEVERAGE,
    WF_TRAIN_RATIO, fit_hmm, identify_states,
    compute_indicators, compute_signals, get_hmm_features,
    _walk_forward_states,
)
from data_loader import fetch_data, prefetch_recent, DAILY_TICKERS, _fetch_prices

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "signals")


# ── 基本面数据获取 ─────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> Dict:
    """拉取基本面数据：营收增速、估值、财报日。失败返回空dict。"""
    try:
        import yfinance as yf
        from datetime import date, timedelta
        t = yf.Ticker(ticker)
        info = t.info
        cal  = t.calendar or {}

        rev_growth = info.get("revenueGrowth")        # 营收YoY增速，如 0.85 = 85%
        ps         = info.get("priceToSalesTrailing12Months")
        pe         = info.get("trailingPE")
        profit_margin = info.get("profitMargins")

        # 下次财报日
        earnings_dates = cal.get("Earnings Date", [])
        next_earnings  = earnings_dates[0] if earnings_dates else None

        # 财报窗口保护：前3天+后3天
        earnings_warning = False
        if next_earnings:
            today = date.today()
            delta = (next_earnings - today).days
            earnings_warning = (-3 <= delta <= 3)

        # 估值风险标记
        valuation_risk = "high" if (ps and ps > 50) else ("medium" if ps and ps > 20 else "low")

        # 营收增速风险
        rev_ok = (rev_growth is not None and rev_growth > 0)

        # 综合基本面评分（0-3）
        fa_score = 0
        if rev_ok: fa_score += 1
        if ps and ps < 20: fa_score += 1
        if profit_margin and profit_margin > 0.10: fa_score += 1

        # 基本面警告列表
        warnings_list = []
        if earnings_warning:
            warnings_list.append(f"⚠️ 财报窗口期（{next_earnings}），建议不开新仓")
        if not rev_ok:
            warnings_list.append("⚠️ 营收同比负增长")
        if valuation_risk == "high":
            warnings_list.append(f"⚠️ 估值偏高 PS={ps:.1f}x")

        return {
            "rev_growth":      round(rev_growth * 100, 1) if rev_growth is not None else None,
            "ps":              round(ps, 2) if ps else None,
            "pe":              round(pe, 2) if pe else None,
            "profit_margin":   round(profit_margin * 100, 1) if profit_margin else None,
            "next_earnings":   str(next_earnings) if next_earnings else None,
            "earnings_warning": earnings_warning,
            "valuation_risk":  valuation_risk,
            "rev_ok":          rev_ok,
            "fa_score":        fa_score,
            "warnings":        warnings_list,
        }
    except Exception:
        return {}


# ── 辅助 ─────────────────────────────────────────────────────

def _regime_labels_by_rank(model, n_states: int) -> dict:
    """返回 {state_id: label}，按 mean_return rank 从低到高标注。"""
    ranks = np.argsort(model.means_[:, 0])   # ranks[0]=lowest return state
    labels = {}
    for rank, state in enumerate(ranks):
        if rank == 0:
            labels[int(state)] = "Bear/Crash"
        elif rank == 1:
            labels[int(state)] = "Bear"
        elif rank == n_states - 1:
            labels[int(state)] = "Bull Run"
        elif rank == n_states - 2:
            labels[int(state)] = "Bull+"
        elif rank == n_states - 3:
            labels[int(state)] = "Warming Up"
        else:
            labels[int(state)] = f"Neutral-{rank}"
    return labels


def _action(is_bull: bool, is_bear: bool, score: int, min_conf: int,
            adx: float, adx_entry: float, in_trade: bool) -> str:
    """
    推断当日操作建议：
      ENTER   — 满足入场条件，当前空仓
      HOLD    — 已持仓，无退出信号
      EXIT    — 已持仓，bear 信号触发
      WATCH   — 接近入场但未满足全部条件
      STAY_OUT— 无持仓，不满足入场
    """
    if in_trade:
        return "EXIT" if is_bear else "HOLD"
    if is_bull and score >= min_conf and adx > adx_entry:
        return "ENTER"
    if is_bull and score >= min_conf - 1:
        return "WATCH"
    return "STAY_OUT"


# ── 单资产信号 ────────────────────────────────────────────────

def generate_signal(ticker: str) -> Dict:
    df = fetch_data(ticker, force_refresh=True)
    p  = TICKER_PARAMS.get(ticker, {})

    n_states  = p.get("n_states", 7)
    bull_top  = p.get("bull_top", 2)
    min_conf  = p.get("min_conf", 9)
    stop_pct  = p.get("stop", -0.08)
    adx_entry = p.get("adx_entry", 25)

    features = get_hmm_features(df)
    n        = len(features)

    # walk-forward states（与回测一致，返回三元组）
    wf_states, is_bull_arr, is_bear_arr = _walk_forward_states(
        features, n, n_states, bull_top=bull_top)

    # 全样本 fit 仅用于 posterior 概率和 regime 标签展示（不参与交易决策）
    full_model              = fit_hmm(features, n_states)
    bull_states, bear_state = identify_states(full_model, bull_top)
    regime_label_map        = _regime_labels_by_rank(full_model, n_states)
    full_states             = full_model.predict(features)

    df = df.copy()
    df["state"]   = wf_states
    df["is_bull"] = is_bull_arr   # walk-forward bull labels（与回测一致）
    df["is_bear"] = is_bear_arr   # walk-forward bear labels（与回测一致）

    df = compute_indicators(df, ticker)
    _, score_series = compute_signals(df)
    df["signal_score"] = score_series

    # 最新 bar
    last = df.iloc[-1]
    last_date   = df.index[-1]
    state_today = int(full_states[-1])   # full_model state for regime label
    is_bull     = bool(last["is_bull"])
    is_bear     = bool(last["is_bear"])
    score       = int(last["signal_score"])
    adx         = float(last["adx"])
    close       = float(last["Close"])
    regime      = regime_label_map.get(state_today, f"State-{state_today}")

    # HMM posterior（最新 bar）
    try:
        posterior = full_model.predict_proba(features[-1:]).flatten().tolist()
    except Exception:
        posterior = [1.0 / n_states] * n_states

    bull_prob = sum(posterior[s] for s in bull_states)
    bear_prob = posterior[bear_state]

    # 信号详情（14 项）
    signal_details = {
        "rsi_ok":        bool(last["rsi"] < 90),
        "momentum_ok":   bool(last["momentum"] > 1.0),
        "vol_ok":        bool(last["volatility"] < 6.0),
        "volume_ok":     bool(last["Volume"] > last["vol_sma20"]),
        "adx_ok":        bool(last["adx"] > 25),
        "above_ema50":   bool(last["Close"] > last["ema50"]),
        "above_ema200":  bool(last["Close"] > last["ema200"]),
        "macd_ok":       bool(last["macd_line"] > last["macd_signal"]),
        "above_bb_mid":  bool(last["Close"] > last["bb_mid"]),
        "stoch_ok":      bool((last["stoch_k"] > last["stoch_d"]) and (last["stoch_k"] < 80)),
        "williams_ok":   bool(last["williams_r"] < -20),
        "cci_ok":        bool(last["cci"] > 0),
        "obv_ok":        bool(last["obv"] > last["obv_ema"]),
        "drawdown_ok":   bool(last["pct_from_high"] > -30),
    }

    # 操作建议（不知道当前实际持仓，给出两种情境）
    action_if_flat = _action(is_bull, is_bear, score, min_conf, adx, adx_entry, in_trade=False)
    action_if_long = _action(is_bull, is_bear, score, min_conf, adx, adx_entry, in_trade=True)

    # vol-targeting scale（若适用）
    vt_scale = None
    if p.get("vol_target", False):
        vt_target = float(df["vol_volatility"].dropna().median())
        entry_rvol = float(last.get("vol_volatility", vt_target))
        if entry_rvol > 0:
            vt_scale = float(np.clip(vt_target / entry_rvol, 0.3, 1.5))

    return {
        "ticker":          ticker,
        "date":            last_date.strftime("%Y-%m-%d"),
        "close":           round(close, 4),
        "state":           state_today,
        "regime":          regime,
        "is_bull":         is_bull,
        "is_bear":         is_bear,
        "signal_score":    score,
        "min_conf":        min_conf,
        "adx":             round(adx, 2),
        "adx_entry":       adx_entry,
        "bull_prob":       round(bull_prob, 4),
        "bear_prob":       round(bear_prob, 4),
        "action_if_flat":  action_if_flat,
        "action_if_long":  action_if_long,
        "stop_pct":        stop_pct,
        "vt_scale":        round(vt_scale, 4) if vt_scale else None,
        "sideways_score":  int(last["sideways_score"]),
        "signal_details":  signal_details,
        "posterior":       [round(p, 4) for p in posterior],
    }


def generate_signal_growth(ticker: str) -> Dict:
    """Signal generator for growth/trend stocks using their optimal EMA strategy."""
    from strategy_growth import GROWTH_STRATEGY, STRATEGY_LABELS
    from backtester import _ema

    df  = fetch_data(ticker, force_refresh=True)
    c   = df["Close"]
    strat = GROWTH_STRATEGY.get(ticker)

    # Build EMA-based signal matching the strategy
    if strat == "ema200_rsi3065":
        from backtester import _ema as _ema2
        e200  = _ema2(c, 200)
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        r14   = 100 - 100/(1+gain/loss.replace(0, float('nan')))
        in_signal = bool((c.iloc[-2] > e200.iloc[-2]) and (r14.iloc[-2] > 30) and (r14.iloc[-2] < 65))
        label = f"价格>EMA200 且 RSI 30-65  (EMA200=${e200.iloc[-1]:.2f} RSI={r14.iloc[-1]:.1f})"
    elif strat in ("ema200",):
        e_fast = _ema(c, 200); e_slow = None
        in_signal = bool(c.iloc[-2] > e_fast.iloc[-2])
        label = f"Price > EMA200  (EMA200=${e_fast.iloc[-1]:.2f})"
    elif strat in ("ema21_50", "ema21_50_vol", "ema21_50_vol20"):
        e_fast = _ema(c, 21); e_slow = _ema(c, 50)
        in_signal = bool(e_fast.iloc[-2] > e_slow.iloc[-2])
        if strat == "ema21_50_vol20":
            vol_ma = df["Volume"].rolling(20).mean()
            vol_ok = bool(df["Volume"].iloc[-2] > vol_ma.iloc[-2])
            in_signal = in_signal and vol_ok
            label = f"EMA21>EMA50 + 量>{vol_ma.iloc[-1]:.0f}  (EMA21=${e_fast.iloc[-1]:.2f} EMA50=${e_slow.iloc[-1]:.2f})"
        else:
            label = f"EMA21 > EMA50  (EMA21=${e_fast.iloc[-1]:.2f} EMA50=${e_slow.iloc[-1]:.2f})"
    elif strat == "ema50_200":
        e_fast = _ema(c, 50); e_slow = _ema(c, 200)
        in_signal = bool(e_fast.iloc[-2] > e_slow.iloc[-2])
        label = f"EMA50 > EMA200  (EMA50=${e_fast.iloc[-1]:.2f} EMA200=${e_slow.iloc[-1]:.2f})"
    elif strat == "ema7_21":
        e_fast = _ema(c, 7); e_slow = _ema(c, 21)
        in_signal = bool(e_fast.iloc[-2] > e_slow.iloc[-2])
        label = f"EMA7 > EMA21  (EMA7=${e_fast.iloc[-1]:.2f} EMA21=${e_slow.iloc[-1]:.2f})"
    elif strat == "ema10_30":
        e_fast = _ema(c, 10); e_slow = _ema(c, 30)
        in_signal = bool(e_fast.iloc[-2] > e_slow.iloc[-2])
        label = f"EMA10 > EMA30  (EMA10=${e_fast.iloc[-1]:.2f} EMA30=${e_slow.iloc[-1]:.2f})"
    elif strat in ("52wh80", "52wh75", "52wh70"):
        pct = 0.80 if strat == "52wh80" else (0.75 if strat == "52wh75" else 0.70)
        h52 = c.rolling(252, min_periods=50).max()
        in_signal = bool(c.iloc[-2] > h52.iloc[-2] * pct)
        label = f"Price > 52wHigh×{pct:.0%}  (52wH=${h52.iloc[-1]:.2f})"
    elif strat == "buyhold":
        in_signal = True
        label = "Buy & Hold"
    else:
        in_signal = True
        label = STRATEGY_LABELS.get(strat, strat)

    action_if_flat = "ENTER" if in_signal else "STAY_OUT"
    action_if_long = "HOLD"  if in_signal else "EXIT"

    # ── 基本面过滤 ────────────────────────────────────────────────
    fa = fetch_fundamentals(ticker)
    # 财报窗口期：强制降级为 WATCH
    if fa.get("earnings_warning") and action_if_flat == "ENTER":
        action_if_flat = "WATCH"

    # ── 14 信号明细（与 HMM 资产保持一致） ──────────────────────
    try:
        from backtester import compute_indicators
        _dfi = compute_indicators(df.copy(), ticker)
        _last = _dfi.iloc[-1]
        _delta = c.diff()
        _gain  = _delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        _loss  = (-_delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        _rsi14 = (100 - 100 / (1 + _gain / _loss.replace(0, float('nan')))).iloc[-1]
        _high_max = c.rolling(252, min_periods=50).max().iloc[-1]
        signal_details = {
            "rsi_ok":       bool(_rsi14 < 90),
            "momentum_ok":  bool(_last.get("momentum", 0) > 1.0),
            "vol_ok":       bool(_last.get("volatility", 999) < 6.0),
            "volume_ok":    bool(_last.get("Volume", 0) > _last.get("vol_sma20", float("inf"))),
            "adx_ok":       bool(_last.get("adx", 0) > 25),
            "above_ema50":  bool(c.iloc[-1] > _last.get("ema50", float("inf"))),
            "above_ema200": bool(c.iloc[-1] > _last.get("ema200", float("inf"))),
            "macd_ok":      bool(_last.get("macd_line", 0) > _last.get("macd_signal", 0)),
            "above_bb_mid": bool(c.iloc[-1] > _last.get("bb_mid", float("inf"))),
            "stoch_ok":     bool((_last.get("stoch_k", 0) > _last.get("stoch_d", 0)) and (_last.get("stoch_k", 0) < 80)),
            "williams_ok":  bool(_last.get("williams_r", 0) < -20),
            "cci_ok":       bool(_last.get("cci", 0) > 0),
            "obv_ok":       bool(_last.get("obv", 0) > _last.get("obv_ema", float("inf"))),
            "drawdown_ok":  bool(_high_max > 0 and (c.iloc[-1] / _high_max - 1) * 100 > -30),
        }
    except Exception:
        signal_details = {}

    return {
        "ticker":         ticker,
        "date":           df.index[-1].strftime("%Y-%m-%d"),
        "close":          round(float(c.iloc[-1]), 4),
        "regime":         label,
        "is_bull":        in_signal,
        "is_bear":        not in_signal,
        "signal_score":   1 if in_signal else 0,
        "min_conf":       1,
        "adx":            0.0,
        "adx_entry":      0,
        "bull_prob":      1.0 if in_signal else 0.0,
        "bear_prob":      0.0 if in_signal else 1.0,
        "action_if_flat": action_if_flat,
        "action_if_long": action_if_long,
        "stop_pct":       -0.20,
        "vt_scale":       None,
        "sideways_score": 0,
        "signal_details":   signal_details,
        "posterior":        [],
        "growth_strategy":  STRATEGY_LABELS.get(strat, strat),
        "fundamentals":     fa,
    }


# ── 主流程 ────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today     = datetime.now().strftime("%Y%m%d")
    out_path  = os.path.join(OUTPUT_DIR, f"signal_{today}.json")

    from strategy_growth import GROWTH_STRATEGY
    HMM_TICKERS    = ["GC=F","SI=F","CL=F","SPY","FXI"]
    GROWTH_TICKERS = ["AAPL","NVDA","META","AMZN","GOOG","MSFT","TSLA","HOOD","PLTR","SOXL","MU","MRVL","AMD","ORCL"]
    tickers = HMM_TICKERS + GROWTH_TICKERS
    prefetch_recent(tickers + ["VIXY", "DXUSD"])
    signals = {}
    errors  = {}

    for ticker in tickers:
        try:
            if ticker in GROWTH_STRATEGY:
                signals[ticker] = generate_signal_growth(ticker)
            else:
                signals[ticker] = generate_signal(ticker)
        except Exception as e:
            errors[ticker] = str(e)

    swing = None
    try:
        from strategy_swing import run_swing, SWING_TICKERS
        res_s = run_swing({t: _fetch_prices(t) for t in SWING_TICKERS})
        swing = {**res_s["current"], "metrics": res_s["metrics"]}
    except Exception as e:
        errors["SWING"] = str(e)

    output = {
        "generated_at": datetime.now().isoformat(),
        "signals":      signals,
        "swing":        swing,
        "errors":       errors,
    }

    # ── 文件输出 ─────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Terminal 输出 ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  HMM Daily Signal Report  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")

    for ticker, sig in signals.items():
        action_flat = sig["action_if_flat"]
        action_long = sig["action_if_long"]
        flag_flat   = "🟢" if action_flat == "ENTER" else ("🔴" if action_flat in ("EXIT","STAY_OUT") else "🟡")
        flag_long   = "🟢" if action_long == "HOLD"  else "🔴"

        print(f"\n  {ticker:6s}  {sig['date']}  Close={sig['close']:.4f}")
        if "state" in sig:
            print(f"  Regime: {sig['regime']:15s}  State={sig['state']}  "
                  f"bull_prob={sig['bull_prob']:.1%}  bear_prob={sig['bear_prob']:.1%}")
            print(f"  Score:  {sig['signal_score']}/14  (min={sig['min_conf']})  "
                  f"ADX={sig['adx']:.1f}  (gate={sig['adx_entry']})  "
                  f"Sideways={sig['sideways_score']}")
            if sig["vt_scale"] is not None:
                print(f"  Vol-target scale: {sig['vt_scale']:.3f}")
        else:
            print(f"  Strategy: {sig.get('growth_strategy', sig['regime'])}")
            print(f"  Signal: {'✅ IN' if sig['is_bull'] else '❌ OUT'}")
        print(f"  Action if FLAT : {flag_flat} {action_flat}")
        print(f"  Action if LONG : {flag_long} {action_long}")

    if swing:
        print(f"\n  波段（连跌3天）空余槽位 {swing['open_slots']}/5")
        for t, v in swing["signals"].items():
            if v["action"] != "STAY_OUT":
                print(f"    {t:6s} {v['action']:8s} 连跌{v['ndown']}天  距SMA200 {v['dist_sma200_pct']:+.1f}%  {v['note'] or ''}")

    if errors:
        print(f"\n  ERRORS: {errors}")

    print(f"\n{'─'*65}")
    print(f"  Output written → {out_path}")
    print(f"{'='*65}\n")

    return output


if __name__ == "__main__":
    run()
