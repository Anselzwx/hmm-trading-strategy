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
from data_loader import fetch_data, DAILY_TICKERS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "signals")


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

    # walk-forward states（与回测一致）
    wf_states = _walk_forward_states(features, n, n_states)

    # 全样本 fit 用于 posterior 概率、state 标签、今日 is_bull/is_bear
    full_model              = fit_hmm(features, n_states)
    bull_states, bear_state = identify_states(full_model, bull_top)
    regime_label_map        = _regime_labels_by_rank(full_model, n_states)

    # 用 full_model predict 对齐 state 语义（wf_states 编号与 full_model 不保证对齐）
    full_states = full_model.predict(features)

    df = df.copy()
    df["state"]   = wf_states          # walk-forward states（与回测一致）
    df["is_bull"] = pd.Series(full_states, index=df.index).isin(bull_states)
    df["is_bear"] = pd.Series(full_states, index=df.index) == bear_state

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


# ── 主流程 ────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today     = datetime.now().strftime("%Y%m%d")
    out_path  = os.path.join(OUTPUT_DIR, f"signal_{today}.json")

    tickers = ["GC=F", "SI=F", "AAPL"]
    signals = {}
    errors  = {}

    for ticker in tickers:
        try:
            signals[ticker] = generate_signal(ticker)
        except Exception as e:
            errors[ticker] = str(e)

    output = {
        "generated_at": datetime.now().isoformat(),
        "signals":      signals,
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
        print(f"  Regime: {sig['regime']:15s}  State={sig['state']}  "
              f"bull_prob={sig['bull_prob']:.1%}  bear_prob={sig['bear_prob']:.1%}")
        print(f"  Score:  {sig['signal_score']}/{14}  (min={sig['min_conf']})  "
              f"ADX={sig['adx']:.1f}  (gate={sig['adx_entry']})  "
              f"Sideways={sig['sideways_score']}")
        if sig["vt_scale"] is not None:
            print(f"  Vol-target scale: {sig['vt_scale']:.3f}")
        print(f"  Action if FLAT : {flag_flat} {action_flat}")
        print(f"  Action if LONG : {flag_long} {action_long}")

    if errors:
        print(f"\n  ERRORS: {errors}")

    print(f"\n{'─'*65}")
    print(f"  Output written → {out_path}")
    print(f"{'='*65}\n")

    return output


if __name__ == "__main__":
    run()
