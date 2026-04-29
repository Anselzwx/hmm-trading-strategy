"""
strategy_d.py — HMM + 布林带突破
价格突破布林带上轨 + HMM 牛态 → 入场
价格跌破布林带中轨 → 出场
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from backtester import (STARTING_CAP, FRICTION_PCT, LEVERAGE, TICKER_PARAMS,
                        MARGIN_PARAMS, _ema, fit_hmm, identify_states,
                        _walk_forward_states, BEAR_CONFIRM)
from data_loader import get_hmm_features

def run_strategy_d(df: pd.DataFrame, ticker: str) -> Dict:
    p        = TICKER_PARAMS.get(ticker, {"n_states":5,"bull_top":2,"stop":-0.08})
    n_states = p.get("n_states", 5)
    bull_top = p.get("bull_top", 2)
    stop     = p.get("stop", -0.08)

    features  = get_hmm_features(df)
    wf_states = _walk_forward_states(features, len(features), n_states)
    full_model              = fit_hmm(features, n_states)
    bull_states, bear_state = identify_states(full_model, bull_top)
    full_states             = full_model.predict(features)

    df = df.copy()
    df["is_bull"] = pd.Series(full_states, index=df.index).isin(bull_states)

    # 布林带
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = mid + 2*std
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - 2*std

    capital     = float(STARTING_CAP)
    position    = 0.0
    in_trade    = False
    entry_price = 0.0
    entry_time  = None
    hold_bars   = 0
    equity_curve = []
    trades = []

    for ts, row in df.iterrows():
        price = float(row["Close"])
        is_bull = bool(row.get("is_bull", False))

        if in_trade:
            hold_bars += 1
            ret = (price - entry_price) / entry_price
            exit_reason = None
            if ret <= stop:
                exit_reason = "StopLoss"
            elif price < float(row["bb_mid"]):
                exit_reason = "BB中轨下穿"
            elif not is_bull:
                exit_reason = "HMM→Bear"
            if exit_reason:
                pnl = (price*(1-FRICTION_PCT) - entry_price)*position*LEVERAGE
                capital += pnl
                trades.append({"entry_time": entry_time, "exit_time": ts,
                                "entry_price": entry_price, "exit_price": price,
                                "pnl": pnl, "hold_bars": hold_bars,
                                "pos_size_pct": 0.8, "exit_reason": exit_reason,
                                "return_pct": ret*100*LEVERAGE})
                position = 0.0; in_trade = False; hold_bars = 0

        if not in_trade and is_bull and price > float(row.get("bb_upper", float("inf"))):
            position    = capital * 0.8 / price
            entry_price = price * (1 + FRICTION_PCT)
            entry_time  = ts; in_trade = True; hold_bars = 0

        mtm = capital + (price - entry_price)*position*LEVERAGE if in_trade else capital
        equity_curve.append(mtm)

    eq = pd.Series(equity_curve, index=df.index)
    total_ret = (eq.iloc[-1]/STARTING_CAP - 1)*100
    pct_ret   = eq.pct_change().dropna()
    ann_ret   = ((eq.iloc[-1]/STARTING_CAP)**(252/len(eq)) - 1)*100
    ann_vol   = pct_ret.std()*np.sqrt(252)*100
    sharpe    = ann_ret/ann_vol if ann_vol > 0 else 0
    dd        = ((eq - eq.cummax())/eq.cummax()*100).min()
    wins      = [t for t in trades if t["pnl"]>0]
    wr        = len(wins)/len(trades)*100 if trades else 0

    return {"equity": eq, "trades": trades,
            "metrics": {"total_return_pct": total_ret, "sharpe": sharpe,
                        "max_drawdown_pct": dd, "win_rate_pct": wr,
                        "ann_return_pct": ann_ret}}
