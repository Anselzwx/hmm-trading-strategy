"""
strategy_c.py — EMA Trend Following (纯均线趋势跟踪，无 HMM)
EMA50 > EMA200 持有，否则空仓
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from backtester import STARTING_CAP, FRICTION_PCT, LEVERAGE, TICKER_PARAMS, _ema

def run_strategy_c(df: pd.DataFrame, ticker: str) -> Dict:
    p    = TICKER_PARAMS.get(ticker, {"stop": -0.08})
    stop = p.get("stop", -0.08)

    df = df.copy()
    df["ema50"]  = _ema(df["Close"], 50)
    df["ema200"] = _ema(df["Close"], 200)
    df["signal"] = (df["ema50"] > df["ema200"]).astype(int)

    capital   = float(STARTING_CAP)
    position  = 0.0
    in_trade  = False
    entry_price = 0.0
    equity_curve = []
    trades = []
    entry_time = None
    hold_bars  = 0

    for ts, row in df.iterrows():
        price = float(row["Close"])
        sig   = int(row["signal"])

        if in_trade:
            hold_bars += 1
            ret = (price - entry_price) / entry_price
            exit_reason = None
            if ret <= stop:
                exit_reason = f"StopLoss"
            elif sig == 0:
                exit_reason = "EMA死叉"
            if exit_reason:
                pnl = (price*(1-FRICTION_PCT) - entry_price) * position * LEVERAGE
                capital += pnl
                trades.append({"entry_time": entry_time, "exit_time": ts,
                                "entry_price": entry_price, "exit_price": price,
                                "pnl": pnl, "hold_bars": hold_bars,
                                "pos_size_pct": 0.8, "exit_reason": exit_reason,
                                "return_pct": ret*100*LEVERAGE})
                position = 0.0; in_trade = False; hold_bars = 0

        if not in_trade and sig == 1:
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
