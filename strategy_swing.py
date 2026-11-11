"""
波段策略：连跌3天买回调（2-5天持仓，多头）

规则
  入场：Close > SMA200 且连续 3 天收跌 → 次日开盘买入
  离场：Close > 前一日 High → 次日开盘卖出；或持仓满 5 天 → 次日开盘卖出
  仓位：5 槽位，每槽 20% 资金，信号先到先得
  成本：单边 5bp
研究记录（2010-2026，16 标的，逐日盯市）：CAGR 16.4%  Sharpe 1.19  MaxDD -19.9%
  分段 2010-15 / 16-20 / 21-23 / 24-26 Sharpe 1.31 / 1.19 / 1.26 / 1.05
"""
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd

SWING_TICKERS = ["AAPL", "NVDA", "META", "AMZN", "GOOG", "MSFT", "TSLA", "HOOD",
                 "PLTR", "SOXL", "MU", "MRVL", "AMD", "ORCL", "SPY", "FXI"]
N_DOWN   = 3
MAX_HOLD = 5
SLOTS    = 5
COST     = 0.0005
START    = "2010-01-01"
SEGMENTS = [("2010", "2015"), ("2016", "2020"), ("2021", "2023"), ("2024", "2026")]


def completed_bars(df: pd.DataFrame) -> pd.DataFrame:
    """去掉美东当日未收盘的K线"""
    now_ny = pd.Timestamp.now(tz="America/New_York")
    last = df.index[-1]
    if last.date() == now_ny.date() and now_ny.hour < 16:
        return df.iloc[:-1]
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Open", "High", "Low", "Close"]].dropna().copy()
    c = d["Close"]
    d["sma200"] = c.rolling(200).mean()
    down = (c.diff() < 0).astype(int)
    d["ndown"] = down.groupby((down == 0).cumsum()).cumsum()
    d["entry_sig"] = (d["ndown"] >= N_DOWN) & (c > d["sma200"])
    d["exit_sig"]  = c > d["High"].shift(1)
    return d.dropna(subset=["sma200"])


def _trades(d: pd.DataFrame, ticker: str) -> list:
    sig, ext = d["entry_sig"].values, d["exit_sig"].values
    o, n = d["Open"].values, len(d)
    out, i = [], 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        j, reason = e, "持仓满5天"
        while True:
            if ext[j]:
                reason = "收盘>前高"
                break
            if j - e + 1 >= MAX_HOLD:
                break
            if j >= n - 1:
                reason = "持仓中"
                break
            j += 1
        xi = j + 1
        if xi >= n:          # 离场信号已出，次日开盘待执行
            out.append(dict(ticker=ticker, entry_date=d.index[e], entry_price=o[e], exit_date=None,
                            exit_price=None, ret=None, days=n - e,
                            exit_reason="持仓中" if reason == "持仓中" else f"明日开盘卖（{reason}）"))
            break
        out.append(dict(ticker=ticker, entry_date=d.index[e], entry_price=o[e], exit_date=d.index[xi],
                        exit_price=o[xi], ret=o[xi] / o[e] - 1 - 2 * COST, days=xi - e, exit_reason=reason))
        i = xi
    return out


def _metrics(pnl: pd.Series, trades: list) -> Dict:
    eq = (1 + pnl).cumprod()
    r = np.array([t["ret"] for t in trades if t["ret"] is not None])
    yrs = len(pnl) / 252
    sd = pnl.std()
    return dict(
        n_trades=len(r),
        win_rate=float((r > 0).mean() * 100) if len(r) else 0.0,
        avg_trade_pct=float(r.mean() * 100) if len(r) else 0.0,
        total_return_pct=float((eq.iloc[-1] - 1) * 100),
        cagr_pct=float((eq.iloc[-1] ** (1 / yrs) - 1) * 100) if yrs > 0 else 0.0,
        sharpe=float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        max_drawdown_pct=float(((eq / eq.cummax()) - 1).min() * 100),
        exposure_pct=float((pnl != 0).mean() * 100),
    )


def run_swing(dfs: Dict[str, pd.DataFrame]) -> Dict:
    """dfs: ticker -> OHLCV（日线）。返回回测结果 + 当前信号"""
    data = {t: add_indicators(completed_bars(df)) for t, df in dfs.items() if df is not None and len(df) > 250}
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    idx = idx[idx >= START]

    cand = sorted((t for tk, d in data.items() for t in _trades(d, tk)),
                  key=lambda x: (x["entry_date"], x["ticker"]))
    taken, busy = [], []
    for t in cand:
        busy = [b for b in busy if b is None or b > t["entry_date"]]
        if len(busy) < SLOTS:
            busy.append(t["exit_date"])
            taken.append(t)

    pnl = pd.Series(0.0, index=idx)
    for t in taken:
        o = data[t["ticker"]]["Open"]
        seg = o[t["entry_date"]:t["exit_date"]] if t["exit_date"] is not None else o[t["entry_date"]:]
        r = (seg.shift(-1) / seg - 1).dropna()
        if t["exit_date"] is None:   # 未平仓：最后一天按收盘盯市
            last = data[t["ticker"]].index[-1]
            r[last] = data[t["ticker"]]["Close"].iloc[-1] / seg.iloc[-1] - 1
        if len(r) == 0:
            continue
        r.iloc[0] -= COST
        if t["exit_date"] is not None:
            r.iloc[-1] -= COST
        common = r.index.intersection(idx)
        pnl[common] += r[common] / SLOTS
    taken = [t for t in taken if t["entry_date"] >= pd.Timestamp(START)]

    segments = []
    for a, b in SEGMENTS:
        p = pnl[a:b]
        if len(p) < 20:
            continue
        m = _metrics(p, [t for t in taken if a <= str(t["entry_date"].year) <= b])
        segments.append(dict(period=f"{a}–{b}", **m))

    return dict(
        equity=(1 + pnl).cumprod(),
        trades=taken,
        metrics=_metrics(pnl, taken),
        segments=segments,
        current=current_state(data, taken),
        computed_at=datetime.now().isoformat(),
    )


def current_state(data: Dict[str, pd.DataFrame], taken: list) -> Dict:
    open_pos = {t["ticker"]: t for t in taken if t["exit_date"] is None}
    n_open = len(open_pos)
    out = {}
    for tk, d in data.items():
        r = d.iloc[-1]
        pos = open_pos.get(tk)
        if pos:
            if pos["exit_reason"].startswith("明日"):
                action = "EXIT"
            else:
                action = "HOLD"
        elif r["entry_sig"]:
            action = "ENTER" if n_open < SLOTS else "WATCH"
        elif r["Close"] > r["sma200"] and r["ndown"] == N_DOWN - 1:
            action = "WATCH"
        else:
            action = "STAY_OUT"
        out[tk] = dict(
            date=str(d.index[-1].date()),
            close=float(r["Close"]),
            ndown=int(r["ndown"]),
            dist_sma200_pct=float((r["Close"] / r["sma200"] - 1) * 100),
            above_sma200=bool(r["Close"] > r["sma200"]),
            action=action,
            exit_trigger=float(r["High"]),      # 持仓时：收盘 > 今日最高价 → 次日开盘卖
            entry_date=str(pos["entry_date"].date()) if pos else None,
            entry_price=float(pos["entry_price"]) if pos else None,
            days_held=int(pos["days"]) if pos else None,
            pos_ret_pct=float((r["Close"] / pos["entry_price"] - 1) * 100) if pos else None,
            note=pos["exit_reason"] if pos else None,
        )
    return dict(signals=out, open_slots=SLOTS - n_open)
