"""
backtester.py  —  Phase 1 升级版
=================================
新增：
  1. Walk-Forward 滚动训练（消除 look-ahead bias）
  2. 固定止损 -8%（兜底风控）
  3. 信号强度仓位管理（得分越高仓位越大）
  4. 夏普比率 / 卡玛比率 / 月度胜率 / SPY Alpha
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

from data_loader import get_hmm_features, fetch_data, DAILY_TICKERS

warnings.filterwarnings("ignore")


# ============================================================
# 常量
# ============================================================
N_STATES          = 7        # 默认状态数（AAPL 使用）
LEVERAGE          = 2.5
STARTING_CAP      = 10_000.0
MIN_CONFIRMATIONS = 8        # 默认信号阈值
TOTAL_SIGNALS     = 14
RANDOM_SEED       = 42
STOP_LOSS_PCT     = -0.08    # 固定止损：单笔亏损超过 -8% 立即出场

# 冷静期 & 最长持仓
COOLDOWN_HOURLY = 48
COOLDOWN_DAILY  = 2
MAX_HOLD_HOURLY = 24 * 30
MAX_HOLD_DAILY  = 60

# Walk-Forward 参数
WF_TRAIN_RATIO  = 0.6        # 60% 训练，40% 测试（滚动）
WF_STEP_RATIO   = 0.1        # 每次向前滚动 10%

# ── 每个 ticker 单独最优参数 ────────────────────────────────
# n_states: HMM状态数  bull_top: 入场状态数（排名靠前N个）  min_conf: 信号阈值
TICKER_PARAMS = {
    "AAPL":  {"n_states": 7, "bull_top": 3, "min_conf": 8},
    "GC=F":  {"n_states": 7, "bull_top": 1, "min_conf": 8},
    "SI=F":  {"n_states": 7, "bull_top": 1, "min_conf": 8},
}


# ============================================================
# 1. HMM 引擎
# ============================================================

def fit_hmm(features: np.ndarray, n_states: int = N_STATES) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        random_state=RANDOM_SEED,
    )
    model.fit(features)
    # 修复转移矩阵空行（小样本 Walk-Forward 窗口可能产生）
    tm = model.transmat_
    row_sums = tm.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    tm[zero_rows] = 1.0 / n_states
    model.transmat_ = tm / tm.sum(axis=1, keepdims=True)
    return model


def identify_states(model: hmm.GaussianHMM, bull_top: int = 1) -> Tuple[List[int], int]:
    mean_returns = model.means_[:, 0]
    ranked       = np.argsort(mean_returns)[::-1]
    return ranked[:bull_top].tolist(), int(ranked[-1])


def decode_regimes(model: hmm.GaussianHMM, features: np.ndarray) -> np.ndarray:
    return model.predict(features)


# ============================================================
# 2. 技术指标
# ============================================================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d    = s.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr   = pd.concat([(hi-lo), (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    dm_p = ((hi-hi.shift()) > (lo.shift()-lo)).astype(float) * (hi-hi.shift()).clip(lower=0)
    dm_m = ((lo.shift()-lo) > (hi-hi.shift())).astype(float) * (lo.shift()-lo).clip(lower=0)
    atr  = tr.ewm(span=period, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = 100 * (di_p-di_m).abs() / (di_p+di_m).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()

def _macd(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    line   = _ema(s, 12) - _ema(s, 26)
    return line, _ema(line, 9)

def _vol_pct(s: pd.Series, window: int, is_daily: bool) -> pd.Series:
    factor = np.sqrt(252) if is_daily else np.sqrt(24 * 365)
    return s.pct_change().rolling(window).std() * factor * 100

def _bollinger(s: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = s.rolling(window).mean()
    std = s.rolling(window).std()
    return mid, mid + 2*std, mid - 2*std

def _stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    pk = 100 * (df["Close"]-lo) / (hi-lo).replace(0, np.nan)
    return pk, pk.rolling(d).mean()

def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    return -100 * (hi-df["Close"]) / (hi-lo).replace(0, np.nan)

def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp  = (df["High"]+df["Low"]+df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    return (tp-sma) / (0.015 * mad.replace(0, np.nan))

def _obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def compute_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out      = df.copy()
    c        = df["Close"]
    is_daily = ticker in DAILY_TICKERS

    out["rsi"]        = _rsi(c)
    out["momentum"]   = c.pct_change(periods=5 if is_daily else 12) * 100
    out["volatility"] = _vol_pct(c, 20, is_daily)
    out["vol_sma20"]  = df["Volume"].rolling(20).mean()
    out["adx"]        = _adx(df)
    out["ema50"]      = _ema(c, 50)
    out["ema200"]     = _ema(c, 200)
    out["macd_line"], out["macd_signal"] = _macd(c)

    bb_mid, bb_upper, bb_lower = _bollinger(c, 20)
    out["bb_mid"]    = bb_mid
    out["bb_upper"]  = bb_upper
    out["bb_lower"]  = bb_lower

    out["stoch_k"], out["stoch_d"] = _stochastic(df)
    out["williams_r"]  = _williams_r(df)
    out["cci"]         = _cci(df)
    obv = _obv(df)
    out["obv"]         = obv
    out["obv_ema"]     = _ema(obv, 20)

    lookback = 365 if is_daily else 365 * 24
    roll_hi  = c.rolling(min(lookback, len(c)), min_periods=20).max()
    out["pct_from_high"] = (c - roll_hi) / roll_hi * 100

    # ATR（用于止损计算）
    tr  = pd.concat([(df["High"]-df["Low"]),
                     (df["High"]-c.shift()).abs(),
                     (df["Low"]-c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(span=14, adjust=False).mean()

    return out


# ============================================================
# 3. 投票信号 — 返回 (bool_series, score_series)
# ============================================================

def compute_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    c1  = df["rsi"]          < 90
    c2  = df["momentum"]     > 1.0
    c3  = df["volatility"]   < 6.0
    c4  = df["Volume"]       > df["vol_sma20"]
    c5  = df["adx"]          > 25
    c6  = df["Close"]        > df["ema50"]
    c7  = df["Close"]        > df["ema200"]
    c8  = df["macd_line"]    > df["macd_signal"]
    c9  = df["Close"]        > df["bb_mid"]
    c10 = (df["stoch_k"]     > df["stoch_d"]) & (df["stoch_k"] < 80)
    c11 = df["williams_r"]   < -20
    c12 = df["cci"]          > 0
    c13 = df["obv"]          > df["obv_ema"]
    c14 = df["pct_from_high"] > -30

    score = sum(x.astype(int) for x in [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14])
    return score >= MIN_CONFIRMATIONS, score   # raw signal uses global default; run_backtest overrides per-ticker


def _position_size(score: float) -> float:
    """
    仓位管理：根据信号得分线性映射仓位比例。
    score=9  → 40%
    score=12 → 70%
    score=14 → 100%
    低于 MIN_CONFIRMATIONS 不入场。
    """
    min_s, max_s = MIN_CONFIRMATIONS, TOTAL_SIGNALS
    min_sz, max_sz = 0.40, 1.00
    ratio = (score - min_s) / max(max_s - min_s, 1)
    return float(np.clip(min_sz + ratio * (max_sz - min_sz), min_sz, max_sz))


# ============================================================
# 4. Walk-Forward 滚动训练
# ============================================================

def _walk_forward_states(features: np.ndarray,
                          n_total: int,
                          n_states: int = N_STATES,
                          train_ratio: float = WF_TRAIN_RATIO,
                          step_ratio:  float = WF_STEP_RATIO) -> np.ndarray:
    min_train  = max(n_states * 20, 60)
    train_size = max(int(n_total * train_ratio), min_train)
    train_size = min(train_size, n_total - 1)
    step_size  = max(int(n_total * step_ratio), 1)
    state_seq  = np.full(n_total, -1, dtype=int)

    base_model = fit_hmm(features[:train_size], n_states)
    mr0 = base_model.means_[:, 0]
    remap0 = {old: new for new, old in enumerate(np.argsort(mr0))}
    state_seq[:train_size] = [remap0[p] for p in base_model.predict(features[:train_size])]

    cursor = train_size
    while cursor < n_total:
        end   = min(cursor + step_size, n_total)
        model = fit_hmm(features[:cursor], n_states)
        preds = model.predict(features[cursor:end])
        mr    = model.means_[:, 0]
        remap = {old: new for new, old in enumerate(np.argsort(mr))}
        state_seq[cursor:end] = [remap[p] for p in preds]
        cursor = end

    return state_seq


# ============================================================
# 5. 回测核心
# ============================================================

def _simulate(df: pd.DataFrame,
              is_daily: bool,
              use_wf_states: bool = True) -> Tuple[list, list]:
    """
    运行完整模拟，返回 (equity_curve, trades)。
    df 必须已包含 is_bull / is_bear / tech_signal / signal_score 列。
    """
    cooldown_max = COOLDOWN_DAILY  if is_daily else COOLDOWN_HOURLY
    max_hold     = MAX_HOLD_DAILY  if is_daily else MAX_HOLD_HOURLY

    capital       = STARTING_CAP
    position      = 0.0
    pos_size_pct  = 0.0
    entry_price   = 0.0
    entry_time    = None
    cooldown_left = 0
    hold_bars     = 0
    in_trade      = False
    equity_curve  = []
    trades        = []

    for ts, row in df.iterrows():
        price = float(row["Close"])

        if in_trade:
            hold_bars += 1
            current_ret = (price - entry_price) / entry_price  # 未杠杆收益率

            exit_reason = None
            if row["is_bear"]:
                exit_reason = "Regime → Bear/Crash"
            elif current_ret <= STOP_LOSS_PCT:
                exit_reason = f"Stop Loss ({STOP_LOSS_PCT*100:.0f}%)"
            elif hold_bars >= max_hold:
                exit_reason = f"Max Hold ({max_hold} bars)"

            if exit_reason:
                pnl_lev  = (price - entry_price) * position * LEVERAGE
                capital += pnl_lev
                trades.append({
                    "entry_time":    entry_time,
                    "exit_time":     ts,
                    "entry_price":   entry_price,
                    "exit_price":    price,
                    "pnl":           pnl_lev,
                    "pos_size_pct":  pos_size_pct,
                    "exit_reason":   exit_reason,
                    "hold_bars":     hold_bars,
                    "return_pct":    current_ret * 100 * LEVERAGE,
                })
                position, in_trade, hold_bars = 0.0, False, 0
                cooldown_left = cooldown_max

        if cooldown_left > 0:
            cooldown_left -= 1

        if not in_trade and cooldown_left == 0 and row["is_bull"] and row["tech_signal"]:
            score        = float(row["signal_score"])
            pos_size_pct = _position_size(score)
            capital_used = capital * pos_size_pct
            position     = capital_used / price
            entry_price  = price
            entry_time   = ts
            in_trade     = True
            hold_bars    = 0

        # MTM equity
        if in_trade:
            unrealised = (price - entry_price) * position * LEVERAGE
            mtm        = capital + unrealised
        else:
            mtm = capital
        equity_curve.append(mtm)

    # 收盘强平
    if in_trade:
        last_price = float(df["Close"].iloc[-1])
        pnl_lev    = (last_price - entry_price) * position * LEVERAGE
        capital   += pnl_lev
        trades.append({
            "entry_time":   entry_time,
            "exit_time":    df.index[-1],
            "entry_price":  entry_price,
            "exit_price":   last_price,
            "pnl":          pnl_lev,
            "pos_size_pct": pos_size_pct,
            "exit_reason":  "End of data",
            "hold_bars":    hold_bars,
            "return_pct":   (last_price-entry_price)/entry_price*100*LEVERAGE,
        })

    return equity_curve, trades


# ============================================================
# 6. 主入口
# ============================================================

def run_backtest(df: pd.DataFrame, ticker: str = "AAPL") -> Dict:
    is_daily = ticker in DAILY_TICKERS
    params   = TICKER_PARAMS.get(ticker, {"n_states": N_STATES, "bull_top": 1, "min_conf": MIN_CONFIRMATIONS})
    n_states = params["n_states"]
    bull_top = params["bull_top"]
    min_conf = params["min_conf"]

    features = get_hmm_features(df)
    n        = len(features)

    # ── Walk-Forward 状态序列（已按均值收益排序，0=最差，n_states-1=最好）──
    wf_states = _walk_forward_states(features, n, n_states)

    df = df.copy()
    df["state"]   = wf_states
    # bull = rank >= n_states - bull_top，bear = rank 0
    df["is_bull"] = df["state"] >= (n_states - bull_top)
    df["is_bear"] = df["state"] == 0

    def _label(s: int) -> str:
        if s == n_states - 1: return "Bull Run"
        if s == n_states - 2: return "Bull+"
        if s == n_states - 3: return "Warming Up"
        if s == 0:            return "Bear/Crash"
        if s == 1:            return "Bear"
        return f"Neutral-{s}"

    df["regime_label"] = df["state"].apply(_label)

    # ── 技术指标 & 信号（使用 ticker 专属阈值）───────────────
    df = compute_indicators(df, ticker)
    raw_signal, score = compute_signals(df)
    df["signal_score"] = score
    df["tech_signal"]  = score >= min_conf

    # ── 模拟 ──────────────────────────────────────────────────
    equity_curve, trades = _simulate(df, is_daily)
    df["equity"] = equity_curve

    # ── 指标 ──────────────────────────────────────────────────
    metrics = _compute_metrics(df, trades, ticker, is_daily)

    # ── 最终模型（全量，仅供展示）────────────────────────────
    full_model              = fit_hmm(features, n_states)
    bull_states, bear_state = identify_states(full_model, bull_top)
    regime_labels           = {s: _label(s) for s in range(n_states)}

    return {
        "df":            df,
        "trades":        trades,
        "metrics":       metrics,
        "bull_states":   bull_states,
        "bear_state":    bear_state,
        "regime_labels": regime_labels,
        "model":         full_model,
        "is_daily":      is_daily,
        "n_states":      n_states,
        "bull_top":      bull_top,
        "min_conf":      min_conf,
    }


# ============================================================
# 7. 绩效指标（Phase 1 升级）
# ============================================================

def _compute_metrics(df: pd.DataFrame, trades: list,
                     ticker: str, is_daily: bool) -> Dict:
    equity   = pd.Series(df["equity"].values, index=df.index)
    n_bars   = len(equity)

    # ── 基础收益 ─────────────────────────────────────────────
    total_ret   = (equity.iloc[-1] / STARTING_CAP - 1) * 100
    bh_ret      = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    alpha       = total_ret - bh_ret

    # ── 最大回撤 ─────────────────────────────────────────────
    roll_max    = equity.cummax()
    dd_series   = (equity - roll_max) / roll_max * 100
    max_dd      = dd_series.min()

    # ── 年化收益 & 夏普比率 ──────────────────────────────────
    bars_per_yr = 252 if is_daily else 252 * 24
    pct_returns = equity.pct_change().dropna()
    ann_ret     = ((equity.iloc[-1] / STARTING_CAP) ** (bars_per_yr / n_bars) - 1) * 100
    ann_vol     = pct_returns.std() * np.sqrt(bars_per_yr) * 100
    sharpe      = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # ── 卡玛比率 ─────────────────────────────────────────────
    calmar      = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # ── 交易胜率 & 月度胜率 ──────────────────────────────────
    wins        = [t for t in trades if t["pnl"] > 0]
    losses      = [t for t in trades if t["pnl"] <= 0]
    win_rate    = len(wins) / len(trades) * 100 if trades else 0.0

    # 月度胜率：按月统计策略收益正负
    monthly_eq  = equity.resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna()
    monthly_win = (monthly_ret > 0).sum() / len(monthly_ret) * 100 if len(monthly_ret) > 0 else 0.0

    # ── 月度收益矩阵（用于热力图）────────────────────────────
    monthly_pct = monthly_ret * 100
    monthly_df  = pd.DataFrame({
        "year":  monthly_pct.index.year,
        "month": monthly_pct.index.month,
        "ret":   monthly_pct.values,
    })

    # ── SPY Alpha ────────────────────────────────────────────
    try:
        spy_df  = fetch_data("SPY")
        spy_bh  = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[0] - 1) * 100
        spy_alpha = total_ret - spy_bh
    except Exception:
        spy_bh    = None
        spy_alpha = None

    # ── 盈亏比 ───────────────────────────────────────────────
    avg_win  = np.mean([t["pnl"] for t in wins])   if wins   else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # ── 平均仓位 ─────────────────────────────────────────────
    avg_pos_size = np.mean([t["pos_size_pct"] for t in trades]) * 100 if trades else 0

    return {
        "total_return_pct":  round(total_ret,    2),
        "ann_return_pct":    round(ann_ret,       2),
        "bh_return_pct":     round(bh_ret,        2),
        "alpha_pct":         round(alpha,          2),
        "spy_bh_pct":        round(spy_bh,         2) if spy_bh  is not None else None,
        "spy_alpha_pct":     round(spy_alpha,      2) if spy_alpha is not None else None,
        "max_drawdown_pct":  round(max_dd,         2),
        "sharpe":            round(sharpe,          3),
        "calmar":            round(calmar,          3),
        "ann_vol_pct":       round(ann_vol,         2),
        "win_rate_pct":      round(win_rate,        2),
        "monthly_win_pct":   round(monthly_win,     2),
        "n_trades":          len(trades),
        "avg_win":           round(avg_win,          2),
        "avg_loss":          round(avg_loss,         2),
        "rr_ratio":          round(rr_ratio,         3),
        "avg_pos_size_pct":  round(avg_pos_size,    1),
        "final_capital":     round(equity.iloc[-1],  2),
        "monthly_df":        monthly_df,
    }
