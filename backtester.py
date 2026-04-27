"""
backtester.py  —  Phase 1 升级版
=================================
新增：
  1. Walk-Forward 滚动训练（消除 look-ahead bias）
  2. 固定止损 -8%（兜底风控）
  3. 信号强度仓位管理（得分越高仓位越大）
  4. 夏普比率 / 卡玛比率 / 月度胜率 / SPY Alpha
  5. 每个 ticker 独立最优参数
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
# 全局默认常量
# ============================================================
N_STATES          = 7
LEVERAGE          = 1.0
STARTING_CAP      = 10_000.0
MIN_CONFIRMATIONS = 9
TOTAL_SIGNALS     = 14
RANDOM_SEED       = 42
STOP_LOSS_PCT     = -0.08

COOLDOWN_HOURLY = 48
COOLDOWN_DAILY  = 2
MAX_HOLD_HOURLY = 24 * 30
MAX_HOLD_DAILY  = 60

# 摩擦成本（统一百分比模型）
# slippage=0.05% per side + commission=0.05% per side = 0.10% per side, 0.20% round trip
FRICTION_PCT    = 0.001   # 0.10% per side; set to 0.0 to disable

# 保证金参数（分资产）
# initial_margin: 开仓所需保证金率（相对 notional）
# maintenance_margin: 维持保证金率，低于此触发强平
MARGIN_PARAMS: Dict[str, Dict] = {
    "AAPL": {"initial_margin": 0.40, "maintenance_margin": 0.25},
    "GC=F": {"initial_margin": 0.15, "maintenance_margin": 0.10},
    "SI=F": {"initial_margin": 0.15, "maintenance_margin": 0.10},
}

WF_TRAIN_RATIO  = 0.6
WF_STEP_RATIO   = 0.1

# ── 每个 ticker 独立最优参数（经 Walk-Forward 验证）──────────
# n_states: HMM状态数
# bull_top: 入场状态数（rank从高到低前N个）
# min_conf: 信号阈值
# stop:     止损比例
# hold_mult: 最大持仓倍数（相对全局MAX_HOLD）
TICKER_PARAMS: Dict[str, Dict] = {
    # AAPL_A2_final: hold_mult=1.25 (75bars), subsample validated 2026-04-21
    "AAPL": {"n_states": 5, "bull_top": 3, "min_conf": 9,  "stop": -0.06, "hold_mult": 1.25, "adx_entry": 25, "regime_reduce": False},
    # Gold_regime_reduce50_final: Regime→reduce50%, price-type stop, validated 2026-04-21
    "GC=F": {"n_states": 7, "bull_top": 2, "min_conf": 9,  "stop": -0.08, "hold_mult": 1.0,  "adx_entry": 25, "regime_reduce": True},
    # Silver_VT1_final: ADX>30, Regime→reduce50%, vol-targeting (rvol median, clip[0.3,1.5]), validated 2026-04-22
    "SI=F": {"n_states": 7, "bull_top": 1, "min_conf": 9,  "stop": -0.06, "hold_mult": 1.0,  "adx_entry": 30, "regime_reduce": True, "vol_target": True},
}

# Regime exit 连续确认 bars（1=原版，2=Gold 定案）
BEAR_CONFIRM: Dict[str, int] = {
    "AAPL": 1,
    "GC=F": 2,
    "SI=F": 1,
}


# ============================================================
# 1. HMM 引擎
# ============================================================

def fit_hmm(features: np.ndarray, n_states: int = N_STATES, n_trials: int = 5) -> hmm.GaussianHMM:
    best_model, best_score = None, -np.inf
    for seed in range(RANDOM_SEED, RANDOM_SEED + n_trials):
        m = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=seed,
        )
        try:
            m.fit(features)
            score = m.score(features)
            if score > best_score:
                best_score = score
                best_model = m
        except Exception:
            continue
    model = best_model
    tm = model.transmat_
    row_sums = tm.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    tm[zero_rows] = 1.0 / n_states
    model.transmat_ = tm / tm.sum(axis=1, keepdims=True)
    return model


def identify_states(model: hmm.GaussianHMM, bull_top: int = 2) -> Tuple[List[int], int]:
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
    line = _ema(s, 12) - _ema(s, 26)
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

    tr  = pd.concat([(df["High"]-df["Low"]),
                     (df["High"]-c.shift()).abs(),
                     (df["Low"]-c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(span=14, adjust=False).mean()

    ema50 = out["ema50"]
    out["ema50_slope"] = (ema50 - ema50.shift(5)) / ema50.shift(5) * 100

    # ── Sideways Score（0-4，越高越横盘）──────────────────────
    # 1. ADX < 18
    sw1 = (out["adx"] < 18).astype(int)

    # 2. |EMA50 slope| < 0.1%
    sw2 = (out["ema50_slope"].abs() < 0.1).astype(int)

    # 3. BB width 在过去 120 bars 的低 30% 分位
    bb_width = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
    bb_low   = bb_width.rolling(120, min_periods=30).quantile(0.30)
    sw3 = (bb_width <= bb_low).astype(int)

    # 4. Breakout failure：突破20日高点后5bars内收益 <= 0
    roll_high_20 = c.rolling(20).max().shift(1)
    breakout     = (c > roll_high_20)
    fwd_ret_5    = c.shift(-5) / c - 1
    sw4 = (breakout & (fwd_ret_5 <= 0)).astype(int).rolling(10, min_periods=1).max()

    out["sideways_score"] = (sw1 + sw2 + sw3 + sw4).clip(0, 4)

    return out


# ============================================================
# 3. 投票信号
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
    return score >= MIN_CONFIRMATIONS, score


def _position_size(score: float, min_conf: int = MIN_CONFIRMATIONS) -> float:
    min_s, max_s = min_conf, TOTAL_SIGNALS
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
    mr0   = base_model.means_[:, 0]
    rmap0 = {old: new for new, old in enumerate(np.argsort(mr0))}
    state_seq[:train_size] = [rmap0[p] for p in base_model.predict(features[:train_size])]

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

def _sideways_cooldown(base_cd: int, sw_score: int, consec_stops: int) -> int:
    """计算止损后 cooldown 长度，综合横盘程度和连续止损次数。"""
    cd = base_cd
    if consec_stops >= 2:
        cd = base_cd * 2          # 连续止损 → 双倍 cooldown
    if sw_score >= 2:
        cd += 3                   # 横盘环境额外 +3 bars
    return cd


def _simulate(df: pd.DataFrame,
              is_daily: bool,
              stop_loss_pct: float = STOP_LOSS_PCT,
              hold_mult: float = 1.0,
              min_conf: int = MIN_CONFIRMATIONS,
              ticker: str = "") -> Tuple[list, list]:
    base_cd           = COOLDOWN_DAILY if is_daily else COOLDOWN_HOURLY
    max_hold          = int((MAX_HOLD_DAILY if is_daily else MAX_HOLD_HOURLY) * hold_mult)
    bear_confirm      = BEAR_CONFIRM.get(ticker, 1)
    tp                = TICKER_PARAMS.get(ticker, {})
    friction_pct      = FRICTION_PCT
    mp                = MARGIN_PARAMS.get(ticker, {"initial_margin": 0.40, "maintenance_margin": 0.25})
    initial_margin    = mp["initial_margin"]
    maint_margin      = mp["maintenance_margin"]
    use_regime_reduce = tp.get("regime_reduce", False)   # Gold + Silver: reduce path
    adx_entry         = tp.get("adx_entry", 25)
    use_vol_target    = tp.get("vol_target", False)       # Silver: rvol-based position scaling
    VT_MIN, VT_MAX    = 0.3, 1.5

    capital       = STARTING_CAP
    position      = 0.0
    pos_size_pct  = 0.0
    entry_price   = 0.0
    stop_price    = 0.0   # price-type stop: fixed at entry, unaffected by partial reduces
    entry_time    = None
    cooldown_left = 0
    hold_bars     = 0
    in_trade      = False
    equity_curve  = []
    trades        = []

    consec_stops            = 0
    after_stopout           = False
    bear_consec             = 0
    regime_reduce_triggered = False   # Lock A: one Regime Reduce per holding cycle
    realised_from_reduce    = 0.0    # PnL already booked from partial reduces
    reduce_time             = None   # bar when reduce was triggered
    reduce_price            = None   # price at reduce trigger

    # vol-targeting: target = full-sample rvol median (Silver only)
    vt_target = float(df["vol_volatility"].dropna().median()) if use_vol_target else None
    vt_scale  = 1.0

    for ts, row in df.iterrows():
        price    = float(row["Close"])
        sw_score = int(row.get("sideways_score", 0))

        # ── 横盘动态参数 ──────────────────────────────────────
        if sw_score <= 1:
            eff_min_conf = min_conf;     pos_scale = 1.0
        elif sw_score == 2:
            eff_min_conf = min_conf + 1; pos_scale = 0.75
        else:
            eff_min_conf = min_conf + 2; pos_scale = 0.50

        entry_min_conf = eff_min_conf + (2 if after_stopout else 0)

        # ── 持仓管理 ─────────────────────────────────────────
        if in_trade:
            hold_bars   += 1
            current_ret  = (price - entry_price) / entry_price

            if row["is_bear"]:
                bear_consec += 1
            else:
                bear_consec = 0

            reduce_reason = None
            exit_reason   = None

            # 层次 2：Maintenance Margin 检查（优先级最高，强平）
            # account equity = capital + unrealised_pnl
            # notional exposure = position × price × LEVERAGE
            notional        = position * price * LEVERAGE
            unrealised_pnl  = (price - entry_price) * position * LEVERAGE
            account_equity  = capital + unrealised_pnl
            if notional > 0 and account_equity < maint_margin * notional:
                exit_reason = "MarginCall"

            if use_regime_reduce:
                # Gold_regime_reduce50_final / Silver_S-R1_provisional:
                # Regime → reduce to 50% once (Lock A); full exit only by price-type Stop/MaxHold
                if bear_consec >= bear_confirm and not regime_reduce_triggered:
                    old_pos          = position
                    new_pos          = position * 0.50
                    reduce_price_net = price * (1 - friction_pct)
                    released         = (old_pos - new_pos) * reduce_price_net
                    realised         = (reduce_price_net - entry_price) * (old_pos - new_pos) * LEVERAGE
                    capital         += released + realised
                    position     = new_pos
                    realised_from_reduce    += realised
                    regime_reduce_triggered  = True
                    reduce_time              = ts
                    reduce_price             = price

                # price-type stop: compare current price to fixed stop_price
                if price <= stop_price:
                    exit_reason = f"StopLoss ({stop_loss_pct*100:.0f}%)"
                elif hold_bars >= max_hold:
                    exit_reason = f"MaxHold ({max_hold} bars)"
            else:
                # AAPL and others: original Regime full exit + return-type stop
                if bear_consec >= bear_confirm:
                    exit_reason = "Regime → Bear/Crash"
                elif current_ret <= stop_loss_pct:
                    exit_reason = f"StopLoss ({stop_loss_pct*100:.0f}%)"
                elif hold_bars >= max_hold:
                    exit_reason = f"MaxHold ({max_hold} bars)"

            if exit_reason:
                exit_price_net = price * (1 - friction_pct)    # pay slippage+commission on exit
                pnl_remaining = (exit_price_net - entry_price) * position * LEVERAGE
                capital      += pnl_remaining
                total_pnl     = realised_from_reduce + pnl_remaining
                is_stop       = ("StopLoss" in exit_reason) or (exit_reason == "MarginCall")
                trades.append({
                    "entry_time":              entry_time,
                    "exit_time":               ts,
                    "entry_price":             entry_price,
                    "exit_price":              price,
                    "pnl":                     total_pnl,
                    "pos_size_pct":            pos_size_pct,
                    "exit_reason":             exit_reason,
                    "reduce_reason":           "RegimeReduce50" if regime_reduce_triggered else "N/A",
                    "reduce_time":             reduce_time,
                    "reduce_price":            reduce_price,
                    "hold_bars":               hold_bars,
                    "return_pct":              (exit_price_net / entry_price - 1) * 100 * LEVERAGE,
                    "sideways_score":          sw_score,
                    "regime_reduce_triggered": regime_reduce_triggered,
                    "vt_scale":                vt_scale,
                })
                position, in_trade, hold_bars  = 0.0, False, 0
                bear_consec                    = 0
                regime_reduce_triggered        = False
                realised_from_reduce           = 0.0
                reduce_time                    = None
                reduce_price                   = None
                if is_stop:
                    consec_stops += 1; after_stopout = True
                    cooldown_left = _sideways_cooldown(base_cd, sw_score, consec_stops)
                else:
                    consec_stops = 0; after_stopout = False
                    cooldown_left = base_cd

        if cooldown_left > 0:
            cooldown_left -= 1

        # ── 开仓判断 ─────────────────────────────────────────
        if (not in_trade and cooldown_left == 0
                and row["is_bull"]
                and row["signal_score"] >= entry_min_conf
                and float(row["adx"]) > adx_entry):
            score        = float(row["signal_score"])
            base_pct     = _position_size(score, eff_min_conf) * pos_scale
            if use_vol_target and vt_target:
                entry_rvol   = float(row.get("vol_volatility", vt_target))
                vt_scale     = float(np.clip(vt_target / entry_rvol, VT_MIN, VT_MAX)) if entry_rvol > 0 else 1.0
            else:
                vt_scale     = 1.0
            pos_size_pct = base_pct * vt_scale

            # 层次 1：Initial Margin 约束
            # notional = capital × pos_size_pct × LEVERAGE
            # 所需保证金 = notional × initial_margin ≤ capital（可用资金）
            # 若超出，缩减 pos_size_pct 使保证金恰好等于 capital
            max_pos_by_margin = 1.0 / (LEVERAGE * initial_margin)  # pos_size_pct 上限
            pos_size_pct      = min(pos_size_pct, max_pos_by_margin)

            position     = capital * pos_size_pct / price
            entry_price  = price * (1 + friction_pct)          # pay slippage+commission on entry
            stop_price   = entry_price * (1 + stop_loss_pct)   # price-type stop, fixed
            entry_time   = ts
            in_trade     = True
            hold_bars    = 0
            after_stopout           = False
            bear_consec             = 0
            regime_reduce_triggered = False
            realised_from_reduce    = 0.0
            reduce_time             = None
            reduce_price            = None

        mtm = capital + (price - entry_price) * position * LEVERAGE if in_trade else capital
        equity_curve.append(mtm)

    if in_trade:
        last_price         = float(df["Close"].iloc[-1])
        last_price_net     = last_price * (1 - friction_pct)
        pnl_remaining      = (last_price_net - entry_price) * position * LEVERAGE
        capital           += pnl_remaining
        trades.append({
            "entry_time":              entry_time,
            "exit_time":               df.index[-1],
            "entry_price":             entry_price,
            "exit_price":              last_price,
            "pnl":                     realised_from_reduce + pnl_remaining,
            "pos_size_pct":            pos_size_pct,
            "exit_reason":             "End of data",
            "reduce_reason":           "RegimeReduce50" if regime_reduce_triggered else "N/A",
            "reduce_time":             reduce_time,
            "reduce_price":            reduce_price,
            "hold_bars":               hold_bars,
            "return_pct":              (last_price - entry_price) / entry_price * 100 * LEVERAGE,
            "sideways_score":          int(df["sideways_score"].iloc[-1]),
            "regime_reduce_triggered": regime_reduce_triggered,
            "vt_scale":                vt_scale,
        })

    return equity_curve, trades


# ============================================================
# 6. 主入口
# ============================================================

def run_backtest(df: pd.DataFrame, ticker: str = "AAPL") -> Dict:
    is_daily = ticker in DAILY_TICKERS
    p        = TICKER_PARAMS.get(ticker, {"n_states": N_STATES, "bull_top": 2,
                                          "min_conf": MIN_CONFIRMATIONS,
                                          "stop": STOP_LOSS_PCT, "hold_mult": 1.0})
    n_states = p["n_states"]
    bull_top = p["bull_top"]
    min_conf = p["min_conf"]
    stop     = p["stop"]
    hold_mult = p["hold_mult"]

    features = get_hmm_features(df)
    n        = len(features)

    wf_states = _walk_forward_states(features, n, n_states)

    df = df.copy()
    df["state"]   = wf_states
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

    df = compute_indicators(df, ticker)
    _, score = compute_signals(df)
    df["signal_score"] = score
    df["tech_signal"]  = score >= min_conf

    # Regime Filter（诊断层，不影响进场）
    ma_ok  = (df["ema50"] > df["ema200"]) & (df["ema50_slope"] > 0)
    adx_ok = df["adx"] > 20
    df["regime_filter"] = (ma_ok & adx_ok).fillna(False)

    equity_curve, trades = _simulate(df, is_daily, stop, hold_mult, min_conf, ticker)
    df["equity"] = equity_curve

    metrics = _compute_metrics(df, trades, ticker, is_daily)

    full_model              = fit_hmm(features, n_states)
    bull_states, bear_state = identify_states(full_model, bull_top)
    regime_labels           = {s: _label(s) for s in range(n_states)}

    # HMM 后验概率（最新 bar）
    try:
        posterior = full_model.predict_proba(features[-1:]).flatten().tolist()
    except Exception:
        posterior = [1.0/n_states] * n_states

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
        "stop":          stop,
        "posterior":     posterior,
    }


# ============================================================
# 7. 绩效指标
# ============================================================

def _compute_metrics(df: pd.DataFrame, trades: list,
                     ticker: str, is_daily: bool) -> Dict:
    equity   = pd.Series(df["equity"].values, index=df.index)
    n_bars   = len(equity)

    total_ret   = (equity.iloc[-1] / STARTING_CAP - 1) * 100
    bh_ret      = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    alpha       = total_ret - bh_ret

    roll_max    = equity.cummax()
    dd_series   = (equity - roll_max) / roll_max * 100
    max_dd      = dd_series.min()

    bars_per_yr = 252 if is_daily else 252 * 24
    pct_returns = equity.pct_change().dropna()
    ann_ret     = ((equity.iloc[-1] / STARTING_CAP) ** (bars_per_yr / n_bars) - 1) * 100
    ann_vol     = pct_returns.std() * np.sqrt(bars_per_yr) * 100
    sharpe      = ann_ret / ann_vol if ann_vol > 0 else 0.0
    calmar      = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    wins        = [t for t in trades if t["pnl"] > 0]
    losses      = [t for t in trades if t["pnl"] <= 0]
    win_rate    = len(wins) / len(trades) * 100 if trades else 0.0

    avg_win  = float(np.mean([t["pnl"] for t in wins]))   if wins   else 0.0
    avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_pos_size = np.mean([t["pos_size_pct"] for t in trades]) * 100 if trades else 0.0

    # Sortino（仅下行波动）
    downside = pct_returns.clip(upper=0)
    down_vol = downside.std() * np.sqrt(bars_per_yr) * 100
    sortino  = ann_ret / down_vol if down_vol > 0 else 0.0

    # Profit Factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy per trade
    expectancy = (win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss) if trades else 0.0

    # 最大连续亏损
    max_consec_loss = cur_consec = 0
    for t in trades:
        if t["pnl"] <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    # 平均持仓周期
    avg_hold = float(np.mean([t["hold_bars"] for t in trades])) if trades else 0.0

    # Tail Ratio
    if len(pct_returns) > 20:
        p95 = float(np.percentile(pct_returns, 95))
        p05 = abs(float(np.percentile(pct_returns, 5)))
        tail_ratio = p95 / p05 if p05 > 0 else float("inf")
    else:
        tail_ratio = 1.0

    # 收益分布偏度/峰度
    skewness = float(pct_returns.skew())
    kurtosis = float(pct_returns.kurt())

    # 最大回撤恢复时间
    in_dd = False; dd_start = 0; max_recovery = 0
    eq_vals = equity.values; eq_max = equity.cummax().values
    for i, (v, mx) in enumerate(zip(eq_vals, eq_max)):
        if v < mx and not in_dd:
            in_dd = True; dd_start = i
        elif v >= mx and in_dd:
            max_recovery = max(max_recovery, i - dd_start)
            in_dd = False

    # Top Trade 贡献度
    all_pnl = sorted([t["pnl"] for t in trades], reverse=True)
    total_gross = sum(all_pnl) if all_pnl else 1
    top5_contrib  = sum(all_pnl[:5])  / max(total_gross, 1) * 100 if len(all_pnl) >= 1 else 0
    top10_contrib = sum(all_pnl[:10]) / max(total_gross, 1) * 100 if len(all_pnl) >= 1 else 0

    # 月度数据（含 BH 对比）
    monthly_eq  = equity.resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna()
    monthly_win = (monthly_ret > 0).sum() / len(monthly_ret) * 100 if len(monthly_ret) > 0 else 0.0

    bh_price     = df["Close"].resample("ME").last()
    bh_monthly   = bh_price.pct_change().dropna() * 100
    strat_monthly = monthly_ret * 100

    monthly_df = pd.DataFrame({
        "year":       strat_monthly.index.year,
        "month":      strat_monthly.index.month,
        "ret":        strat_monthly.values,
    })
    # 对齐 BH 月度
    bh_aligned = bh_monthly.reindex(strat_monthly.index).fillna(0)
    monthly_df["bh_ret"]    = bh_aligned.values
    monthly_df["alpha_ret"] = monthly_df["ret"] - monthly_df["bh_ret"]

    # 出场原因归因
    exit_attr = {}
    if trades:
        tdf = pd.DataFrame(trades)
        for reason, grp in tdf.groupby("exit_reason"):
            exit_attr[reason] = {
                "count":   len(grp),
                "win_r":   round((grp["pnl"] > 0).mean() * 100, 1),
                "avg_pnl": round(grp["pnl"].mean(), 1),
                "total_pnl": round(grp["pnl"].sum(), 1),
            }

    try:
        spy_df    = fetch_data("SPY")
        spy_bh    = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[0] - 1) * 100
        spy_alpha = total_ret - spy_bh
    except Exception:
        spy_bh    = None
        spy_alpha = None

    return {
        "total_return_pct":  round(total_ret,      2),
        "ann_return_pct":    round(ann_ret,         2),
        "bh_return_pct":     round(bh_ret,          2),
        "alpha_pct":         round(alpha,            2),
        "spy_bh_pct":        round(spy_bh,           2) if spy_bh   is not None else None,
        "spy_alpha_pct":     round(spy_alpha,        2) if spy_alpha is not None else None,
        "max_drawdown_pct":  round(max_dd,           2),
        "max_recovery_bars": max_recovery,
        "sharpe":            round(sharpe,            3),
        "sortino":           round(sortino,           3),
        "calmar":            round(calmar,            3),
        "tail_ratio":        round(min(tail_ratio, 99.0), 3),
        "ann_vol_pct":       round(ann_vol,           2),
        "win_rate_pct":      round(win_rate,          2),
        "monthly_win_pct":   round(monthly_win,       2),
        "n_trades":          len(trades),
        "avg_win":           round(avg_win,           2),
        "avg_loss":          round(avg_loss,          2),
        "rr_ratio":          round(rr_ratio,          3),
        "profit_factor":     round(min(profit_factor, 99.0), 3),
        "expectancy":        round(expectancy,        2),
        "max_consec_loss":   max_consec_loss,
        "avg_hold_bars":     round(avg_hold,          1),
        "skewness":          round(skewness,          3),
        "kurtosis":          round(kurtosis,          3),
        "avg_pos_size_pct":  round(avg_pos_size,     1),
        "final_capital":     round(equity.iloc[-1],   2),
        "top5_contrib_pct":  round(top5_contrib,     1),
        "top10_contrib_pct": round(top10_contrib,    1),
        "monthly_df":        monthly_df,
        "exit_attribution":  exit_attr,
    }
