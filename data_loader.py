"""
data_loader.py
--------------
通用 OHLCV 数据加载器，支持任意 yfinance ticker。

- 股票 / ETF（AAPL 等）：用 1h 间隔，55 天分块拉取（yfinance 限制）
- 贵金属（GC=F / SI=F）：yfinance 对期货 1h 数据支持差，改用 1d 日线
  覆盖 730 天，数据量足够训练 HMM。
"""

import os
import pickle
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


CACHE_DIR       = os.path.dirname(__file__)
CACHE_TTL_HOURS = 1
LOOKBACK_DAYS   = 730
CHUNK_DAYS      = 55        # 1h 模式下每块不超过 60 天

# 用日线的 ticker（期货符号在 1h 下数据质量差）
DAILY_TICKERS = {"GC=F", "SI=F"}

# 资产显示名
ASSET_LABELS = {
    "AAPL":  "Apple Inc. (AAPL)",
    "GC=F":  "Gold Futures (GOLD)",
    "SI=F":  "Silver Futures (SILVER)",
}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> str:
    safe = ticker.replace("=", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f".cache_{safe}.pkl")


def _cache_is_fresh(ticker: str) -> bool:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _load_cache(ticker: str) -> pd.DataFrame:
    with open(_cache_path(ticker), "rb") as f:
        return pickle.load(f)


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    with open(_cache_path(ticker), "wb") as f:
        pickle.dump(df, f)


def _flatten(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _keep_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    return raw[cols].copy()


# ---------------------------------------------------------------------------
# 下载逻辑
# ---------------------------------------------------------------------------

def _fetch_hourly(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """分块拉取 1h 数据并拼接。"""
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        raw = yf.download(
            ticker,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if not raw.empty:
            chunks.append(_keep_ohlcv(_flatten(raw)))
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _fetch_daily(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """拉取日线数据（贵金属期货用）。"""
    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return pd.DataFrame()
    return _keep_ohlcv(_flatten(raw))


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def fetch_data(ticker: str = "AAPL", force_refresh: bool = False) -> pd.DataFrame:
    """
    返回带以下列的 DataFrame：
        Open, High, Low, Close, Volume,
        returns, range_pct, vol_volatility   ← HMM 特征
    """
    if not force_refresh and _cache_is_fresh(ticker):
        return _load_cache(ticker)

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

    if ticker in DAILY_TICKERS:
        # 日线模式：volume 对期货意义不大但保留结构一致性
        raw = _fetch_daily(ticker, start_dt, end_dt)
        # 日线没有盘中成交量波动特征，用价格波动率替代 vol_volatility
        vol_window = 5   # 5 日滚动
    else:
        raw = _fetch_hourly(ticker, start_dt, end_dt)
        vol_window = 24  # 24 小时滚动

    if raw.empty:
        raise RuntimeError(
            f"yfinance 未返回 {ticker} 的任何数据，请检查网络或 ticker 符号。"
        )

    df = raw.copy()
    df.dropna(inplace=True)

    # 特征 1：对数收益率 (%)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1)) * 100

    # 特征 2：价格区间占收盘价比例
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # 特征 3：成交量对数的滚动标准差（期货日线时用价格收益率 std 替代）
    if ticker in DAILY_TICKERS:
        df["vol_volatility"] = df["returns"].rolling(vol_window, min_periods=3).std()
    else:
        log_vol = np.log(df["Volume"].replace(0, np.nan))
        df["vol_volatility"] = log_vol.rolling(vol_window, min_periods=6).std()

    df.dropna(inplace=True)
    _save_cache(ticker, df)
    return df


def get_hmm_features(df: pd.DataFrame) -> "np.ndarray":
    return df[["returns", "range_pct", "vol_volatility"]].values
