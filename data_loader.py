"""
data_loader.py
--------------
OHLCV 数据加载器，使用 Financial Modeling Prep API。

- AAPL / GCUSD / SIUSD：全部使用日线，10年历史（2016-01-01 至今）
- 缓存 1 小时，避免重复请求
"""

import os
import pickle
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


CACHE_DIR       = os.path.dirname(__file__)
CACHE_TTL_HOURS = 1
FMP_API_KEY     = "1aqqbJ9eixJ0cr8RGJd5LC9fXjdH5k1p"
FMP_START       = "2016-01-01"
FMP_BASE        = "https://financialmodelingprep.com/stable/historical-price-eod/full"

# 所有 ticker 统一日线
DAILY_TICKERS = {"GC=F", "SI=F", "AAPL"}

# ticker → FMP symbol 映射
FMP_SYMBOL = {
    "AAPL": "AAPL",
    "GC=F": "GCUSD",
    "SI=F": "SIUSD",
}

ASSET_LABELS = {
    "AAPL": "Apple Inc. (AAPL)",
    "GC=F": "Gold Futures (GOLD)",
    "SI=F": "Silver Futures (SILVER)",
}


# ---------------------------------------------------------------------------
# 缓存
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


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

def _fetch_fmp(ticker: str) -> pd.DataFrame:
    symbol  = FMP_SYMBOL.get(ticker, ticker)
    end_str = datetime.now().strftime("%Y-%m-%d")
    url = (f"{FMP_BASE}?symbol={symbol}"
           f"&from={FMP_START}&to={end_str}&apikey={FMP_API_KEY}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"FMP 未返回 {symbol} 数据，请检查 API key 或 symbol。")

    df = pd.DataFrame(data)
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()
    return df


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

    df = _fetch_fmp(ticker)

    # 特征 1：对数收益率 (%)
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1)) * 100

    # 特征 2：价格区间占收盘价比例
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # 特征 3：5日滚动收益率标准差
    df["vol_volatility"] = df["returns"].rolling(5, min_periods=3).std()

    df.dropna(inplace=True)
    _save_cache(ticker, df)
    return df


def get_hmm_features(df: pd.DataFrame) -> "np.ndarray":
    return df[["returns", "range_pct", "vol_volatility"]].values
