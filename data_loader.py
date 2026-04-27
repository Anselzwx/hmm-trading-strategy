"""
data_loader.py
--------------
OHLCV 数据加载器，使用 Financial Modeling Prep API。

- AAPL / GCUSD / SIUSD：全部使用日线，10年历史（2016-01-01 至今）
- 缓存 1 小时，避免重复请求
- 宏观因子：从 investing_macro_data.db 读取，forward-fill 到每个交易日
"""

import os
import pickle
import sqlite3
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


CACHE_DIR       = os.path.dirname(__file__)
CACHE_TTL_HOURS = 1
FMP_API_KEY     = "1aqqbJ9eixJ0cr8RGJd5LC9fXjdH5k1p"
FMP_START       = "2016-01-01"
FMP_BASE        = "https://financialmodelingprep.com/stable/historical-price-eod/full"

MACRO_DB = os.environ.get(
    "MACRO_DB_PATH",
    "/Users/zhaowenxuan/Desktop/公司文件/黄金交接/investing_macro_data.db"
)

# 宏观指标表名 → 输出列名
MACRO_TABLES = {
    "美国CPI月率":          "cpi_mom",
    "美国核心CPI月率":      "core_cpi_mom",
    "美国核心PCE物价指数月率": "core_pce_mom",
    "美国初请失业金人数":   "jobless_claims",
    "美国ISM制造业PMI":     "ism_pmi",
}

# 所有 ticker 统一日线
DAILY_TICKERS = {"GC=F", "SI=F", "AAPL", "NVDA", "META"}

# ticker → FMP symbol 映射
FMP_SYMBOL = {
    "AAPL": "AAPL",
    "GC=F": "GCUSD",
    "SI=F": "SIUSD",
    "NVDA": "NVDA",
    "META": "META",
}

ASSET_LABELS = {
    "AAPL": "Apple Inc. (AAPL)",
    "GC=F": "Gold Futures (GOLD)",
    "SI=F": "Silver Futures (SILVER)",
    "NVDA": "NVIDIA Corp. (NVDA)",
    "META": "Meta Platforms (META)",
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
# 宏观数据
# ---------------------------------------------------------------------------

def _parse_value(s) -> float:
    """把 '0.3%' / '227K' / '52.6' 等字符串转成 float。"""
    if s is None:
        return float("nan")
    s = str(s).strip().replace("%", "").replace("K", "e3").replace("M", "e6").replace("B", "e9")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_macro() -> pd.DataFrame:
    """
    从 investing_macro_data.db 读取关键宏观指标，
    解析今值，以发布日期为索引，forward-fill 到日频。
    返回 DataFrame，index 为 DatetimeIndex（日频）。
    """
    if not os.path.exists(MACRO_DB):
        return pd.DataFrame()

    conn = sqlite3.connect(MACRO_DB)
    series = {}
    for table, col in MACRO_TABLES.items():
        try:
            df = pd.read_sql(f'SELECT datetime, 今值 FROM "{table}"', conn, parse_dates=["datetime"])
            df = df.dropna(subset=["今值"])
            df["value"] = df["今值"].apply(_parse_value)
            df = df.dropna(subset=["value"])
            df = df.set_index("datetime")["value"].sort_index()
            # 去重：同一天取最后一条
            df = df[~df.index.duplicated(keep="last")]
            series[col] = df
        except Exception:
            pass
    conn.close()

    if not series:
        return pd.DataFrame()

    macro = pd.DataFrame(series)
    # 重采样到日频，forward-fill（月度数据填充到每个交易日）
    macro = macro.resample("D").last().ffill()
    return macro


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
        returns, range_pct, vol_volatility,   ← 价格特征
        cpi_mom, core_cpi_mom, core_pce_mom,  ← 宏观特征（forward-fill）
        jobless_claims, ism_pmi
    """
    if not force_refresh and _cache_is_fresh(ticker):
        return _load_cache(ticker)

    df = _fetch_fmp(ticker)

    # 价格特征
    df["returns"]       = np.log(df["Close"] / df["Close"].shift(1)) * 100
    df["range_pct"]     = (df["High"] - df["Low"]) / df["Close"] * 100
    df["vol_volatility"] = df["returns"].rolling(5, min_periods=3).std()

    # 拼入宏观特征
    macro = load_macro()
    if not macro.empty:
        macro.index = macro.index.tz_localize(None)
        df = df.join(macro, how="left")
        # 对宏观列做标准化（z-score），避免量纲差异影响 HMM
        for col in MACRO_TABLES.values():
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                if sigma > 0:
                    df[col] = (df[col] - mu) / sigma
                df[col] = df[col].ffill()

    df.dropna(inplace=True)
    _save_cache(ticker, df)
    return df


def get_hmm_features(df: pd.DataFrame) -> "np.ndarray":
    base = ["returns", "range_pct", "vol_volatility"]
    macro_cols = [c for c in MACRO_TABLES.values() if c in df.columns]
    return df[base + macro_cols].values
