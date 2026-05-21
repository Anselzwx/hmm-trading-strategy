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
FMP_START       = "2008-01-01"
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
DAILY_TICKERS = {"GC=F", "SI=F", "CL=F", "AAPL", "NVDA", "META",
                 "AMZN", "GOOG", "MSFT", "TSLA", "HOOD", "SPY", "FXI", "PLTR"}

# ticker → FMP symbol 映射
FMP_SYMBOL = {
    "AAPL": "AAPL", "GC=F": "GCUSD", "SI=F": "SIUSD", "CL=F": "CLUSD",
    "NVDA": "NVDA", "META": "META",
    "AMZN": "AMZN", "GOOG": "GOOG", "MSFT": "MSFT",
    "TSLA": "TSLA", "HOOD": "HOOD", "SPY":  "SPY",
    "FXI":  "FXI",  "PLTR": "PLTR",
}

ASSET_LABELS = {
    "AAPL": "Apple Inc. (AAPL)",
    "GC=F": "Gold Futures (GOLD)",
    "SI=F": "Silver Futures (SILVER)",
    "CL=F": "Crude Oil Futures (WTI)",
    "NVDA": "NVIDIA Corp. (NVDA)",
    "META": "Meta Platforms (META)",
    "AMZN": "Amazon (AMZN)",
    "GOOG": "Alphabet / Google (GOOG)",
    "MSFT": "Microsoft (MSFT)",
    "TSLA": "Tesla (TSLA)",
    "HOOD": "Robinhood (HOOD)",
    "SPY":  "S&P 500 ETF (SPY)",
    "FXI":  "China Large-Cap ETF (FXI)",
    "PLTR": "Palantir (PLTR)",
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
# 财报日过滤
# ---------------------------------------------------------------------------

def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """
    从 FMP 拉取历史财报日期，返回 DatetimeIndex。
    仅适用于股票（商品期货无财报，返回空）。
    """
    if ticker not in FMP_SYMBOL or ticker in ("GC=F", "SI=F", "CL=F"):
        return pd.DatetimeIndex([])
    symbol = FMP_SYMBOL.get(ticker, ticker)
    try:
        url = (f"https://financialmodelingprep.com/stable/earnings-calendar"
               f"?symbol={symbol}&from={FMP_START}"
               f"&to={datetime.now().strftime('%Y-%m-%d')}&apikey={FMP_API_KEY}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DatetimeIndex([])
        dates = pd.to_datetime([d["date"] for d in data if "date" in d], errors="coerce")
        return dates.dropna()
    except Exception:
        return pd.DatetimeIndex([])


def earnings_blackout_mask(index: pd.DatetimeIndex,
                           earnings_dates: pd.DatetimeIndex,
                           pre_days: int = 3,
                           post_days: int = 1) -> pd.Series:
    """
    返回 bool Series（True = 财报前后封锁期，不允许开新仓）。
    index: 回测 DataFrame 的时间索引。
    """
    mask = pd.Series(False, index=index)
    for ed in earnings_dates:
        start = ed - pd.Timedelta(days=pre_days)
        end   = ed + pd.Timedelta(days=post_days)
        mask.loc[(index >= start) & (index <= end)] = True
    return mask


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def _load_market_features() -> pd.DataFrame:
    """加载 VIX（VIXY）和美元指数（DXUSD）作为市场特征。"""
    frames = {}
    for sym, col in [("VIXY", "vix_ret"), ("DXUSD", "dxy_ret")]:
        try:
            raw = _fetch_fmp(sym)
            ret = np.log(raw["Close"] / raw["Close"].shift(1)) * 100
            frames[col] = ret
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    mkt = pd.DataFrame(frames)
    mkt = mkt.resample("D").last().ffill()
    # z-score 标准化
    for col in mkt.columns:
        mu, sigma = mkt[col].mean(), mkt[col].std()
        if sigma > 0:
            mkt[col] = (mkt[col] - mu) / sigma
    return mkt


def fetch_data(ticker: str = "AAPL", force_refresh: bool = False) -> pd.DataFrame:
    """
    返回带以下列的 DataFrame：
        Open, High, Low, Close, Volume,
        returns, range_pct, vol_volatility,   ← 价格特征
        cpi_mom, core_cpi_mom, core_pce_mom,  ← 宏观特征（forward-fill）
        jobless_claims, ism_pmi,
        vix_ret, dxy_ret                       ← 市场特征
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
        for col in MACRO_TABLES.values():
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                if sigma > 0:
                    df[col] = (df[col] - mu) / sigma
                df[col] = df[col].ffill()

    # 拼入市场特征（VIX、美元指数），跳过自身
    if ticker not in ("VIXY", "DXUSD"):
        mkt = _load_market_features()
        if not mkt.empty:
            mkt.index = mkt.index.tz_localize(None)
            df = df.join(mkt, how="left")
            for col in mkt.columns:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0.0)  # 早期无数据填0（中性）

    # 只对核心价格特征做 dropna，宏观/市场特征已 ffill
    df.dropna(subset=["returns", "range_pct", "vol_volatility"], inplace=True)
    _save_cache(ticker, df)
    return df


def get_hmm_features(df: pd.DataFrame) -> "np.ndarray":
    base = ["returns", "range_pct", "vol_volatility"]
    macro_cols = [c for c in MACRO_TABLES.values() if c in df.columns]
    mkt_cols   = [c for c in ("vix_ret", "dxy_ret") if c in df.columns]
    return df[base + macro_cols + mkt_cols].values
