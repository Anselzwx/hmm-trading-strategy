"""
CNN Fear & Greed Index（美股恐惧与贪婪指数）

CNN 官网/接口对非浏览器请求返回 451/418，这里读 fgscore.com 的转载页：
页面服务端渲染，Next.js payload 里带 CNN 原始数据 —— 总分、昨日/周/月/季对比、
7 个子指标分数与评级、近 120 个交易日历史。免 key。
"""
import json
import re
from typing import Optional

import pandas as pd
import requests

URL = "https://fgscore.com/en/markets/sp500"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# key → (中文名, 衡量什么, 恐惧方向的含义)
INDICATORS = {
    "market_momentum_sp500": ("市场动量", "S&P 500 vs 125 日均线", "指数跌破/远低于均线"),
    "stock_price_strength":  ("股价强度", "NYSE 52 周新高 vs 新低数量", "新低多于新高"),
    "stock_price_breadth":   ("股价广度", "McClellan 成交量总和指数", "下跌股成交量占优"),
    "put_call_options":      ("期权 Put/Call", "5 日平均 Put/Call 比", "看跌期权买盘多"),
    "market_volatility_vix": ("市场波动率", "VIX vs 50 日均线", "VIX 高于均线"),
    "safe_haven_demand":     ("避险需求", "股票 vs 国债 20 日收益差", "国债跑赢股票"),
    "junk_bond_demand":      ("垃圾债需求", "垃圾债 vs 投资级债利差", "利差走阔"),
}


def band(v: float) -> str:
    if v < 25:
        return "极度恐惧"
    if v < 45:
        return "恐惧"
    if v <= 55:
        return "中性"
    if v <= 75:
        return "贪婪"
    return "极度贪婪"


def band_color(v: float) -> str:
    if v < 25:
        return "#ff5252"
    if v < 45:
        return "#fb923c"
    if v <= 55:
        return "#94a3b8"
    if v <= 75:
        return "#4ade80"
    return "#00c864"


def _payload(html: str) -> str:
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, flags=re.S)
    return "".join(json.loads('"' + c + '"') for c in chunks)


def _json_after(s: str, key: str):
    i = s.find(f'"{key}":')
    if i < 0:
        return None
    return json.JSONDecoder().raw_decode(s, i + len(key) + 3)[0]


def _ts(ms) -> Optional[pd.Timestamp]:
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("America/New_York") if ms else None


def fetch() -> dict:
    """返回 {value, rating, timestamp, compare{yesterday,week,month,quarter},
             indicators[{key,name,desc,fear_means,score,rating,timestamp}], history(Series)}"""
    resp = requests.get(URL, headers={"User-Agent": _UA}, timeout=20)
    resp.raise_for_status()
    s = _payload(resp.text)

    cur = _json_after(s, "current")
    inds = _json_after(s, "cnnIndicators")
    if not cur or not inds:
        raise ValueError("页面结构变化：未找到 current / cnnIndicators")
    compare = _json_after(s, "compare") or {}

    pts = _json_after(s, "points") or []
    hist = pd.Series({pd.Timestamp(p["t"], unit="ms").normalize(): float(p["v"]) for p in pts},
                     dtype=float).sort_index()

    out_inds = []
    for r in inds:
        name, desc, fear = INDICATORS.get(r["key"], (r["key"], "", ""))
        out_inds.append(dict(key=r["key"], name=name, desc=desc, fear_means=fear,
                             score=float(r["score"]), rating=r.get("rating", ""),
                             timestamp=_ts(r.get("timestamp"))))
    return dict(value=float(cur["value"]), rating=cur.get("classification", ""),
                timestamp=_ts(cur.get("timestamp")), compare=compare,
                indicators=out_inds, history=hist, source=URL)


if __name__ == "__main__":
    d = fetch()
    print(d["value"], d["rating"], d["timestamp"], d["compare"])
    for r in d["indicators"]:
        print(f'{r["name"]:8s} {r["score"]:6.1f} {r["rating"]:14s} {r["timestamp"]}')
    print(d["history"].tail())
