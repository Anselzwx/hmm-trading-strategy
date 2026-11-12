"""
盘前 / 盘后 / 夜盘实时报价（CNBC 报价接口，免费、批量、无需 key）

个股：常规收盘价 + 盘前(PRE_MKT) / 盘后(POST_MKT) 最新成交
期货：@GC.1 @SI.1 @CL.1 近 24 小时交易，直接给最新价
夜盘（美东 20:00–04:00）：个股无免费报价，用纳指/标普期货代表大盘方向
"""
from datetime import datetime
from typing import Dict, List
import requests

_URL = "https://quote.cnbc.com/quote-html-webservice/quote.htm"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

CNBC_SYMBOL = {"GC=F": "@GC.1", "SI=F": "@SI.1", "CL=F": "@CL.1"}
INDEX_FUTURES = {"NQ": "@ND.1", "ES": "@SP.1"}
EXT_LABEL = {"PRE_MKT": "盘前", "POST_MKT": "盘后", "POST_MKT_PREV": "盘后"}
ALERT_PCT = 3.0


def _f(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def fetch_quotes(tickers: List[str]) -> Dict[str, dict]:
    """返回 {ticker: {reg_close, reg_chg_pct, ext_session, ext_last, ext_chg_pct, last, move_pct, ...}}"""
    sym = {t: CNBC_SYMBOL.get(t, t) for t in tickers}
    sym.update({k: v for k, v in INDEX_FUTURES.items()})
    resp = requests.get(_URL, params={
        "partnerId": 2, "requestMethod": "quick", "exthrs": 1, "noform": 1,
        "fund": 1, "output": "json", "symbols": "|".join(sym.values()),
    }, headers={"User-Agent": _UA}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()["QuickQuoteResult"]["QuickQuote"]
    if isinstance(rows, dict):
        rows = [rows]
    by_sym = {r.get("symbol"): r for r in rows}

    out = {}
    for t, s in sym.items():
        r = by_sym.get(s)
        if not r or _f(r.get("last")) is None:
            continue
        is_fut = s.startswith("@")
        reg = _f(r.get("last"))
        q = dict(
            symbol=s, name=r.get("name"), is_future=is_fut,
            reg_close=reg, reg_chg_pct=_f(r.get("change_pct")),
            reg_time=r.get("last_time"),
            ext_session=None, ext_last=None, ext_chg_pct=None, ext_time=None,
        )
        e = r.get("ExtendedMktQuote") or {}
        if not is_fut and _f(e.get("last")) is not None:
            q.update(ext_session=EXT_LABEL.get(e.get("type"), e.get("type")),
                     ext_last=_f(e.get("last")), ext_chg_pct=_f(e.get("change_pct")),
                     ext_time=e.get("last_time"))
        # 期货：最新价相对前结算；个股：盘前/盘后价相对常规收盘
        q["last"] = q["ext_last"] if q["ext_last"] is not None else reg
        q["move_pct"] = q["reg_chg_pct"] if is_fut else (q["ext_chg_pct"] or 0.0)
        out[t] = q
    return out


def assess(ticker: str, quote: dict, trend_sig: dict = None, swing_sig: dict = None) -> List[str]:
    """把盘后/夜盘变动映射成对现有信号的提示（不改变主信号本身）"""
    if not quote:
        return []
    mv = quote.get("move_pct") or 0.0
    last = quote.get("last")
    notes = []
    big_down, big_up = mv <= -ALERT_PCT, mv >= ALERT_PCT
    if trend_sig:
        af, al = trend_sig.get("action_if_flat"), trend_sig.get("action_if_long")
        if af == "ENTER" and big_down:
            notes.append(f"趋势 ENTER，但{_sess(quote)}已跌 {mv:+.1f}% → 建议开盘后观察再买，不要开盘直接追")
        if af == "ENTER" and big_up:
            notes.append(f"趋势 ENTER，但{_sess(quote)}已涨 {mv:+.1f}% → 开盘可能高开，入场成本会明显高于信号价")
        if al == "HOLD" and big_down:
            notes.append(f"若持仓：{_sess(quote)} {mv:+.1f}%，开盘大概率跳空低开，注意风险")
        if al == "EXIT" and big_down:
            notes.append(f"若持仓：本就该卖，{_sess(quote)}再跌 {mv:+.1f}%，开盘卖出价会更低")
    if swing_sig and last and not quote.get("is_future"):
        c0, nd, dist = swing_sig.get("close"), swing_sig.get("ndown", 0), swing_sig.get("dist_sma200_pct", 0)
        sma200 = c0 / (1 + dist / 100) if c0 else None
        reg, reg_d = quote.get("reg_close"), str(quote.get("reg_time") or "")[:10]
        c = c0
        # 信号数据之后若已有新的常规收盘，先把这一天补进连跌计数
        if reg and reg_d > swing_sig.get("date", ""):
            nd = nd + 1 if reg < c0 else 0
            c = reg
            if nd >= 3 and sma200 and reg > sma200 and swing_sig.get("action") not in ("HOLD", "EXIT"):
                notes.append(f"波段：按 {reg_d} 最新收盘 ${reg:,.2f} 已连跌 {nd} 天且在 SMA200 上方 → 已触发买入（app 信号数据尚未更新）")
        if swing_sig.get("action") == "ENTER" and big_down:
            notes.append(f"波段 ENTER，{_sess(quote)} {mv:+.1f}% → 按规则明日开盘买（买跌策略本就接受跳空），仓位可减半")
        if swing_sig.get("action") == "HOLD" and big_down:
            notes.append(f"波段持仓中，{_sess(quote)} {mv:+.1f}%，策略无硬止损，按规则持有至离场条件")
        if quote.get("ext_last") is not None and sma200:
            if nd == 2 and last < c and last > sma200:
                notes.append(f"波段预览：已连跌 2 天，若下一交易日收在 ${last:,.2f} 附近将触发连跌 3 天买入")
            if c > sma200 >= last:
                notes.append(f"波段预览：{_sess(quote)}价 ${last:,.2f} 已低于 SMA200（约 ${sma200:,.2f}），若收盘确认将停止波段买入")
    if not notes and (big_down or big_up):
        notes.append(f"{_sess(quote)}变动 {mv:+.1f}%，开盘可能大幅跳空")
    return notes


def _sess(q: dict) -> str:
    if q.get("is_future"):
        return "期货最新价"
    return q.get("ext_session") or "盘后"


def market_session(now: datetime = None) -> str:
    """当前美股时段（按美东时间）"""
    import pandas as pd
    ny = pd.Timestamp(now or datetime.now()).tz_localize("Asia/Shanghai").tz_convert("America/New_York") \
        if now else pd.Timestamp.now(tz="America/New_York")
    if ny.weekday() >= 5:
        return "周末休市"
    hm = ny.hour * 60 + ny.minute
    if 4 * 60 <= hm < 9 * 60 + 30:
        return "盘前"
    if 9 * 60 + 30 <= hm < 16 * 60:
        return "常规交易"
    if 16 * 60 <= hm < 20 * 60:
        return "盘后"
    return "夜盘"
