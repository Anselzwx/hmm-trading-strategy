"""
monitor.py — 每日监控 + Email 报警
=====================================
用法：
    python monitor.py

触发条件：
    1. 任一资产出现 ENTER 信号（空仓时）
    2. 任一资产出现 EXIT / MarginCall 信号（持仓时）
    3. 任一资产 HMM bear_prob > 60%（预警）
    4. 脚本运行出错（错误报警）

配置：
    设置环境变量 GMAIL_APP_PASSWORD，或直接修改下方 CONFIG。
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
"""

from __future__ import annotations

import os
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from typing import Dict, List

from signal_generator import run as generate_signals

# ── 配置 ─────────────────────────────────────────────────────
CONFIG = {
    "smtp_host":   "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "anselwilliam789@gmail.com",
    "recipient":   "anselwilliam789@gmail.com",
    "app_password": os.environ.get("GMAIL_APP_PASSWORD", ""),  # 从环境变量读取
}

# 报警阈值
BEAR_PROB_ALERT  = 0.60   # bear_prob 超过此值发预警
ACTION_ALERTS    = {"ENTER", "EXIT", "MarginCall"}


# ── 邮件发送 ─────────────────────────────────────────────────

def send_email(subject: str, body_html: str) -> None:
    if not CONFIG["app_password"]:
        print(f"[monitor] EMAIL SKIPPED — GMAIL_APP_PASSWORD not set.")
        print(f"  Subject: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = CONFIG["sender"]
    msg["To"]      = CONFIG["recipient"]
    msg.attach(MIMEText(body_html, "html"))

    with smtplib.SMTP(CONFIG["smtp_host"], CONFIG["smtp_port"]) as server:
        server.ehlo()
        server.starttls()
        server.login(CONFIG["sender"], CONFIG["app_password"])
        server.sendmail(CONFIG["sender"], CONFIG["recipient"], msg.as_string())

    print(f"[monitor] Email sent: {subject}")


# ── 报警判断 ─────────────────────────────────────────────────

def _check_alerts(signals: Dict) -> List[Dict]:
    alerts = []
    for ticker, sig in signals.items():
        action_flat = sig.get("action_if_flat", "")
        action_long = sig.get("action_if_long", "")
        bear_prob   = sig.get("bear_prob", 0.0)

        if action_flat in ACTION_ALERTS:
            alerts.append({
                "ticker":   ticker,
                "type":     "ACTION",
                "level":    "HIGH",
                "message":  f"Signal if FLAT: {action_flat}",
                "sig":      sig,
            })
        if action_long in ACTION_ALERTS:
            alerts.append({
                "ticker":   ticker,
                "type":     "ACTION",
                "level":    "HIGH",
                "message":  f"Signal if LONG: {action_long}",
                "sig":      sig,
            })
        if bear_prob >= BEAR_PROB_ALERT:
            alerts.append({
                "ticker":   ticker,
                "type":     "BEAR_PROB",
                "level":    "WARN",
                "message":  f"bear_prob={bear_prob:.1%} ≥ {BEAR_PROB_ALERT:.0%}",
                "sig":      sig,
            })
    return alerts


# ── HTML 邮件构建 ─────────────────────────────────────────────

def _sig_row(sig: Dict) -> str:
    action_flat = sig.get("action_if_flat", "—")
    action_long = sig.get("action_if_long", "—")
    color_flat  = "#d4edda" if action_flat == "ENTER" else ("#f8d7da" if action_flat in ("EXIT","STAY_OUT") else "#fff3cd")
    color_long  = "#f8d7da" if action_long in ("EXIT","MarginCall") else "#d4edda"
    vt          = f"vt_scale={sig['vt_scale']:.3f}" if sig.get("vt_scale") else ""
    return f"""
    <tr>
      <td><b>{sig['ticker']}</b></td>
      <td>{sig['date']}</td>
      <td>{sig['close']}</td>
      <td>{sig['regime']}</td>
      <td>{sig['signal_score']}/{14} (min={sig['min_conf']})</td>
      <td>{sig['adx']:.1f} (gate={sig['adx_entry']})</td>
      <td>{sig['bull_prob']:.1%}</td>
      <td>{sig['bear_prob']:.1%}</td>
      <td style="background:{color_flat}">{action_flat}</td>
      <td style="background:{color_long}">{action_long}</td>
      <td>{vt}</td>
    </tr>"""


def _build_alert_email(alerts: List[Dict], signals: Dict) -> tuple[str, str]:
    date_str = datetime.now().strftime("%Y-%m-%d")
    levels   = [a["level"] for a in alerts]
    subject  = f"[HMM] {'🔴 ACTION' if 'HIGH' in levels else '🟡 WARN'} — {date_str}"

    alert_rows = ""
    for a in alerts:
        bg = "#f8d7da" if a["level"] == "HIGH" else "#fff3cd"
        alert_rows += f'<tr style="background:{bg}"><td><b>{a["ticker"]}</b></td><td>{a["level"]}</td><td>{a["message"]}</td></tr>'

    sig_rows = "".join(_sig_row(s) for s in signals.values())

    body = f"""
<html><body style="font-family:monospace;font-size:13px;">
<h2>HMM Daily Signal Report — {date_str}</h2>

<h3>Alerts</h3>
<table border="1" cellpadding="4" cellspacing="0">
  <tr><th>Ticker</th><th>Level</th><th>Message</th></tr>
  {alert_rows}
</table>

<h3>Full Signal Table</h3>
<table border="1" cellpadding="4" cellspacing="0">
  <tr>
    <th>Ticker</th><th>Date</th><th>Close</th><th>Regime</th>
    <th>Score</th><th>ADX</th><th>BullProb</th><th>BearProb</th>
    <th>If FLAT</th><th>If LONG</th><th>VT</th>
  </tr>
  {sig_rows}
</table>

<p style="color:gray;font-size:11px;">Generated {datetime.now().isoformat()}</p>
</body></html>"""

    return subject, body


def _build_daily_email(signals: Dict) -> tuple[str, str]:
    date_str = datetime.now().strftime("%Y-%m-%d")
    subject  = f"[HMM] Daily Report — {date_str} — No Alerts"
    sig_rows = "".join(_sig_row(s) for s in signals.values())

    body = f"""
<html><body style="font-family:monospace;font-size:13px;">
<h2>HMM Daily Signal Report — {date_str}</h2>
<p>No actionable alerts today.</p>

<h3>Signal Table</h3>
<table border="1" cellpadding="4" cellspacing="0">
  <tr>
    <th>Ticker</th><th>Date</th><th>Close</th><th>Regime</th>
    <th>Score</th><th>ADX</th><th>BullProb</th><th>BearProb</th>
    <th>If FLAT</th><th>If LONG</th><th>VT</th>
  </tr>
  {sig_rows}
</table>

<p style="color:gray;font-size:11px;">Generated {datetime.now().isoformat()}</p>
</body></html>"""

    return subject, body


# ── 主流程 ────────────────────────────────────────────────────

def run():
    print(f"\n[monitor] {datetime.now().strftime('%Y-%m-%d %H:%M')} — starting")

    try:
        output  = generate_signals()
        signals = output.get("signals", {})
        errors  = output.get("errors", {})

        if errors:
            subject = f"[HMM] ERROR — {datetime.now().strftime('%Y-%m-%d')}"
            body    = f"<html><body><h3>Signal generation errors:</h3><pre>{errors}</pre></body></html>"
            send_email(subject, body)
            return

        alerts = _check_alerts(signals)

        if alerts:
            subject, body = _build_alert_email(alerts, signals)
        else:
            subject, body = _build_daily_email(signals)

        send_email(subject, body)

        # terminal 摘要
        print(f"[monitor] Alerts: {len(alerts)}")
        for a in alerts:
            print(f"  [{a['level']}] {a['ticker']}: {a['message']}")
        if not alerts:
            print("  No alerts.")

    except Exception:
        tb = traceback.format_exc()
        print(f"[monitor] EXCEPTION:\n{tb}")
        try:
            subject = f"[HMM] CRASH — {datetime.now().strftime('%Y-%m-%d')}"
            body    = f"<html><body><h3>monitor.py crashed:</h3><pre>{tb}</pre></body></html>"
            send_email(subject, body)
        except Exception:
            pass


if __name__ == "__main__":
    run()
