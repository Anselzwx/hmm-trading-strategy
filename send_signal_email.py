"""
send_signal_email.py — 每日 HMM 信号邮件推送
=============================================
用法：
    python send_signal_email.py

功能：
    1. 调用 signal_generator.py 生成当日所有资产信号
    2. 将信号格式化为 HTML 报告
    3. 通过 Gmail SMTP 发送到 zhao.wenxu@northeastern.edu

环境变量：
    HMM_SENDER_EMAIL    — 发件人 Gmail 地址（需开启应用密码）
    HMM_SENDER_PASSWORD — Gmail 应用密码（16位，非账户密码）

示例（macOS/Linux）：
    export HMM_SENDER_EMAIL="your.sender@gmail.com"
    export HMM_SENDER_PASSWORD="xxxx xxxx xxxx xxxx"
    python send_signal_email.py

定时任务（每个交易日 22:00 执行）：
    crontab -e
    0 22 * * 1-5 cd /Users/zhaowenxuan/Desktop/工作/HMM && python send_signal_email.py
"""

from __future__ import annotations

import os
import smtplib
import warnings
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")

from signal_generator import generate_signal
from data_loader import ASSET_LABELS

RECIPIENT    = "anselwilliam789@gmail.com"
TICKERS      = ["AAPL", "GC=F", "SI=F", "CL=F",
                 "NVDA", "META", "AMZN", "GOOG",
                 "MSFT", "TSLA", "HOOD", "SPY", "FXI", "PLTR"]

# ── 颜色映射 ──────────────────────────────────────────────────

ACTION_COLOR = {
    "ENTER":    "#22c55e",   # 绿
    "HOLD":     "#3b82f6",   # 蓝
    "EXIT":     "#ef4444",   # 红
    "WATCH":    "#f59e0b",   # 橙
    "STAY_OUT": "#6b7280",   # 灰
}
ACTION_EMOJI = {
    "ENTER":    "🟢",
    "HOLD":     "🔵",
    "EXIT":     "🔴",
    "WATCH":    "🟡",
    "STAY_OUT": "⚫",
}


def _row_html(sig: dict) -> str:
    action = sig["action_if_flat"]
    color  = ACTION_COLOR.get(action, "#6b7280")
    emoji  = ACTION_EMOJI.get(action, "")
    label  = ASSET_LABELS.get(sig["ticker"], sig["ticker"])
    stop   = sig["stop_pct"] * 100

    return f"""
    <tr>
      <td style="padding:8px 12px;font-weight:600;">{label}</td>
      <td style="padding:8px 12px;">{sig['close']:.4f}</td>
      <td style="padding:8px 12px;">{sig['regime']}</td>
      <td style="padding:8px 12px;text-align:center;">{sig['signal_score']}/4</td>
      <td style="padding:8px 12px;text-align:center;">{sig['bull_prob']:.0%}</td>
      <td style="padding:8px 12px;text-align:center;">{sig['bear_prob']:.0%}</td>
      <td style="padding:8px 12px;text-align:center;">{sig['sideways_score']}</td>
      <td style="padding:8px 12px;font-weight:700;color:{color};">{emoji} {action}</td>
      <td style="padding:8px 12px;color:#ef4444;">{stop:+.1f}%</td>
    </tr>"""


def _error_row(ticker: str, err: str) -> str:
    label = ASSET_LABELS.get(ticker, ticker)
    return f"""
    <tr style="background:#fef2f2;">
      <td style="padding:8px 12px;font-weight:600;">{label}</td>
      <td colspan="8" style="padding:8px 12px;color:#ef4444;">⚠ {err}</td>
    </tr>"""


def build_html(signals: dict, errors: dict, generated_at: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = ""
    for ticker in TICKERS:
        if ticker in signals:
            rows += _row_html(signals[ticker])
        elif ticker in errors:
            rows += _error_row(ticker, errors[ticker])

    # ENTER/EXIT 摘要
    enter_list = [ASSET_LABELS.get(t, t) for t, s in signals.items()
                  if s["action_if_flat"] == "ENTER"]
    exit_list  = [ASSET_LABELS.get(t, t) for t, s in signals.items()
                  if s["action_if_flat"] == "EXIT"]

    summary_html = ""
    if enter_list:
        summary_html += f"""
        <div style="background:#f0fdf4;border-left:4px solid #22c55e;padding:10px 16px;margin-bottom:8px;">
          <strong>🟢 建议开仓：</strong> {', '.join(enter_list)}
        </div>"""
    if exit_list:
        summary_html += f"""
        <div style="background:#fef2f2;border-left:4px solid #ef4444;padding:10px 16px;margin-bottom:8px;">
          <strong>🔴 建议平仓：</strong> {', '.join(exit_list)}
        </div>"""
    if not enter_list and not exit_list:
        summary_html = """
        <div style="background:#f9fafb;border-left:4px solid #6b7280;padding:10px 16px;margin-bottom:8px;">
          ⚫ 当日无明确入场/出场信号，保持观望。
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size:14px; color:#111; background:#fff; margin:0; padding:20px; }}
  h1 {{ font-size:20px; margin-bottom:4px; }}
  p.sub {{ color:#6b7280; margin:0 0 16px; }}
  table {{ border-collapse:collapse; width:100%; margin-top:16px; }}
  th {{ background:#1e293b; color:#fff; padding:8px 12px; text-align:left; font-size:12px; }}
  tr:nth-child(even) {{ background:#f8fafc; }}
  .footer {{ margin-top:20px; color:#9ca3af; font-size:12px; }}
</style>
</head>
<body>
  <h1>📊 HMM 每日交易信号报告</h1>
  <p class="sub">{date_str} &nbsp;|&nbsp; 生成于 {generated_at}</p>

  {summary_html}

  <table>
    <thead>
      <tr>
        <th>资产</th>
        <th>最新价</th>
        <th>HMM 状态</th>
        <th>信号分</th>
        <th>看多概率</th>
        <th>看空概率</th>
        <th>横盘评分</th>
        <th>操作建议</th>
        <th>止损位</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <div class="footer">
    <p>* 操作建议基于 HMM Walk-Forward 信号 + 200MA 过滤 + 财报日封锁。</p>
    <p>* 信号仅供参考，不构成投资建议。实盘操作请结合风险管理。</p>
    <p>* 查看完整分析：<a href="https://hmm-trading.streamlit.app/">hmm-trading.streamlit.app</a></p>
  </div>
</body>
</html>"""


def send_email(html_body: str, subject: str) -> None:
    sender   = os.environ.get("HMM_SENDER_EMAIL")
    password = os.environ.get("HMM_SENDER_PASSWORD")

    if not sender or not password:
        raise RuntimeError(
            "请设置环境变量 HMM_SENDER_EMAIL 和 HMM_SENDER_PASSWORD\n"
            "  export HMM_SENDER_EMAIL='your.sender@gmail.com'\n"
            "  export HMM_SENDER_PASSWORD='xxxx xxxx xxxx xxxx'  # Gmail 应用密码"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = RECIPIENT
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, RECIPIENT, msg.as_string())


def run():
    print(f"\n{'='*60}")
    print(f"  HMM 每日信号邮件  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    signals, errors = {}, {}
    for ticker in TICKERS:
        print(f"  [{ticker:6s}] 生成信号...", end=" ", flush=True)
        try:
            signals[ticker] = generate_signal(ticker)
            action = signals[ticker]["action_if_flat"]
            print(f"✓  {action}")
        except Exception as e:
            errors[ticker] = str(e)
            print(f"✗  {e}")

    generated_at = datetime.now().isoformat()
    html = build_html(signals, errors, generated_at)

    date_str = datetime.now().strftime("%Y-%m-%d")
    subject  = f"📊 HMM 信号报告 {date_str}"

    print(f"\n  发送邮件 → {RECIPIENT} ...", end=" ", flush=True)
    try:
        send_email(html, subject)
        print("✓ 已发送")
    except Exception as e:
        print(f"✗ 失败：{e}")
        # 降级：将 HTML 保存到本地
        fallback_path = os.path.join(
            os.path.dirname(__file__), "signals",
            f"email_{date_str}.html"
        )
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML 已保存 → {fallback_path}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
