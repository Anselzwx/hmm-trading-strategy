#!/bin/bash
# 每日自动更新（cron：北京时间周二至周六 06:30，美股收盘后）
# precompute → signal_generator → 只提交 results/ signals/ 并推送；Streamlit Cloud 自动读到新数据
set -o pipefail
cd "$(dirname "$0")" || exit 1
PY=/opt/anaconda3/bin/python3
exec >> auto_update.log 2>&1
echo "===== $(date '+%F %T') 开始 ====="

LOCK=/tmp/hmm_auto_update.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "上一次更新还在跑，跳过"
    exit 0
fi
trap 'rmdir "$LOCK"' EXIT

$PY precompute.py 2>&1 | grep -v "not converging" | grep -E "处理|失败|⚠️|完成|CAGR|Error|Traceback" || { echo "❌ precompute 失败"; exit 1; }
$PY signal_generator.py 2>&1 | grep -E "Output written|Error|Traceback" || { echo "❌ signal_generator 失败"; exit 1; }

git add results/ signals/
if git diff --cached --quiet; then
    echo "没有数据变化，不提交"
else
    git commit -q -m "Data update $(date +%F)（自动）" && git push -q origin main && echo "✅ 已推送 $(git log --oneline -1)" || echo "❌ 提交/推送失败"
fi
echo "===== $(date '+%F %T') 结束 ====="
