"""
precompute.py
-------------
预先计算三个资产的回测结果并保存到 results/ 目录。
在本地跑完后把 results/ 一起推送到 GitHub，
Streamlit Cloud 启动时直接读取，秒开。

运行方式：
    python precompute.py
"""

import os
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from data_loader import fetch_data
from backtester  import run_backtest

TICKERS   = ["AAPL", "GC=F", "SI=F"]
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")


def safe_filename(ticker: str) -> str:
    return ticker.replace("=", "_").replace("/", "_")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"  处理 {ticker} ...")
        print(f"{'='*50}")

        print(f"  [1/2] 拉取数据...")
        df = fetch_data(ticker, force_refresh=True)
        print(f"        数据量：{len(df)} bars  ({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")

        print(f"  [2/2] Walk-Forward 回测（约 30-60s）...")
        result = run_backtest(df, ticker)

        m = result["metrics"]
        print(f"        总收益：{m['total_return_pct']:+.1f}%  "
              f"夏普：{m['sharpe']:.2f}  "
              f"最大回撤：{m['max_drawdown_pct']:.1f}%  "
              f"交易：{m['n_trades']} 笔")

        # 保存
        out_path = os.path.join(OUT_DIR, f"{safe_filename(ticker)}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"        已保存 → {out_path}")

    # 写一个时间戳文件，让 app 显示"数据更新时间"
    ts_path = os.path.join(OUT_DIR, "computed_at.txt")
    with open(ts_path, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n✅ 全部完成！结果保存在 results/ 目录")
    print(f"   下一步：git add results/ && git commit -m 'update results' && git push")


if __name__ == "__main__":
    main()
