import os
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from data_loader    import fetch_data
from backtester     import run_backtest
from backtester_v2  import run_backtest_v2
from strategy_c     import run_strategy_c
from strategy_d     import run_strategy_d
from strategy_growth import run_strategy_growth, GROWTH_STRATEGY, STRATEGY_LABELS

TICKERS = ["AAPL", "GC=F", "SI=F", "CL=F", "NVDA", "META", "AMZN", "GOOG", "MSFT", "TSLA", "HOOD", "SPY", "FXI", "PLTR", "SOXL"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def safe_filename(ticker: str) -> str:
    return ticker.replace("=", "_").replace("/", "_")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"  处理 {ticker} ...")
        print(f"{'='*50}")

        print(f"  [1/4] 拉取数据...")
        df = fetch_data(ticker, force_refresh=True)
        print(f"        数据量：{len(df)} bars  ({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")

        is_growth = ticker in GROWTH_STRATEGY

        if is_growth:
            strat_label = STRATEGY_LABELS[GROWTH_STRATEGY[ticker]]
            print(f"  [2/4] 策略A（成长股最优：{strat_label}）...")
            res_g = run_strategy_growth(df, ticker)
            # 成长股：把 growth 结果挂在主 result 上，并填充 a/b/c/d 槽位
            # compute_indicators already called inside run_strategy_growth
            from backtester import compute_indicators as _ci
            _full_df = _ci(df.copy(), ticker)
            _full_df["regime_label"] = "N/A"
            _full_df["is_bull"]      = False
            _full_df["signal_score"] = 0
            result = {
                "df":                    _full_df,
                "trades":                res_g["trades"],
                "metrics":               res_g["metrics"],
                "equity_growth":         res_g["equity"],
                "is_growth":             True,
                "growth_strategy_type":  res_g["strategy_type"],
                "growth_strategy_label": res_g["strategy_label"],
                # HMM 字段置空
                "bull_states": [], "bear_state": None,
                "regime_labels": {}, "model": None,
                "is_daily": True, "n_states": 0,
                "bull_top": 0, "min_conf": 0, "stop": 0, "posterior": [],
            }
            mg = res_g["metrics"]
            print(f"        总收益：{mg['total_return_pct']:+.1f}%  夏普：{mg['sharpe']:.2f}  MaxDD：{mg['max_drawdown_pct']:.1f}%  交易：{mg['n_trades']}笔")
        else:
            print(f"  [2/4] 策略A（HMM信号投票）...")
            result = run_backtest(df, ticker)
            m = result["metrics"]
            print(f"        总收益：{m['total_return_pct']:+.1f}%  夏普：{m['sharpe']:.2f}  MaxDD：{m['max_drawdown_pct']:.1f}%  交易：{m['n_trades']}笔")

        print(f"  [3/4] 策略B（Trailing Stop）...")
        try:
            res_b = run_backtest_v2(df, ticker)
            result["equity_b"]  = res_b["equity"]
            result["trades_b"]  = res_b["trades"]
            result["metrics_b"] = res_b["metrics"]
            mb = res_b["metrics"]
            print(f"        总收益：{mb['total_return_pct']:+.1f}%  夏普：{mb['sharpe']:.2f}")
        except Exception as e:
            print(f"        ⚠️ 失败：{e}")
            result["equity_b"] = result["trades_b"] = result["metrics_b"] = None

        print(f"  [3/4] 策略C（EMA趋势跟踪）...")
        try:
            res_c = run_strategy_c(df, ticker)
            result["equity_c"]  = res_c["equity"]
            result["trades_c"]  = res_c["trades"]
            result["metrics_c"] = res_c["metrics"]
            mc = res_c["metrics"]
            print(f"        总收益：{mc['total_return_pct']:+.1f}%  夏普：{mc['sharpe']:.2f}")
        except Exception as e:
            print(f"        ⚠️ 失败：{e}")
            result["equity_c"] = result["trades_c"] = result["metrics_c"] = None

        print(f"  [4/4] 策略D（HMM+布林带）...")
        try:
            res_d = run_strategy_d(df, ticker)
            result["equity_d"]  = res_d["equity"]
            result["trades_d"]  = res_d["trades"]
            result["metrics_d"] = res_d["metrics"]
            md = res_d["metrics"]
            print(f"        总收益：{md['total_return_pct']:+.1f}%  夏普：{md['sharpe']:.2f}")
        except Exception as e:
            print(f"        ⚠️ 失败：{e}")
            result["equity_d"] = result["trades_d"] = result["metrics_d"] = None

        out_path = os.path.join(OUT_DIR, f"{safe_filename(ticker)}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"        已保存 → {out_path}")

    ts_path = os.path.join(OUT_DIR, "computed_at.txt")
    with open(ts_path, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n✅ 全部完成！")


if __name__ == "__main__":
    main()
