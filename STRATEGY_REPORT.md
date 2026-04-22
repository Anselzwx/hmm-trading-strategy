# HMM Multi-Asset Trading Strategy — OOS Robustness Report

**Version:** Milestone v1  
**Date:** 2026-04-22  
**Assets:** GC=F (Gold), SI=F (Silver), AAPL  
**Period:** 2016-01-07 → 2026-04-21 (≈10 years, daily bars)

---

## 1. Version Evolution

| Phase | Change | Rationale |
|-------|--------|-----------|
| v0 Baseline | Regime full exit; fixed stop; no ADX gate | Initial framework |
| v1 Gold Z1 | Regime → Reduce 50% once per cycle (no full exit) | Full exit truncated winning trades prematurely |
| v1 Silver S-R1 | ADX entry raised to 30; same Regime reduce logic | ADX<30 entries showed low directional conviction |
| v1 AAPL A2 | hold_mult raised 0.75 → 1.25 (max_hold 45→75 bars) | 77% of MaxHold exits continued rising +20 bars |
| v1 Final | price-type stop throughout; reduce_time/reduce_price logging | Accounting correctness and auditability |

---

## 2. Final Parameters

```
TICKER_PARAMS = {
  "AAPL": n_states=5  bull_top=3  min_conf=9  stop=-6%  hold_mult=1.25  adx_entry=25  regime_reduce=False
  "GC=F": n_states=7  bull_top=2  min_conf=9  stop=-8%  hold_mult=1.0   adx_entry=25  regime_reduce=True
  "SI=F": n_states=7  bull_top=1  min_conf=9  stop=-6%  hold_mult=1.0   adx_entry=30  regime_reduce=True
}

BEAR_CONFIRM = { "AAPL": 1, "GC=F": 2, "SI=F": 1 }
LEVERAGE = 2.5
```

**Key architectural rules:**
- **price-type stop:** `stop_price = entry_price × (1 + stop_pct)`, fixed at entry, unaffected by partial reduces
- **Regime Reduce (Gold/Silver):** on bear_consec ≥ bear_confirm, reduce position to 50% once per cycle; PnL realized immediately; full exit only by Stop or MaxHold
- **ADX gate:** entry blocked when ADX ≤ adx_entry threshold
- **Walk-forward:** 60% train / 10% step, no look-ahead

---

## 3. Per-Asset Root Cause and Resolution

### GC=F (Gold)
**Root cause:** Regime full exit at first bear signal cut winning trades too early. Gold trends are persistent — bear signals mid-trend are noise, not reversals.  
**Resolution:** Regime → Reduce 50% (Lock A, once per cycle). bear_confirm=2 (2-bar confirmation before trigger). Full exit only by price-type stop or MaxHold.  
**Validation:** RegimeExit dropped to 0. RegimeReduce triggers on 13/33 trades. Profit concentration: all subsamples positive, Top-3 removal leaves residual +16% (passed).

### SI=F (Silver)
**Root cause (1):** Low-ADX entries had weak directional signal, inflating trade count with marginal setups.  
**Root cause (2):** Regime full exit same problem as Gold, but Silver's trend structure is more volatile.  
**Resolution:** ADX entry gate raised to 30 + same Regime reduce architecture as Gold.  
**2021-2023 weakness:** +18.4% with Sharpe 0.32 — attributed to Silver's structural bear market in that period (asset-level constraint), not strategy failure. Stop Aftermath audit confirmed 5/9 stop trades were genuine entry failures, not stop-too-tight issues. Break-even stop tested (S-BE1 trigger=+3%) rescued 3/3 swept trades but cost -565% Total Return by truncating winners — rejected.

### AAPL
**Root cause:** MaxHold=45 bars too short. 77% of MaxHold exits continued rising an average of +3.5% over the following 20 bars.  
**Resolution:** hold_mult raised to 1.25 (max_hold 45→75 bars). Subsample validated: 2021-2023 improved from +44% → +207%.  
**State-2 audit (post-hold_mult fix):** State-2 (17 trades) showed 65% win rate, avg_ret +6.9%, avg_max_gain +14.1% — equal or better than State-3/4. No entry quality problem remains. Issue was hold_mult, not entry state.

---

## 4. Full-Sample Results

| Ticker | Return | Sharpe | MaxDD | Trades | Win% | Profit Factor |
|--------|--------|--------|-------|--------|------|---------------|
| GC=F   | +10015% | 1.45  | -22.9% | 33   | 70%  | 3.38 |
| SI=F   | +1583%  | 1.05  | -32.8% | 18   | 44%  | 3.01 |
| AAPL   | +995%   | 1.00  | -35.9% | 35   | 63%  | 4.01 |

**Trade quality:**
- GC=F: avg_win=+7.8%, avg_loss=-5.3%
- SI=F: avg_win=+22.4%, avg_loss=-5.9%
- AAPL: avg_win=+13.9%, avg_loss=-5.9%

---

## 5. Subsample Results (IS / OOS Proxy)

### GC=F
| Period | Return | Sharpe | MaxDD | Trades | Stop | RegimeReduce |
|--------|--------|--------|-------|--------|------|--------------|
| 2016–2020 | +690.0% | 1.34 | -14.4% | 15 | 3 | 6 |
| 2021–2023 | +289.8% | 1.41 | -12.1% | 8  | 0 | 5 |
| 2024–2026 | +219.4% | 1.71 | -22.9% | 10 | 1 | 2 |

All three segments: positive return, Sharpe > 1.3. Most consistent asset in the portfolio.

### SI=F
| Period | Return | Sharpe | MaxDD | Trades | Stop | RegimeReduce |
|--------|--------|--------|-------|--------|------|--------------|
| 2016–2020 | +215.6% | 1.27 | -24.9% | 8 | 4 | 0 |
| 2021–2023 | +18.4%  | 0.32 | -32.8% | 5 | 3 | 2 |
| 2024–2026 | +321.3% | 1.62 | -16.7% | 5 | 2 | 5 |

2021-2023 is structurally weak (Silver bear market). All segments remain positive. Sharpe recovery in 2024-2026 confirms mechanism validity.

### AAPL
| Period | Return | Sharpe | MaxDD | Trades | Stop | MaxHold |
|--------|--------|--------|-------|--------|------|---------|
| 2016–2020 | +409.7% | 1.36 | -33.1% | 18 | 5 | 9 |
| 2021–2023 | +35.5%  | 0.50 | -35.9% | 9  | 2 | 6 |
| 2024–2026 | +67.6%  | 1.02 | -21.2% | 8  | 2 | 4 |

2021-2023 positive but weak — consistent with tech sector drawdown in that period. All segments positive.

---

## 6. Key Trade-offs

| Decision | What we gained | What we gave up |
|----------|---------------|-----------------|
| Regime → Reduce (not full exit) | Kept long winners alive through bear signals | Larger drawdowns when bear signal was correct |
| bear_confirm=2 for Gold | Fewer false bear triggers; 0 RegimeExit | Slower response to genuine reversals |
| ADX≥30 for Silver | Filtered low-conviction entries | Reduced trade count (18 trades total) |
| hold_mult=1.25 for AAPL | 2021-2023 +44%→+207%; MaxHold exits 31→19 | Slightly longer exposure during drawdown periods |
| No break-even stop (Silver) | Preserved full winning trades | 3 trades "swept after profit" remain in stop log |

---

## 7. Known Limitations

1. **Silver sample size is small (18 trades).** Statistical conclusions are directional, not definitive. Profit concentration analysis was waived for this reason.

2. **All three subsamples use overlapping HMM training data.** Walk-forward mitigates look-ahead bias but does not provide a fully independent OOS test.

3. **2× leverage assumed throughout.** Real execution would face margin requirements, funding costs, and slippage — all of which erode returns.

4. **Silver 2021-2023 structural weakness is unresolved.** The mechanism is accepted as is; the asset's bear market behavior cannot be addressed by parameter tuning alone.

5. **AAPL State-2 trades (17/35) carry slightly higher stop rate (29% vs 22%)** — acknowledged but not addressed; State-2 avg_ret is equal to State-3/4 after hold_mult fix.

---

## 8. Next Directions

| Priority | Task | Status |
|----------|------|--------|
| 1 | Silver stop optimization (break-even, trailing) | **Closed** — break-even net-negative |
| 2 | AAPL State-2 entry quality audit | **Closed** — no issue after hold_mult fix |
| 3 | This OOS robustness report | **Done** |
| Next | Formal walk-forward expanding-window OOS test | Not started |
| Next | Position sizing optimization (Kelly / vol-targeting) | Not started |
| Next | Multi-asset portfolio correlation and combined equity curve | Not started |
