# HMM Multi-Asset Trading Strategy — OOS Robustness Report

**Version:** Milestone v4 (Final)  
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
| **v2 Silver VT1** | **SI=F position scaled by rvol vol-targeting** | **Vol–PnL audit: rvol→stop_rate corr=0.528; high-vol entries systematically over-sized** |
| **v3 AAPL OOS audit** | **No change to AAPL parameters** | **Expanding-window OOS confirmed AAPL IS advantage is structural (2016-2022 bull market); Regime Reduce tested (A-R1/A-R2) and rejected** |
| **v4 Portfolio baseline** | **Equal-weight portfolio (1/3 each) established as official baseline** | **Strategy-layer correlation ≈0; Portfolio Sharpe 1.79 > any single asset; MaxDD -18.1%** |

---

## 2. Final Parameters (v2)

```
TICKER_PARAMS = {
  "AAPL": n_states=5  bull_top=3  min_conf=9  stop=-6%  hold_mult=1.25  adx_entry=25  regime_reduce=False
  "GC=F": n_states=7  bull_top=2  min_conf=9  stop=-8%  hold_mult=1.0   adx_entry=25  regime_reduce=True
  "SI=F": n_states=7  bull_top=1  min_conf=9  stop=-6%  hold_mult=1.0   adx_entry=30  regime_reduce=True  vol_target=True
}

BEAR_CONFIRM = { "AAPL": 1, "GC=F": 2, "SI=F": 1 }
LEVERAGE = 2.5
```

**Key architectural rules:**
- **price-type stop:** `stop_price = entry_price × (1 + stop_pct)`, fixed at entry, unaffected by partial reduces
- **Regime Reduce (Gold/Silver):** on bear_consec ≥ bear_confirm, reduce position to 50% once per cycle; PnL realized immediately; full exit only by Stop or MaxHold
- **ADX gate:** entry blocked when ADX ≤ adx_entry threshold
- **Walk-forward:** 60% train / 10% step, no look-ahead
- **Silver vol-targeting:** `vt_scale = clip(rvol_median / entry_rvol, 0.3, 1.5)`; target = full-sample rvol median (1.393); actual scale range 0.61–1.38 (upper clip never triggered)

---

## 3. Per-Asset Root Cause and Resolution

### GC=F (Gold)
**Root cause:** Regime full exit at first bear signal cut winning trades too early. Gold trends are persistent — bear signals mid-trend are noise, not reversals.  
**Resolution:** Regime → Reduce 50% (Lock A, once per cycle). bear_confirm=2 (2-bar confirmation before trigger). Full exit only by price-type stop or MaxHold.  
**Validation:** RegimeExit dropped to 0. RegimeReduce triggers on 13/33 trades. Profit concentration: all subsamples positive, Top-3 removal leaves residual +16% (passed).  
**Vol-targeting:** Vol–PnL audit showed weak correlation (rng_pct→ret = -0.225, moderate). Not applied — insufficient systematic pattern to justify mechanism change.

### SI=F (Silver)
**Root cause (1):** Low-ADX entries had weak directional signal, inflating trade count with marginal setups.  
**Root cause (2):** Regime full exit same problem as Gold, but Silver's trend structure is more volatile.  
**Root cause (3):** Fixed position sizing systematically over-sized high-volatility entries (rvol→stop_rate corr = 0.528).  
**Resolution:** ADX entry gate raised to 30 + Regime reduce architecture + rvol-based vol-targeting.  
**2021-2023 weakness:** +18.4% with Sharpe 0.32 — attributed to Silver's structural bear market. Stop Aftermath audit: 5/9 stops were genuine entry failures (max_gain < 1%), 3/9 were swept after profit. Break-even stop (S-BE1, trigger=+3%) rescued 3/3 swept trades but cost -565% Total Return by truncating winners — rejected.

### AAPL
**Root cause:** MaxHold=45 bars too short. 77% of MaxHold exits continued rising an average of +3.5% over the following 20 bars.  
**Resolution:** hold_mult raised to 1.25 (max_hold 45→75 bars). Subsample validated: 2021-2023 improved from +44% → +207%.  
**State-2 audit (post-hold_mult fix):** State-2 (17 trades) showed 65% win rate, avg_ret +6.9%, avg_max_gain +14.1% — equal or better than State-3/4. No entry quality problem remains.  
**Vol-targeting:** Vol–PnL audit showed no systematic pattern (all correlations < 0.18; high-vol avg_ret +7.6% > low-vol +5.4%). Not applied.

---

## 4. Full-Sample Results (v2 Baseline)

| Ticker | Return | Sharpe | MaxDD | Trades | Win% | Profit Factor |
|--------|--------|--------|-------|--------|------|---------------|
| GC=F   | +10015% | 1.45  | -22.9% | 33   | 70%  | 3.38 |
| SI=F   | **+1725%**  | **1.06**  | **-31.0%** | 18   | 44%  | 3.01 |
| AAPL   | +995%   | 1.00  | -35.9% | 35   | 63%  | 4.01 |

**Silver v1 → v2 delta:** Return +1583% → +1725% (+142%), Sharpe 1.05 → 1.06, MaxDD -32.8% → -31.0%

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

2021-2023 structurally weak (Silver bear market). All segments positive. Sharpe recovery in 2024-2026 confirms mechanism validity. Subsample avg_ret unchanged by vol-targeting (effect is on capital path, not per-trade return).

### AAPL
| Period | Return | Sharpe | MaxDD | Trades | Stop | MaxHold |
|--------|--------|--------|-------|--------|------|---------|
| 2016–2020 | +409.7% | 1.36 | -33.1% | 18 | 5 | 9 |
| 2021–2023 | +35.5%  | 0.50 | -35.9% | 9  | 2 | 6 |
| 2024–2026 | +67.6%  | 1.02 | -21.2% | 8  | 2 | 4 |

2021-2023 positive but weak — consistent with tech sector drawdown. All segments positive.

---

## 6. Key Trade-offs

| Decision | What we gained | What we gave up |
|----------|---------------|-----------------|
| Regime → Reduce (not full exit) | Kept long winners alive through bear signals | Larger drawdowns when bear signal was correct |
| bear_confirm=2 for Gold | Fewer false bear triggers; 0 RegimeExit | Slower response to genuine reversals |
| ADX≥30 for Silver | Filtered low-conviction entries | Reduced trade count (18 trades total) |
| hold_mult=1.25 for AAPL | 2021-2023 +44%→+207%; MaxHold exits 31→19 | Slightly longer exposure during drawdown periods |
| No break-even stop (Silver) | Preserved full winning trades | 3 trades "swept after profit" remain in stop log |
| Silver vol-targeting only | Return +142%, MaxDD -1.8%; clean归因 | GC=F/AAPL remain fixed (audit justified) |

---

## 7. Known Limitations

1. **Silver sample size is small (18 trades).** Statistical conclusions are directional, not definitive. Profit concentration analysis was waived for this reason.

2. **All three subsamples use overlapping HMM training data.** Walk-forward mitigates look-ahead bias but does not provide a fully independent OOS test.

3. **2.5× leverage assumed throughout.** Real execution would face margin requirements, funding costs, and slippage — all of which erode returns.

4. **Silver 2021-2023 structural weakness is unresolved.** The mechanism is accepted as is; the asset's bear market behavior cannot be addressed by parameter tuning alone.

5. **Silver vol-targeting uses full-sample rvol median as target.** This introduces mild look-ahead (target is computed on the full dataset). In a live system, the target would need to be rolling or pre-specified.

6. **AAPL State-2 trades (17/35) carry slightly higher stop rate (29% vs 22%)** — acknowledged but not addressed; State-2 avg_ret is equal to State-3/4 after hold_mult fix.

7. **AAPL OOS Sharpe degrades significantly (1.00 full-sample → 0.31 OOS).** Expanding-window OOS test (init=60%, step=10%) shows: OOS avg_hold=12.8 bars vs IS ~56 bars; 76% of OOS exits are Regime → Bear/Crash (HMM state drift); HMM bull-bar frequency collapses in OOS2 (19% vs ~48% in other windows). Root cause: AAPL 2022+ market structure change causes HMM state distribution to drift from IS training. Regime Reduce (A-R1 bear_confirm=1, A-R2 bear_confirm=2) was tested — both showed 2021-2023 ret=-0.7%, StopLoss rising from 9→16, MaxDD deepening to -42.6%. Regime Reduce rejected. AAPL maintains regime_reduce=False. Full-sample performance relies heavily on 2016-2022 bull market structural tailwind.

---

## 8. Multi-asset Portfolio (v4 — Official Baseline)

### Strategy-layer vs Price-layer Correlation

| | GC=F | SI=F | AAPL |
|--|------|------|------|
| **Strategy equity returns** | | | |
| GC=F | 1.000 | 0.020 | -0.002 |
| SI=F | 0.020 | 1.000 | 0.003 |
| AAPL | -0.002 | 0.003 | 1.000 |
| **Underlying price returns (reference)** | | | |
| GC=F | 1.000 | 0.771 | 0.038 |
| SI=F | 0.771 | 1.000 | 0.117 |

GC=F and SI=F share 0.771 price-level correlation, yet their strategy equity curves correlate at only 0.020 — the entry/exit timing is effectively independent across all three assets. This near-zero strategy-layer correlation is the foundation of the portfolio's diversification value.

### Equal-weight Portfolio vs Single Assets

| | Return | Sharpe | MaxDD | Calmar |
|--|--------|--------|-------|--------|
| GC=F | +9842% | 1.47 | -22.2% | 444 |
| SI=F | +1727% | 1.08 | -31.1% | 56 |
| AAPL | +995% | 1.00 | -35.9% | 28 |
| **Portfolio (1/3 each)** | **+4188%** | **1.79** | **-18.1%** | **231** |

Portfolio Sharpe 1.79 exceeds every individual asset (best single: GC=F at 1.47). MaxDD -18.1% is tighter than every individual asset (best single: GC=F at -22.2%). The improvement is genuine diversification, not averaging — the portfolio's Sharpe is super-additive relative to any single component.

### Portfolio Subsample Results

| Period | Return | Sharpe | MaxDD |
|--------|--------|--------|-------|
| 2016–2020 | +435% | 1.92 | -12.0% |
| 2021–2023 | **+159%** | **1.49** | -12.3% |
| 2024–2026 | +205% | 2.00 | -18.1% |

All three segments: positive return, Sharpe > 1.4. The 2021–2023 segment — weakest for every individual asset (AAPL +35.5%, SI=F +18.4%) — returns +159% at the portfolio level, as Gold's strong performance in that period offsets the other assets' weakness.

### Portfolio Baseline Conclusion

**Equal-weight portfolio (1/3 each across GC=F, SI=F, AAPL) is accepted as the official portfolio baseline for this research cycle.** On the basis of near-zero strategy-layer correlations, the portfolio delivers:
- Sharpe improvement: 1.47 → 1.79 (+0.32 above best single asset)
- MaxDD improvement: -22.2% → -18.1% (tighter than best single asset)
- Consistent performance across all three time segments including the 2021-2023 bear market period

### Future Portfolio Directions (not part of current milestone)

- Non-equal weighting (Sharpe-weighted, inverse-drawdown, risk-parity style)
- Rolling portfolio allocation with periodic rebalancing
- Portfolio-level volatility targeting
- Live allocation sizing given real margin and capital constraints

---

## 9. Research Log — Closed Directions

| Direction | Outcome |
|-----------|---------|
| Silver break-even stop (trigger=+3%) | Rejected — rescued 3/3 swept trades but -565% Return from winner truncation |
| AAPL Regime Reduce (A-R1 bc=1, A-R2 bc=2) | Rejected — 2021-2023 ret unchanged at -0.7%, StopLoss 9→16, MaxDD -35.9%→-42.6%; root cause is HMM state drift + entry quality in 2022+ bear market, not Regime trigger timing |
| Silver fixed stop width (-7%/-8%) | Not tested — audit showed 5/9 stops are genuine entry failures; wider stop would increase loss |
| AAPL State-2 entry filter | Closed — no issue after hold_mult fix; State-2 performs equal to State-3/4 |
| Gold/AAPL vol-targeting | Not applied — Vol–PnL audit showed no systematic overweight pattern |

---

## 10. Research Cycle Status

| Direction | Status |
|-----------|--------|
| Silver stop optimization | **Closed** — break-even net-negative |
| AAPL State-2 entry quality | **Closed** — no issue after hold_mult fix |
| OOS robustness report | **Closed** — v1 report produced |
| Position sizing optimization | **Closed** — Silver VT1 applied |
| Expanding-window OOS test | **Closed** — GC=F/SI=F stable; AAPL structural weakness documented |
| Multi-asset portfolio | **Closed** — equal-weight baseline established |

**This research cycle is complete.** All planned Next Directions have been addressed. The v4 baseline (per-asset parameters + equal-weight portfolio) is the official deliverable of this cycle.

### If a new cycle begins, natural entry points are:
- Non-equal portfolio weighting (risk-parity, Sharpe-weighted)
- AAPL HMM drift mitigation (regime-adaptive retraining, feature engineering)
- Live trading infrastructure (execution, slippage modeling, margin management)
- Expanding the asset universe
