# Backtest and Monte Carlo — multi-strategy Quant sleeve (generated 2026-10-24)

**Window:** 2023-10-23 → 2026-10-22 (3 years, 2 further years of warm-up for the 252-day signals).
**What was tested:** only the rule-based parts of the fund (strategies, walk-forward IC weights, ≥3-votes entry gate, vol-managed
sizing, the real `plan_orders` with the 5-trades/day cap, the A16 drift band, trim pairing, A7/A12, the −10% stop and the −8% trailing stop).
LLM agents are excluded on purpose: a language model has read the future, so any LLM backtest leaks look-ahead.
**Fills:** next open after the signal close · **costs:** 0 (A20) except row H · **risk-free rate:** 0 · stocks only.

## Headline

Every number below is the **mean over six staggered start dates** (the same strategy started in six consecutive weeks), with the Sharpe range
shown. That range is the noise floor: the same rules gave Sharpe 0.73 to 1.54 for A depending only on which Monday the fund started, and up to
0.86 wide for other variants. **Differences smaller than that are not evidence of anything.**

Quant as deployed (A) averaged **17.3% CAGR, Sharpe 1.06, max drawdown -17.0%**
against SPY 22.4% / 1.41 / -18.8%. It did **not** beat SPY on Sharpe over this window
(first start: 1126 fills, 278 forced exits, 25 drift trims; 25 of them already had a new-name buy in the same plan,
0 needed a paired buy).
Ensemble-only (B) averaged Sharpe 1.35, the legacy Quant score (D) 0.84, the as-deployed blend (A) 1.06. The start-date ranges do not overlap, which is the only kind of gap worth noticing. A sits below B: the legacy half of the blend may be diluting the ensemble, which makes `ENSEMBLE['blend']` the one parameter worth a look (not tuned here; tuning it on this window would be curve-fitting).
Three years of one mostly bullish regime is far too little to separate skill from luck; read the rankings below as direction, not proof.

![equity](equity_drawdown.png)

## 0. Old vs new settings (Oct 24)

N0 = the Oct 23 deployment (IC-weighted strategies, Quant 75% ensemble, −8% trailing stop). N1 = the new settings: shrinkage weights
(equal prior + λ·max(t, 0), never all-zero, PEAD cap 30%), Quant = 100% ensemble, −12% trailing stop. N2 = N1 plus the regime allocator
(family weights × SPY-state multipliers, renormalised). Same six staggered starts, same rule-based sleeve, next-open fills, zero cost.

| Variant | CAGR | Vol | Sharpe (range over 6 starts) | Max DD | Turnover/yr |
|---|---:|---:|---:|---:|---:|
| N0. OLD deployed (Oct 23: IC weights, blend 0.75, -8% trail) | 17.3% | 16.4% | 1.06 (0.73–1.54) | -17.0% | 12.3x |
| N1. NEW (shrinkage, 100% ensemble, -12% trail), no regime allocator | 16.5% | 15.1% | 1.09 (0.57–1.43) | -18.5% | 7.9x |
| N2. NEW + regime allocator | 15.0% | 14.6% | 1.03 (0.87–1.25) | -20.2% | 7.4x |
| SPY (buy & hold) | 22.4% | 15.2% | 1.41 (1.33–1.50) | -18.8% | – |
| RSP (buy & hold) | 17.9% | 13.8% | 1.27 (1.19–1.36) | -17.8% | – |
| MTUM (buy & hold) | 27.3% | 23.2% | 1.16 (1.11–1.24) | -21.0% | – |

**These settings were chosen after the first backtest, on the same 2023–26 window: they are in-sample, not significant, and the Sharpe
range over the six starts is the noise floor. Do not read N1/N2 > N0 as skill.** The 15-day live race cannot tell them apart either.

Regime state per week (read at Friday close; 156 weeks; calm_uptrend 81 wks (52%), drawdown_high_vol 16 wks (10%), rebound 0 wks (0%), normal 59 wks (38%)). Week-by-week CSV: `regime_weekly.csv`.

| State at Friday close | weeks | N0 avg weekly return | N1 avg weekly return | N2 avg weekly return |
|---|---:|---:|---:|---:|
| calm_uptrend | 81 | 0.22% | 0.12% | 0.16% |
| drawdown_high_vol | 16 | 0.20% | 0.21% | 0.19% |
| rebound | 0 | n/a | n/a | n/a |
| normal | 59 | 0.55% | 0.65% | 0.52% |

- 2023-10-27 → 2023-11-17: **normal** (4 wk)
- 2023-11-24 → 2024-02-09: **calm_uptrend** (12 wk)
- 2024-02-16 → 2024-03-08: **normal** (4 wk)
- 2024-03-15 → 2024-04-19: **calm_uptrend** (6 wk)
- 2024-04-26 → 2024-05-10: **normal** (3 wk)
- 2024-05-17 → 2024-07-19: **calm_uptrend** (10 wk)
- 2024-07-26 → 2024-07-26: **normal** (1 wk)
- 2024-08-02 → 2024-08-09: **drawdown_high_vol** (2 wk)
- 2024-08-16 → 2024-09-27: **normal** (7 wk)
- 2024-10-04 → 2024-11-01: **calm_uptrend** (5 wk)
- 2024-11-08 → 2024-11-29: **normal** (4 wk)
- 2024-12-06 → 2024-12-13: **calm_uptrend** (2 wk)
- 2024-12-20 → 2025-02-07: **normal** (8 wk)
- 2025-02-14 → 2025-02-21: **calm_uptrend** (2 wk)
- 2025-02-28 → 2025-02-28: **normal** (1 wk)
- 2025-03-07 → 2025-05-09: **drawdown_high_vol** (10 wk)
- 2025-05-16 → 2025-05-16: **normal** (1 wk)
- 2025-05-23 → 2025-05-23: **drawdown_high_vol** (1 wk)
- 2025-05-30 → 2025-06-06: **normal** (2 wk)
- 2025-06-13 → 2025-10-10: **calm_uptrend** (18 wk)
- 2025-10-17 → 2025-10-31: **normal** (3 wk)
- 2025-11-07 → 2025-11-14: **calm_uptrend** (2 wk)
- 2025-11-21 → 2025-12-05: **normal** (3 wk)
- 2025-12-12 → 2026-01-30: **calm_uptrend** (8 wk)
- 2026-02-06 → 2026-02-13: **normal** (2 wk)
- 2026-02-20 → 2026-02-20: **calm_uptrend** (1 wk)
- 2026-02-27 → 2026-03-06: **normal** (2 wk)
- 2026-03-13 → 2026-03-13: **calm_uptrend** (1 wk)
- 2026-03-20 → 2026-04-02: **drawdown_high_vol** (3 wk)
- 2026-04-10 → 2026-04-24: **normal** (3 wk)
- 2026-05-01 → 2026-05-29: **calm_uptrend** (5 wk)
- 2026-06-05 → 2026-07-17: **normal** (7 wk)
- 2026-07-24 → 2026-07-24: **calm_uptrend** (1 wk)
- 2026-07-31 → 2026-08-21: **normal** (4 wk)
- 2026-08-28 → 2026-10-16: **calm_uptrend** (8 wk)


## 1. Variants vs benchmarks

| Variant | CAGR | Vol | Sharpe (range over 6 starts) | Max DD | Turnover/yr |
|---|---:|---:|---:|---:|---:|
| A. Quant as deployed (legacy + ensemble), weekly | 17.3% | 16.4% | 1.06 (0.73–1.54) | -17.0% | 12.3x |
| B. Ensemble only (IC-weighted) | 18.9% | 13.7% | 1.35 (1.13–1.58) | -13.2% | 8.2x |
| C. Ensemble only, equal-weight strategies | 17.7% | 14.4% | 1.21 (0.90–1.63) | -17.5% | 10.7x |
| D. Legacy technical score only (no ensemble) | 15.1% | 18.9% | 0.84 (0.68–1.10) | -18.9% | 13.5x |
| E. A, daily rebalance | 20.0% | 18.4% | 1.09 (0.93–1.39) | -19.2% | 23.4x |
| F. A without the >=3-votes gate | 20.1% | 16.4% | 1.20 (0.82–1.36) | -16.3% | 11.3x |
| G. A without the vol-managed overlay | 19.4% | 19.8% | 0.99 (0.80–1.30) | -23.3% | 12.7x |
| H. A with 5 bp cost per side | 17.4% | 16.5% | 1.06 (0.89–1.22) | -17.8% | 12.1x |
| SPY (buy & hold) | 22.4% | 15.2% | 1.41 (1.33–1.50) | -18.8% | – |
| MTUM (buy & hold) | 27.3% | 23.2% | 1.16 (1.11–1.24) | -21.0% | – |
| RSP (buy & hold) | 17.9% | 13.8% | 1.27 (1.19–1.36) | -17.8% | – |

A = the OCTOBER 23 Quant agent before the LLM sees it (`rule_score` = legacy technical score blended with the ensemble by
0.75 × the allocated ensemble weight; IC-weighted strategies, −8% trailing stop). The Oct 24 settings are in section 0. B/C isolate the ensemble. D is the old Quant score. E–H test one design choice at a time.
Turnover is one-way traded value per year ÷ average NAV; MTUM and RSP are buy-and-hold ETFs on the same open-to-close basis.

## 2. Profit-taking without cash and the trailing stop (B1 / B2 ablation, on A)

| Variant | CAGR | Vol | Sharpe (range over 6 starts) | Max DD | Turnover/yr |
|---|---:|---:|---:|---:|---:|
| P1. A: trim pairing ON,  trailing stop ON | 17.3% | 16.4% | 1.06 (0.73–1.54) | -17.0% | 12.3x |
| P2. A: trim pairing OFF, trailing stop ON | 17.3% | 16.4% | 1.06 (0.73–1.54) | -17.0% | 12.3x |
| P3. A: trim pairing ON,  trailing stop OFF | 21.1% | 17.4% | 1.19 (0.88–1.35) | -20.8% | 8.8x |
| P4. A: trim pairing OFF, trailing stop OFF (−10% stop only) | 21.1% | 17.4% | 1.19 (0.88–1.35) | -20.8% | 8.8x |
| P5. A: daily, trim pairing ON,  trailing ON | 20.0% | 18.4% | 1.09 (0.93–1.39) | -19.2% | 23.4x |
| P6. A: daily, trim pairing OFF, trailing ON | 20.0% | 18.4% | 1.09 (0.93–1.39) | -19.2% | 23.4x |
| P7. A: daily, trim pairing ON,  trailing OFF | 24.1% | 19.8% | 1.20 (1.09–1.36) | -21.7% | 24.3x |
| P8. A: daily, trim pairing OFF, trailing OFF | 24.1% | 19.8% | 1.20 (1.09–1.36) | -21.7% | 24.3x |

Counts in P1 (weekly, first start): 25 drift trims (25 already redeployed by the plan's own new-name buys), 0 paired buys,
278 forced exits; P4 (stop only): 75 forced exits.
The trailing stop forces roughly the extra exits you see in that difference, which is where most of the turnover comes from.

## 3. Each strategy on its own

Long-only top-15 by one strategy's sector-neutral z, equal weight, weekly, next-open fills, no costs. IC is the cross-sectional
Spearman correlation of z with the next 5-day return, daily; the t-stat uses n/5 observations because 5-day windows overlap.
**Treat the standalone CAGRs as inflated:** the universe is today's S&P 500, which contains names admitted *because* they rallied
(survivorship / index-inclusion bias hits momentum hardest). The fully constrained fund simulation in section 1 earns far less.

| Strategy | Standalone Sharpe | CAGR | Mean IC | IC t | Turnover/yr |
|---|---:|---:|---:|---:|---:|
| xs_momentum_12_1 | 1.95 | 94.0% | 0.0220 | 1.60 | 3.6x |
| ts_trend | 2.03 | 67.7% | 0.0192 | 1.34 | 4.2x |
| industry_momentum | 1.42 | 28.2% | 0.0002 | 0.03 | 6.3x |
| residual_momentum | 1.94 | 54.3% | 0.0244 | 1.71 | 4.3x |
| high_52w | 1.66 | 27.5% | -0.0012 | -0.09 | 12.3x |
| st_reversal_1w | 1.25 | 34.9% | 0.0061 | 0.58 | 23.6x |
| pead_sue | 1.66 | 42.7% | 0.0149 | 2.98 | 6.1x |
| earnings_premium | 1.39 | 29.9% | -0.0098 | -1.09 | 20.8x |
| low_beta | 1.48 | 19.6% | -0.0277 | -1.37 | 1.5x |
| high_volume_premium | 1.56 | 30.3% | -0.0050 | -0.91 | 21.5x |
| sector_rotation | 0.67 | 13.5% | -0.0020 | -0.14 | 6.0x |

![strategies](strategies.png)

Not backtestable (no point-in-time history): `eps_revisions`, `quality_gp` (gross-margin/ROA proxy), `insider_buying`, `news_surprise`
(only today's snapshot exists; FinBERT/news history does not). `turn_of_month` is a tilt, tested as a market-timing statistic:
SPY averaged 8.5 bp/day inside the 136-day window versus 8.9 bp outside (t = -0.05).
`etf_trend_200dma` is tested on the 14 leveraged long ETFs, equal weight, in/out of the 200dma:

| Leveraged-ETF filter | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|
| leveraged ETFs, 200dma filter | 1.02 | 39.9% | -39.7% |
| leveraged ETFs, always held | 1.21 | 58.5% | -53.8% |

## 4. Ablation: drop one strategy (ensemble-only variant B, base Sharpe 1.35)

| Dropped strategy | Sharpe w/o it | Δ Sharpe | CAGR | Max DD |
|---|---:|---:|---:|---:|
| xs_momentum_12_1 | 1.34 | -0.00 | 18.3% | -14.8% |
| ts_trend | 1.14 | -0.21 | 15.2% | -13.4% |
| industry_momentum | 1.33 | -0.01 | 20.2% | -18.3% |
| residual_momentum | 1.16 | -0.19 | 16.2% | -13.1% |
| high_52w | 1.12 | -0.22 | 16.3% | -17.3% |
| st_reversal_1w | 1.31 | -0.04 | 20.7% | -17.7% |
| pead_sue | 1.30 | -0.04 | 19.1% | -16.1% |
| earnings_premium | 0.86 | -0.49 | 12.3% | -18.0% |
| low_beta | 0.90 | -0.44 | 15.0% | -22.3% |
| high_volume_premium | 1.07 | -0.27 | 16.6% | -19.2% |
| sector_rotation | 1.37 | +0.03 | 20.8% | -21.0% |

With 11 strategies and a 3-year window most Δ Sharpe values are inside the noise of the test; a drop that *raises* Sharpe is a
reason to look, not to delete. Ensemble weights on the last day: none cleared t ≥ 1.

![weights](weights.png)

## 5. Monte Carlo — 15 trading days, 10,000 block-bootstrap paths

Blocks of 5 days drawn jointly from the fund's and SPY's daily returns (their correlation is kept); 3 blocks per path.
Drawdown is measured inside the 15-day path from its own running peak.

| Path source | P(beat SPY) | P(DD ≤ −7%) | P(DD ≤ −10%) | P(loss) | 15d return p5/p50/p95 | Excess p5/p50/p95 |
|---|---:|---:|---:|---:|---:|---:|
| A. Quant as deployed | 47% | 5% | 2% | 39% | -5.2% / 1.0% / 7.6% | -5.9% / -0.2% / 5.0% |
| D. Legacy only | 48% | 9% | 2% | 39% | -6.0% / 1.2% / 8.2% | -6.3% / -0.2% / 5.6% |
| SPY alone (vs itself, for scale) | 0% | 3% | 1% | 33% | -4.0% / 1.4% / 7.1% | 0.0% / 0.0% / 0.0% |

![mc](montecarlo.png)

P(beat SPY) near 50% means the sleeve is indistinguishable from the index over a 15-day race, which is the honest prior for the
class competition: with this horizon, the P&L ranking is mostly noise.

## 6. Known biases and limits (state these on the slide)

- **Survivorship:** the universe is today's S&P 500 constituents; names that were dropped from the index (usually losers) are absent, so every long-only number is flattered.
- **Frozen fundamentals / no fundamentals at all:** quality, estimate revisions and the Research agent cannot be replayed point-in-time, so they are not in the backtest. Sector/industry labels are today's GICS.
- **No historical news:** `news_surprise`, the Sentiment agent and the politician-trades context are not backtested; the C4 news veto is not exercised.
- **Earnings data:** EPS surprises come from yfinance's history (reported values, some restatement risk); `earnings_premium` uses the realised report dates, which companies announce weeks ahead (small look-ahead if a date moved).
- **Not in the loop:** LSTM / Kronos / LightGBM, regime rules, leveraged-ETF trading, the mean-variance optimiser (a rule-based weighting stands in), PM/Risk/CIO judgement. This is the Quant sleeve, not the fund.
- **Single regime, 3 years:** one bull-market window, ablations are multiple comparisons on the same data; none of the differences is claimed significant.
- **Design parameters** (blend 0.75 (old group), 20% cap, 60-session IC, −8% trailing stop (old group; −12% in section 0), 15 names, 5-day cooldown after a stop) were fixed in advance, not tuned on this sample.
- **Monte Carlo** resamples this same in-sample history; it states the spread of 15-day outcomes, not a forecast.

_Generated by `backtest/run_backtest.py` in 381s from `data/cache/bt/` (prices + earnings history)._
