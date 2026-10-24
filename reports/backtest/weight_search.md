# Walk-forward weight search (train 2018–2021, test 2022–2026)

**Fast proxy (so thousands of configurations are affordable): every Friday close, score = Quant (the ensemble) blended with the Macro sector tilt,
hold the top 15 eligible stocks equal weight to the next Friday's next-open (fills at the next open, zero cost). The full fund engine (5 trades/day,
stops, vol-managed sizing) is NOT in the proxy; the finalists are re-run through the real `simulate` to confirm.

Grid (coarse, 10% steps):  family prior over momentum / earnings / reversal / defensive (per-strategy cap c in {20%, 30%} must be feasible),
                           shrinkage lambda in {0.5, 1, 2}, regime allocator on/off, Macro-tilt share m in {0, .1, .2, .3, .4} of the Quant+Macro blend.
Selection: NOT the single best. Each configuration is scored by the mean TRAIN Sharpe of itself plus all its one-step neighbours (a 10% shift between two
families, adjacent lambda / macro share, regime flip, cap flip); the best neighbourhoods are the "robust region". A finalist whose TEST Sharpe is below 0.8x
its TRAIN Sharpe is rejected. Not backtestable: Sentiment, Research, and the Macro LLM (only its Python sector tilt is here).


4590 configurations. Across the whole grid the train and test Sharpes have Spearman rank correlation **+0.01** (median train 0.68, median test 1.12); 59% of configurations beat the current setup in the test period.
Best single configuration on TRAIN (what a naive search would pick): mom 10/earn 40/rev 20/def 30, λ0.5, regime on, macro 0%, cap 30%: train 1.32, test 1.35.

| Robust-region finalists | Train Sharpe | Test Sharpe | Test/Train | Neigh. mean / min (train) | Max DD train / test | Deflated Sharpe (train, N=4590) | PSR test (SR>0) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mom 0/earn 60/rev 30/def 10, λ0.5, regime on, macro 0%, cap 30% | 1.22 | 1.27 | 1.05 | 1.17 / 1.05 | -37% / -27% | 0.88 | 1.00 | pass |
| mom 10/earn 40/rev 20/def 30, λ0.5, regime off, macro 0%, cap 30% | 1.24 | 1.42 | 1.14 | 1.16 / 1.06 | -28% / -17% | 0.89 | 1.00 | pass |
| mom 10/earn 30/rev 30/def 30, λ0.5, regime on, macro 0%, cap 30% | 1.24 | 1.47 | 1.19 | 1.15 / 1.01 | -29% / -19% | 0.89 | 1.00 | pass |
| mom 10/earn 50/rev 30/def 10, λ0.5, regime on, macro 0%, cap 30% | 1.23 | 1.19 | 0.97 | 1.12 / 0.95 | -32% / -27% | 0.89 | 0.99 | pass |
| mom 20/earn 40/rev 30/def 10, λ0.5, regime on, macro 0%, cap 30% | 1.14 | 1.26 | 1.11 | 1.11 / 0.95 | -32% / -23% | 0.84 | 1.00 | pass |
| mom 10/earn 60/rev 30/def 0, λ0.5, regime on, macro 0%, cap 30% | 1.14 | 1.25 | 1.10 | 1.11 / 0.95 | -36% / -30% | 0.84 | 1.00 | pass |

| Reference (same proxy) | Train Sharpe | Test Sharpe | Max DD train / test | PSR test (SR>0) |
|---|---:|---:|---:|---:|
| current (regime on) | 0.69 | 1.07 | -34% / -22% | 0.99 |
| current (regime off) | 0.54 | 1.02 | -32% / -22% | 0.99 |
| SPY weekly buy&hold | 0.87 | 0.79 | -32% / -21% | 0.95 |

**Winner (first finalist passing the 0.8x rule):** mom 0/earn 60/rev 30/def 10, λ0.5, regime on, macro 0%, cap 30%. Test Sharpe 1.27 vs current 1.07: **APPLY** (rule: apply only if the test-period Sharpe beats the current setup).

| Full-engine confirmation (mean of 2 starts) | train Sharpe | test Sharpe | train Max DD | test Max DD |
|---|---:|---:|---:|---:|
| current (N2: equal prior, lambda 0.5, regime on) | 0.79 | 0.19 | -33% | -23% |
| winner | 1.01 | 0.48 | -29% | -17% |

Caveats: in-sample selection on 4 years of one market, survivorship-biased universe (today's S&P 500), the proxy ignores stops, the 5-trade cap and vol-managed sizing, Sentiment/Research are absent. A strategy that wins here is a hypothesis for the next out-of-sample year, not evidence.
