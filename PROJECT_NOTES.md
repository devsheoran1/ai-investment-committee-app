# PROJECT_NOTES — shared state

Last updated: 2026-10-17. Update this file whenever a decision, limit or owner changes.

**Order of authority:** (1) Prof. Manero's brief, `Mif_project_investment_AI_manager_tool_v1-4.pdf`
→ (2) our IPS questionnaire → (3) the IPS Default column. Where they conflict, the brief wins,
and the conflict is logged below.

## How to run

```bash
cd ~/Downloads/ai_investment_committee
/opt/miniconda3/bin/python3 -m pip install -r requirements.txt   # the Python that has torch/yfinance
/opt/miniconda3/bin/python3 -m streamlit run app.py
```

- Terminal: `python orchestrator.py --provider mock` (propose only) · `--execute` (commit) · `--snapshot` (after-close revalue) · `--reset`
- Test against a throwaway fund: `FUND_DB_PATH=/tmp/test.db streamlit run app.py`
- Live fund = `finance.db` (repo root). Synthetic-data days go to `finance_synthetic_sandbox.db`.

## Brief requirements → status

| Brief | Requirement | Status | Where |
|---|---|---|---|
| §1 | Init mode: research, pick 10–20 names, deploy $1M | ✅ | `orchestrator.py` (mode `init`), `analytics.optimise` |
| §1 | Daily mode, every change risk-assessed | ✅ | PM ⇄ Risk loop on the post-trade book |
| §1 | Max 5 trades/day | ✅ | `analytics.plan_orders` + `committee/trader.py` (init day exempt; see C1) |
| §1 | UI: portfolio now, P&L since init, trades day by day, reasoning per change | ✅ | `app.py` tabs; `reason` stored per trade |
| §2 | Streamlit + SQLite `finance.db` | ✅ | `database.py` |
| §2/§8.2 | Data / Strategy / Risk / Execution agents | ✅ | `agents.py` maps them onto our 8 |
| §4 | News researched every morning before the open | ✅ | Yahoo RSS + Google News RSS (`data.get_news`) |
| §5 | yfinance, list of tickers that are known to have data | ✅ | S&P 500 from Wikipedia, checked per run (514/514 downloaded on Oct 17) |
| §6 | Small token footprint, no big blobs | ✅ | screening funnel → ~38 names reach the agents |
| §6.6 | Keys in `.env`, never committed | ✅ | `.gitignore` |
| §8.1 | Tables simulation / positions / transactions / daily_snapshots | ✅ | plus runs, messages, regime_state |
| §8.3 | Sidebar date, Start New Simulation, Advance Day; Total Value + Cash; Trading Floor (Run Agents → reasoning → Execute) / Portfolio / Performance (Plotly) | ✅ | `app.py` |
| Fig. 2 | KPIs: Total Return, Max Drawdown, Win Rate; daily snapshots table | ✅ | Win Rate = % of up days (see C6) |
| §10.2 | Max 3 runs/day | ✅ | `RunLimitError` (not applied to mock/synthetic) |
| §10.3 | Stocks and leveraged ETFs only | ✅ | |
| §10.4 | Sequential trade CSV | ✅ | `trades.csv`, rewritten after every execution |
| §10.5 | Daily forum post 17:00–18:00 Madrid | ✅ | `report.forum_post()`; shown in the dashboard's Trades tab |
| §11 | Repo with app.py, database.py, agents.py, prices.csv, requirements.txt | ✅ | `prices.csv` refreshed on every real download |
| §11 | Summary of the 15-day run (Word doc) | ✅ | `python report.py --final` → `reports/final_summary.docx` |
| §11 | Class presentation | ⏳ | |

## Prof. Manero's answers (Oct 20) — rules
1. Init day may exceed 5 trades; the cap applies from day 2 (C1 resolved).
2. S&P 500 incl. Nasdaq listings is fine (C2 resolved).
3. Leveraged longs (30% total / 15% each) and inverse hedges (25%, risk-off only) are allowed **provided the constraints
   are enforced by our Risk Manager agent**: the hard-limit check is called from `committee/risk_manager.py` and appears
   as the Risk Manager's verdict in the transcript and report.
4. Transaction costs = 0 (A20).
5. Runs are manual: no schedulers or daemons, only the button and the CLI.

## Dry run Oct 19–20 (archived: `archive/dry_run_2026-10-19_20/`, presentation material)
The first live-style book, built under the old rules, then archived and the live fund reset to $1M on Oct 20 (no init;
live init is Oct 22 14:30 Madrid). 7 runs, 20 trades (17 buys / 3 sells). What broke and why:
- **Book drifted to 12 → toward 9 names and 8.3% cash** (limit then 5%, now 3%): limits were checked on the target book,
  then the 5-trade cap and $25k minimum trimmed it and the executable book was never re-checked. → Stage 1.
- **Sentiment scored S 0.0 on all 20 trades**: the agent answered "no material news" for ~90% of names and `validate()`
  turned "no headlines" into 0.0 too. → Stage 2.
- **Quant scored Q 0.0 on 9 of 20 trades** (gpt-oss-20b laziness). → Stage 4.
- **One PM note pasted on every trade**, mentioning FAS, which never traded. → Stage 3.
- **Timestamps mixed**: the report showed the run at 08:14 and its trades at 13:34. → Stage 3 (all Madrid time).
- **Runs 4–7 crashed with `KeyError: 'ABNB'`**: Yahoo returned only 382/513 tickers, the big-gap retry never fired, and a
  held name without a price crashed `plan_orders` (and would have dropped out of NAV). → fixed in Stage 1.
- `orchestrator.py --reset` used to continue into a committee run (= an init). Now it only resets.

## Stage 1 (Oct 20): hard risk limits hold on the executable book
- Limits now: **cash ≤ 3%** (≥ 97% invested), **hard floor 10 holdings, target 12–15** (init aims for 14),
  B2 ETF list expanded to 14 leveraged + 4 inverse (screener keeps ADV > $10m).
- The Risk Manager checks the **post-trade book** (after the 5-trade cap, the $25k minimum and cash upsizing):
  `committee/risk_manager.py: check_book()` = `analytics.risk_report` (metrics) + `classify()` (verdict). That is the
  Risk Manager enforcing the limits, per Prof. Manero's rule 3, and it is what the transcript shows.
  - Hard (always reject; an LLM "approve" is overridden and the override logged): holdings < 10, cash > 3%, name > 20%,
    sector > 35%, leveraged > 30% total / 15% each / regime cap, inverse > 25% or outside risk-off, vol > 25%, VaR > 5%, C11.
  - Soft (may reject, C6.3): effective positions < 4, corr > 0.9 among the top 3, a position > 25% of 20d ADV (new ADV check).
  - Everything else is a warning (beta band, corr > 0.8, concentration...). A rejection on warnings only is overridden to approve.
  - 3 rounds, then mechanical de-risk: offending positions shrunk pro rata (`analytics.shrink_to_limits`), freed weight
    to compliant names, then leverage room cut if still needed. Logged as "mechanical de-risk (IPS C6.5)".
- `plan_orders`: A7 floor on the executable book (an unpaired discretionary exit that leaves < 10 names is dropped;
  stop-loss / news-veto / C11 exits always go; the next run puts replacement buys first); A12 cash ≤ 3% by upsizing the
  day's BUYs within the 20% / 35% / ETF caps, then topping up held names if a slot is free; otherwise flagged "CASH".
- Vol-targeted leverage sleeve: ETF cap = base cap × (1y median 20d vol ÷ current 20d vol), capped at IPS limits, applied in
  the optimiser; shown per ETF in the risk report (`etf_effective_caps`). Test: doubling vol halves the cap.
- C6.4 replaces the "held 5 days" exit: a leveraged/inverse ETF with return < 0, > 12% off its peak since entry, or lagging
  leverage × underlying by > 2% needs the PM's explicit re-approval (same path as invalidation responses).
- C8.5 robustness: missing tickers re-downloaded in batches; a held name without a live price is valued at its last
  known close and is untradeable that day (no more `KeyError`).
- Tests: `tests/test_risk_limits.py` (every hard rule, soft vs warning, override, ETF caps, C11 ladder, A7 cases a/b/c, cash
  upsizing, missing price).

## Stage 2 (Oct 20): Sentiment produces real signal (C4.1–C4.6)
- News: up to 10 headlines per name over 5 days, each with source + Madrid timestamp + URL, from Yahoo Finance RSS,
  Google News (general + a Reuters/Bloomberg/WSJ query) and SEC EDGAR 8-K (per-company Atom feed, CIK from the S&P 500 table).
  A relevance filter drops headlines that don't name the company/ticker (Yahoo's per-ticker feed mixes in market stories);
  max 4 per source. SEC requires a contact in the User-Agent: set `SEC_USER_AGENT` in `.env` (placeholder by default).
- FinBERT (`ProsusAI/finbert`, local, MPS) scores every headline (P(pos) − P(neg)), cached per headline in
  `data/cache/finbert_headlines.pkl`. The LLM sees the FinBERT scores and judges surprise vs expectation; final = LLM score,
  both logged. ~9 s for ~40 names on the M2 (17 s first load).
- No headlines → `score: null`, shown "n/a" (never 0.0). The composite redistributes the missing agent's weight pro rata
  (C5.2) and the trade reason says so. A failed call → null for every name (counts toward D12).
- Sanity guard: > 70% of names with headlines scored exactly 0.0 → one retry on gpt-oss-120b (logged in the model field).
- Intraday re-check (C4.5, run slot 3): `python orchestrator.py --recheck` / dashboard button. Headlines refreshed for held
  names only; a held name ≤ −0.8 convenes the full committee with that name as a forced exit (Risk → CIO → Trader).
- Mock coverage on Oct 20 data: 42 of 43 names with a non-zero score (was ~0 in the dry run).
- Tests: `tests/test_sentiment.py`.

## Conflicts and how we resolved them

| # | Conflict | Resolution |
|---|---|---|
| C1 | Brief: 5 trades/day vs initialisation needs 10–20 buys | ✅ RESOLVED Oct 20: Prof. Manero confirmed the init day may exceed 5 trades; the cap applies from day 2. |
| C2 | Brief §1 "NYSE shares" vs §10.3 "stocks and leveraged ETFs" vs our S&P 500 | ✅ RESOLVED Oct 20: Prof. Manero confirmed the S&P 500 incl. Nasdaq listings is fine. |
| C3 | Brief mega-prompt: 40% max position, 2 trades/day, $10k, 1 week, 10 ETFs from CSV | That section is the generic template. Competition rules (§10) and §1 win: $1M, 5 trades, 20% (IPS A8, stricter than 40%) |
| C4 | Brief §2 `finance.db` vs §8.1 `finance_game.db` | `finance.db` |
| C5 | Brief CSV format has no ticker column | Added `ticker` and `reason`: positions can't be tracked without them |
| C6 | Brief "Win Rate" undefined | % of up days. Switch to % of profitable round-trips once there are enough closed trades |
| C7 | IPS A12 cash ≤ 5% vs A16 ±5% drift band | When exits leave cash > 5%, top-ups inside the drift band are allowed |
| C8 | IPS B9 left blank; default is 40% quality/value/low-vol, against A14 (momentum) | Funnel is price-only and momentum-led: 12-1m 35%, 3m 25%, trend 25%, low vol 15%. Quality comes in at Research |
| C9 | IPS C1.5 macro tilt +0.5 *and* 15% macro weight would count macro twice | Tilt goes through the 15% weight, sized so it equals ±0.5 on the 0–10 composite |
| C10 | IPS A6 beta "any" vs C1.4 beta bands per regime | Bands are soft concerns for Risk, not hard limits |
| C11 | IPS A4 −10% drawdown, no de-risking action given | ✅ Signed off Oct 20 (graduated): at −7% from peak NAV the leveraged sleeve is capped at 10% and no new leveraged buys; at −10% all leveraged longs are exited and, in risk-off, the inverse hedge is ≥ 15%. Restrictions lift after 2 consecutive days back above −7%. `config.DD_*`, `analytics.drawdown_state`, enforced as a hard limit by the Risk Manager. |
| C12 | IPS A15 still contains the template text "[STOCK]" | Ignored; mean-variance with the diversification penalty (default) |
| C13 | IPS C5.3 fixed size bands vs A15 mean-variance | Optimiser sizes; C5.3 used only as the entry filter |
| C14 | IPS B8 "short recent IPOs" vs long-only stack, B6 1-year history, S&P 500 membership | Not built. Downside only via inverse ETFs in risk-off |
| C15 | IPS B11 shortlist 40 vs Groq free-tier 8K tokens/min | 30 stocks + holdings + ETFs ≈ 38 names |
| C16 | IPS A3 25% vol vs 30% in 3x ETFs | 25% vol stays a hard limit, so the optimiser uses less than 30% leverage when needed |
| C17 | IPS C8 Trader = deterministic code | Done: 7 LLM agents + 1 rule-based Trader (say so in the presentation) |
| C18 | IPS D12 "neutral" ambiguous | "No signal" = failed or all-zero scores. A neutral *regime* does not count |
| C19 | IPS B7 / C4.2 Bloomberg, FactSet, Reuters | Not available to us. Free sources: Yahoo RSS, Google News RSS; earnings calendar still to add |
| C20 | IPS A19 "real-time price" | yfinance last price (~15 min delayed in market hours; previous close pre-open) |

## IPS → implementation

Implemented in code (see `config.py`, which notes where each number comes from):
A1–A3, A5–A10, A11, A12, A16–A18, A20 · B1–B6, B8 (vol filter), B10–B12 · C1.1–C1.5 (regime and tilts
as Python rules, LLM explains, ≤1 tilt override) · C2.1–C2.7 (flags in Python, insufficient-data
exclusion) · C3.3–C3.4 · C4.1, C4.3, C4.4, C4.5 (hourly refresh) · C5.1–C5.5 · C6.1–C6.5 · C7 · C8.1–C8.6 ·
D1–D5, D10–D12 · E1–E5 (daily report `reports/daily_<date>.md`, built after every execution and Advance Day).

Not yet built:
- **B7 earnings calendar + consensus** (feeds C4.3 and the C5.3 earnings fast-track)
- **C0.1 Devil's Advocate** agent
- **D6–D9 20-day replay** in a separate DB
- B2 extra ETFs by AUM (needs a list)
- C1.2 FOMC/CPI/jobs calendar

## Agent output formats (contracts between teammates)

- **Sentiment output (Oct 20, owner Mer):** `sentiment[t].score` may be **null** (no news / no answer = n/a). New keys
  `finbert` (avg FinBERT of the name's headlines), `llm_score`, `n_headlines`, `based_on` (headlines the score rests on),
  optional `retried_on`. Input `news[t]` is now a list of dicts `{title, source, ts (Madrid), ts_utc, url, finbert}`
  (was a list of strings). Composite rows gain `na` and `weights_used`; `quant`/`sentiment`/`research` may be null.
- **Risk Manager output (Oct 20, owner Dev):** `required_changes` is now a **list** (was a string); new keys
  `hard_breaches`, `soft_rejects`, `warnings`, `limit_checks` (pass/fail table rows: rule, value, limit, status, kind),
  `override`, `llm_decision`. `soft_concerns` kept for compatibility (= soft_rejects + warnings).
- **PM:** `equity_exposure` now 0.97–1.00 (cash ≤ 3%). `invalidated_holdings` may now include leveraged-ETF reviews
  (`type: leveraged_review`, IPS C6.4) that need an exit/hold answer.

Changed on Oct 17, so tell the owners:
- **Macro**: regime/tilts now come from rules. The LLM returns `confidence, tilt_override, watch_list, rationale`;
  validate() adds `regime, equity_exposure_target, max_leveraged, inverse_hedge_range, sector_tilts`.
- **Research**: same schema; `rating` can now be `insufficient_data`; gets a `flags` list. Stocks only.
- **PM**: `equity_exposure` is now 0.95–1.00.
- **Trader**: no LLM any more; output is `approved_orders, rejected, execution_notes, flags`.
- Quant, Sentiment, Risk, CIO: unchanged.
- Scores are still on −1..+1 internally. The UI also shows the composite on the IPS 0–10 scale.
  Moving every agent to 0–10 (IPS C2.1) is an open team decision.

## CLAUDE.md skill gaps (#1–#8) status — 2026-10-17
#1 5-trade cap ✅ · #2 init/daily + 3 runs/day ✅ · #3 universe + funnel ✅ (S&P 500, not Russell 1000, per Dev's Oct 17 instruction)
· #4 trade CSV + prices.csv ✅ · #5 agents.py / finance.db ✅ · #6 daily report + 15-day summary ✅ · #7 20-day replay ⏳ · #8 IPS mapping ✅.
Note: CLAUDE.md is a zipped skill whose §3 still describes the pre-Oct-17 code; regenerate it.

## Add-ons shipped Oct 18 (research shortlist items 1 and 2 of the Kronos/reflection session)

**Kronos forecaster + model horse race** (`forecasters/`, `vendor/kronos/`)
- Kronos-mini (4.1M params, `NeoQuasar/Kronos-mini` + `Kronos-Tokenizer-2k`, MIT) vendored from
  github.com/shiyu-coder/Kronos, **commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`** (only `model/`, plus LICENSE).
  Two local changes, listed in `vendor/kronos/SOURCE.txt`: relative import in `kronos.py`, and a
  `return_samples` flag so we get per-path outputs (needed for `p_up` and dispersion).
- `forecasters/kronos_fc.py`: 400-day OHLCV lookback, 5-day horizon, 5 sampled paths (3 in the backtest),
  MPS with CPU retry. Returns `exp_ret_5d`, `p_up`, `dispersion` per ticker. Cached per data date. Any failure returns None
  and the committee runs LSTM-only (tested with a bogus model id).
- `forecasters/horse_race.py`: LSTM vs Kronos vs naive 5d-momentum on the SAME 6 non-overlapping 5-day origins x shortlist
  (~228 ticker-origins). Scores hit rate, mean rank IC, IC t-stat. Blend weights = max(IC, 0), normalised over LSTM and Kronos.
- `data._download` now keeps High/Low (Kronos needs OHLC). `forecaster.forecast` exposes `oos_preds`.
- Cost on the M2 (8GB): +29 s per run (target < 60 s), whole mock run 32 s, peak RSS about 540 MB.
- **First scoreboard (Oct 17 data, 38 names):** LSTM hit 54% / IC +0.008 (t +0.1); Kronos hit 50-55% / IC -0.04 to -0.05
  (t -0.6 to -0.8); naive 5d momentum hit 57% / IC +0.137 (t +2.2). Kronos did NOT beat the LSTM or the baseline out of sample,
  and the simplest momentum rule beat both. With 6 origins these t-stats are weak evidence: say so in the presentation.

**Reflection memory** (`reflection.py`, `committee/reflector.py`, table `reflections`)
- Every executed trade stores a call: thesis, invalidation text + price level, entry price/date, regime, sector, signals.
  PM can supply `trade_notes`; otherwise a deterministic default fills both, so no trade is stored without them.
- Each run first grades calls whose 5-day horizon has elapsed, using only prices <= the as-of date (unit-tested against
  look-ahead), vs SPY. Verdict: right / wrong / flat (+/-1% excess band; a SELL is right if the stock then lagged SPY).
  One batched gpt-oss-20b call (`Reflection Analyst`, stage `reflection`) writes the lessons; mock lessons are deterministic.
- PM and CIO get <= 8 lessons (<= 2400 chars, about 600 tokens): same ticker > same sector > same regime > recent, plus the
  committee's running hit rate. Held names whose invalidation price is hit are flagged; the PM must answer each with
  "exit" or "hold" + reason. Unanswered = treated as an exit candidate.
- UI: "Lessons learned" expander in the Committee tab, horse-race table in the Trading Floor transcript, both in the daily report.
- Tests: `tests/test_kronos.py`, `tests/test_reflection.py`, plus a seeded integration run (3 fake past trades, mock x2).

**Real Groq runs of this session (Oct 18, throwaway DB, gpt-oss-120b/20b):** init + 2 daily runs with a seeded book.
Per run: 9-12 LLM calls, about 23-27K prompt + 5-7K completion tokens, 123-150 s wall, 4-7 rate-limit (429) waits per run
(handled automatically; free tier is 8K tokens/min per model), 0 hard failures after the fixes below. Mock run: 32 s, ~540 MB peak.
Real lessons read like: "Oil breakout signal drove XOM above SPY in neutral regime, validating breakout thesis in Energy."

**Bugs found by the real Groq runs and fixed (they would have bitten in the live window):**
1. Init could end with 6-9 names and a breached 35% sector cap: the optimiser now guarantees >= 10 names at initialisation,
   gives each a floor weight above the $25K minimum trade (no dust dropped at the end), and widens the candidate pool when the
   caps are infeasible instead of falling back to a portfolio that breaches them. Regression test: `tests/test_optimiser.py`.
2. The CIO LLM rejected a risk-approved *initial* portfolio ("maintaining current holdings", there were none). On the init run
   the CIO can now dissent but not stand the fund down (Brief: start fully invested).
3. gpt-oss-20b sometimes returns every Quant score as 0.0: a parseable all-zero reply now falls back to the rule-based
   momentum/LSTM score, flagged in the note. A real failure/timeout still gives neutral (IPS D11).
4. Groq 413 "request too large" on the Quant call (prompt + max_tokens counted against the minute limit): Quant prompt is now a
   column table (names once, not per ticker), default output budget 3000, and a 413 retries with a halved budget.
   `llm.usage` now also counts `rate_limited`.

**Contract changes (all optional / additive, nothing removed):**
- Quant output: `model_scoreboard` (str); each `signals[t]` may carry `kronos_ret`, `kronos_p_up`. Quant prompt gets
  `kronos_ret`, `kronos_p_up`, `agree` per ticker and a one-line scoreboard.
- PM output: `trade_notes`, `invalidation_responses`. PM/CIO prompt: `lessons`, `committee_track_record`, `invalidated_holdings`.
- New agent `Reflection Analyst` (not in ROSTER, so the "8-member committee" count is unchanged).
- `database.execute_trade(..., meta=None)` new optional arg. `llm.usage` gains `rate_limited`.
- Owners to tell: Quant (Kronos fields), PM (thesis discipline), Risk/CIO (lessons block).

## Known issues / watch list

- yfinance `Ticker.news` returns nothing in v1.7, so we use RSS (fixed Oct 17).
- Fundamentals now come only for the ~30 shortlisted stocks; failures show as missing data, never as synthetic numbers.
- Groq test runs (Oct 17, test DB): about 100–120 s, 9–11 calls, ~27K tokens, no rate-limit failures.
  Run 1 traded nothing: the low-beta shortlist missed the C1.4 band. Fixed by putting the beta band
  inside the optimiser; soft concerns can now cause a rejection only in round 1.
- gpt-oss-20b sometimes leaves Quant scores at 0.0 with no comment for 30–40% of names (lazy output).
  Watch this; options are splitting the Quant call in two batches or using the 120b model for Quant.
- Screener's low-vol factor (15%) pulls toward low-beta names, which works against the C1.4 beta bands.
  Consider moving that 15% to 3m momentum.
- The ±5% drift band can leave cash at 5–6% for a day after exits when there's no trade slot to reinvest.

- (add-on session) see the list below
- **Blend weights follow the spec literally:** with LSTM IC +0.008 (t +0.1) the LSTM gets 100% of the model blend. That is
  noise-level skill; the Quant prompt carries the t-stat so the agent can discount it. A minimum-t floor is worth considering.
- **Existing LSTM leakage caveat:** `forecaster._build_dataset` scales each stock by its full-sample volatility, including the
  test window. Mild, but its OOS numbers are slightly flattering vs Kronos (which normalises per window).
- Reflection needs 5 trading days of history before the first lesson exists; the live window is short, so expect only a
  handful of graded calls (about 3 to 8 per week). Present it as a demo of the mechanism, not as alpha.
- Mock runs re-buy a name the day after its stop-loss (no cooldown rule in the IPS). Worth deciding whether stopped-out names sit out N days.

## Owners

| Agent / area | Owner |
|---|---|
| Macro Strategist + `data.get_macro` | |
| Equity Research + fundamentals | |
| Quant + LSTM (`forecaster.py`) | |
| Sentiment + news | |
| PM + optimiser | |
| Risk Manager + risk engine | |
| CIO, Trader, dashboard | |
