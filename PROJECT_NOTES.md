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

## Stage 3 (Oct 20): human-readable reasons, names, Madrid time (E1, E2, E7, D13, C0.4)
- **Trade Rationale Writer** (`committee/rationale_writer.py`, gpt-oss-20b, one call after execution): 2–3 plain-English
  sentences per trade (what the company is, why now, portfolio effect, exit trigger = invalidation price) + one Risk
  Manager line. Facts only from that trade's agent outputs. Guard: text naming another ticker or a price not in the facts
  falls back to a deterministic template (also used in mock and on failure). Logged as stage `rationale`.
- Reasons are per trade (C5.5): forced reason → PM's per-trade thesis → that name's strongest signal. The PM's batch
  rationale is no longer pasted onto every row (dry run: FAS mentioned on trades that had nothing to do with FAS).
- `trades.csv` (E1): date, hour_madrid, ticker, name, side, price, shares, total, reason, signals. Scores live in
  `signals` only. `transactions` gained `name`, `signals`, `proposed_at` (auto-migrated on open).
- Names everywhere ("PSX · Phillips 66", `data.label`): app tables, daily report, forum post, Word summary.
- Time (D13): all stored timestamps were already Madrid; the "run 08:14 / trades 13:34" in the dry run was a **5.3 h gap
  between proposal and Execute**, booked at stale proposal prices. Now: trades are priced and stamped at execution time,
  the report shows Madrid with ET in brackets, and a proposal older than 2 h is refused (`expired`).
  `scripts/migrate_madrid_time.py` normalises any non-Madrid timestamp and backfills names / proposal times; run on the
  archived dry run: 0 timestamps changed, 5 stale executions flagged.
- Tests: `tests/test_reasons.py`.

## Stage 5 (Oct 20): daily market awareness (C0.1, C0.3, B7, C1.2, C2.8, B9)
- **Gemini key works** (tested Oct 20): `gemini-3.5-flash` answers in ~2 s. The 2.5 models in our config were retired for
  new keys, so `config.PROVIDERS["gemini"]` now uses `gemini-3.5-flash` / `gemini-3.5-flash-lite`.
- **Morning Brief** (`morning_brief.py`): one Gemini Flash call pre-open (fallback gpt-oss-120b on Groq, then a template:
  never blocks a run). Inputs: ES/NQ/YM futures, Euro Stoxx/DAX/FTSE/Nikkei/Hang Seng, VIX, 10y, DXY, WTI, sector ETFs,
  ~40 market headlines + held-name headlines, earnings due, macro calendar. Output: 2 paragraphs + tickers in play.
  Prepended (≤ 900 chars) to every agent's prompt via `Agent.run`, and logged as stage `brief`.
  *Doc vs spec:* IPS C0.3 says "gpt-oss-20b, one paragraph"; the Oct 20 spec says Gemini Flash, two paragraphs. We follow
  the spec (newer); flagged here. `config.JUNIOR_GEMINI_FALLBACK` (off by default) lets a junior agent that hits 3+ Groq 429s
  retry on Gemini.
- **Earnings (B7)**: `data.get_earnings` (yfinance): next date/time, trading days to report, `reports_soon` (≤ 5 days), last 4
  quarters' EPS est/actual/surprise, 30-day EPS-estimate revisions (0q, +1y), recommendation trend. Feeds Research (C2.8
  earnings preview line), Sentiment, PM (`earnings_soon`) and the **C5.3 fast track** (reporting soon, Sentiment ≥ +0.7,
  Quant ≥ 5/10 → eligible, capped at 10% NAV in the optimiser).
- **PEAD** (`analytics.pead`): surprise %, 3-day announcement return minus SPY, days since report; drift = +1 / −1 when both
  agree, 0 if mixed, decaying linearly over 20 trading days; replay-safe. Fed to the screener (Stage 2), Quant (input column
  `pead_drift`, role_prompt untouched), the PM and the trade `signals`.
- **Macro calendar (C1.2)** in `config.MACRO_CALENDAR`: FOMC Oct 27–28 (decision Oct 28 14:00 ET), PCE + Q3 GDP Oct 29,
  ECI Oct 30, NFP Nov 6, CPI Nov 10 (sources: Fed via fedratecalc, BEA, BLS schedule pages).
- **Devil's Advocate (C0.1)**: after the PM's first proposal, one gpt-oss-120b call on the top 3 BUYs (bear case, evidence,
  proceed/downsize/drop); the PM then answers each objection in `da_responses` (unanswered = "NOT ANSWERED"), and the CIO
  sees both. Advisory, no veto. In the roster and transcript.
- **Regime rules (C1.3)**: verified exactly (2-of-4 risk-off, all-3 risk-on, downgrade immediate, upgrade after 2 days); test added.
- **Two-stage screener (B9)**: see C8. Quality uses ROE, net margin and D/E (vs sector via the sector-neutral z-score);
  "margin trend YoY" would need quarterly statements for 60 names per run, so it's left out of the screen (Research still flags it).
- Cost: + ~2 LLM calls per run (DA + PM answer) + 1 Gemini call; first run of the day ~65 s in mock (earnings/fundamentals for 60 names).
- Tests: `tests/test_market_awareness.py`.

## Stage 4 (Oct 20): Quant reliability + model honesty (C3.2–C3.7)
- **LSTM rebuilt to C3.5** (`forecaster.py`): 60 days of returns, volume change, 20-day vol and SPY returns; two heads
  (P(up over 5 days), BCE + 5-day return, Huber, kept for the IC). **Leakage fixed**: every series is scaled by its vol
  over the training window only. **Monthly walk-forward** over the last 250 trading days (12 retrains, each on labels known
  before its block; the newest block model is the live model, ≤ 1 month old). ~65 s on the M2 → cached per data date.
  `PRICE_HISTORY` raised to 3y so the first block still trains on ≥ 1 year.
- **LightGBM** (`forecasters/lgbm.py`): 1/3/12-1m momentum, 20d vol, RSI-14, distance from 50dma, industry momentum (1m,
  GICS sub-industry peers), 1-week reversal, post-earnings drift; same monthly walk-forward. Runs in a **subprocess**
  (`lgbm_worker.py`) because LightGBM's OpenMP runtime aborts the process when PyTorch's is loaded too. Needed
  `conda install -c conda-forge llvm-openmp` on the M2 (libomp). ~2 s.
- **Horse race** (`forecasters/horse_race.py`): LSTM vs Kronos-mini vs LightGBM vs naive 5d momentum on the same **20**
  non-overlapping 5-day origins (was 6). Kronos uses 1 sampled path per backtest origin to keep it to ~40 s (cached daily).
  **C3.4 weights for every model**: trust hit rate over its own walk-forward set (≥ 100 predictions) vs the "always up"
  baseline: < 52% or not beating it → 0; 52–55% → 25%; > 55% → 50%; plus the IC t-stat floor (t < 1 → 0); models capped at
  50% in total. Table in the daily report and the Trading Floor transcript.
- **Quant agent code** (role_prompt untouched, Jayesh is rewriting it): Python computes the C3.2 signals (12-1m momentum,
  trend vs 50/200dma, RSI-14, MACD 12/26/9 at low weight, 3m vol as a risk adjustment) into a `breakdown` and a
  `rule_score`; `validate()` enforces C3.3 (RSI > 75 halves a positive score; RSI < 25 never positive unless above the
  200dma) and adds `flags` (overbought / oversold_watch / below_200dma) and `model_weights`.
- **C3.7 zero-score guard**: > 30% exact 0.0 → split the batch in two and retry → gpt-oss-120b → rule-based score. The
  path taken is stored in `quant_fallback` and appended to every trade's `signals` ("Q from rule-based ...").
- First honest numbers (Oct 20, 3y data, 43-name shortlist, 20 common origins): LSTM hit 50%, IC −0.150 (t −2.6),
  trust-hit 53% vs always-up 57%; Kronos hit 47%, IC −0.029 (t −0.8); LightGBM hit 54%, IC +0.019 (t +0.3), trust-hit 54% vs
  56%; naive 5d momentum hit 50%, IC −0.070. **No model passes C3.4 → all model weights 0, Quant runs on technicals.**
  Short-horizon returns are close to unpredictable; the gate is doing its job, and that is the presentation point.
- Cost: cold first run of the day ~112 s in mock (3y download + walk-forward + 20 Kronos origins), cached runs ~3–30 s.
- Tests: `tests/test_quant_models.py`.

## Stage 6 (Oct 20): Part D operations + 20-day replay (D1–D13)
- Verified and tested: D10 pre-flight (missing / non-positive / > 20% jump excluded; 3x ETFs allowed 20% × leverage),
  D11 (an agent that raises = no signal, logged in `failed_agents`, run continues), D12 (2+ of the 4 signal agents silent →
  "Insufficient information", no new trades, forced exits still go), C8.5 (no live price → untradeable, valued at last
  close), C8.6 (> 20% NAV jump vs last snapshot → Trader halts).
- **Replay**: `python orchestrator.py --replay 20 [--provider mock|groq]` → `finance_replay.db`, `trades_replay.csv`,
  `reports/replay/` (+ `summary.json`). Never touches `finance.db`. Day d decides on prices up to the **previous close**
  and fills at d's **open**, stamped 15:30 Madrid (D8). Models retrained walk-forward on each day's data (lighter LSTM:
  120-day walk-forward, retrain every 30 days). Fundamentals = today's snapshot, frozen (D7, small look-ahead, stated in
  the summary); estimate revisions dropped; earnings history cut at d; Sentiment n/a (no historical news, weight
  redistributed); no Morning Brief; Kronos skipped (too slow per day). Reflection grading runs inside the replay.
  Known bias: today's S&P 500 constituents (survivorship). Dashboard: "Fund: Live / Replay" toggle (replay is read-only).
- Reflection entry date is now the **execution** date (was the data date), so a 5-day call is measured from the fill.
- Test proving no look-ahead: run a replay day, multiply every price on/after d by 3 with model caches bypassed, assert
  identical orders. `tests/test_operations.py`.

## Stage 7 (Oct 20): reporting (E3–E6) + performance attribution
- **Daily report** adds: Morning Brief, Devil's Advocate objections next to the PM's answers, top headlines per held name
  (source, FinBERT, LLM score), the Risk Manager's hard-limit pass/fail table on the post-trade book (+ warnings and any
  override), agent failures (D11), the horse-race table, lessons learned, and the attribution line.
- **Forum post (E4)**: holdings with names + 3 lines: today's trades and the lead reason, return vs SPY in bps,
  tomorrow's watch list (Morning Brief tickers in play + macro releases in the next 5 days).
- **Final Word report (E5)**: NAV vs SPY chart, KPIs, attribution with a waterfall chart, graded reflection calls, the
  latest horse race, best/worst picks and largest driver/drag, limitations, daily evolution, every trade with its reason.
- **Dashboard (E6)**: beta (post-trade book), cash %, leveraged %, days per regime, committee hit rate, attribution waterfall.
- **Attribution** (`attribution.py`): daily Brinson-Fachler vs SPY on start-of-day holdings, SPDR sector ETFs as sector
  benchmarks and SPY sector weights (yfinance, Morningstar sectors mapped to GICS, cached with as-of date). Parts: sector
  calls (allocation), stock picking (selection incl. interaction), leverage sleeve split into beta (leverage × underlying)
  and decay/tracking, cash drag, plus "SPY vs sector ETFs" tracking so the parts sum exactly. Linked over days with
  Carino smoothing. The gap to the NAV return (fills at execution prices) is shown as execution/timing. Selection is also
  attributed to the agent whose weighted score drove each position at entry. Test: parts sum to the total within 1 bp.
- Runtime fix found while testing the replay: the LSTM trains **2.5× faster on CPU than on MPS** on the M2 (12 s vs 30 s per
  fit), so `config.LSTM_DEVICE` defaults to cpu; replay uses a 60-day walk-forward per day. One replay day ≈ 1 min in mock.
- Fixed: on the init day every buy was labelled a "replacement to keep 10 holdings" (replenish rule now daily runs only).
- Tests: `tests/test_attribution.py`.

## Stage 8 (Oct 20): close-out
- **Kill switch**: a file named `HALT` in the repo root makes `--execute` and the dashboard's Execute refuse (analysis still runs).
- **Backups**: `backup.sh` copies `finance.db`, `trades.csv`, `prices.csv`, `reports/` to `backups/<YYYY-MM-DD_HHMM>/`
  (gitignored); called automatically after every live execute and snapshot (never for tests or the replay).
- **NYSE calendar** (`nyse_calendar.py`, 2026 holidays + early closes hardcoded): no snapshot is logged on a weekend or
  holiday, and execution on a non-trading day is refused.
- **Prompt logging**: every agent message now stores the full system prompt, user prompt, raw output, tokens and latency
  (`messages` columns, auto-migrated) → "what this agent saw" for the dashboard and the presentation.
- Tests: `tests/test_closeout.py`.

## Stage 8 results (Oct 20–21)
- **20-day replay** (`--replay 20`, mock agents, prices only, no news): Sep 23 → Oct 20, **−3.71% vs SPY −0.56% (−315 bps)**,
  max drawdown −5.53%, 51 trades, committee hit rate 47% on 41 graded calls (15 right / 17 wrong / 9 flat). Ran 53 min, 0 errors.
  Mock agents = rule-based scores, so this tests the mechanics (limits, floor, cap, reflection, attribution), not the LLMs.
- **Real Groq run** on `/tmp/test.db`: init 382 s, 10 calls, ~34.6K tokens, 9 rate-limit (429) waits, 14 trades; daily 566 s,
  8 calls, ~27.8K tokens, 4 waits, 3 trades. **Sentiment failed both times with 413 "request too large"** (43 names × 10
  headlines + brief > Groq's 8K tokens/min request size). Fixed: batches sized on the whole request (≤ 14K chars), 6 headlines
  per name to the LLM. Re-tested on Groq: 8 batches, 34/43 names scored, 9 n/a (no news), 194 s, 8 waits.
- **Runs exceed 4 minutes on the Groq free tier** (the spec's threshold). Proposed fallback, NOT switched on (needs Dev's OK):
  set `config.JUNIOR_GEMINI_FALLBACK = True` (junior agents move to `gemini-3.5-flash-lite` after 3 Groq 429s), or route
  Sentiment straight to Gemini Flash (its batches are the biggest token user).

## Price accuracy (Oct 21): `prices.py`
- History: yfinance chunks of 50 → per-ticker retry (2×) → Stooq closes. Coverage logged ("511/513 priced; missing: …").
- Execution: Finnhub quote vs yfinance; gap > 1% = "price unverified", order not executed (Trader logs it, C8.5). Finnhub down = "single-source price".
- 21:45 snapshot: Polygon grouped-daily close vs yfinance, > 0.5% gaps logged, Polygon wins. Per-holding source stored in `daily_snapshots.price_sources` (new column, auto-migrated).
- Held name with no price: last verified close, "stale price", no trading. Data quality box in dashboard header + daily report.
- Throttles: Finnhub 55/min (free 60), Polygon 5/min. Keys `FINNHUB_API_KEY`, `POLYGON_API_KEY` in `.env`, never logged.
- Keys tested Oct 21: Finnhub OK. Polygon OK for older days but **403 on same/previous day** (free tier = delayed EOD) → at 21:45 the snapshot will usually fall back to yfinance (flagged single-source). Watch item.
- Tests: `tests/test_prices.py` (4/4). No live/LLM re-run done for this change.

## Limits update (Oct 21, Dev)
- Max portfolio vol 35% (was 25%, IPS A3). VaR95 1d stays 5%. Cash 0-5% (was <= 3%) = min 95% invested (A12); PM exposure 0.95-1.00.
- Regime: any ONE risk-off signal (VIX > 25, SPY < 50 & 200dma, HYG/LQD > 1% below 20dma, VIX +30% 1d) blocks new leveraged buys
  (plan_orders skip + Risk Manager hard check); 2 of 4 = full risk-off (unchanged). `macro_regime` now returns `lev_buy_block` (contract addition).
- Tests: risk_limits fixtures moved to 5.1% cash / 36% vol; new test_one_risk_off_signal_blocks_leveraged_buys.

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
| C8 | IPS B9 weights vs A14 momentum focus | ✅ Decided Oct 20 (option B, momentum/trend capped at 50%): two-stage screener. Stage 1 (all ~500, price only, sector-neutral): 12-1m vol-scaled momentum 39 · industry momentum 24 · trend 25 · 1-week reversal 12 → top 60. Stage 2: Stage-1 score 57 · revisions + post-earnings drift 27 · quality 16 → top 30. Effective 22/14/14/27/16/7, value 0, low-vol replaced by vol-scaling. Missing Stage-2 input reweights pro rata. `config.SCREEN_STAGE1/2`. |
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

- **Quant output (Stage 4, owner Jayesh; consumer PM, owner Rima):** each `signals[t]` gains `breakdown`
  {momentum_12_1, trend, macd, rsi, models}, `flags`, `rule_score`; top level gains `model_weights` {lstm, kronos, lgbm} and
  optional `quant_fallback`. Score / comment unchanged. Model card fields: `walk_forward_days`, `always_up_hit_rate`.
- **Stage 5 (Oct 20):** new agent **Devil's Advocate** (in ROSTER; owner TBD): input `ideas` (top 3 BUY orders), output
  `{objections: {ticker: {bear_case, evidence, verdict}}}`. **PM** gains `da_responses` (required when objections exist) and
  sees `earnings_soon`, `pead`, `earnings_fast_track_eligible`. **Research** gains optional `ratings[t].earnings_preview`
  (C2.8). **Quant** input table gains a `pead_drift` column (output unchanged). Every agent's prompt starts with the
  MORNING BRIEF. Composite rows gain `pead`.
- **Orders / trades (Oct 20):** each order carries `facts`, `why`, `name`, `signals_text`; `reason` is plain English.
  New agent `Trade Rationale Writer` (not in ROSTER): input `facts[]`, output `{rationales: {ticker: text}, fallback_used_for}`.
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

## Owners (IPS C0.2)

| Agent / area | Owner |
|---|---|
| C1 Macro Strategist (+ `data.get_macro`, regime rules) | Rima |
| C2 Equity Research (+ fundamentals, earnings preview) | Marina |
| C3 Quant (+ LSTM, Kronos, LightGBM, horse race) | Jayesh |
| C4 Sentiment (+ news, FinBERT) | Mer |
| C5 Portfolio Manager (+ optimiser, plan_orders) | Rima |
| C6 Risk Manager (+ risk engine, check_book) | Dev |
| C7 CIO | Mer |
| C8 Trader (rule-based execution) | Marina |
| Devil's Advocate, Trade Rationale Writer, Morning Brief | team (unassigned) |
| Presentation intro + ending | Dev + Jayesh |

## Oct 21: Streamlit Cloud read-only viewer
- Cloud runs `app.py` from `main` with `VIEWER_MODE=1` (also auto-on if orchestrator imports fail): no Run/Execute/Snapshot/Reset/re-check/report buttons; sidebar shows last update time.
- `requirements.txt` = light viewer deps; `requirements-local.txt` = torch/transformers/lightgbm/openai etc. Laptop installs both. `huggingface_hub` pin removed.
- `finance.db`, `trades.csv`, `reports/` now tracked (replay reports still ignored). `publish.sh` commits+pushes them; called automatically after live `--execute` (with trades), `--snapshot`, and the app's Execute/Snapshot buttons. Opt out with `PUBLISH=0`; never fires with `FUND_DB_PATH` set or on the replay fund.

## Oct 21 (pm): UI + resilience pass
- Theme lightened (navy #1B3354, panels #234068, accent #6BA3EC). Data quality moved to a collapsed expander at the page bottom. Positions: Total cost + Market value.
- Agents always run on LLM_PROVIDER (groq); `gemini` removed from the AI-brain dropdown and remapped in `_agent_provider` (runs #2-3 had gone out on gemini via the dropdown). Gemini = Morning Brief only.
- Agent failure fallbacks (D11): PM = 2×composite + C5.3 + forced exits; DA = proceed; **CIO = approve Risk-passed orders meeting C5.3 + forced exits** (was: reject → forced exits only on daily runs). Logged as "CIO — rules fallback"; cards show "⚠ Agent failed → …". Message content gains a `fallback` field (schema add, non-breaking).
- Decision Room: Python steps shown as Screener (Marina), Regime Engine (Rima), Model Lab (Jayesh; horse race + LSTM merged, model card in expander), tagged "Rules engine". Display-only; DB still logs agent "System".
- Single-source prices: snapshot path = Polygon grouped(today) + yfinance, never Finnhub. Snapshot at 21:45 Madrid = 15:45 ET, before the close, so Polygon free tier returns nothing → every name yfinance-only → SINGLE. Finnhub failures were swallowed silently; now logged.

## Oct 21 (late): official close booking
- `record_prior_close()` runs at the start of every live committee run: if the last completed NYSE session (America/New_York, `nyse_calendar.last_completed_session`, early closes respected) has no `kind='close'` row, it books it from Polygon grouped (prev day) + yfinance close, <= 0.5% cross-check (`SNAP_TOL`; > 0.5% → Polygon, flagged). Skipped if trades executed after that close (book changed) or fund not initialised.
- `daily_snapshots.kind` added ('close' | 'intraday', migration). Intraday marks never overwrite a close row.
- 21:45 button renamed "Intraday mark (not the close)" (21:45 Madrid = 15:45 ET most weeks). Finnhub = live trade prices only. Finnhub key verified working (AAPL quote OK).
- Tests: `tests/test_prior_close.py` (DST gap week, half day, idempotence, book-changed skip).

## Oct 21: testing-handoff fixes (mock/tests only)
- Performance: one point per session, cumulative return in %, %-formatted axes; daily-return / rolling-vol charts show placeholder text until enough sessions.
- Risk numbers labelled by state: Portfolio tab = CURRENT (executed book only); Decision Room limit grid = PROPOSED / EXECUTED post-trade book.
- Trades: decided time, price source, price time; executions outside 09:30–close ET are labelled "last close (off-hours)" (`nyse_calendar.is_open`).
- Execution is idempotent: `db.claim_run` atomically moves proposed → executing; rerun/double click/race = one execution.
- Beta: risk_report used ddof 1/0 (+0.4% bias), fixed; matches an independent 1y OLS slope. New `notional_exposure` + `etf_notional` (e.g. FAS 10% × 3 = 30%).
- CIO → Risk: if the order list changes after Risk's verdict, Risk re-checks the new post-trade book (stage `risk_recheck`); hard breach = forced exits only.
- Decision Room: status row (each stage ok/fallback/failed, CIO decision, risk verdict, executed yes/no); agent minutes behind a diagnostics toggle.
- Committee reliability shows "x of n valid".
- **CONTRACT CHANGE (flag to team):** `trades.csv` gains 3 columns: `decided_at, price_source, price_time`. `transactions` table gains `price_source, price_time` (auto-migrated). risk_report JSON gains `notional_exposure`, `etf_notional`.
- Tests: `tests/test_handoff.py` (7 tests, all pass); existing suites pass.

## Oct 21 (2): follow-up robustness tests (mock/tests only)
- API failures: llm.py retry loop verified, not rebuilt. 429/503/timeout → bounded exponential backoff (≤60 s per wait, LLM_MAX_RETRIES=6), then LLMError → agent logged `FAILED (...)` + rules fallback (CIO: extra `rules` message "rules fallback"). Never logged under an LLM model name.
- Fix: a SELL clamped to 0 shares no longer logs a 0-share "trade" (it used to count toward the 5/day cap).
- Tests: `tests/test_followup.py` (11): retries/backoff, PM+CIO down, reconciliation (NAV, cash ≥ 0, DB = trades.csv = fills), init exemption, 5/day cap, rejects not counted, 3 runs/day (errored runs don't count; replay exempt by design), restart mid-execution / mid-run, duplicate same-day snapshots, <2 sessions placeholder.
- Open risk: a process killed mid-execution leaves the run in `executing` with a partial book (fail-closed, no double execution). Recovery is manual: inspect, then re-run the committee.
