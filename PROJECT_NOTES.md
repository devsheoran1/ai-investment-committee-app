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

## Conflicts and how we resolved them

| # | Conflict | Resolution |
|---|---|---|
| C1 | Brief: 5 trades/day vs initialisation needs 10–20 buys | Init day exempt (IPS D1). **Confirm with Prof. Manero.** |
| C2 | Brief §1 "NYSE shares" vs §10.3 "stocks and leveraged ETFs" vs our S&P 500 | S&P 500 (includes Nasdaq listings) + US leveraged ETFs. Worth a one-line check with the professor |
| C3 | Brief mega-prompt: 40% max position, 2 trades/day, $10k, 1 week, 10 ETFs from CSV | That section is the generic template. Competition rules (§10) and §1 win: $1M, 5 trades, 20% (IPS A8, stricter than 40%) |
| C4 | Brief §2 `finance.db` vs §8.1 `finance_game.db` | `finance.db` |
| C5 | Brief CSV format has no ticker column | Added `ticker` and `reason`: positions can't be tracked without them |
| C6 | Brief "Win Rate" undefined | % of up days. Switch to % of profitable round-trips once there are enough closed trades |
| C7 | IPS A12 cash ≤ 5% vs A16 ±5% drift band | When exits leave cash > 5%, top-ups inside the drift band are allowed |
| C8 | IPS B9 left blank; default is 40% quality/value/low-vol, against A14 (momentum) | Funnel is price-only and momentum-led: 12-1m 35%, 3m 25%, trend 25%, low vol 15%. Quality comes in at Research |
| C9 | IPS C1.5 macro tilt +0.5 *and* 15% macro weight would count macro twice | Tilt goes through the 15% weight, sized so it equals ±0.5 on the 0–10 composite |
| C10 | IPS A6 beta "any" vs C1.4 beta bands per regime | Bands are soft concerns for Risk, not hard limits |
| C11 | IPS A4 −10% drawdown, no de-risking action given ("don't cut in half") | Interpreted as: no new leveraged longs and existing ones exited while below −10%. **Needs your sign-off** |
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
