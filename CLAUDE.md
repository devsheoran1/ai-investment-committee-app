# AI Investment Committee: project brain (CLAUDE.md)

Read this file and `PROJECT_NOTES.md` before doing anything.
- `PROJECT_NOTES.md` = live status: brief checklist, conflicts C1–C20, IPS mapping, agent contracts, known issues.
- This file = how to work and what to do next.
- Don't re-ask anything answered in either file.

## 0. How to work (non-negotiable)
- **User:** Dev, Master in Finance (IE Business School), ex-portfolio analyst. Use pro finance terms, skip the basics. Be casual and direct, no filler.
- **Bias to shipping.** The default action is: make the next concrete change, test it, report in 3 lines or fewer, then wait for "next".
- **Non-trivial choices:** give options with trade-offs, your pick and why, then let Dev decide. At most one question per turn.
- **One executor:** Claude Code does all the code. The claude.ai chat is for thinking and docs only.
- **After every material change**, update `PROJECT_NOTES.md`.
- **Keep diffs small.** Test with `--provider mock` (or `FUND_DB_PATH=/tmp/test.db`), then one real Groq run. No giant rewrites.
- **Never commit `.env` or any key.** The repo is private and each teammate uses their own Groq key. The old key leaked and must be rotated.
- **Goal:** a disciplined, risk-controlled, honestly evaluated system. Over 15 days the P&L is mostly noise, so push back on curve-fitting.
- **Inter-agent JSON schemas are contracts between teammates.** Flag any change in `PROJECT_NOTES.md`.

## 1. Brief (Prof. Jaume Manero; the PDF is `Mif_project_investment_AI_manager_tool_v1-4.pdf`)
- **Fund:** autonomous multi-agent paper-trading fund with $1M.
- **Init mode:** pick 10–20 names and deploy fully invested. The init day is exempt from the 5-trade cap; confirmation with Manero is still pending.
- **Daily mode:** max 5 trades/day and max 3 runs/day.
- **Instruments:** stocks and leveraged ETFs only.
- **Logging:** sequential trade CSV with a reason for every trade, plus a daily forum post.
- **Deliverables:** `app.py`, `database.py`, `agents.py`, `prices.csv`, `requirements.txt`, `finance.db`, the repo/zip without keys, a 15-day summary (Word), and a class presentation.

## 2. Locked decisions
- **LLM:** Groq free tier.
  - `openai/gpt-oss-120b` for the senior agents (PM, Risk, CIO); `openai/gpt-oss-20b` for the junior agents.
  - Providers are abstracted (groq/ollama/gemini/anthropic/mock).
- **Universe:** S&P 500 (from Wikipedia, `data/universe_sp500.csv`) plus US leveraged ETFs. This replaced the Russell 1000 plan on Oct 17, on Dev's instruction.
- **Screening funnel:** price-only and momentum-led, ending at ~30 stocks + holdings + ETFs ≈ 38 names. That size is set by Groq's 8K tokens/min limit.
- **Order of authority:** brief > IPS questionnaire > IPS Default column. Conflicts are logged in `PROJECT_NOTES.md`.
- **Agents:** 7 LLM agents + 1 rule-based Trader. Macro regime and tilts come from Python rules; the LLM explains them.
- **Division of labour:** the LLM judges; Python does all the maths and enforces the hard limits.
- **Machine:** Mac M2, 8GB RAM. Use `/opt/miniconda3/bin/python3 -m streamlit run app.py`.

## 3. Code map (regenerated Oct 20, end of the Oct 20 spec)
- **Pipeline** (`orchestrator.py`): data → two-stage screener (`screener.py`) → regime rules → Morning Brief (`morning_brief.py`,
  Gemini Flash) → reflection grading (`reflection.py`) → Macro / Research / Quant / Sentiment (`committee/`) → composite →
  PM → Devil's Advocate → PM answers → optimiser → `plan_orders` (≤ 5 trades, cash ≤ 3% upsizing, 10-name floor) →
  Risk Manager `check_book` on the post-trade book (3 rounds, then mechanical de-risk) → CIO → rule-based Trader.
  Execute is a separate step (`execute_run`: execution-time prices, 2 h stale guard, HALT kill switch, Trade Rationale
  Writer, backup). Other entry points: `--recheck` (C4.5 intraday news), `--snapshot` (21:45 after-close), `--replay N`, `--reset`.
- **Models** (`forecaster.py`, `forecasters/`): walk-forward LSTM (CPU), Kronos-mini (vendored), LightGBM (subprocess),
  horse race with C3.4 weights. FinBERT in `finbert.py`. Attribution in `attribution.py`. Calendar in `nyse_calendar.py`.
- **Data** (`data.py`): S&P 500 + 14 leveraged / 4 inverse ETFs, yfinance OHLCV (3y), fundamentals, earnings (B7), news
  (Yahoo/Google/SEC 8-K), macro + sector ETFs.
- **Runtime outputs:** live `finance.db`; replay `finance_replay.db`; `trades.csv` (E1 columns), `prices.csv`;
  `reports/daily_<date>.md`, `reports/final_summary.docx`; `backups/<ts>/` after every execute/snapshot.
- **Tests:** `tests/` (one file per stage); full status in `PROJECT_NOTES.md`.

## 4. Next up (in order)
1. **Live init Oct 22, 14:30 Madrid** (Dev): `streamlit run app.py` → Run Agents → read everything → Execute.
2. Dev rewrites the Quant (C3) and Risk (C6) `role_prompt`s; the code around them is final.
3. Stage 9 dashboard redesign (Streamlit themed) if not finished; then the presentation.
4. Decide: stop-loss cooldown (re-buying a name the day after its stop-loss); 0–10 scale for all agents.
5. Rotate the leaked Groq key; set `SEC_USER_AGENT` in `.env` to a team contact.

## 5. Research-backed add-ons (one per cycle; default order)
1. **Reflection memory** (from TradingAgents, ~98K stars): take the realized return vs SPY of past calls, turn it into a short lesson, and feed it to the PM.
2. **Thesis + exit/invalidation per trade**, re-checked daily. This fixes the Alpha Arena failure where LLMs abandoned their own exits.
3. **Bull/Bear debate** (TradingAgents), 1–2 rounds before the PM. It also covers IPS C0.1 (Devil's Advocate).
4. **Decision cards** in the daily reports (daily_stock_analysis style).
5. **Safe mode:** freeze trading after 3 consecutive agent failures.
6. **FinBERT** (ProsusAI/finbert) for Sentiment.
7. **Forecaster horse race:** LSTM vs Kronos-mini (NeoQuasar, MIT) vs LightGBM, all scored OOS on hit rate and IC. Report honestly: tree ensembles usually win.
8. **TTM + Log-HAR vol forecast** for Risk.
- **Skip:** LangGraph, paid data APIs, FinRL/RL, 7B+ local LLMs, crypto/prediction-market repos.

## 6. Dashboard (after the core work)
- **(a)** Themed Streamlit + Plotly, about 1 day.
- **(b)** Static HTML/JS reading a JSON export, with a scrubbable replay timeline. About 3–4 days. **Recommended.**
- **(c)** React + FastAPI: overkill.
- **Look:** dark navy, one accent, serif NAV numbers.

## 7. Open items for Dev
- Email Manero about whether the init day is exempt (C1), and about the universe (C2).
- Sign off on C11.
- Decide: dashboard route; whether to move all agents to the 0–10 scale.
- Rotate the Groq key.
- Fill in the owners table in `PROJECT_NOTES.md`.

## 8. Default action when invoked with no specific ask
1. Take the first unfinished item in §4.
2. Implement it as a small diff.
3. Test in mock mode.
4. Update `PROJECT_NOTES.md`.
5. Report in 3 lines or fewer, then wait for "next".
