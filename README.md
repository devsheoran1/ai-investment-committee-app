# AI Investment Committee

A paper-trading fund run by an **8-member AI investment committee** (7 LLM agents + a
rule-based Trader) for Prof. Manero's MiF Deep Learning competition. Universe: the
**S&P 500 + leveraged ETFs**, screened down to ~38 names per run. Max 5 trades/day,
max 3 runs/day, $1M starting capital. See **PROJECT_NOTES.md** for decisions, the IPS
mapping and open items.

```
 DATA + TOOLS (Python)            S&P 500 + ETFs · screening funnel · fundamentals · news · macro · LSTM
        │                                regime rules (IPS C1.3) · forced exits (stop-loss, news, ETF clock)
 ANALYSTS                         🌍 Macro Strategist   📑 Equity Research
        │                         🧠 Quant Analyst      📰 Sentiment Analyst
        ▼
 BLEND (Python)                   composite score per stock
        │
 PM ⇄ RISK NEGOTIATION            📈 Portfolio Manager → optimiser → ≤5 orders → 🛡️ Risk Manager
        │                         (Risk reviews the post-trade book; reject → PM revises, 2 rounds, then override)
        ▼
 SIGN-OFF                         👔 CIO  (approve = trade, reject = keep portfolio)
        │
 EXECUTION                        💹 Trader (rules) → "Execute Trades" button → finance.db + trades.csv
```

---

## 1. Setup (Mac)

The project runs on the Miniconda Python at `/opt/miniconda3/bin/python3` (it already has torch and yfinance).

```bash
cd ~/Downloads/ai_investment_committee
/opt/miniconda3/bin/python3 -m pip install -r requirements.txt
cp .env.example .env        # paste GROQ_API_KEY=... (free: https://console.groq.com/keys)
/opt/miniconda3/bin/python3 -m streamlit run app.py
```

**Daily routine (Madrid time):** 14:30 → *Run Agents*, read the reasoning, *Execute Trades*.
17:00–18:00 → forum post. 21:45 → *Advance Day* (after-close snapshot).

Terminal-only: `python orchestrator.py --provider mock` (propose) · `--execute` · `--snapshot` · `--reset`.
Pick `mock` to test offline without spending any API calls.

---

## 2. How an agent works (the 30-second version)

```
agent = role (system prompt) + inputs from Python tools + a JSON answer
```

1. **Python tools do all the maths**: downloading data, ratios, VaR, beta, the LSTM, the optimiser.
2. **The LLM only judges**: it reads a compact brief and answers in a fixed JSON format
   ("bullish/bearish, score, why").
3. **Python validates every answer** before anyone else sees it: clips scores, fills missing
   tickers, and enforces hard rules. For example, the Risk Manager *cannot* approve a portfolio
   that breaches a hard limit, and the CIO *cannot* overrule a Risk veto, whatever the model says.
4. **The orchestrator** (`orchestrator.py`) passes each agent's output to the next agent.
   That hand-off is what makes it "multi-agent".

Why this design: LLMs are great at judgment and terrible at arithmetic. Keeping the numbers
in Python makes the fund auditable. Every message is saved, so any meeting can be replayed
in the *Committee room* tab.

---

## 3. The team

| Agent | File | Model tier | Python tools it reads | Can it block? |
|---|---|---|---|---|
| 👔 CIO | `committee/cio.py` | smart | everything | Yes, final sign-off |
| 🌍 Macro Strategist | `committee/macro_strategist.py` | fast | VIX, 10y/3m yields, SPY trend, sector ETF momentum | No |
| 📑 Equity Research | `committee/equity_research.py` | fast | P/E, EV/EBITDA, ROE, margins, growth, leverage | No |
| 🧠 Quant Analyst | `committee/quant_analyst.py` | fast | momentum, RSI, moving averages, **LSTM forecasts + model card** | No |
| 📰 Sentiment Analyst | `committee/sentiment_analyst.py` | fast | Yahoo Finance headlines | No |
| 📈 Portfolio Manager | `committee/portfolio_manager.py` | smart | composite scores, current portfolio, Risk feedback | Sets convictions & exposure; optimiser sets weights |
| 🛡️ Risk Manager | `committee/risk_manager.py` | smart | vol, VaR/CVaR, beta, drawdown, concentration, sector limits | Yes, veto |
| 💹 Trader | `committee/trader.py` | rules (no LLM) | order list, 5-trade cap, cash | Rejects invalid orders |

Senior roles use the bigger model (`openai/gpt-oss-120b` on Groq) and junior roles the faster
one (`openai/gpt-oss-20b`). Change this in `config.py`.

### Suggested split for the group
Each person **owns 1–2 agents**: their role prompt, their Python tool, and their slide in the
final presentation. The JSON schema in each agent file is the "contract" between teammates.
Don't change it without telling the person downstream.

- Person A: Macro Strategist + macro data (`data.get_macro`)
- Person B: Equity Research + fundamentals (`data.get_fundamentals`)
- Person C: Quant Analyst + **the LSTM** (`forecaster.py`), the core Deep Learning content
- Person D: Sentiment Analyst (upgrade idea: FinBERT run locally)
- Person E: PM + optimiser (`analytics.optimise`)
- Person F: Risk Manager + risk engine (`analytics.risk_report`)
- Whoever's left: CIO, Trader, dashboard (`app.py`)

---

## 4. The deep-learning piece (`forecaster.py`)

- A 2-layer LSTM (PyTorch) is trained on all 12 stocks pooled together.
- Input: the last 30 days of volatility-scaled returns for the stock and for the S&P 500.
- Target: the stock's next-5-day return. Loss: Huber.
- It uses early stopping and a **time-based train/test split with a gap**, so there is no
  look-ahead.
- It reports an **out-of-sample hit rate and information coefficient**. The Quant Analyst is told
  to trust the model only as much as those numbers justify.

Expect a hit rate around 50–55%. That's normal: short-horizon returns are close to unpredictable,
and saying so with evidence is a stronger presentation point than a suspiciously good backtest.
Runs on CPU or the M2 GPU (MPS) in roughly 20–60 seconds.

---

## 5. Free tier: what to expect (Groq)

As of October 2026, the free plan limits are **per model**: about 30 requests/min,
1,000 requests/day and 8,000 tokens/min. Check your actual limits at
https://console.groq.com/settings/limits.

- One meeting = 8–12 calls, split across two models. Daily limits are no issue.
- The **8K tokens/min** limit is the one you'll hit. When that happens, the app automatically
  waits and retries, so a meeting takes about 1–3 minutes. That's normal; leave it alone.
- Want it faster or smarter for the final demo? Set `LLM_PROVIDER=anthropic` (paid Claude API)
  in `.env`. Nothing else changes.

| Provider | `.env` setting | Cost | Notes |
|---|---|---|---|
| Groq | `LLM_PROVIDER=groq` | Free | Default |
| Ollama (local) | `LLM_PROVIDER=ollama` | Free | Install from ollama.com, then `ollama pull llama3.2:3b`. Weaker on 8GB RAM |
| Gemini | `LLM_PROVIDER=gemini` | Free tier | Check model names in `config.py` |
| Claude | `LLM_PROVIDER=anthropic` | Paid | Best quality |
| Mock | `LLM_PROVIDER=mock` | Free | No AI, for offline testing |

---

## 6. Files

| File | What it does |
|---|---|
| `config.py` | Every setting, with the brief/IPS section each number comes from |
| `data.py` | S&P 500 universe, yfinance prices (→ `prices.csv`), fundamentals, RSS news, macro |
| `screener.py` | Screening funnel: hard filters → momentum ranking → shortlist |
| `analytics.py` | Technicals, regime rules, composite, optimiser, risk report, 5-trade order planner |
| `forecaster.py` | The LSTM |
| `agents.py` | Brief deliverable: maps Data/Strategy/Risk/Execution agents onto `committee/` |
| `committee/` | The 8 agents, one file each (`base.py` explains the pattern) |
| `orchestrator.py` | The meeting: Run Agents (propose) and Execute (commit) |
| `database.py` | SQLite `finance.db` (brief schema) + `trades.csv` export |
| `app.py` | Streamlit dashboard (brief §8.3 layout) |
| `PROJECT_NOTES.md` | Decisions, conflicts, IPS mapping, owners |

## 7. Ideas to go further
- Run the meeting daily with a scheduler, so the NAV-vs-SPY chart builds up a track record.
- Let agents call tools themselves (native tool-calling), e.g. Research pulls a 10-K section on demand.
- Add a 🐻 Devil's Advocate agent that argues against the PM's top 3 bets before Risk reviews.
- Replace keyword sentiment with FinBERT, a small transformer that runs fine on an M2.
- Track each analyst's hit rate over time and give the more accurate ones more weight in the blend.
