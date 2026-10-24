"""
Central configuration. Everything you might want to tweak lives here.

Switching the "brain" of the agents = change LLM_PROVIDER in your .env file:
    groq      -> free cloud API (default, recommended)
    ollama    -> free local model on your laptop (slower / weaker on 8GB RAM)
    gemini    -> Google AI Studio free tier
    anthropic -> Claude API (paid, best quality)
    mock      -> no AI at all; rule-based fake answers so the app runs offline
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# LLM provider
# --------------------------------------------------------------------------- #
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Every provider below speaks the OpenAI-compatible chat API, so one client
# works for all of them. "fast" = junior analysts, "smart" = senior roles.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        # Free plan limits are PER MODEL, so using two different models
        # doubles the throughput we get for free.
        "fast": "openai/gpt-oss-20b",
        "smart": "openai/gpt-oss-120b",
        "extra": {"reasoning_effort": "low"},  # keeps token use (and rate limits) down
        "json_mode": True,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "fast": "llama3.2:3b",   # fits in 8GB RAM. `ollama pull llama3.2:3b`
        "smart": "llama3.2:3b",
        "extra": {},
        "json_mode": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        # Check https://ai.google.dev/gemini-api/docs/models for current free models
        "fast": "gemini-3.5-flash-lite",      # 2.5 models are retired for new keys (checked Oct 20)
        "smart": "gemini-3.5-flash",
        "extra": {},
        "json_mode": True,
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "key_env": "ANTHROPIC_API_KEY",
        "fast": "claude-haiku-5-5",
        "smart": "claude-sonnet-5-5",
        "extra": {},
        "json_mode": False,
    },
    "mock": {"base_url": None, "key_env": None, "fast": "mock", "smart": "mock",
             "extra": {}, "json_mode": False},
}

# Optional per-agent model override, e.g. {"Risk Manager": "openai/gpt-oss-120b"}
MODEL_OVERRIDES: dict[str, str] = {}

LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 6

# --------------------------------------------------------------------------- #
# Fund / mandate
# Source of truth, in this order: (1) the professor's brief (PDF v1-4),
# (2) our IPS questionnaire, (3) defaults. "Brief" / "IPS A8" etc. in the
# comments say where each number comes from.
# --------------------------------------------------------------------------- #
FUND_NAME = "AI Investment Committee"
STARTING_CAPITAL = 1_000_000.0          # Brief §10: $1,000,000 competition allocation
BENCHMARK = "SPY"                       # IPS A1 default

# Universe = S&P 500 constituents (downloaded from Wikipedia, cached weekly)
# + leveraged ETFs. Brief §10.3: "Trading is restricted to stocks and leveraged ETFs."
UNIVERSE_SOURCE = "sp500"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
LEVERAGED_ETFS = {                      # IPS B2 (Oct 20 list; screener keeps only ADV > $10m). 3x unless noted
    "TQQQ": "ProShares UltraPro QQQ (3x Nasdaq-100)", "UPRO": "ProShares UltraPro S&P 500 (3x)",
    "SPXL": "Direxion Daily S&P 500 Bull 3x", "SOXL": "Direxion Daily Semiconductor Bull 3x",
    "TECL": "Direxion Daily Technology Bull 3x", "FAS": "Direxion Daily Financial Bull 3x",
    "TNA": "Direxion Daily Small Cap Bull 3x", "LABU": "Direxion Daily S&P Biotech Bull 3x",
    "ERX": "Direxion Daily Energy Bull 2x", "CURE": "Direxion Daily Healthcare Bull 3x",
    "DRN": "Direxion Daily Real Estate Bull 3x", "NAIL": "Direxion Daily Homebuilders Bull 3x",
    "DPST": "Direxion Daily Regional Banks Bull 3x", "UDOW": "ProShares UltraPro Dow30 (3x)",
}
INVERSE_ETFS = {                        # IPS A11: hedges only, risk-off only
    "SQQQ": "ProShares UltraPro Short QQQ (-3x)", "SPXU": "ProShares UltraPro Short S&P 500 (-3x)",
    "SOXS": "Direxion Daily Semiconductor Bear 3x", "SDOW": "ProShares UltraPro Short Dow30 (-3x)",
}
ETF_NAMES = {**LEVERAGED_ETFS, **INVERSE_ETFS}
LEVERAGE_FACTOR = {**{t: 3.0 for t in LEVERAGED_ETFS}, "ERX": 2.0,
                   **{t: -3.0 for t in INVERSE_ETFS}}
# Underlying index proxy per ETF (IPS C6.4 decay check: ETF return vs leverage x underlying)
ETF_UNDERLYING = {"TQQQ": "QQQ", "UPRO": "SPY", "SPXL": "SPY", "SOXL": "SOXX", "TECL": "XLK", "FAS": "XLF",
                  "TNA": "IWM", "LABU": "XBI", "ERX": "XLE", "CURE": "XLV", "DRN": "XLRE", "NAIL": "ITB",
                  "DPST": "KRE", "UDOW": "DIA", "SQQQ": "QQQ", "SPXU": "SPY", "SOXS": "SOXX", "SDOW": "DIA"}

PRICE_HISTORY = "3y"     # 12-1m momentum + the 250-day monthly walk-forward of the LSTM/LightGBM (C3.4/C3.5)

# Screening funnel (IPS Part B). Python screens ~510 names down to a shortlist
# so the agents never see more than a few dozen tickers (Brief §6: token budget).
SCREEN = {
    "min_price": 1.0,                   # IPS B3
    "min_adv_usd": 10e6,                # IPS B4: 20-day average dollar volume
    "min_history_days": 252,            # IPS B6: 1 year of clean data
    "max_vol_3m_stocks": 0.80,          # IPS B8 default (stocks only; 3x ETFs are exempt)
    "sector_neutral": True,             # IPS B10 default
    "shortlist_size": 30,               # IPS B11 default is 40; 30 keeps Groq free tier workable
    # IPS B9 was left blank. The default (30/15/25/15/15 incl. quality + value)
    # contradicts A14 (momentum focus) and needs fundamentals for all 500 names,
    # which yfinance can't serve in one run. So the funnel is price-based and
    # momentum-led; quality/valuation come in at the Research step instead.
    "factor_weights": {"mom_12_1": 0.35, "mom_3m": 0.25, "trend_200dma": 0.25, "low_vol": 0.15},
}

# --------------------------------------------------------------------------- #
# Risk limits. Python enforces these; the Risk Manager LLM cannot override them.
# --------------------------------------------------------------------------- #
RISK_LIMITS = {
    "max_single_weight": 0.20,          # IPS A8 (Brief mega-prompt says 40%; ours is stricter)
    "max_sector_weight": 0.35,          # IPS A9
    "min_cash_weight": 0.0,             # Brief §8.2 constraint 1: cash >= 0
    "max_cash_weight": 0.05,            # IPS A12 (Oct 21): cash 0-5%, i.e. >= 95% invested, post-trade book
    "max_portfolio_vol": 0.35,          # IPS A3 (Oct 21: raised from 25%)
    "max_var_95_1d": 0.05,              # IPS A5
    "max_beta": None,                   # IPS A6: "any" (regime bands in C1.4 are soft checks)
    "max_leveraged_total": 0.30,        # IPS A10
    "max_leveraged_each": 0.15,         # IPS A10
    "max_inverse_total": 0.25,          # IPS A11 (only in risk-off)
    "max_drawdown": -0.10,              # IPS A4: from peak NAV (graduated, see DD_* below)
}
STOP_LOSS = -0.10                       # IPS A17: -10% from average cost -> forced sell
# IPS C6.4 (Oct 20): no max holding period while a leveraged ETF is in profit; re-approval needed when
# its return < 0, it is > 12% below its peak since entry, or it lags leverage x underlying by > decay tolerance.
LEV_REVIEW_PEAK_DD = -0.12
LEV_REVIEW_DECAY_TOL = 0.02
# Vol-targeted leverage sleeve: ETF cap = base cap x (1y median 20d vol / current 20d vol), never above the IPS caps
LEV_VOL_WINDOW = 20
LEV_VOL_TARGET_LOOKBACK = 252
LEV_BASE_CAP_EACH = 0.15                # IPS A10 per leveraged ETF
INV_BASE_CAP_EACH = 0.25                # IPS A11 inverse total (single hedge ETF can carry it)
# IPS A4 / C11 (signed off Oct 20, graduated)
DD_WARN, DD_HARD = -0.07, -0.10         # from peak NAV
DD_WARN_LEV_CAP = 0.10                  # at -7%: leveraged sleeve <= 10%, no new leveraged buys
DD_HARD_INVERSE_MIN = 0.15              # at -10%: all leveraged longs out; in risk-off inverse hedge >= 15%
DD_RECOVERY_DAYS = 2                    # restrictions lift after 2 consecutive days back above -7%
NEWS_SELL_TRIGGER = -0.8                # IPS C4.4
DRIFT_TOLERANCE = 0.05                  # IPS A16: resizes smaller than this are skipped

# Trading rules (Brief §1 + §10)
MAX_TRADES_PER_DAY = 5                  # Brief §1: "5 trades (5 stocks)"
INIT_DAY_EXEMPT_FROM_CAP = True         # IPS D1: initialisation needs 10-20 buys (confirm with Prof. Manero)
MAX_RUNS_PER_DAY = 3                    # Brief §10.2
MIN_HOLDINGS, MAX_HOLDINGS = 10, 20     # Brief §1: "10 or 20"; 10 is a hard floor every day (IPS A7)
TARGET_HOLDINGS = (12, 15)              # IPS A7 (Oct 20): buffer so one exit never breaches the floor
TARGET_HOLDINGS_INIT = 14
TARGET_CASH = 0.01                      # where upsizing aims when cash would exceed the 5% cap
MAX_DAILY_TURNOVER = 0.25               # IPS A18, one-way (buys + sells) / 2 / NAV
MIN_TRADE_NOTIONAL = 25_000             # IPS C8.4
TRANSACTION_COST_BPS = 0                # IPS A20: nil unless the professor sets one
NAV_JUMP_HALT = 0.20                    # IPS C8.6: >20% NAV move vs last snapshot = data error
PRICE_JUMP_EXCLUDE = 0.20               # IPS D10: per-ticker pre-flight check

# Exposure per regime (IPS C1.4). Always >= 95% invested (A12: cash <= 5%).
# Any ONE risk-off signal blocks new leveraged buys (Oct 21); 2 of 4 = full risk-off (analytics.macro_regime).
REGIME_RULES = {
    "risk_on":  {"equity_exposure": 0.98, "max_leveraged": 0.30, "inverse": (0.0, 0.0),  "beta": (1.2, 1.6)},
    "neutral":  {"equity_exposure": 0.98, "max_leveraged": 0.10, "inverse": (0.0, 0.0),  "beta": (0.9, 1.2)},
    "risk_off": {"equity_exposure": 0.98, "max_leveraged": 0.0,  "inverse": (0.10, 0.15), "beta": (0.3, 0.7)},
}
RISK_OFF_HIGH_VIX = 30                  # IPS C1.4: inverse hedge up to 25% if VIX > 30

MAX_RISK_ROUNDS = 3                     # IPS C6.5 (Oct 20)
# IPS C6.3: the only soft concerns allowed to reject
SOFT_REJECT = {"min_effective_positions": 4, "max_top3_pair_corr": 0.90, "max_adv_fraction": 0.25}

# --------------------------------------------------------------------------- #
# Daily market awareness (Stage 5, Oct 20)
# --------------------------------------------------------------------------- #
MORNING_BRIEF_PROVIDER = "gemini"       # C0.3 / spec: Gemini Flash free tier; falls back to gpt-oss-120b on Groq
JUNIOR_GEMINI_FALLBACK = False          # if True, a junior agent hitting repeated Groq 429s retries on Gemini Flash
JUNIOR_GEMINI_FALLBACK_AFTER_429 = 3
# C1.2 macro calendar for the live window (Madrid dates; sources: federalreserve.gov via fedratecalc, BLS, BEA)
MACRO_CALENDAR = [
    {"date": "2026-10-28", "event": "FOMC rate decision (14:00 ET / 19:00 Madrid), press conference 14:30 ET"},
    {"date": "2026-10-27", "event": "FOMC meeting day 1"},
    {"date": "2026-10-29", "event": "PCE / Personal Income & Outlays (Sep) + Q3 GDP advance, 08:30 ET"},
    {"date": "2026-10-30", "event": "Employment Cost Index Q3, 08:30 ET"},
    {"date": "2026-11-06", "event": "Employment Situation / NFP (Oct), 08:30 ET"},
    {"date": "2026-11-10", "event": "CPI (Oct), 08:30 ET"},
]
EARNINGS_SOON_DAYS = 5                  # B7: flag names reporting within 5 trading days
PEAD_WINDOW_DAYS = 20                   # post-earnings drift decays to 0 over 20 trading days
FAST_TRACK = {"min_sentiment": 0.7, "min_quant_10": 5.0, "max_weight": 0.10}   # C5.3 earnings fast track

# Two-stage screener (IPS B9, decided Oct 20). Momentum/trend capped at 50% of the effective weight.
SCREEN_STAGE1 = {"weights": {"mom_12_1_volscaled": 0.39, "industry_mom_1m": 0.24, "trend": 0.25, "reversal_1w": 0.12},
                 "keep": 60, "per_sector": 5, "overall": 5}      # top 5 per GICS sector + the 5 best overall
SCREEN_STAGE2 = {"weights": {"stage1": 0.57, "revisions_drift": 0.27, "quality": 0.16}, "keep": 30,
                 "min_per_sector": 1, "max_per_sector": 5}

# 20-day replay (IPS D6-D9, Stage 6). Separate DB, decisions on data up to the previous close, fills at the open.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY = {"db": os.path.join(_HERE, "finance_replay.db"), "trades_csv": os.path.join(_HERE, "trades_replay.csv"),
          "report_dir": os.path.join(_HERE, "reports", "replay"), "lstm_wf_days": 60, "lstm_retrain_every": 30,
          "fill_time_madrid": "15:30"}

# How the composite blends analyst scores (IPS C5.2, must sum to 1)
SIGNAL_WEIGHTS = {"quant": 0.40, "sentiment": 0.30, "research": 0.15, "macro": 0.15}

# --------------------------------------------------------------------------- #
# Deep learning model
# --------------------------------------------------------------------------- #
LSTM_DEVICE = os.getenv("LSTM_DEVICE", "cpu")    # cpu is 2.5x faster than MPS for this small LSTM (measured Oct 20)
LSTM = {
    "lookback": 30,       # days of returns fed to the network
    "horizon": 5,         # predict cumulative return over next 5 trading days
    "hidden": 32,
    "layers": 2,
    "epochs": 25,
    "lr": 1e-3,
    "seed": 42,
}

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.getenv("FUND_DB_PATH", os.path.join(BASE_DIR, "finance.db"))   # Brief §2: "finance.db"
TRADES_CSV = os.getenv("FUND_TRADES_CSV", os.path.join(BASE_DIR, "trades.csv"))   # Brief §10.4: sequential trade log
PRICES_CSV = os.path.join(BASE_DIR, "prices.csv")   # Brief §11 deliverable
UNIVERSE_CSV = os.path.join(DATA_DIR, "universe_sp500.csv")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Multi-strategy ensemble inside the Quant agent (Oct 23) + profit-taking without cash
# --------------------------------------------------------------------------- #
ENSEMBLE = {
    "enabled": True,
    "blend": 1.0,                  # rule_score = (1 - blend) x legacy technical/model score + blend x ensemble
                                   # Oct 24: 0.75 -> 1.0 (Quant = 100% ensemble). Chosen AFTER seeing the 2023-26 backtest
                                   # (in-sample, not significant); the legacy score only survives as the fallback and tracker
    "weight_mode": "shrinkage",    # "shrinkage": w ~ 1 + shrink_lambda x max(t, 0), never all-zero | "ic_legacy": the Oct 23 rule
    "shrink_lambda": 0.5,          # a validated strategy with t = 2 gets 2x the equal prior, t = 4 gets 3x (before the caps)
    "ic_sessions": 60,             # rolling window of out-of-sample ICs (5-day forward return)
    "ic_horizon": 5,
    "weight_cap": 0.20,            # per-strategy cap
    "family_caps": {"pead_sue": 0.30},   # PEAD is the best-documented anomaly here: 30%; every other strategy 20%
    "min_t": 1.0,                  # t-stat below this -> weight 0
    "unvalidated": "equal",        # snapshot-only signals (no point-in-time history): "equal" prior or "zero"
    "snapshot_prior": 0.5,         # snapshot-only prior = 0.5 x 1/N ...
    "snapshot_cap": 0.05,          # ... capped at 5% each, replaced by the measured IC once 60 sessions exist
    "vote_z": 0.5,                 # a strategy "agrees" when its z >= +0.5
    "min_votes": 3,                # new entries need >= 3 agreeing strategies (fewer if fewer apply, e.g. ETFs)
    "tom_tilt": 0.03,              # turn-of-month: +0.03 on the Quant score, never a vote
    "z_to_score": 1.5,             # ensemble z -> score: clip(z / 1.5, -1, 1)
}
VOL_MANAGED = {"enabled": True, "lookback": 63, "floor": 0.6, "cap": 1.25}   # optimiser overlay (Moreira-Muir style)
TRAILING_STOP = -0.12                   # exit if a STOCK falls 12% from its peak since entry (Oct 24: was -8%; plus A17 -10% from cost)
ENTRY_COMPOSITE = 6.3                   # Oct 24: new-name bar on the 0-10 composite (was 7); the "beat the weakest holding by >= 1" rule stays
PAIR_TRIMS = True                       # A16 drift trims are paired with a buy of the best unheld idea
# "Top" US politicians whose disclosed BUYS the Sentiment agent is shown (editable; matched on the member's name)
POLITICIAN_SIGNAL = False   # Oct 23: off (no QUIVER_API_KEY, STOCK Act filings lag up to 45 days); not read by Sentiment
POLITICIAN_WATCHLIST = ["Pelosi", "Crenshaw", "Gottheimer", "McCaul", "Khanna", "Tuberville", "Greene", "Green, Mark",
                        "Sessions", "Fallon", "Cleo Fields", "Wasserman", "Rouzer"]

# --------------------------------------------------------------------------- #
# Regime allocator over strategy families (Oct 24). Python decides; the Quant LLM may tilt ONE family by +-20%/day.
# --------------------------------------------------------------------------- #
STRATEGY_FAMILIES = {
    "momentum":  ["xs_momentum_12_1", "ts_trend", "residual_momentum", "industry_momentum", "high_52w"],
    "earnings":  ["pead_sue", "eps_revisions", "earnings_premium"],
    "reversal":  ["st_reversal_1w"],
    "defensive": ["quality_gp", "low_beta"],
}
REGIME_ALLOC = {
    "enabled": True,
    # SPY-based state, first match wins in this order. Settings were chosen AFTER the first backtest (in-sample).
    "dd_from_peak": -0.05, "peak_window": 252, "rebound_from_low": 0.05, "low_window": 63, "vol_window": 20, "vol_median_window": 252,
    "multipliers": {   # family weight x multiplier, then renormalised to the same total and re-capped
        "calm_uptrend":      {"momentum": 1.2, "earnings": 1.0, "reversal": 0.5, "defensive": 0.8},
        "drawdown_high_vol": {"momentum": 0.5, "earnings": 1.5, "reversal": 1.5, "defensive": 1.5},
        "rebound":           {"momentum": 0.3, "earnings": 1.2, "reversal": 2.0, "defensive": 1.0},
        "normal":            {"momentum": 1.0, "earnings": 1.0, "reversal": 1.0, "defensive": 1.0},
    },
}
LLM_TILT_MAX = 0.20                     # Quant may scale ONE family's weight by 1 +- 0.20 per day, with a written reason; logged
SNAPSHOT_Z_LOG = os.getenv("FUND_SNAPSHOT_Z_LOG", os.path.join(_HERE, "reports", "snapshot_z_log.csv"))   # daily z of snapshot-only signals

# Soft hedge (Oct 24): ANY ONE C1.3 risk-off signal (short of full risk-off) lets Quant/Macro propose 0-10% SPXU; the PM
# decides; the Risk Manager enforces only the 25% inverse cap (A11) there. Full risk-off keeps the old IPS A11 range.
SOFT_HEDGE = {"enabled": True, "ticker": "SPXU", "max": 0.10}
