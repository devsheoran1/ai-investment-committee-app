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
        "fast": "gemini-2.5-flash-lite",
        "smart": "gemini-2.5-flash",
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
LEVERAGED_ETFS = {                      # IPS B2 default list (3x unless noted)
    "TQQQ": "Nasdaq-100", "UPRO": "S&P 500", "SOXL": "Semiconductors",
    "TECL": "Technology", "FAS": "Financials", "TNA": "Russell 2000",
    "LABU": "Biotech", "ERX": "Energy (2x)",
}
INVERSE_ETFS = {"SQQQ": "Nasdaq-100 (-3x)", "SPXU": "S&P 500 (-3x)"}   # IPS A11: hedges only
LEVERAGE_FACTOR = {**{t: 3.0 for t in LEVERAGED_ETFS}, "ERX": 2.0,
                   **{t: -3.0 for t in INVERSE_ETFS}}

PRICE_HISTORY = "2y"     # enough for 12-1m momentum + the LSTM; keeps the S&P 500 download fast

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
    "max_cash_weight": 0.05,            # IPS A12: "not more than 5%"
    "max_portfolio_vol": 0.25,          # IPS A3 default
    "max_var_95_1d": 0.05,              # IPS A5
    "max_beta": None,                   # IPS A6: "any" (regime bands in C1.4 are soft checks)
    "max_leveraged_total": 0.30,        # IPS A10
    "max_leveraged_each": 0.15,         # IPS A10
    "max_inverse_total": 0.25,          # IPS A11 (only in risk-off)
    "max_drawdown": -0.10,              # IPS A4: from peak NAV
}
STOP_LOSS = -0.10                       # IPS A17: -10% from average cost -> forced sell
LEVERAGED_MAX_HOLD_DAYS = 5             # IPS C6.4 default / C5.4
NEWS_SELL_TRIGGER = -0.8                # IPS C4.4
DRIFT_TOLERANCE = 0.05                  # IPS A16: resizes smaller than this are skipped

# Trading rules (Brief §1 + §10)
MAX_TRADES_PER_DAY = 5                  # Brief §1: "5 trades (5 stocks)"
INIT_DAY_EXEMPT_FROM_CAP = True         # IPS D1: initialisation needs 10-20 buys (confirm with Prof. Manero)
MAX_RUNS_PER_DAY = 3                    # Brief §10.2
MIN_HOLDINGS, MAX_HOLDINGS = 10, 20     # Brief §1: "10 or 20"
TARGET_HOLDINGS_INIT = 15
MAX_DAILY_TURNOVER = 0.25               # IPS A18, one-way (buys + sells) / 2 / NAV
MIN_TRADE_NOTIONAL = 25_000             # IPS C8.4
TRANSACTION_COST_BPS = 0                # IPS A20: nil unless the professor sets one
NAV_JUMP_HALT = 0.20                    # IPS C8.6: >20% NAV move vs last snapshot = data error
PRICE_JUMP_EXCLUDE = 0.20               # IPS D10: per-ticker pre-flight check

# Exposure per regime (IPS C1.4). Always >= 95% invested (A12).
REGIME_RULES = {
    "risk_on":  {"equity_exposure": 0.98, "max_leveraged": 0.30, "inverse": (0.0, 0.0),  "beta": (1.2, 1.6)},
    "neutral":  {"equity_exposure": 0.98, "max_leveraged": 0.10, "inverse": (0.0, 0.0),  "beta": (0.9, 1.2)},
    "risk_off": {"equity_exposure": 0.98, "max_leveraged": 0.0,  "inverse": (0.10, 0.15), "beta": (0.3, 0.7)},
}
RISK_OFF_HIGH_VIX = 30                  # IPS C1.4: inverse hedge up to 25% if VIX > 30

MAX_RISK_ROUNDS = 2                     # IPS C6.5

# How the composite blends analyst scores (IPS C5.2, must sum to 1)
SIGNAL_WEIGHTS = {"quant": 0.40, "sentiment": 0.30, "research": 0.15, "macro": 0.15}

# --------------------------------------------------------------------------- #
# Deep learning model
# --------------------------------------------------------------------------- #
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
