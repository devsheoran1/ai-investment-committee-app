"""
Market data layer (the brief's "Data Agent" tools).

    universe      S&P 500 constituents from Wikipedia (cached weekly) + leveraged ETFs
    prices        yfinance batch download of the whole universe (Brief §5), exported to prices.csv
    fundamentals  yfinance .info, only for the shortlist (Research reads these)
    news          Yahoo Finance per-ticker RSS, Google News RSS as fallback (Brief §4)
    macro         VIX, rates, credit, dollar, oil, sector ETF momentum (IPS C1.2)

If Yahoo is down: last good download -> synthetic prices in a separate sandbox DB.
Fundamentals and news never fall back to fake data on a real-data day: a missing
value is reported as missing, so agents say "insufficient data" instead of guessing.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import pickle
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import numpy as np
import pandas as pd

import config

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Sector ETF per GICS sector (Wikipedia's S&P 500 table uses GICS names)
SECTOR_ETFS = {
    "Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Health Care": "XLV", "Consumer Staples": "XLP", "Consumer Discretionary": "XLY",
    "Industrials": "XLI", "Utilities": "XLU", "Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
LEVERAGED_SECTOR = "Leveraged ETF"
INVERSE_SECTOR = "Inverse ETF"

# Used only if Wikipedia is unreachable and nothing is cached yet
_FALLBACK_UNIVERSE = {
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AMZN": "Consumer Discretionary", "GOOGL": "Communication Services", "META": "Communication Services",
    "JPM": "Financials", "V": "Financials", "XOM": "Energy", "JNJ": "Health Care",
    "UNH": "Health Care", "PG": "Consumer Staples", "KO": "Consumer Staples", "CAT": "Industrials",
    "LIN": "Materials", "NEE": "Utilities", "PLD": "Real Estate", "HD": "Consumer Discretionary",
}

DATA_SOURCE = {"prices": "unknown", "synthetic": False, "universe": "unknown"}  # shown in the UI
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (IE MiF student project; paper trading research)"}


# --------------------------------------------------------------------------- #
# caching helpers
# --------------------------------------------------------------------------- #
def _cache_path(name: str) -> str:
    return os.path.join(config.CACHE_DIR, f"{dt.date.today().isoformat()}_{name}.pkl")


def _cached(name: str, fn, max_age_min: float | None = None, keep_if=lambda v: True):
    """Day-scoped pickle cache. max_age_min refreshes intraday (prices, news),
    so the 21:45 after-close run doesn't reuse the 14:30 pre-open download."""
    path = _cache_path(name)
    if os.path.exists(path):
        fresh = max_age_min is None or (time.time() - os.path.getmtime(path)) / 60 < max_age_min
        if fresh:
            with open(path, "rb") as f:
                return pickle.load(f)
    val = fn()
    if keep_if(val):
        with open(path, "wb") as f:
            pickle.dump(val, f)
    return val


def _key(prefix: str, tickers: list[str]) -> str:
    return f"{prefix}_{len(tickers)}_{zlib.crc32(','.join(sorted(tickers)).encode()):08x}"


def _latest_cached_real(name: str):
    files = sorted(f for f in os.listdir(config.CACHE_DIR) if f.endswith(f"_{name}.pkl"))
    if not files:
        return None
    with open(os.path.join(config.CACHE_DIR, files[-1]), "rb") as f:
        return pickle.load(f), files[-1][:10]


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #
def _download_sp500() -> pd.DataFrame:
    import requests
    r = requests.get(config.SP500_URL, headers=_HTTP_HEADERS, timeout=20)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    df = pd.DataFrame({
        "symbol": t["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),  # BRK.B -> BRK-B
        "name": t["Security"].astype(str),
        "sector": t["GICS Sector"].astype(str),
        "industry": t["GICS Sub-Industry"].astype(str),                      # industry momentum (B9)
        "cik": t["CIK"].astype(str).str.zfill(10),                           # SEC EDGAR 8-K feed (C4.2)
        "asset_type": "stock",
    })
    if len(df) < 450:
        raise RuntimeError(f"S&P 500 table looks wrong ({len(df)} rows)")
    df["fetched"] = dt.date.today().isoformat()
    return df


def get_universe() -> pd.DataFrame:
    """S&P 500 (refreshed weekly) + leveraged/inverse ETFs. Columns: symbol, name, sector, asset_type."""
    path = config.UNIVERSE_CSV
    stocks = None
    if os.path.exists(path):
        cached = pd.read_csv(path)
        age = (dt.date.today() - dt.date.fromisoformat(str(cached["fetched"].iloc[0]))).days
        if age <= 7 and {"industry", "cik"} <= set(cached.columns):
            stocks, DATA_SOURCE["universe"] = cached, f"S&P 500 (Wikipedia, {cached['fetched'].iloc[0]})"
    if stocks is None:
        try:
            stocks = _download_sp500()
            stocks.to_csv(path, index=False)
            DATA_SOURCE["universe"] = f"S&P 500 (Wikipedia, {stocks['fetched'].iloc[0]})"
        except Exception as e:  # noqa: BLE001
            if os.path.exists(path):
                stocks = pd.read_csv(path)
                DATA_SOURCE["universe"] = f"S&P 500 (cached {stocks['fetched'].iloc[0]}; refresh failed)"
            else:
                stocks = pd.DataFrame({"symbol": list(_FALLBACK_UNIVERSE), "name": list(_FALLBACK_UNIVERSE),
                                       "sector": list(_FALLBACK_UNIVERSE.values()), "asset_type": "stock"})
                DATA_SOURCE["universe"] = f"FALLBACK 18 stocks (Wikipedia failed: {str(e)[:60]})"
    etfs = pd.DataFrame(
        [{"symbol": t, "name": d, "sector": LEVERAGED_SECTOR, "industry": LEVERAGED_SECTOR, "cik": None,
          "asset_type": "leveraged_etf"} for t, d in config.LEVERAGED_ETFS.items()]
        + [{"symbol": t, "name": d, "sector": INVERSE_SECTOR, "industry": INVERSE_SECTOR, "cik": None,
            "asset_type": "inverse_etf"} for t, d in config.INVERSE_ETFS.items()])
    for col in ("industry", "cik"):
        if col not in stocks.columns:
            stocks[col] = None
    out = pd.concat([stocks[["symbol", "name", "sector", "industry", "cik", "asset_type"]], etfs], ignore_index=True)
    return out.drop_duplicates("symbol").reset_index(drop=True)


_NAMES: dict = {}


def names_map() -> dict[str, str]:
    """ticker -> company/ETF name (IPS E7), from data/universe_sp500.csv + config.ETF_NAMES."""
    if not _NAMES:
        try:
            u = pd.read_csv(config.UNIVERSE_CSV)
            _NAMES.update(dict(zip(u["symbol"], u["name"])))
        except (OSError, KeyError, ValueError):
            pass
        _NAMES.update(config.ETF_NAMES)
    return _NAMES


def label(ticker: str) -> str:
    """'PSX · Phillips 66' (IPS E7)."""
    n = names_map().get(ticker)
    return f"{ticker} · {n}" if n else ticker


# --------------------------------------------------------------------------- #
# prices (OHLCV for the whole universe in one batch)
# --------------------------------------------------------------------------- #
def _download(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Chunks of 50 -> per-ticker retry -> Stooq (prices.py). Oct 20: Yahoo's single batch returned 382/513."""
    import prices
    fields = prices.download_history(tickers, period)
    close = fields["close"]
    if config.BENCHMARK not in close.columns or close[config.BENCHMARK].isna().all():
        raise RuntimeError("benchmark missing from download")
    have = [t for t in tickers if t in close.columns and close[t].notna().sum() > 0]
    if len(have) < 0.5 * len(tickers):
        raise RuntimeError(f"only {len(have)}/{len(tickers)} tickers downloaded")
    idx = close[config.BENCHMARK].dropna().index     # trading days only
    return {k: v.reindex(idx)[[t for t in have if t in v.columns]].sort_index() for k, v in fields.items()}


def _synthetic_prices(tickers: list[str], days: int = 504, seed: int = 7) -> pd.DataFrame:
    """Correlated GBM with mild momentum so the demo has something to find."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    mkt = rng.normal(0.0004, 0.010, days)
    rets = np.zeros((days, n))
    for i, tk in enumerate(tickers):
        lev = config.LEVERAGE_FACTOR.get(tk)
        if lev:
            rets[:, i] = lev * mkt
            continue
        beta, idio_vol, drift = rng.uniform(0.6, 1.4), rng.uniform(0.008, 0.02), rng.uniform(-0.0002, 0.0007)
        eps = rng.normal(0, idio_vol, days)
        for t in range(1, days):  # small AR(1) term = learnable signal
            eps[t] += 0.05 * eps[t - 1]
        rets[:, i] = mkt if tk == config.BENCHMARK else drift + beta * mkt + eps
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)


def get_market_data(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """{'close','open','volume'} DataFrames (dates x tickers). Real data if at all
    possible: fresh download (<60 min old) -> last good download -> synthetic."""
    if tickers is None:
        tickers = get_universe()["symbol"].tolist()
    # + the ETFs' underlying indices (IPS C6.4 decay check); they are never ranked or traded
    tickers = list(dict.fromkeys(list(tickers) + [config.BENCHMARK] + sorted(set(config.ETF_UNDERLYING.values()))))
    name = _key("market", tickers)
    try:
        md = _cached(name, lambda: _download(tickers, config.PRICE_HISTORY), max_age_min=60)
        DATA_SOURCE.update(prices=f"yfinance ({len(md['close'].columns)}/{len(tickers)} tickers)", synthetic=False)
    except Exception as e:  # noqa: BLE001
        stale = _latest_cached_real(name)
        if stale is not None:
            md, day = stale
            DATA_SOURCE.update(prices=f"yfinance (cached from {day}; today's download failed)", synthetic=False)
        else:
            close = _synthetic_prices(tickers)
            md = {"close": close, "open": close.shift(1).fillna(close), "high": close * 1.005, "low": close * 0.995,
                  "volume": pd.DataFrame(5e6, index=close.index, columns=close.columns)}
            DATA_SOURCE.update(prices=f"SYNTHETIC (yfinance failed: {str(e)[:80]})", synthetic=True)
    if not DATA_SOURCE["synthetic"]:
        export_prices_csv(md["close"])
    return md


def get_prices(tickers: list[str] | None = None) -> pd.DataFrame:
    """Close prices only (kept for callers that just need closes)."""
    return get_market_data(tickers)["close"]


def last_known_prices(tickers: list[str]) -> dict[str, float]:
    """Fallback valuation for held names Yahoo didn't return today (IPS C8.5): last non-null close
    from prices.csv, else from the most recent cached download. Never used for trading."""
    out: dict[str, float] = {}
    if os.path.exists(config.PRICES_CSV):
        try:
            p = pd.read_csv(config.PRICES_CSV, index_col=0, usecols=lambda c: c == "date" or c in tickers)
            for t in p.columns:
                s = p[t].dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
        except (ValueError, OSError):
            pass
    missing = [t for t in tickers if t not in out]
    if missing:
        files = sorted(f for f in os.listdir(config.CACHE_DIR) if "_market_" in f)
        for f in reversed(files):
            try:
                with open(os.path.join(config.CACHE_DIR, f), "rb") as fh:
                    c = pickle.load(fh)["close"]
            except Exception:  # noqa: BLE001
                continue
            for t in [t for t in missing if t in c.columns and c[t].notna().any()]:
                out[t] = float(c[t].dropna().iloc[-1])
            missing = [t for t in tickers if t not in out]
            if not missing:
                break
    return out


def export_prices_csv(close: pd.DataFrame) -> None:
    """Brief §11 deliverable: prices.csv (date x ticker closing prices)."""
    try:
        close.round(4).to_csv(config.PRICES_CSV, index_label="date")
    except OSError:
        pass


def is_synthetic() -> bool:
    return bool(DATA_SOURCE.get("synthetic", False))


# --------------------------------------------------------------------------- #
# fundamentals (shortlist only)
# --------------------------------------------------------------------------- #
FUNDAMENTAL_FIELDS = {
    "marketCap": "mkt_cap_bn", "trailingPE": "pe_ttm",
    "forwardPE": "pe_fwd", "priceToBook": "pb", "enterpriseToEbitda": "ev_ebitda",
    "returnOnEquity": "roe", "profitMargins": "net_margin",
    "revenueGrowth": "rev_growth", "earningsGrowth": "eps_growth",
    "debtToEquity": "debt_to_equity", "dividendYield": "div_yield",
    "freeCashflow": "fcf_bn",
}
CORE_METRICS = ("pe_fwd", "ev_ebitda", "pb", "roe", "net_margin", "rev_growth", "eps_growth",
                "debt_to_equity", "fcf_bn")   # IPS C2.2 list


def _fetch_fundamentals(ticker: str) -> dict:
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    if not info or (info.get("regularMarketPrice") is None and info.get("marketCap") is None):
        raise RuntimeError("empty info")
    out = {}
    for src, dst in FUNDAMENTAL_FIELDS.items():
        v = info.get(src)
        if dst in ("mkt_cap_bn", "fcf_bn") and v is not None:
            v = round(v / 1e9, 1)
        elif isinstance(v, float):
            v = round(v, 3)
        out[dst] = v
    return out


def _synthetic_fundamentals(ticker: str) -> dict:
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    return {
        "mkt_cap_bn": round(float(rng.uniform(20, 3000)), 1),
        "pe_ttm": round(float(rng.uniform(10, 45)), 1), "pe_fwd": round(float(rng.uniform(9, 35)), 1),
        "pb": round(float(rng.uniform(1.5, 30)), 1), "ev_ebitda": round(float(rng.uniform(7, 30)), 1),
        "roe": round(float(rng.uniform(0.05, 0.6)), 3), "net_margin": round(float(rng.uniform(0.05, 0.45)), 3),
        "rev_growth": round(float(rng.uniform(-0.05, 0.3)), 3), "eps_growth": round(float(rng.uniform(-0.1, 0.4)), 3),
        "debt_to_equity": round(float(rng.uniform(10, 200)), 1), "div_yield": round(float(rng.uniform(0, 0.035)), 3),
        "fcf_bn": round(float(rng.uniform(-2, 100)), 1), "synthetic": True,
    }


def get_fundamentals(tickers: list[str], sectors: dict[str, str]) -> dict[str, dict]:
    """Stocks only (IPS C2.5: leveraged ETFs never reach Research)."""
    synthetic = is_synthetic()

    def one(t):
        if synthetic:
            return t, _synthetic_fundamentals(t)
        try:
            return t, _fetch_fundamentals(t)
        except Exception:  # noqa: BLE001
            return t, {}

    def fetch_all():
        with ThreadPoolExecutor(max_workers=6) as ex:
            return dict(ex.map(one, tickers))

    raw = _cached(_key("fundamentals", tickers), fetch_all,
                  keep_if=lambda v: sum(bool(x) for x in v.values()) >= 0.5 * max(1, len(v)))
    out = {}
    for t in tickers:
        d = dict(raw.get(t) or {})
        d["sector"] = sectors.get(t, "Unknown")
        n_missing = sum(d.get(k) is None for k in CORE_METRICS)
        d["missing_core_metrics"] = n_missing
        d["insufficient_data"] = n_missing > 2           # IPS C2.7
        out[t] = d
    return out


def research_flags(fundamentals: dict[str, dict]) -> dict[str, list[str]]:
    """IPS C2.4 flags, computed in Python so the analyst interprets, never calculates."""
    by_sector: dict[str, list[float]] = {}
    for d in fundamentals.values():
        if d.get("debt_to_equity") is not None:
            by_sector.setdefault(d["sector"], []).append(float(d["debt_to_equity"]))
    med = {s: float(np.median(v)) for s, v in by_sector.items()}
    flags = {}
    for t, d in fundamentals.items():
        f = []
        if d.get("fcf_bn") is not None and d["fcf_bn"] < 0:
            f.append("negative free cash flow")
        if d.get("debt_to_equity") is not None and d["sector"] in med and d["debt_to_equity"] > med[d["sector"]]:
            f.append("D/E above sector median (shortlist)")
        if d.get("net_margin") is not None and d["net_margin"] < 0:
            f.append("negative net margin")
        flags[t] = f
    return flags


# --------------------------------------------------------------------------- #
# news (Brief §4: research news every morning before the open)
# --------------------------------------------------------------------------- #
MADRID_TZ = "Europe/Madrid"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "AIInvestmentCommittee research-project@example.com")  # SEC asks for a contact


def _to_madrid(ts) -> str:
    from zoneinfo import ZoneInfo
    return ts.astimezone(ZoneInfo(MADRID_TZ)).strftime("%Y-%m-%d %H:%M") if ts is not None else ""


def _parse_feed(xml_text: str, max_age_days: int, default_source: str) -> list[dict]:
    """RSS (<item>) or Atom (<entry>) -> [{title, source, ts (Madrid), ts_utc, url}] within the age window."""
    root = ElementTree.fromstring(xml_text)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)
    out = []
    atom = "{http://www.w3.org/2005/Atom}"
    items = list(root.iter("item")) or list(root.iter(f"{atom}entry"))
    for item in items:
        title = (item.findtext("title") or item.findtext(f"{atom}title") or "").strip()
        when = item.findtext("pubDate") or item.findtext(f"{atom}updated")
        ts = None
        try:
            ts = parsedate_to_datetime(when) if when and not when[:4].isdigit() else \
                dt.datetime.fromisoformat(when.replace("Z", "+00:00")) if when else None
        except (TypeError, ValueError):
            ts = None
        if not title or (ts is not None and ts < cutoff):
            continue
        src = (item.findtext("source") or "").strip() or default_source
        if default_source == "Google News" and " - " in title:            # Google appends " - Publisher"
            title, pub = title.rsplit(" - ", 1)
            src = src if src != "Google News" else pub
        link = item.findtext("link") or ""
        if not link:
            le = item.find(f"{atom}link")
            link = le.get("href", "") if le is not None else ""
        out.append({"title": title[:160], "source": src[:40], "ts": _to_madrid(ts),
                    "ts_utc": ts.isoformat() if ts else "", "url": link})
    return out


def _fetch_news(ticker: str, name: str = "", cik: str | None = None, n: int = 10, max_age_days: int = 5) -> list[dict]:
    """IPS C4.2 free stack: Yahoo Finance RSS, Google News (general + Reuters/Bloomberg/WSJ), SEC EDGAR 8-K.
    Up to n headlines over max_age_days, newest first, each with source and Madrid timestamp."""
    import requests
    from urllib.parse import quote_plus
    q_name = quote_plus(f'"{name}"') if name and name != ticker else quote_plus(ticker)
    feeds = [
        (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", "Yahoo Finance", _HTTP_HEADERS),
        (f"https://news.google.com/rss/search?q={quote_plus(ticker)}+stock+when:{max_age_days}d&hl=en-US&gl=US&ceid=US:en",
         "Google News", _HTTP_HEADERS),
        (f"https://news.google.com/rss/search?q={q_name}+(site:reuters.com+OR+site:bloomberg.com+OR+site:wsj.com)"
         f"+when:{max_age_days}d&hl=en-US&gl=US&ceid=US:en", "Google News", _HTTP_HEADERS),
    ]
    if cik:
        feeds.append((f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include"
                      f"&count=10&output=atom", "SEC EDGAR 8-K",
                      {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}))
    items, seen = [], set()
    for url, src, headers in feeds:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code != 200:
                continue
            for it in _parse_feed(r.text, max_age_days, src):
                key = "".join(ch for ch in it["title"].lower() if ch.isalnum())[:80]
                if key and key not in seen:
                    seen.add(key)
                    items.append(it)
        except Exception:  # noqa: BLE001
            continue
    # relevance: Yahoo's per-ticker feed mixes in market-wide stories; keep headlines that name the company or
    # ticker (SEC filings are the company's own by definition). Max 4 per source so one feed can't crowd out the rest.
    keys = _name_keys(ticker, name)
    tick_re = re.compile(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])")       # case-sensitive: "MAR" != "Mar 2027"
    items = [it for it in items if it["source"] == "SEC EDGAR 8-K" or tick_re.search(it["title"])
             or any(k in it["title"].lower() for k in keys)]
    items.sort(key=lambda it: it["ts_utc"], reverse=True)
    per_src, out = {}, []
    for it in items:
        if per_src.get(it["source"], 0) < 4:
            out.append(it)
            per_src[it["source"]] = per_src.get(it["source"], 0) + 1
    return out[:n]


_SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "holdings", "plc", "ltd", "group",
             "the", "class", "a", "b", "c", "(the)", "&", "incorporated", "limited"}


def _name_keys(ticker: str, name: str) -> list[str]:
    """Lower-case tokens a relevant headline should contain: the ticker and the company's distinctive name."""
    keys = []                       # the ticker itself is matched case-sensitively in _fetch_news
    words = [w for w in name.replace(",", " ").split() if w.lower() not in _SUFFIXES]
    if words:
        first = words[0].lower()
        keys.append(first if len(first) > 3 else " ".join(w.lower() for w in words[:2]))
    return keys


def get_news(tickers: list[str], names: dict | None = None, ciks: dict | None = None,
             refresh: bool = False) -> dict[str, list[dict]]:
    """Up to 10 headlines per ticker (5 days). Refreshed hourly, or on demand (refresh=True: the C4.5 intraday
    re-check). An all-empty result is never cached, so one bad fetch can't blank sentiment for the day."""
    names, ciks = names or {}, ciks or {}

    def one(t):
        c = ciks.get(t)
        return _fetch_news(t, names.get(t, ""), c if isinstance(c, str) and c.strip("0") and c != "None" else None)

    def fetch_all():
        with ThreadPoolExecutor(max_workers=8) as ex:
            return dict(zip(tickers, ex.map(one, tickers)))
    if is_synthetic():
        return {t: [] for t in tickers}
    return _cached(_key("news_v2", tickers), fetch_all, max_age_min=0 if refresh else 60,
                   keep_if=lambda v: any(v.values()))


# --------------------------------------------------------------------------- #
# earnings (IPS B7 / C2.8): dates, EPS estimate/actual/surprise, revisions, recommendation trend
# --------------------------------------------------------------------------- #
def _fetch_earnings(ticker: str) -> dict:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    out: dict = {}
    try:
        ed = tk.earnings_dates
        if ed is not None and len(ed):
            ed = ed.sort_index()
            now = pd.Timestamp.now(tz=ed.index.tz)
            past = ed[ed.index <= now].dropna(subset=["Reported EPS"]).tail(4)
            fut = ed[ed.index > now]
            out["last4"] = [{"date": str(i.date()), "est": None if pd.isna(r["EPS Estimate"]) else round(float(r["EPS Estimate"]), 3),
                             "actual": None if pd.isna(r["Reported EPS"]) else round(float(r["Reported EPS"]), 3),
                             "surprise_pct": None if pd.isna(r["Surprise(%)"]) else round(float(r["Surprise(%)"]), 2),
                             "time_et": i.strftime("%H:%M")} for i, r in past.iterrows()]
            if len(fut):
                out["next_date"] = str(fut.index[0].date())
                out["next_time_et"] = fut.index[0].strftime("%H:%M")
    except Exception:  # noqa: BLE001
        pass
    try:
        et = tk.eps_trend
        if et is not None and len(et):
            def chg(period):
                if period in et.index and et.loc[period, "30daysAgo"]:
                    return round(float(et.loc[period, "current"] / et.loc[period, "30daysAgo"] - 1), 4)
                return None
            out["rev_30d_0q"], out["rev_30d_1y"] = chg("0q"), chg("+1y")
            out["current_estimate"] = round(float(et.loc["0q", "current"]), 3) if "0q" in et.index else None
    except Exception:  # noqa: BLE001
        pass
    try:
        rc = tk.recommendations
        if rc is not None and len(rc):
            out["rec_trend"] = {r["period"]: {k: int(r[k]) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
                                for _, r in rc.head(2).iterrows()}
    except Exception:  # noqa: BLE001
        pass
    return out


def get_earnings(tickers: list[str], asof=None) -> dict[str, dict]:
    """Per stock: next report date (+ trading days to it, reports_soon flag), last 4 quarters' EPS est/actual/surprise,
    30-day EPS-estimate revisions (current quarter, next year), recommendation trend. Cached daily."""
    if is_synthetic() or not tickers:
        return {t: {} for t in tickers}

    def fetch_all():
        with ThreadPoolExecutor(max_workers=6) as ex:
            return dict(zip(tickers, ex.map(_fetch_earnings, tickers)))
    raw = _cached(_key("earnings", tickers), fetch_all, keep_if=lambda v: sum(bool(x) for x in v.values()) >= 0.3 * max(1, len(v)))
    today = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.today().normalize()
    out = {}
    for t in tickers:
        d = dict(raw.get(t) or {})
        if asof is not None:              # replay: only reports known by then; estimate revisions are today's -> drop
            hist = [r for r in d.get("last4", []) if pd.Timestamp(r["date"]) < today]
            fut = [r["date"] for r in (raw.get(t) or {}).get("last4", []) if pd.Timestamp(r["date"]) >= today]
            d = {"last4": hist, **({"next_date": min(fut)} if fut else {})}
        if d.get("next_date"):
            n = len(pd.bdate_range(today, pd.Timestamp(d["next_date"]))) - 1
            d["trading_days_to_report"] = n
            d["reports_soon"] = 0 <= n <= config.EARNINGS_SOON_DAYS
        out[t] = d
    return out


# --------------------------------------------------------------------------- #
# macro (IPS C1.2)
# --------------------------------------------------------------------------- #
MACRO_SYMBOLS = ["^VIX", "^TNX", "^IRX", "HYG", "LQD", "DX-Y.NYB", "CL=F"]


def _pct(s: pd.Series, n: int) -> float:
    s = s.dropna()
    return round(float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100, 2) if len(s) > n else 0.0


def sector_etf_close() -> pd.DataFrame:
    """Close prices of the 11 SPDR sector ETFs (+ SPY), same cache as get_macro. For attribution."""
    sym = MACRO_SYMBOLS + [config.BENCHMARK] + list(SECTOR_ETFS.values())
    px = _cached("macro_px_v2", lambda: _download(sym, "2y")["close"])
    return px[[c for c in list(SECTOR_ETFS.values()) + [config.BENCHMARK] if c in px.columns]]


def get_macro(asof=None) -> dict:
    sym = MACRO_SYMBOLS + [config.BENCHMARK] + list(SECTOR_ETFS.values())
    synthetic = False
    try:
        if is_synthetic():
            raise RuntimeError("synthetic day")
        px = _cached("macro_px_v2", lambda: _download(sym, "2y")["close"], max_age_min=60)
        if asof is not None:              # replay: only data strictly before the decision day
            px = px[px.index < pd.Timestamp(asof)]
    except Exception:  # noqa: BLE001
        px = _synthetic_prices(sym, days=252, seed=11)
        px["^VIX"] = 15 + (px["^VIX"] / px["^VIX"].iloc[0] - 1) * 40
        px["^TNX"] = 4.2 + (px["^TNX"] / px["^TNX"].iloc[0] - 1) * 3
        px["^IRX"] = 3.9 + (px["^IRX"] / px["^IRX"].iloc[0] - 1) * 2
        synthetic = True

    def col(s):
        return px[s].dropna() if s in px.columns else pd.Series([np.nan, np.nan])

    spy, vix, tnx = col(config.BENCHMARK), col("^VIX"), col("^TNX")
    hyg_lqd = (col("HYG") / col("LQD")).dropna()
    macro = {
        "vix_level": round(float(vix.iloc[-1]), 2),
        "vix_1d_change_pct": _pct(vix, 1),
        "vix_5d_change_pct": _pct(vix, 5),
        "vix_1y_percentile": round(float((vix < vix.iloc[-1]).mean()), 2),
        "us10y_yield_pct": round(float(tnx.iloc[-1]), 2),
        "us10y_5d_change_bp": round(float(tnx.iloc[-1] - tnx.iloc[-6]) * 100) if len(tnx) > 6 else 0,
        "us3m_yield_pct": round(float(col("^IRX").iloc[-1]), 2),
        "spy_vs_50dma_pct": round(float(spy.iloc[-1] / spy.tail(50).mean() - 1) * 100, 2),
        "spy_vs_200dma_pct": round(float(spy.iloc[-1] / spy.tail(200).mean() - 1) * 100, 2),
        "spy_3m_return_pct": _pct(spy, 63),
        "hyg_lqd_vs_20dma_pct": (round(float(hyg_lqd.iloc[-1] / hyg_lqd.tail(20).mean() - 1) * 100, 2)
                                 if len(hyg_lqd) >= 20 else None),
        "dxy_level": round(float(col("DX-Y.NYB").iloc[-1]), 2) if "DX-Y.NYB" in px.columns else None,
        "dxy_5d_change_pct": _pct(col("DX-Y.NYB"), 5) if "DX-Y.NYB" in px.columns else None,
        "wti_level": round(float(col("CL=F").iloc[-1]), 2) if "CL=F" in px.columns else None,
        "wti_5d_change_pct": _pct(col("CL=F"), 5) if "CL=F" in px.columns else None,
        "synthetic": synthetic,
    }
    macro["curve_10y_3m_bp"] = round((macro["us10y_yield_pct"] - macro["us3m_yield_pct"]) * 100)
    macro["sector_1m_momentum_pct"] = {s: _pct(col(e), 21) for s, e in SECTOR_ETFS.items() if e in px.columns}
    macro["sector_3m_momentum_pct"] = {s: _pct(col(e), 63) for s, e in SECTOR_ETFS.items() if e in px.columns}
    return macro
