"""
The "tools" the agents use. Pure Python / numpy — this is where ALL the maths
happens. LLMs never compute numbers; they only read these outputs and judge.

    technicals       Quant inputs (Brief §8.2 Data Agent: incl. 5-day returns)
    macro_regime     IPS C1.3 regime rules + C1.5 sector tilts (deterministic)
    composite        IPS C5.2 blend of analyst scores
    optimise         mean-variance on PM convictions, IPS limits as constraints
    risk_report      every IPS Part A limit checked in code
    plan_orders      turns a target portfolio into <= 5 trades (Brief §1)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config

TRADING_DAYS = 252
REGIME_ORDER = {"risk_off": 0, "neutral": 1, "risk_on": 2}


# --------------------------------------------------------------------------- #
# Technicals (Quant Analyst)
# --------------------------------------------------------------------------- #
def _rsi(series: pd.Series, n: int = 14) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / n).mean().iloc[-1]
    loss = -delta.clip(upper=0).ewm(alpha=1 / n).mean().iloc[-1]
    return float(100 - 100 / (1 + gain / loss)) if loss > 0 else 100.0


def technicals(prices: pd.DataFrame, tickers: list[str]) -> dict[str, dict]:
    out = {}
    for t in tickers:
        p = prices[t].dropna()
        r = np.log(p).diff().dropna()
        out[t] = {
            "ret_5d_pct": round(float(p.iloc[-1] / p.iloc[-6] - 1) * 100, 1),
            "ret_1m_pct": round(float(p.iloc[-1] / p.iloc[-21] - 1) * 100, 1),
            "ret_3m_pct": round(float(p.iloc[-1] / p.iloc[-63] - 1) * 100, 1),
            "mom_12_1_pct": round(float(p.iloc[-21] / p.iloc[-252] - 1) * 100, 1) if len(p) > 252 else None,
            "px_vs_50dma_pct": round(float(p.iloc[-1] / p.tail(50).mean() - 1) * 100, 1),
            "px_vs_200dma_pct": round(float(p.iloc[-1] / p.tail(200).mean() - 1) * 100, 1),
            "rsi_14": round(_rsi(p), 1),
            "vol_3m_ann_pct": round(float(r.tail(63).std() * np.sqrt(TRADING_DAYS)) * 100, 1),
        }
    return out


# --------------------------------------------------------------------------- #
# Macro regime (IPS C1.3 / C1.5) — rules in code, the Macro LLM explains them
# --------------------------------------------------------------------------- #
def macro_regime(macro: dict) -> dict:
    """Raw regime from today's data (before the 2-day upgrade rule)."""
    hl = macro.get("hyg_lqd_vs_20dma_pct")
    off = {
        "VIX > 25": macro["vix_level"] > 25,
        "SPY below 50dma and 200dma": macro["spy_vs_50dma_pct"] < 0 and macro["spy_vs_200dma_pct"] < 0,
        "HYG/LQD > 1% below 20dma": hl is not None and hl < -1.0,
        "VIX up > 30% in one day": macro["vix_1d_change_pct"] > 30,
    }
    on = {
        "VIX < 20": macro["vix_level"] < 20,
        "SPY above 50dma": macro["spy_vs_50dma_pct"] > 0,
        "HYG/LQD >= 20dma": hl is not None and hl >= 0,
    }
    if sum(off.values()) >= 2:
        raw = "risk_off"
    elif all(on.values()):
        raw = "risk_on"
    else:
        raw = "neutral"
    return {"raw_regime": raw, "risk_off_signals": off, "risk_on_signals": on}


def apply_hysteresis(raw: str, prev_effective: str | None, prev_raw: str | None) -> tuple[str, str]:
    """IPS C1.3: downgrades apply immediately; upgrades need the signal on 2 days in a row."""
    if prev_effective is None:
        return raw, "first observation"
    if REGIME_ORDER[raw] <= REGIME_ORDER[prev_effective]:
        return raw, ("downgrade applied immediately" if raw != prev_effective else "unchanged")
    if prev_raw is not None and REGIME_ORDER[prev_raw] >= REGIME_ORDER[raw]:
        return raw, "upgrade confirmed (2 days in a row)"
    return prev_effective, f"upgrade to {raw} pending (needs a 2nd day)"


def sector_tilts(macro: dict) -> dict[str, str]:
    """IPS C1.5: top 3 sectors by 1m ETF momentum overweight, bottom 3 underweight."""
    mom = macro.get("sector_1m_momentum_pct", {})
    ranked = sorted(mom, key=mom.get, reverse=True)
    tilts = {s: "neutral" for s in ranked}
    for s in ranked[:3]:
        tilts[s] = "overweight"
    for s in ranked[-3:]:
        tilts[s] = "underweight"
    return tilts


# --------------------------------------------------------------------------- #
# Post-earnings announcement drift (Stage 5, Oct 20)
# --------------------------------------------------------------------------- #
def pead(earnings: dict[str, dict], close: pd.DataFrame, asof=None) -> dict[str, dict]:
    """For names that reported in the last 20 trading days: eps_surprise_pct, announcement_return (3-day return
    around the report minus SPY), days_since_report, and drift_score = +1 if surprise AND announcement return are
    both positive, -1 if both negative, 0 if mixed, decaying linearly to 0 over PEAD_WINDOW_DAYS. Uses only prices
    <= asof (replay-safe)."""
    if asof is not None:
        close = close[close.index <= pd.Timestamp(asof)]
    out = {}
    if config.BENCHMARK not in close.columns or close.empty:
        return out
    spy = close[config.BENCHMARK]
    for t, e in (earnings or {}).items():
        last = (e or {}).get("last4") or []
        if not last or t not in close.columns:
            continue
        rep = last[-1]
        d = pd.Timestamp(rep["date"])
        if d > close.index[-1]:
            continue
        pos = int(close.index.searchsorted(d))
        if rep.get("time_et", "00:00") >= "16:00":                  # after the close: reaction is the next session
            pos += 1
        days = len(close) - 1 - pos
        if days < 0 or days > config.PEAD_WINDOW_DAYS or pos < 1 or pos + 1 > len(close) - 1:
            continue
        p = close[t]
        ann = float(p.iloc[pos + 1] / p.iloc[pos - 1] - 1) - float(spy.iloc[pos + 1] / spy.iloc[pos - 1] - 1)
        sp = rep.get("surprise_pct")
        sign = 0
        if sp is not None and sp > 0 and ann > 0:
            sign = 1
        elif sp is not None and sp < 0 and ann < 0:
            sign = -1
        out[t] = {"eps_surprise_pct": sp, "announcement_return": round(ann, 4), "days_since_report": int(days),
                  "drift_score": round(sign * (1 - days / config.PEAD_WINDOW_DAYS), 3)}
    return out


# --------------------------------------------------------------------------- #
# Covariance with shrinkage (no sklearn needed)
# --------------------------------------------------------------------------- #
def shrunk_cov(returns: pd.DataFrame, shrink: float = 0.3) -> pd.DataFrame:
    """Shrink sample covariance toward a constant-correlation target."""
    s = returns.cov() * TRADING_DAYS
    vol = np.sqrt(np.diag(s))
    corr = returns.corr().values
    n = len(vol)
    avg_corr = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = avg_corr * np.outer(vol, vol)
    np.fill_diagonal(target, vol ** 2)
    out = (1 - shrink) * s.values + shrink * target
    return pd.DataFrame(out, index=s.index, columns=s.columns)


# --------------------------------------------------------------------------- #
# Composite score (IPS C5.2) — shown on the IPS's 0-10 scale as well
# --------------------------------------------------------------------------- #
TILT_TO_SCORE = {"overweight": 2 / 3, "neutral": 0.0, "underweight": -2 / 3}  # x 15% = +/-0.5 on 0-10 (C1.5)


def to_10(x: float) -> float:
    return round((float(x) + 1) * 5, 1)


def composite_scores(tickers, sectors, asset_types, research, quant, sentiment, tilts, regime) -> dict[str, dict]:
    """IPS C5.2 blend. An agent with no signal for a name (score None: no news, failed call, ETF without
    fundamentals) is left out and its weight is redistributed pro rata to the others (Oct 20)."""
    W = config.SIGNAL_WEIGHTS
    out = {}
    for t in tickers:
        kind = asset_types.get(t, "stock")
        qv = (quant.get(t) or {}).get("score")
        sv = (sentiment.get(t) or {}).get("score")
        q = None if qv is None else float(qv)
        s_ = None if sv is None else float(sv)
        if kind == "stock":
            r_row = research.get(t) or {}
            insufficient = r_row.get("rating") == "insufficient_data"
            rv = r_row.get("score")
            r = None if (rv is None or insufficient or not r_row) else float(rv)
            m = TILT_TO_SCORE.get(str(tilts.get(sectors.get(t), "neutral")).lower(), 0.0)
        else:                                       # IPS C2.5: ETFs have no fundamentals
            r, insufficient = None, False
            m = {"risk_on": 1.0, "neutral": 0.0, "risk_off": -1.0}[regime] if kind == "leveraged_etf" else \
                (1.0 if regime == "risk_off" else -1.0)
        parts = {"quant": q, "sentiment": s_, "research": r, "macro": m}
        avail = {k: v for k, v in parts.items() if v is not None}
        tot = sum(W[k] for k in avail) or 1.0
        weights_used = {k: round(W[k] / tot, 3) for k in avail}
        comp = float(np.clip(sum(W[k] * v for k, v in avail.items()) / tot, -1, 1))
        na = [k for k, v in parts.items() if v is None]
        out[t] = {"type": kind, "research": None if r is None else round(r, 2), "quant": None if q is None else round(q, 2),
                  "sentiment": None if s_ is None else round(s_, 2), "macro": round(m, 2), "composite": round(comp, 3),
                  "composite_10": to_10(comp), "insufficient_data": insufficient, "na": na,
                  "weights_used": weights_used}
    return out


# --------------------------------------------------------------------------- #
# Leverage sleeve, liquidity and drawdown state (IPS A4/C11, A10/A11, C6.3)
# --------------------------------------------------------------------------- #
def etf_caps(prices: pd.DataFrame, tickers, asset_types: dict[str, str]) -> dict[str, float]:
    """Vol-targeted cap per leveraged/inverse ETF: base cap x (1y median 20d vol / current 20d vol),
    never above the IPS caps. Normal vol -> cap unchanged; a vol spike shrinks it."""
    out = {}
    w = config.LEV_VOL_WINDOW
    for t in tickers:
        k = asset_types.get(t, "stock")
        if k not in ("leveraged_etf", "inverse_etf") or t not in prices.columns:
            continue
        r = np.log(prices[t].dropna()).diff().dropna()
        if len(r) < w + 5:
            continue
        roll = r.rolling(w).std().dropna().tail(config.LEV_VOL_TARGET_LOOKBACK)
        cur, target = float(roll.iloc[-1]), float(roll.median())
        base = config.LEV_BASE_CAP_EACH if k == "leveraged_etf" else config.INV_BASE_CAP_EACH
        ipscap = config.RISK_LIMITS["max_leveraged_each"] if k == "leveraged_etf" else config.RISK_LIMITS["max_inverse_total"]
        out[t] = round(float(min(ipscap, base * (target / cur))) if cur > 0 else base, 4)
    return out


def adv20(market: dict, tickers) -> dict[str, float]:
    """20-day average dollar volume per ticker (IPS C6.3 liquidity check)."""
    c, v = market["close"], market.get("volume")
    out = {}
    for t in tickers:
        if v is not None and t in c.columns and t in v.columns:
            out[t] = float((c[t] * v[t]).dropna().tail(20).mean() or 0.0)
    return out


def drawdown_state(equity_history: list[float], nav: float) -> dict:
    """IPS A4 / C11 graduated rule. equity_history = prior daily closing NAVs (oldest first).
    level: 'ok' | 'warn' (<= -7%) | 'hard' (<= -10%). Restrictions lift only after DD_RECOVERY_DAYS
    consecutive days back above -7%."""
    hist = list(equity_history) + [nav]
    peak, dds = config.STARTING_CAPITAL, []
    for x in hist:
        peak = max(peak, x)
        dds.append(x / peak - 1)
    dd = dds[-1]
    if dd <= config.DD_HARD:
        level = "hard"
    elif dd <= config.DD_WARN:
        level = "warn"
    else:
        level = "ok"
        breached = [i for i, d in enumerate(dds) if d <= config.DD_WARN]
        if breached:
            above = len(dds) - 1 - breached[-1]            # consecutive days above -7% since the last breach
            if above < config.DD_RECOVERY_DAYS:
                level = "warn"                              # still recovering: restrictions stay on
    return {"drawdown": round(dd, 4), "level": level,
            "lev_cap": 0.0 if level == "hard" else config.DD_WARN_LEV_CAP if level == "warn" else None,
            "no_new_leveraged": level in ("warn", "hard"),
            "inverse_min_risk_off": config.DD_HARD_INVERSE_MIN if level == "hard" else None}


# --------------------------------------------------------------------------- #
# Optimiser (Portfolio Manager's tool)
# --------------------------------------------------------------------------- #
def optimise(convictions: dict[str, float], prices: pd.DataFrame, sectors: dict[str, str],
             asset_types: dict[str, str], equity_exposure: float, regime: str, vix: float,
             max_names: int, risk_aversion: float = 4.0, diversification: float = 0.5,
             lev_cap_override: float | None = None, min_weight: float = 0.01,
             ensure_min: bool = False, caps: dict[str, float] | None = None,
             inverse_min: float | None = None, lookback: int = 252) -> dict[str, float]:
    """Mean-variance on the PM's convictions, maximise
        w'mu - (risk_aversion/2) w'Sigma w - diversification * ||w - w_eq||^2
    with every IPS limit as a constraint. Only the top `max_names` convictions
    are eligible (Brief: 10-20 holdings). Positions below `min_weight` (the
    $25k minimum trade) are removed and the problem re-solved, never re-scaled,
    so sector/position caps still hold exactly."""
    L, R = config.RISK_LIMITS, config.REGIME_RULES[regime]
    lev_cap = min(L["max_leveraged_total"], R["max_leveraged"])
    if lev_cap_override is not None:
        lev_cap = min(lev_cap, lev_cap_override)
    inv_lo, inv_hi = R["inverse"]
    if regime == "risk_off" and vix > config.RISK_OFF_HIGH_VIX:
        inv_hi = L["max_inverse_total"]
    if regime == "risk_off" and inverse_min:                 # C11 hard level: hedge >= 15% in risk-off
        inv_lo, inv_hi = max(inv_lo, inverse_min), max(inv_hi, inverse_min)
    caps = caps or {}
    equity_exposure = float(np.clip(equity_exposure, 1 - L["max_cash_weight"], 1 - L["min_cash_weight"]))

    def kind(t):
        return asset_types.get(t, "stock")

    # New money only goes to names with a positive view; if fewer than 10 qualify
    # (Brief: 10-20 holdings) the next-best non-negative names fill the gap.
    ranked = sorted([t for t in convictions if kind(t) == "stock"], key=lambda t: -convictions[t])
    stocks = [t for t in ranked if convictions[t] > 0]
    if len(stocks) < config.MIN_HOLDINGS:
        stocks += [t for t in ranked if -0.5 < convictions[t] <= 0][:config.MIN_HOLDINGS - len(stocks)]
    min_names = config.TARGET_HOLDINGS[0] if ensure_min else config.MIN_HOLDINGS
    if ensure_min and len(stocks) < min_names:     # initialisation aims for 12-15 names (IPS A7)
        stocks += [t for t in ranked if t not in stocks][:min_names - len(stocks)]
    floor_conv = -0.99 if ensure_min else -0.5
    pool = [t for t in ranked if t not in stocks and convictions[t] > floor_conv]   # extra names if the caps are infeasible
    lev = sorted([t for t in convictions if kind(t) == "leveraged_etf" and convictions[t] > 0],
                 key=lambda t: -convictions[t]) if lev_cap > 0 else []
    inv = sorted([t for t in convictions if kind(t) == "inverse_etf"], key=lambda t: -convictions[t]) \
        if inv_hi > 0 else []
    n_etf = min(len(lev), 2) + min(len(inv), 1)
    chosen = stocks[:max(min_names, max_names - n_etf)] + lev[:2] + inv[:1]
    tickers = [t for t in chosen if t in prices.columns]
    if not tickers:
        return {}

    # If the sector/position caps can't be met with this candidate set (e.g. 6 names in 3 sectors cannot
    # satisfy a 35% sector cap), widen the set with the next-best names instead of falling back to a
    # portfolio that breaches the caps.
    for extra in (0, 4, 8, 14):
        keep = list(dict.fromkeys(tickers + [t for t in pool[:extra] if t in prices.columns]))
        for _ in range(4):
            w, ub, ok = _solve(keep, convictions, prices, sectors, kind, equity_exposure, lev_cap, inv_lo, inv_hi,
                               risk_aversion, diversification, lookback, beta_band=R["beta"], caps=caps,
                               lb=min_weight * 1.02 if ensure_min else 0.0)
            small = [t for t, x in zip(keep, w) if x < min_weight]
            if not small or len(keep) - len(small) < min_names:
                break
            keep = [t for t in keep if t not in small]
        if ok:
            break
    w = np.minimum(np.clip(w, 0, None), ub)
    # floor (not round) so the target never needs more cash than we have
    return {t: math.floor(float(x) * 1e4) / 1e4 for t, x in zip(keep, w) if x >= min_weight * 0.999}


def _solve(tickers, convictions, prices, sectors, kind, equity_exposure, lev_cap, inv_lo, inv_hi,
           risk_aversion, diversification, lookback, beta_band=None, lb=0.0, caps=None):
    L = config.RISK_LIMITS
    buf = 0.995   # small buffer under each cap so whole-share rounding never breaches it
    rets = np.log(prices[tickers]).diff().dropna().tail(lookback)
    cov = shrunk_cov(rets).values
    conv = np.array([float(np.clip(convictions[t], -1, 1)) for t in tickers])
    mu = 0.07 + 0.10 * conv   # prior market return + conviction alpha

    n = len(tickers)
    ub = []
    for t in tickers:
        k = kind(t)
        cap = (caps or {}).get(t, 1.0)                     # vol-targeted ETF cap (never above IPS)
        ub.append(buf * (min(L["max_single_weight"], cap) if k == "stock" else
                         min(L["max_leveraged_each"], lev_cap, cap) if k == "leveraged_etf" else min(inv_hi, cap)))
    ub = np.array(ub)
    bounds = [(min(lb, 0.5 * b), b) for b in ub]   # lb: every selected name must clear the minimum trade size
    cons = [{"type": "eq", "fun": lambda w: w.sum() - equity_exposure}]
    for sec in {sectors.get(t, "Unknown") for t in tickers if kind(t) == "stock"}:
        idx = [i for i, t in enumerate(tickers) if kind(t) == "stock" and sectors.get(t) == sec]
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: buf * L["max_sector_weight"] - w[idx].sum()})
    li = [i for i, t in enumerate(tickers) if kind(t) == "leveraged_etf"]
    if li:
        cons.append({"type": "ineq", "fun": lambda w: buf * lev_cap - w[li].sum()})
    ii = [i for i, t in enumerate(tickers) if kind(t) == "inverse_etf"]
    if ii:
        cons.append({"type": "ineq", "fun": lambda w: buf * inv_hi - w[ii].sum()})
        if inv_lo > 0:
            cons.append({"type": "ineq", "fun": lambda w: w[ii].sum() - inv_lo})

    w0 = np.minimum(ub, equity_exposure / n)
    w0 *= equity_exposure / w0.sum()
    w_eq = w0.copy()

    def neg_util(w):
        return -(w @ mu - 0.5 * risk_aversion * w @ cov @ w - diversification * np.sum((w - w_eq) ** 2))

    res = None
    if beta_band is not None and config.BENCHMARK in prices.columns:
        # IPS C1.4: regime beta band. Portfolio beta is linear in w, so it is a plain constraint.
        r = np.log(prices[tickers + [config.BENCHMARK]]).diff().dropna().tail(lookback)
        m = r[config.BENCHMARK].values
        betas = np.array([np.cov(r[t].values, m)[0, 1] / m.var() for t in tickers])
        lo, hi = beta_band
        bcons = cons + [{"type": "ineq", "fun": lambda w: w @ betas - lo},
                        {"type": "ineq", "fun": lambda w: hi - w @ betas}]
        res = minimize(neg_util, w0, method="SLSQP", bounds=bounds, constraints=bcons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            res = None          # band unreachable with this shortlist: Risk sees it as a soft concern
    if res is None:
        res = minimize(neg_util, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
    return (res.x if res.success else w0), ub, bool(res.success)


# --------------------------------------------------------------------------- #
# Risk (Risk Manager's tool)
# --------------------------------------------------------------------------- #
def risk_report(weights: dict[str, float], prices: pd.DataFrame, sectors: dict[str, str],
                asset_types: dict[str, str], regime: str, nav_drawdown: float = 0.0,
                lookback: int = 252, adv: dict | None = None, nav: float | None = None,
                caps: dict | None = None, dd_state: dict | None = None,
                current_weights: dict | None = None) -> dict:
    """Risk METRICS of a book (pure maths). The pass/fail verdict on these numbers is the Risk Manager's
    (committee/risk_manager.py: check_book), per Prof. Manero's rule 3."""
    tickers = [t for t, w in weights.items() if w > 0]
    base = {"regime": regime, "fund_drawdown_from_peak": round(nav_drawdown, 4), "dd_state": dd_state or {},
            "etf_effective_caps": caps or {}, "weights": {t: round(w, 4) for t, w in weights.items() if w > 0},
            "current_weights": {t: round(w, 4) for t, w in (current_weights or {}).items()}}
    if not tickers:
        return {**base, "note": "100% cash", "n_positions": 0, "cash_weight": 1.0, "portfolio_vol_ann": 0.0,
                "var_95_1d": 0.0, "cvar_95_1d": 0.0, "beta_vs_spy": 0.0, "effective_n_positions": 0.0,
                "leveraged_etf_weight": 0.0, "inverse_etf_weight": 0.0, "sector_weights": {}, "max_pairwise_corr": 0.0,
                "top3_max_pair_corr": 0.0, "adv_fraction": {}}

    w = np.array([weights[t] for t in tickers])
    rets = prices[tickers + [config.BENCHMARK]].pct_change().dropna().tail(lookback)
    port = rets[tickers].values @ w
    bench = rets[config.BENCHMARK].values
    vol = float(port.std() * np.sqrt(TRADING_DAYS))
    var95 = float(-np.percentile(port, 5))
    cvar95 = float(-port[port <= np.percentile(port, 5)].mean())
    beta = float(np.cov(port, bench)[0, 1] / bench.var())
    cum = np.cumprod(1 + port)
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    sector_w: dict[str, float] = {}
    for t, x in zip(tickers, w):
        sector_w[sectors.get(t, "Unknown")] = sector_w.get(sectors.get(t, "Unknown"), 0.0) + float(x)
    hhi = float((w / w.sum()) @ (w / w.sum()))
    corr = rets[tickers].corr() if len(tickers) > 1 else None
    max_pair = float(np.max(corr.values[np.triu_indices(len(tickers), 1)])) if corr is not None else 0.0
    top3 = [t for t, _ in sorted(zip(tickers, w), key=lambda kv: -kv[1])[:3]]
    top3_corr = float(np.max(corr.loc[top3, top3].values[np.triu_indices(len(top3), 1)])) \
        if corr is not None and len(top3) > 1 else 0.0
    adv_frac = {}
    if adv and nav:
        for t, x in zip(tickers, w):
            if adv.get(t):
                adv_frac[t] = round(float(x * nav / adv[t]), 4)
    lev_w = {t: float(x) for t, x in zip(tickers, w) if asset_types.get(t) == "leveraged_etf"}
    inv_w = {t: float(x) for t, x in zip(tickers, w) if asset_types.get(t) == "inverse_etf"}
    return {
        **base,
        "portfolio_vol_ann": round(vol, 4), "var_95_1d": round(var95, 4), "cvar_95_1d": round(cvar95, 4),
        "beta_vs_spy": round(beta, 2), "max_drawdown_1y_backtest": round(mdd, 4),
        "n_positions": len(tickers), "effective_n_positions": round(1 / hhi, 1),
        "max_pairwise_corr": round(max_pair, 2), "top3_holdings": top3, "top3_max_pair_corr": round(top3_corr, 2),
        "cash_weight": round(1 - float(w.sum()), 4),
        "leveraged_etf_weight": round(sum(lev_w.values()), 4), "leveraged_by_etf": {k: round(v, 4) for k, v in lev_w.items()},
        "inverse_etf_weight": round(sum(inv_w.values()), 4), "inverse_by_etf": {k: round(v, 4) for k, v in inv_w.items()},
        "sector_weights": {k: round(v, 4) for k, v in sorted(sector_w.items(), key=lambda kv: -kv[1])},
        "adv_fraction": adv_frac,
    }


# --------------------------------------------------------------------------- #
def plan_orders(target_w: dict[str, float], positions: dict[str, dict], px: dict[str, float],
                nav: float, cash: float, forced_exits: dict[str, str], init_mode: bool,
                composite: dict[str, dict], untradeable: dict[str, str], sectors: dict | None = None,
                asset_types: dict | None = None, caps: dict | None = None, lev_cap: float | None = None,
                no_new_leveraged: bool = False) -> tuple[list[dict], list[dict]]:
    """Target book -> the orders that will actually execute (<= 5/day after init). Returns (orders, skipped).

    Order of business (IPS C5.5): forced exits; replacement buys if the book is below the 12-name target
    (A7); then score gain x weight moved. Then, on the EXECUTABLE book:
      - holdings floor (A7): a discretionary exit that would leave < 10 names must be paired with a new-name
        buy the same day, or it is dropped. Forced exits (stop-loss, news veto, C11) always go through.
      - cash <= 3% (A12): if the cap leaves too much cash, the BUYs already in the plan are upsized
        (20% name / 35% sector / ETF caps), then held names topped up if a slot is free; otherwise flagged.
    Sells always precede buys."""
    L = config.RISK_LIMITS
    sectors, asset_types, caps = sectors or {}, asset_types or {}, caps or {}
    held_names = {t for t, p in positions.items() if p.get("shares", 0) > 0}
    cands, skipped = [], []
    exit_proceeds = sum(p["shares"] * px.get(t, 0.0) for t, p in positions.items()
                        if t in forced_exits or target_w.get(t, 0.0) == 0)
    cash_heavy = (cash + exit_proceeds) / nav > L["max_cash_weight"]       # A12 beats the A16 drift band
    survivors = len([t for t in held_names if t not in forced_exits])
    replenish = survivors < config.TARGET_HOLDINGS[0]                     # A7: replacement buys first
    for t in sorted(set(target_w) | set(positions)):
        held = float(positions.get(t, {}).get("shares", 0))
        if t not in px:                                                    # C8.5: never crash on a missing price
            if held > 0 or target_w.get(t, 0) > 0:
                skipped.append({"ticker": t, "reason": "untradeable today: no valid live price (C8.5)"})
            continue
        cur_val = held * px[t]
        tgt = 0.0 if t in forced_exits else float(target_w.get(t, 0.0))
        diff_val = tgt * nav - cur_val
        dw = diff_val / nav
        if t in untradeable:
            if abs(dw) > 0:
                skipped.append({"ticker": t, "reason": f"untradeable today: {untradeable[t]}"})
            continue
        price = px[t]
        forced = t in forced_exits
        if not forced:
            if held > 0 and tgt > 0 and abs(dw) < config.DRIFT_TOLERANCE and not (cash_heavy and dw > 0):
                if abs(dw) * nav >= config.MIN_TRADE_NOTIONAL:
                    skipped.append({"ticker": t, "reason": f"resize {dw:+.1%} within ±{config.DRIFT_TOLERANCE:.0%} drift band"})
                continue
            if abs(diff_val) < config.MIN_TRADE_NOTIONAL and not (held > 0 and tgt == 0):
                continue
        side = "BUY" if diff_val > 0 else "SELL"
        if side == "BUY" and no_new_leveraged and asset_types.get(t) == "leveraged_etf":
            skipped.append({"ticker": t, "reason": "C11 drawdown rule: no new leveraged buys"})
            continue
        shares = int(math.floor(abs(diff_val) / price))
        if side == "SELL":
            shares = int(held) if tgt == 0 else min(shares, int(held))
        if shares <= 0:
            continue
        is_new = side == "BUY" and held <= 0
        score = abs((composite.get(t) or {}).get("composite", 0.0))
        prio = (1e6 if forced else 0) + (5e5 if replenish and is_new else 0) + abs(dw) * (score + 0.1)
        cands.append({"ticker": t, "side": side, "shares": shares, "price": round(price, 4),
                      "notional": round(shares * price, 2), "target_weight": round(tgt, 4),
                      "current_weight": round(cur_val / nav, 4), "forced": forced, "is_new": is_new,
                      "full_exit": side == "SELL" and shares >= held, "forced_reason": forced_exits.get(t, ""),
                      "replacement": bool(replenish and is_new), "priority": prio})

    cands.sort(key=lambda o: -o["priority"])
    chosen, turnover = [], 0.0
    for o in cands:
        if not init_mode:
            if len(chosen) >= config.MAX_TRADES_PER_DAY:
                skipped.append({"ticker": o["ticker"], "reason": f"daily cap of {config.MAX_TRADES_PER_DAY} trades reached"})
                continue
            if not o["forced"] and turnover + o["notional"] / nav / 2 > config.MAX_DAILY_TURNOVER + 1e-9:
                skipped.append({"ticker": o["ticker"], "reason": f"would exceed {config.MAX_DAILY_TURNOVER:.0%} daily turnover"})
                continue
        if o["is_new"] and _post_count(positions, chosen) >= config.MAX_HOLDINGS:
            skipped.append({"ticker": o["ticker"], "reason": f"would exceed {config.MAX_HOLDINGS} holdings"})
            continue
        chosen.append(o)
        turnover += o["notional"] / nav / 2

    # A7 floor on the executable book: drop unpaired discretionary exits (lowest priority first)
    while _post_count(positions, chosen) < config.MIN_HOLDINGS:
        disc = [o for o in chosen if o["full_exit"] and not o["forced"]]
        if not disc:
            break
        drop = min(disc, key=lambda o: o["priority"])
        chosen.remove(drop)
        skipped.append({"ticker": drop["ticker"], "reason": f"exit would leave < {config.MIN_HOLDINGS} holdings "
                        "without a paired buy today (IPS A7): not executed"})

    # Cash >= 0 (Brief §8.2): sells fund buys; trim the lowest-priority buys if short
    sells = [o for o in chosen if o["side"] == "SELL"]
    buys = [o for o in chosen if o["side"] == "BUY"]
    avail = cash + sum(o["notional"] for o in sells)
    final_buys = []
    for o in sorted(buys, key=lambda o: -o["priority"]):
        if o["notional"] > avail:
            sh = int(math.floor(avail / o["price"]))
            if sh * o["price"] < config.MIN_TRADE_NOTIONAL:
                skipped.append({"ticker": o["ticker"], "reason": "not enough cash after higher-priority trades"})
                continue
            o = {**o, "shares": sh, "notional": round(sh * o["price"], 2)}
        avail -= o["notional"]
        final_buys.append(o)

    # A12 cash <= 3% on the executable book: upsize the planned BUYs, then top up held names if a slot is free
    if avail / nav > L["max_cash_weight"]:
        post_val = _post_values(positions, sells + final_buys, px)
        excess = avail - config.TARGET_CASH * nav
        lev_total = sum(v for t, v in post_val.items() if asset_types.get(t) == "leveraged_etf")
        lev_room_total = 0.995 * (lev_cap if lev_cap is not None else L["max_leveraged_total"]) * nav - lev_total

        def room(t):
            k = asset_types.get(t, "stock")
            name_cap = L["max_single_weight"] if k == "stock" else min(L["max_leveraged_each"], caps.get(t, 1.0)) \
                if k == "leveraged_etf" else min(L["max_inverse_total"], caps.get(t, 1.0))
            r = 0.995 * name_cap * nav - post_val.get(t, 0.0)
            if k == "stock":
                sec_val = sum(v for u, v in post_val.items() if sectors.get(u) == sectors.get(t))
                r = min(r, 0.995 * L["max_sector_weight"] * nav - sec_val)
            elif k == "leveraged_etf":
                r = min(r, lev_room_total)
            return max(0.0, r)

        # spread the excess across today's BUYs pro rata to their size (not "fill the first one to its cap"),
        # a few passes so what one name can't take (cap/sector room) goes to the others
        for _ in range(4):
            eligible = [o for o in final_buys if not (no_new_leveraged and asset_types.get(o["ticker"]) == "leveraged_etf")
                        and room(o["ticker"]) >= o["price"]]
            base = sum(o["notional"] for o in eligible)
            if excess < 1 or not eligible or base <= 0:
                break
            pool = excess
            for o in eligible:
                want = pool * o["notional"] / base
                add_sh = int(math.floor(min(room(o["ticker"]), want, excess) / o["price"]))
                if add_sh <= 0:
                    continue
                j = next(i for i, x in enumerate(final_buys) if x["ticker"] == o["ticker"])
                cur = final_buys[j]
                final_buys[j] = {**cur, "shares": cur["shares"] + add_sh,
                                 "notional": round((cur["shares"] + add_sh) * cur["price"], 2), "upsized": True}
                post_val[o["ticker"]] = post_val.get(o["ticker"], 0.0) + add_sh * o["price"]
                if asset_types.get(o["ticker"]) == "leveraged_etf":
                    lev_room_total -= add_sh * o["price"]
                excess -= add_sh * o["price"]
                avail -= add_sh * o["price"]
        slots = len(cands) if init_mode else config.MAX_TRADES_PER_DAY - len(sells) - len(final_buys)
        if avail / nav > L["max_cash_weight"] and slots > 0:
            in_plan = {o["ticker"] for o in final_buys}
            for t in sorted((t for t in held_names - set(forced_exits) if t in px and t not in in_plan
                             and t not in untradeable and asset_types.get(t, "stock") == "stock"),
                            key=lambda t: -(composite.get(t) or {}).get("composite", 0.0)):
                if slots <= 0 or excess <= config.MIN_TRADE_NOTIONAL:
                    break
                amt = min(room(t), excess)
                sh = int(math.floor(amt / px[t]))
                if sh * px[t] < config.MIN_TRADE_NOTIONAL:
                    continue
                final_buys.append({"ticker": t, "side": "BUY", "shares": sh, "price": round(px[t], 4),
                                   "notional": round(sh * px[t], 2), "target_weight": round((post_val.get(t, 0) + sh * px[t]) / nav, 4),
                                   "current_weight": round(positions[t]["shares"] * px[t] / nav, 4), "forced": False,
                                   "is_new": False, "full_exit": False, "forced_reason": "", "replacement": False,
                                   "priority": 0.0, "topup_for_cash": True})
                post_val[t] = post_val.get(t, 0.0) + sh * px[t]
                excess -= sh * px[t]; avail -= sh * px[t]; slots -= 1
        if avail / nav > L["max_cash_weight"]:
            skipped.append({"ticker": "CASH", "reason": f"cash would stay at {avail / nav:.1%} > {L['max_cash_weight']:.0%}: "
                            "no room left in today's buys (name/sector/ETF caps) and no free trade slot (flagged, IPS A12)"})

    orders = sorted(sells, key=lambda o: -o["notional"]) + sorted(final_buys, key=lambda o: -o["notional"])
    return orders, skipped


def _post_values(positions: dict, orders: list[dict], px: dict) -> dict[str, float]:
    sh = {t: p.get("shares", 0) for t, p in positions.items()}
    for o in orders:
        sh[o["ticker"]] = sh.get(o["ticker"], 0) + (o["shares"] if o["side"] == "BUY" else -o["shares"])
    return {t: s * px[t] for t, s in sh.items() if s > 0 and t in px}


def _post_count(positions: dict, orders: list[dict]) -> int:
    sh = {t: p.get("shares", 0) for t, p in positions.items()}
    for o in orders:
        sh[o["ticker"]] = sh.get(o["ticker"], 0) + (o["shares"] if o["side"] == "BUY" else -o["shares"])
    return sum(1 for s in sh.values() if s > 0)


def shrink_to_limits(target: dict[str, float], rr: dict, sectors: dict, asset_types: dict,
                     caps: dict | None = None, lev_cap: float | None = None) -> dict[str, float]:
    """Mechanical de-risk after the last Risk round (IPS C6.5): shrink the offending positions pro rata and
    hand the freed weight to the compliant stocks with room. Keeps total exposure (cash <= 3%)."""
    L = config.RISK_LIMITS
    w = dict(target)
    caps = caps or {}
    lev_cap = L["max_leveraged_total"] if lev_cap is None else lev_cap
    for _ in range(6):
        freed, offenders = 0.0, set()
        for t, x in list(w.items()):
            k = asset_types.get(t, "stock")
            cap = L["max_single_weight"] if k == "stock" else min(L["max_leveraged_each"], caps.get(t, 1.0)) \
                if k == "leveraged_etf" else min(L["max_inverse_total"], caps.get(t, 1.0))
            if x > cap * 0.995:
                freed += x - cap * 0.995; w[t] = cap * 0.995; offenders.add(t)
        sec = {}
        for t, x in w.items():
            if asset_types.get(t, "stock") == "stock":
                sec.setdefault(sectors.get(t), []).append(t)
        for s_, names in sec.items():
            tot = sum(w[t] for t in names)
            if tot > L["max_sector_weight"] * 0.995:
                f = L["max_sector_weight"] * 0.995 / tot
                for t in names:
                    freed += w[t] * (1 - f); w[t] *= f; offenders.add(t)
        lev = [t for t in w if asset_types.get(t) == "leveraged_etf"]
        lev_tot = sum(w[t] for t in lev)
        if lev_tot > lev_cap * 0.995 + 1e-9:
            f = (lev_cap * 0.995) / lev_tot if lev_tot > 0 else 0
            for t in lev:
                freed += w[t] * (1 - f); w[t] *= f; offenders.add(t)
        if "volatility" in " ".join(rr.get("breaches", [])) or "VaR" in " ".join(rr.get("breaches", [])):
            for t in lev:
                freed += w[t] * 0.3; w[t] *= 0.7; offenders.add(t)
        if freed <= 1e-6:
            break
        recv = [t for t in w if t not in offenders and asset_types.get(t, "stock") == "stock"]
        if not recv:
            break
        base = sum(w[t] for t in recv)
        for t in recv:
            w[t] += freed * (w[t] / base if base > 0 else 1 / len(recv))
    return {t: math.floor(x * 1e4) / 1e4 for t, x in w.items() if x > 0}
