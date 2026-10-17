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
    W = config.SIGNAL_WEIGHTS
    out = {}
    for t in tickers:
        kind = asset_types.get(t, "stock")
        q = float((quant.get(t) or {}).get("score", 0) or 0)
        s = float((sentiment.get(t) or {}).get("score", 0) or 0)
        if kind == "stock":
            r_row = research.get(t) or {}
            r = float(r_row.get("score", 0) or 0)
            m = TILT_TO_SCORE.get(str(tilts.get(sectors.get(t), "neutral")).lower(), 0.0)
            comp = W["research"] * r + W["quant"] * q + W["sentiment"] * s + W["macro"] * m
            insufficient = r_row.get("rating") == "insufficient_data"
        else:
            # IPS C2.5: ETFs have no fundamentals -> re-weight Quant/Sentiment/Macro to 100%
            r = None
            if kind == "leveraged_etf":
                m = {"risk_on": 1.0, "neutral": 0.0, "risk_off": -1.0}[regime]
            else:
                m = 1.0 if regime == "risk_off" else -1.0
            tot = W["quant"] + W["sentiment"] + W["macro"]
            comp = (W["quant"] * q + W["sentiment"] * s + W["macro"] * m) / tot
            insufficient = False
        comp = float(np.clip(comp, -1, 1))
        out[t] = {"type": kind, "research": None if r is None else round(r, 2), "quant": round(q, 2),
                  "sentiment": round(s, 2), "macro": round(m, 2), "composite": round(comp, 3),
                  "composite_10": to_10(comp), "insufficient_data": insufficient}
    return out


# --------------------------------------------------------------------------- #
# Optimiser (Portfolio Manager's tool)
# --------------------------------------------------------------------------- #
def optimise(convictions: dict[str, float], prices: pd.DataFrame, sectors: dict[str, str],
             asset_types: dict[str, str], equity_exposure: float, regime: str, vix: float,
             max_names: int, risk_aversion: float = 4.0, diversification: float = 0.5,
             lev_cap_override: float | None = None, min_weight: float = 0.01,
             lookback: int = 252) -> dict[str, float]:
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
    equity_exposure = float(np.clip(equity_exposure, 1 - L["max_cash_weight"], 1 - L["min_cash_weight"]))

    def kind(t):
        return asset_types.get(t, "stock")

    # New money only goes to names with a positive view; if fewer than 10 qualify
    # (Brief: 10-20 holdings) the next-best non-negative names fill the gap.
    ranked = sorted([t for t in convictions if kind(t) == "stock"], key=lambda t: -convictions[t])
    stocks = [t for t in ranked if convictions[t] > 0]
    if len(stocks) < config.MIN_HOLDINGS:
        stocks += [t for t in ranked if -0.5 < convictions[t] <= 0][:config.MIN_HOLDINGS - len(stocks)]
    lev = sorted([t for t in convictions if kind(t) == "leveraged_etf" and convictions[t] > 0],
                 key=lambda t: -convictions[t]) if lev_cap > 0 else []
    inv = sorted([t for t in convictions if kind(t) == "inverse_etf"], key=lambda t: -convictions[t]) \
        if inv_hi > 0 else []
    n_etf = min(len(lev), 2) + min(len(inv), 1)
    chosen = stocks[:max(config.MIN_HOLDINGS, max_names - n_etf)] + lev[:2] + inv[:1]
    tickers = [t for t in chosen if t in prices.columns]
    if not tickers:
        return {}

    keep = list(tickers)
    w = None
    for _ in range(4):
        w, ub = _solve(keep, convictions, prices, sectors, kind, equity_exposure, lev_cap, inv_lo, inv_hi,
                       risk_aversion, diversification, lookback, beta_band=R["beta"])
        small = [t for t, x in zip(keep, w) if x < min_weight]
        if not small or len(keep) - len(small) < config.MIN_HOLDINGS:
            break
        keep = [t for t in keep if t not in small]
    w = np.minimum(np.clip(w, 0, None), ub)
    # floor (not round) so the target never needs more cash than we have
    return {t: math.floor(float(x) * 1e4) / 1e4 for t, x in zip(keep, w) if x >= min_weight * 0.999}


def _solve(tickers, convictions, prices, sectors, kind, equity_exposure, lev_cap, inv_lo, inv_hi,
           risk_aversion, diversification, lookback, beta_band=None):
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
        ub.append(buf * (L["max_single_weight"] if k == "stock" else
                         min(L["max_leveraged_each"], lev_cap) if k == "leveraged_etf" else inv_hi))
    ub = np.array(ub)
    bounds = [(0.0, b) for b in ub]
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
    return (res.x if res.success else w0), ub


# --------------------------------------------------------------------------- #
# Risk (Risk Manager's tool)
# --------------------------------------------------------------------------- #
def risk_report(weights: dict[str, float], prices: pd.DataFrame, sectors: dict[str, str],
                asset_types: dict[str, str], regime: str, nav_drawdown: float = 0.0,
                lookback: int = 252) -> dict:
    L, R = config.RISK_LIMITS, config.REGIME_RULES[regime]
    tickers = [t for t, w in weights.items() if w > 0]
    if not tickers:
        return {"note": "100% cash", "breaches": ["portfolio is 100% cash (IPS A12: max 5%)"],
                "soft_concerns": [], "portfolio_vol_ann": 0.0, "cash_weight": 1.0}

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
    cash = 1 - float(w.sum())
    lev_w = {t: float(x) for t, x in zip(tickers, w) if asset_types.get(t) == "leveraged_etf"}
    inv_w = {t: float(x) for t, x in zip(tickers, w) if asset_types.get(t) == "inverse_etf"}
    corr = rets[tickers].corr().values if len(tickers) > 1 else np.array([[1.0]])
    max_pair_corr = float(np.max(corr[np.triu_indices(len(tickers), 1)])) if len(tickers) > 1 else 0.0

    eps = 1e-4
    breaches = []
    for t, x in zip(tickers, w):
        lim = L["max_single_weight"] if asset_types.get(t, "stock") == "stock" else L["max_leveraged_each"]
        if asset_types.get(t) == "inverse_etf":
            lim = L["max_inverse_total"]
        if x > lim + eps:
            breaches.append(f"{t} weight {x:.1%} > limit {lim:.0%}")
    for s, x in sector_w.items():
        if s in ("Leveraged ETF", "Inverse ETF"):
            continue
        if x > L["max_sector_weight"] + eps:
            breaches.append(f"{s} sector {x:.1%} > limit {L['max_sector_weight']:.0%}")
    if cash < L["min_cash_weight"] - eps:
        breaches.append(f"cash {cash:.2%} < 0 (Brief: cash >= 0)")
    if cash > L["max_cash_weight"] + eps:
        breaches.append(f"cash {cash:.1%} > maximum {L['max_cash_weight']:.0%} (IPS A12)")
    lev_cap = min(L["max_leveraged_total"], R["max_leveraged"])
    if sum(lev_w.values()) > lev_cap + eps:
        breaches.append(f"leveraged ETFs {sum(lev_w.values()):.1%} > {lev_cap:.0%} allowed in {regime}")
    if inv_w and regime != "risk_off":
        breaches.append(f"inverse ETFs held outside risk-off (IPS A11): {list(inv_w)}")
    if sum(inv_w.values()) > L["max_inverse_total"] + eps:
        breaches.append(f"inverse ETFs {sum(inv_w.values()):.1%} > {L['max_inverse_total']:.0%}")
    if vol > L["max_portfolio_vol"]:
        breaches.append(f"volatility {vol:.1%} > limit {L['max_portfolio_vol']:.0%}")
    if var95 > L["max_var_95_1d"]:
        breaches.append(f"1d VaR95 {var95:.2%} > limit {L['max_var_95_1d']:.2%}")
    if L.get("max_beta") is not None and beta > L["max_beta"]:
        breaches.append(f"beta {beta:.2f} > limit {L['max_beta']:.2f}")
    n_pos = len(tickers)
    if not (config.MIN_HOLDINGS <= n_pos <= config.MAX_HOLDINGS):
        breaches.append(f"{n_pos} holdings outside {config.MIN_HOLDINGS}-{config.MAX_HOLDINGS} (Brief §1)")

    soft = []
    lo, hi = R["beta"]
    if not lo - 0.005 <= beta <= hi + 0.005:
        soft.append(f"beta {beta:.2f} outside the {regime} band {lo}-{hi} (IPS C1.4)")
    if 1 / hhi < 8:
        soft.append(f"effective positions {1 / hhi:.1f} < 8 (IPS C6.3)")
    if max_pair_corr > 0.8:
        soft.append(f"max pairwise correlation {max_pair_corr:.2f} > 0.8")
    if nav_drawdown <= L["max_drawdown"]:
        soft.append(f"fund drawdown {nav_drawdown:.1%} breached {L['max_drawdown']:.0%} (IPS A4): "
                    f"no new leveraged longs")

    return {
        "portfolio_vol_ann": round(vol, 4),
        "var_95_1d": round(var95, 4),
        "cvar_95_1d": round(cvar95, 4),
        "beta_vs_spy": round(beta, 2),
        "max_drawdown_1y_backtest": round(mdd, 4),
        "fund_drawdown_from_peak": round(nav_drawdown, 4),
        "n_positions": n_pos,
        "effective_n_positions": round(1 / hhi, 1),
        "max_pairwise_corr": round(max_pair_corr, 2),
        "cash_weight": round(cash, 4),
        "leveraged_etf_weight": round(sum(lev_w.values()), 4),
        "inverse_etf_weight": round(sum(inv_w.values()), 4),
        "sector_weights": {k: round(v, 4) for k, v in sorted(sector_w.items(), key=lambda kv: -kv[1])},
        "regime": regime,
        "breaches": breaches,
        "soft_concerns": soft,
    }


# --------------------------------------------------------------------------- #
# Order planning: target portfolio -> at most 5 trades (Brief §1, IPS C5.5)
# --------------------------------------------------------------------------- #
def plan_orders(target_w: dict[str, float], positions: dict[str, dict], px: dict[str, float],
                nav: float, cash: float, forced_exits: dict[str, str], init_mode: bool,
                composite: dict[str, dict], untradeable: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Returns (orders, skipped). Forced exits go first, then the trades that move
    the most weight toward the best-scored names. Sells always precede buys."""
    cands, skipped = [], []
    # IPS A12 (cash <= 5%) beats A16 (drift band): if exits would leave too much cash,
    # top-ups of existing holdings are allowed even inside the drift band.
    exit_proceeds = sum(p["shares"] * px.get(t, 0.0) for t, p in positions.items()
                        if t in forced_exits or target_w.get(t, 0.0) == 0)
    cash_heavy = (cash + exit_proceeds) / nav > config.RISK_LIMITS["max_cash_weight"]
    for t in sorted(set(target_w) | set(positions)):
        held = float(positions.get(t, {}).get("shares", 0))
        cur_val = held * px.get(t, 0.0)
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
        shares = int(math.floor(abs(diff_val) / price))
        if side == "SELL":
            shares = int(held) if tgt == 0 else min(shares, int(held))
        if shares <= 0:
            continue
        score = abs((composite.get(t) or {}).get("composite", 0.0))
        cands.append({"ticker": t, "side": side, "shares": shares, "price": round(price, 4),
                      "notional": round(shares * price, 2), "target_weight": round(tgt, 4),
                      "current_weight": round(cur_val / nav, 4), "forced": forced,
                      "forced_reason": forced_exits.get(t, ""),
                      "priority": (1e6 if forced else 0) + abs(dw) * (score + 0.1)})

    cands.sort(key=lambda o: -o["priority"])
    chosen, turnover = [], 0.0
    n_holdings = len([t for t, p in positions.items() if p.get("shares", 0) > 0])
    for o in cands:
        if not init_mode:
            if len(chosen) >= config.MAX_TRADES_PER_DAY:
                skipped.append({"ticker": o["ticker"], "reason": f"daily cap of {config.MAX_TRADES_PER_DAY} trades reached"})
                continue
            if not o["forced"] and turnover + o["notional"] / nav / 2 > config.MAX_DAILY_TURNOVER + 1e-9:
                skipped.append({"ticker": o["ticker"], "reason": f"would exceed {config.MAX_DAILY_TURNOVER:.0%} daily turnover"})
                continue
        is_new = o["side"] == "BUY" and positions.get(o["ticker"], {}).get("shares", 0) <= 0
        if is_new and n_holdings >= config.MAX_HOLDINGS:
            skipped.append({"ticker": o["ticker"], "reason": f"would exceed {config.MAX_HOLDINGS} holdings"})
            continue
        if is_new:
            n_holdings += 1
        if o["side"] == "SELL" and o["shares"] >= positions.get(o["ticker"], {}).get("shares", 0):
            n_holdings -= 1
        chosen.append(o)
        turnover += o["notional"] / nav / 2

    # Cash >= 0 (Brief §8.2): sells fund buys; trim the smallest-priority buys if short
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
    orders = sorted(sells, key=lambda o: -o["notional"]) + sorted(final_buys, key=lambda o: -o["notional"])
    return orders, skipped
