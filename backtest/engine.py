"""
Rule-based backtest of the fund's Quant/ensemble sleeve (Oct 23). No LLM anywhere (LLM backtests leak look-ahead:
the model has read the future). What is real: the strategy code, the walk-forward IC weights, the entry gate, the vol
overlay, the 10-20 name / 20% / 35% limits, `analytics.plan_orders` (5 trades/day, A16 drift band, trim pairing,
A7 floor, A12 cash) and the stop rules. What is simplified: no Research/Sentiment/Macro agents, no LSTM/Kronos/LightGBM,
no PM/Risk/CIO LLM judgement, stocks only (the leveraged-ETF filter is tested standalone), weights from a rule instead
of the mean-variance optimiser. Fills at the next open, zero costs (A20) unless a cost is passed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import analytics
import config
import strategies
from strategies.ensemble import forward_returns, rank_ic, weights_from_table

BACKTESTABLE = ["xs_momentum_12_1", "ts_trend", "industry_momentum", "residual_momentum", "high_52w", "st_reversal_1w",
                "pead_sue", "earnings_premium", "low_beta", "high_volume_premium", "sector_rotation"]
Z_TO_SCORE = config.ENSEMBLE["z_to_score"]


@dataclass
class Prep:
    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    sector: pd.Series
    asset_type: pd.Series
    stocks: list
    z: dict            # strategy -> z panel (dates x all tickers)
    ic: pd.DataFrame   # dates x strategy (cross-sectional rank IC against the 5d forward return)
    legacy: pd.DataFrame
    eligible: pd.DataFrame
    vol63: pd.DataFrame
    names: list


def legacy_technical(close: pd.DataFrame, stocks: list) -> pd.DataFrame:
    """Vectorised copy of Quant's legacy sub-scores (momentum z, trend, MACD, RSI) and its vol adjustment."""
    c = close[stocks]
    mom = c.shift(21) / c.shift(252) - 1
    mz = ((mom.sub(mom.mean(axis=1), axis=0)).div(mom.std(axis=1).replace(0, np.nan), axis=0) / 2).clip(-1, 1).fillna(0)
    trend = (np.where(c > c.rolling(50).mean(), 0.5, -0.5) + np.where(c > c.rolling(200).mean(), 0.5, -0.5))
    trend = pd.DataFrame(trend, index=c.index, columns=c.columns)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    macd_s = pd.DataFrame(np.where(hist > 0, 0.5, -0.5), index=c.index, columns=c.columns)
    d = c.diff()
    rs = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + rs)
    rsi_s = pd.DataFrame(np.where(rsi > 75, -1.0, np.where((rsi >= 50) & (rsi <= 70), 0.3, 0.0)), index=c.index, columns=c.columns)
    tech = 0.40 * mz + 0.30 * trend + 0.10 * macd_s + 0.20 * rsi_s
    vol = c.pct_change(fill_method=None).rolling(63).std() * np.sqrt(252)
    out = tech / (1 + (vol - 0.40).clip(lower=0) / 0.40)
    return out.clip(-1, 1).where(mom.notna())


def prepare(px: dict, uni: pd.DataFrame, earn_events: dict | None = None) -> Prep:
    close, openp, volume = px["close"], px["open"], px["volume"]
    uni = uni[uni["symbol"].isin(close.columns)]
    at = {t: a for t, a in zip(uni["symbol"], uni["asset_type"])}
    d = strategies.build_data(close, volume, uni, at, benchmark="SPY", earn_events=earn_events or None)
    stocks = d.stocks()
    strats = [s for s in strategies.registry(BACKTESTABLE) if s.name != "pead_sue" or earn_events]
    if not earn_events:
        strats = [s for s in strats if s.name not in ("pead_sue", "earnings_premium")]
    z = {s.name: s.z(d).reindex(columns=stocks) for s in strats}
    fwd = forward_returns(close, config.ENSEMBLE["ic_horizon"])[stocks]
    ic = pd.DataFrame({k: rank_ic(v, fwd) for k, v in z.items()})
    adv = (close * volume).rolling(20).mean()
    elig = (close[stocks] >= 5) & (adv[stocks] >= config.SCREEN["min_adv_usd"]) & (close[stocks].notna().rolling(252).sum() >= 252)
    vol63 = close[stocks].pct_change(fill_method=None).rolling(63).std() * np.sqrt(252)
    return Prep(close=close, open=openp, volume=volume, sector=d.sector, asset_type=d.asset_type, stocks=stocks, z=z, ic=ic,
                legacy=legacy_technical(close, stocks), eligible=elig, vol63=vol63, names=list(z))


def weight_history(P: Prep, exclude: tuple = (), equal: bool = False, cfg: dict | None = None) -> pd.DataFrame:
    """Per-date ensemble weights, learned only from ICs whose forward return was realised by that date."""
    cfg = {**config.ENSEMBLE, **(cfg or {})}
    names = [n for n in P.names if n not in exclude]
    if equal:
        return pd.DataFrame(1.0 / len(names), index=P.close.index, columns=names)
    known = P.ic[names].shift(cfg["ic_horizon"])                     # IC of day k is only known from k + horizon
    roll = known.rolling(cfg["ic_sessions"], min_periods=2)
    mean, sd, n = roll.mean(), roll.std(ddof=1), roll.count().fillna(0)
    t = mean / (sd / np.sqrt((n / cfg["ic_horizon"]).clip(lower=1.0))).replace(0, np.nan)
    shrunk = mean.clip(lower=0) * (1 - 1 / t ** 2).clip(lower=0)
    rows = []
    hist = {k: True for k in names}
    for i in range(len(known)):
        table = {k: {"n": int(n.iat[i, j]), "t": None if pd.isna(t.iat[i, j]) else float(t.iat[i, j]),
                     "shrunk_ic": 0.0 if pd.isna(shrunk.iat[i, j]) else float(shrunk.iat[i, j]),
                     "validated": int(n.iat[i, j]) >= cfg["ic_sessions"]} for j, k in enumerate(names)}
        rows.append(weights_from_table(table, hist, cfg)[0])
    return pd.DataFrame(rows, index=known.index).reindex(columns=names).fillna(0.0)


def ensemble_panels(P: Prep, W: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(score -1..1, votes_long, allocated weight) per date from weights W."""
    num = den = None
    votes = None
    for k in W.columns:
        z = P.z[k]
        w = W[k].values[:, None]
        a = z.fillna(0.0).values * w
        b = z.notna().values * w
        v = (z >= config.ENSEMBLE["vote_z"]).astype(int).values
        num = a if num is None else num + a
        den = b if den is None else den + b
        votes = v if votes is None else votes + v
    with np.errstate(invalid="ignore", divide="ignore"):
        ens_z = np.where(den > 0, num / den, np.nan)
    score = pd.DataFrame(np.clip(ens_z / Z_TO_SCORE, -1, 1), index=P.close.index, columns=P.stocks)
    return score, pd.DataFrame(votes, index=P.close.index, columns=P.stocks), W.sum(axis=1)


def as_deployed(P: Prep, score: pd.DataFrame, alloc: pd.Series) -> pd.DataFrame:
    """Quant's rule_score: (1 - b) x legacy technical + b x ensemble, b = blend x allocated weight (live code path)."""
    b = (config.ENSEMBLE["blend"] * alloc).clip(0, 1).values[:, None]
    e = score.reindex(index=P.legacy.index)
    return ((1 - b) * P.legacy + b * e.where(e.notna(), P.legacy)).where(P.legacy.notna() | e.notna())


# ----------------------------------------------------------------------------------------------- portfolio rule
def cap_book(w: dict, sector: dict, total: float, name_cap=0.20, sec_cap=0.35) -> dict:
    """Scale to `total`, then enforce the name and sector caps, handing the excess to the names with room."""
    s = sum(w.values())
    if s <= 0:
        return {}
    w = {k: v / s * total for k, v in w.items()}
    for _ in range(12):
        for k in w:
            w[k] = min(w[k], name_cap)
        for sec in {sector.get(k) for k in w}:
            members = [k for k in w if sector.get(k) == sec]
            tot = sum(w[k] for k in members)
            if tot > sec_cap:
                for k in members:
                    w[k] *= sec_cap / tot
        gap = total - sum(w.values())
        if gap < 1e-4:
            break
        room = {k: min(name_cap - w[k], max(0.0, sec_cap - sum(w[m] for m in w if sector.get(m) == sector.get(k))))
                for k in w}
        rs = sum(r for r in room.values() if r > 0)
        if rs <= 1e-6:
            break
        for k, r in room.items():
            if r > 0:
                w[k] += gap * r / rs if gap * r / rs < r else r
    return {k: math.floor(v * 1e4) / 1e4 for k, v in w.items() if v > 0}


def simulate(P: Prep, score: pd.DataFrame, votes: pd.DataFrame | None, *, freq: str = "W", n_names: int = 15,
             pair_trims: bool = True, trailing: bool = True, stop: bool = True, vote_gate: bool = True,
             vol_managed: bool = True, cost_bps: float = 0.0, start: str = "2023-10-23", end: str | None = None,
             cooldown_days: int = 5) -> dict:
    idx = P.close.index
    i0 = int(idx.searchsorted(pd.Timestamp(start)))
    i1 = len(idx) - 1 if end is None else int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    sector = P.sector.to_dict()
    at = P.asset_type.to_dict()
    pos: dict[str, dict] = {}
    cash, pending, first = config.STARTING_CAPITAL, [], True
    nav_hist, traded, n_trades, n_forced, cool = {}, 0.0, 0, 0, {}
    n_trims = n_pairs = n_absorbed = 0
    fills_by_day: dict = {}
    fills: list = []
    min_cash = cash
    gate = config.ENSEMBLE["min_votes"]
    week_end = pd.Series(idx.isocalendar().week.values, index=idx)
    last_of_week = week_end.ne(week_end.shift(-1))
    for i in range(max(i0 - 1, 0), i1 + 1):
        day = idx[i]
        if pending and i >= i0:                                   # fills at today's open
            orow = P.open.iloc[i]
            crow = P.close.iloc[i]
            for o in sorted(pending, key=lambda o: o["side"] != "SELL"):
                t = o["ticker"]
                px = orow.get(t)
                px = crow.get(t) if px is None or pd.isna(px) else px
                if px is None or pd.isna(px):
                    continue
                if o["side"] == "SELL":
                    sh = min(o["shares"], pos.get(t, {}).get("shares", 0))
                    if sh <= 0:
                        continue
                    cash += sh * px * (1 - cost_bps / 1e4)
                    fills.append((day, t, "SELL", sh, float(px)))
                    pos[t]["shares"] -= sh
                    if pos[t]["shares"] <= 0:
                        del pos[t]
                else:
                    sh = min(o["shares"], int(cash // (px * (1 + cost_bps / 1e4))))
                    if sh <= 0:
                        continue
                    cash -= sh * px * (1 + cost_bps / 1e4)
                    fills.append((day, t, "BUY", sh, float(px)))
                    p = pos.setdefault(t, {"shares": 0, "avg_cost": px, "opened_date": str(day.date())})
                    p["avg_cost"] = (p["avg_cost"] * p["shares"] + px * sh) / (p["shares"] + sh)
                    p["shares"] += sh
                traded += sh * px
                n_trades += 1
                fills_by_day[day] = fills_by_day.get(day, 0) + 1
            pending = []
        crow = P.close.iloc[i]
        min_cash = min(min_cash, cash)
        value = sum(p["shares"] * crow[t] for t, p in pos.items() if not pd.isna(crow[t]))
        nav = cash + value
        if i >= i0:
            nav_hist[day] = nav
        if i == i1:
            break
        forced = {}
        for t, p in pos.items():
            px = crow[t]
            if pd.isna(px):
                continue
            if stop and px / p["avg_cost"] - 1 <= config.STOP_LOSS:
                forced[t] = "stop-loss"
            elif trailing:
                dd, _ = analytics.trailing_drawdown(p["avg_cost"], px, P.close[t].loc[p["opened_date"]:day].values)
                if dd <= config.TRAILING_STOP:
                    forced[t] = "trailing stop"
        for t in forced:
            cool[t] = i + cooldown_days
        decision = first or freq == "D" or bool(last_of_week.iat[i])
        if not (decision or forced):
            continue
        srow = score.iloc[i]
        ok = P.eligible.iloc[i] & srow.notna()
        cand = srow[ok].sort_values(ascending=False)
        rank = cand.rank(ascending=False)
        keep = [t for t in pos if t not in forced and t in rank.index and rank[t] <= 25 and cand[t] > -0.05]
        vrow = votes.iloc[i] if votes is not None else None
        pool = [t for t in cand.index if t not in pos and cool.get(t, -1) < i and cand[t] > 0.05
                and (not vote_gate or vrow is None or vrow[t] >= gate)]
        want = (config.TARGET_HOLDINGS_INIT if first else n_names)
        names = keep + pool[:max(0, want - len(keep))]
        if len(names) < config.MIN_HOLDINGS:                      # A7 floor: fill from the next best, gate or not
            names += [t for t in cand.index if t not in names and t not in forced and cool.get(t, -1) < i][:config.MIN_HOLDINGS - len(names)]
        if not names:
            continue
        raw = {t: max(float(cand[t]), 0.05) for t in names}
        if vol_managed:
            vv = P.vol63.iloc[i][names]
            med = float(P.vol63.iloc[i][P.eligible.iloc[i]].median())
            for t in names:
                if not pd.isna(vv[t]) and vv[t] > 0:
                    raw[t] *= float(np.clip(med / vv[t], config.VOL_MANAGED["floor"], config.VOL_MANAGED["cap"]))
        target = cap_book(raw, sector, 0.97)
        pxd = {t: float(crow[t]) for t in set(target) | set(pos) if not pd.isna(crow[t])}
        comp = {t: {"composite": float(srow[t]) if not pd.isna(srow[t]) else 0.0} for t in pxd}
        pos_in = {t: dict(p) for t, p in pos.items()}
        orders, _ = analytics.plan_orders(target, pos_in, pxd, nav, cash, forced, first, comp, {}, sectors=sector,
                                          asset_types=at, buy_ideas=pool if pair_trims else None)
        n_forced += sum(1 for o in orders if o.get("forced"))
        n_trims += sum(1 for o in orders if o["side"] == "SELL" and not o.get("forced") and not o.get("full_exit"))
        n_pairs += sum(1 for o in orders if o.get("pair_of"))
        n_absorbed += sum(1 for o in orders if o["side"] == "SELL" and not o.get("forced") and not o.get("full_exit")
                          and any(b["side"] == "BUY" and b.get("is_new") for b in orders))
        pending = orders
        first = False
    nav_s = pd.Series(nav_hist)
    return {"nav": nav_s, "turnover_one_way": traded / 2, "trades": n_trades, "forced_exits": n_forced, "trims": n_trims, "paired_buys": n_pairs, "trims_absorbed": n_absorbed,
            "avg_nav": float(nav_s.mean()), "fills_by_day": fills_by_day, "fills": fills, "min_cash": min_cash}


# ----------------------------------------------------------------------------------------------------- metrics
def metrics(nav: pd.Series, traded_one_way: float | None = None) -> dict:
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = float(r.std() * np.sqrt(252))
    dd = nav / nav.cummax() - 1
    out = {"CAGR": cagr, "Vol": vol, "Sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
           "MaxDD": float(dd.min()), "Total": float(nav.iloc[-1] / nav.iloc[0] - 1)}
    if traded_one_way is not None:
        out["Turnover/yr"] = traded_one_way / float(nav.mean()) / yrs
    return out


def buy_hold(px: pd.DataFrame, openp: pd.DataFrame, ticker: str, start: str, end: str | None = None) -> pd.Series:
    i0 = int(px.index.searchsorted(pd.Timestamp(start)))
    c = px[ticker].iloc[i0:]
    if end:
        c = c[c.index <= pd.Timestamp(end)]
    base = float(openp[ticker].iloc[i0]) if not pd.isna(openp[ticker].iloc[i0]) else float(c.iloc[0])
    return config.STARTING_CAPITAL * c / base


def standalone(P: Prep, name: str, top: int = 15, start: str = "2023-10-23") -> dict:
    """Long-only top-N by one strategy's z, equal weight, weekly, signal at Friday close, hold Monday open -> next
    Monday open. Sharpe, turnover and the mean rank IC with its t-stat."""
    idx = P.close.index
    z = P.z[name]
    wk = pd.Series(idx.isocalendar().week.values, index=idx)
    dec = [i for i in range(len(idx) - 1) if wk.iat[i] != wk.iat[i + 1] and idx[i] >= pd.Timestamp(start) - pd.Timedelta(days=7)]
    rets, prev, turn = {}, set(), 0.0
    for a, b in zip(dec[:-1], dec[1:]):
        row = z.iloc[a].where(P.eligible.iloc[a]).dropna()
        if len(row) < top * 2:
            continue
        hold = set(row.nlargest(top).index)
        o = P.open.iloc[a + 1]
        c = P.open.iloc[b + 1] if b + 1 < len(idx) else P.close.iloc[b]
        r = (c[list(hold)] / o[list(hold)] - 1).dropna()
        if len(r):
            rets[idx[b]] = float(r.mean())
            turn += len(hold - prev) / top / 2 if prev else 0.0
            prev = hold
    r = pd.Series(rets)
    r = r[r.index >= pd.Timestamp(start)]
    ic = P.ic[name].dropna()
    ic = ic[ic.index >= pd.Timestamp(start)]
    t = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic) / config.ENSEMBLE["ic_horizon"]))) if len(ic) > 2 and ic.std() > 0 else np.nan
    return {"Sharpe": float(r.mean() / r.std() * np.sqrt(52)) if len(r) > 5 and r.std() > 0 else np.nan,
            "CAGR": float((1 + r).prod() ** (52 / len(r)) - 1) if len(r) else np.nan,
            "MeanIC": float(ic.mean()) if len(ic) else np.nan, "IC_t": t, "Turnover/yr": turn / max(len(r) / 52, 1e-9)}


def etf_filter_standalone(P: Prep, start: str = "2023-10-23") -> dict:
    """Leveraged ETFs held (equal weight) only while above their 200dma, else cash, vs always holding them."""
    etf = [t for t in P.close.columns if P.asset_type.get(t) == "leveraged_etf"]
    c = P.close[etf]
    above = c > c.rolling(200, min_periods=150).mean()
    r = c.pct_change(fill_method=None)
    held = above.shift(1)                                         # trade on the next bar
    flt = (r.where(held).mean(axis=1)).fillna(0.0)
    raw = r.mean(axis=1).fillna(0.0)
    sel = flt.index >= pd.Timestamp(start)
    out = {}
    for lab, s in (("leveraged ETFs, 200dma filter", flt), ("leveraged ETFs, always held", raw)):
        s = s[sel]
        out[lab] = {"Sharpe": float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else np.nan,
                    "CAGR": float((1 + s).prod() ** (252 / len(s)) - 1), "MaxDD": float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min())}
    return out


def tom_study(P: Prep, start: str = "2023-10-23") -> dict:
    from strategies.turn_of_month import flags
    f = flags(P.close.index)
    r = P.close["SPY"].pct_change()
    r, f = r[r.index >= pd.Timestamp(start)], f[f.index >= pd.Timestamp(start)]
    a, b = r[f == 1], r[f == 0]
    t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
    return {"window_days": int(len(a)), "mean_window_bp": float(a.mean() * 1e4), "mean_other_bp": float(b.mean() * 1e4), "t": float(t)}


# ----------------------------------------------------------------------------------------------- Monte Carlo
def monte_carlo(strat: pd.Series, bench: pd.Series, paths: int = 10_000, horizon: int = 15, block: int = 5,
                seed: int = 7) -> dict:
    """Stationary-block bootstrap of daily (strategy, benchmark) return PAIRS (keeps their correlation), 15-day paths."""
    df = pd.concat([strat, bench], axis=1).dropna()
    s, b = df.iloc[:, 0].values, df.iloc[:, 1].values
    n = len(df)
    rng = np.random.default_rng(seed)
    nb = math.ceil(horizon / block)
    starts = rng.integers(0, n - block, size=(paths, nb))
    pick = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, -1)[:, :horizon]
    ps, pb = s[pick], b[pick]
    cs = np.cumprod(1 + ps, axis=1)
    cb = np.cumprod(1 + pb, axis=1)
    ret, bret = cs[:, -1] - 1, cb[:, -1] - 1
    dd = (cs / np.maximum.accumulate(np.concatenate([np.ones((paths, 1)), cs], axis=1), axis=1)[:, 1:] - 1).min(axis=1)
    q = lambda x: {f"p{k}": float(np.percentile(x, k)) for k in (5, 25, 50, 75, 95)}
    return {"paths": paths, "horizon": horizon, "block": block,
            "P(beat benchmark)": float((ret > bret).mean()), "P(DD <= -7%)": float((dd <= -0.07).mean()),
            "P(DD <= -10%)": float((dd <= -0.10).mean()), "P(loss)": float((ret < 0).mean()),
            "return": q(ret), "benchmark_return": q(bret), "excess": q(ret - bret), "max_dd": q(dd), "_ret": ret, "_excess": ret - bret}
