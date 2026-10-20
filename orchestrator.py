"""
The investment committee meeting: who speaks when, and what Python enforces.

    0. Guards                 max 3 runs/day (Brief §10.2), pre-flight data checks (IPS D10, C8.6)
    1. Data + screen          S&P 500 + leveraged ETFs -> hard filters -> ranked shortlist
    2. Regime (Python)        IPS C1.3 rules with the 2-day upgrade rule
    3. Analysts               Macro -> Research -> Quant -> Sentiment (a failed agent = no signal, D11)
    4. Blend + rules          composite (C5.2), forced exits (C5.4), entry rule (C5.3), D12 check
    5. PM <-> Risk            PM convictions -> optimiser -> <=5 orders -> Risk reviews the POST-TRADE book
    6. CIO sign-off           approve = trade; reject = only forced exits (stop-loss etc.)
    7. Trader (rules)         validity checks -> 5-trade cap -> cash >= 0 -> sells before buys

Brief §8.3 Trading Floor: "Run Agents" proposes, "Execute Trades" commits.
`run_committee(execute=False)` stores the proposal; `execute_run(run_id)` commits it.

Terminal:  python orchestrator.py [--provider mock] [--execute] [--reset] [--snapshot]
"""
from __future__ import annotations

import json
import math
import os

import pandas as pd

import analytics
import config
import data
import finbert
import database as db
import forecaster
import morning_brief
import reflection
import report
from forecasters import horse_race, kronos_fc, lgbm
import screener
from committee import (ChiefInvestmentOfficer, EquityResearchAnalyst, MacroStrategist,
                       DevilsAdvocate, PortfolioManager, QuantAnalyst, ReflectionAnalyst, RiskManager, SentimentAnalyst,
                       Trader)
from llm import LLMClient


class RunLimitError(RuntimeError):
    pass


def _last_prices(close, tickers) -> dict[str, float]:
    out = {}
    for t in tickers:
        if t in close.columns:
            s = close[t].dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
    return out


def _preflight(close, tickers) -> dict[str, str]:
    """IPS D10: per-ticker check. Returns {ticker: reason} for names that fail."""
    bad = {}
    for t in tickers:
        if t not in close.columns:
            bad[t] = "no price data"
            continue
        s = close[t].dropna()
        if len(s) < 2 or not math.isfinite(float(s.iloc[-1])) or s.iloc[-1] <= 0:
            bad[t] = "missing or non-positive latest price"
            continue
        jump = abs(float(s.iloc[-1] / s.iloc[-2] - 1))
        limit = config.PRICE_JUMP_EXCLUDE * max(1.0, abs(config.LEVERAGE_FACTOR.get(t, 1.0)))
        if jump > limit:
            bad[t] = f"price moved {jump:.0%} vs prior close (> {limit:.0%}): likely data error"
    return bad


def run_committee(provider: str | None = None, progress=None, execute: bool = False,
                  extra_forced: dict | None = None, _skip_run_limit: bool = False, asof=None) -> dict:
    """asof (replay, IPS D6-D9): decide on data strictly BEFORE that date (previous close); news/brief n/a,
    fundamentals frozen (today's snapshot), estimate revisions dropped, Kronos skipped, lighter LSTM walk-forward."""
    replay = asof is not None
    emit = progress or (lambda stage, msg: print(f"[{stage}] {msg}"))
    llm = LLMClient(provider)

    # ------------------------------------------------------- 1. data + guards -- #
    emit("data", "Loading S&P 500 universe + leveraged ETFs and downloading prices")
    universe = data.get_universe()
    market = data.get_market_data(universe["symbol"].tolist())
    if replay:
        cut = pd.Timestamp(asof)
        market = {k: v[v.index < cut] for k, v in market.items()}
    synthetic = data.is_synthetic()
    db.use_synthetic(synthetic)
    db.init_db()
    db.mark_stale_runs()
    if not synthetic and llm.provider != "mock" and not _skip_run_limit and not replay and db.runs_today() >= config.MAX_RUNS_PER_DAY:
        raise RunLimitError(f"Already {db.runs_today()} runs today; the brief allows {config.MAX_RUNS_PER_DAY}.")
    if synthetic:
        emit("data", "⚠️ Yahoo unreachable — using SYNTHETIC prices in a separate sandbox fund")

    close = market["close"]
    sectors = dict(zip(universe["symbol"], universe["sector"]))
    asset_types = dict(zip(universe["symbol"], universe["asset_type"]))
    positions = db.get_positions()
    holdings = list(positions)
    init_mode = not db.is_initialized()
    mode = "init" if init_mode else "daily"

    # ------------------------------------------------------- 2. regime ------- #
    macro = data.get_macro(asof)
    reg = analytics.macro_regime(macro)
    prev_eff, prev_raw = db.previous_regime()
    regime, regime_note = analytics.apply_hysteresis(reg["raw_regime"], prev_eff, prev_raw)
    db.save_regime(reg["raw_regime"], regime, regime_note)
    regime_info = {**reg, "regime": regime, "note": regime_note, "previous_regime": prev_eff}
    rules = config.REGIME_RULES[regime]

    # ------------------------------------------------------- screening ------- #
    def stage2_inputs(top):   # B9 Stage 2: revisions + post-earnings drift, quality (fetched for the top 60 only)
        import numpy as _np
        earn = data.get_earnings(top, asof=asof)
        fund = data.get_fundamentals(top, sectors)
        drift = analytics.pead(earn, close)
        out = {}
        for t in top:
            e, f_ = earn.get(t) or {}, fund.get(t) or {}
            revs = [x for x in (e.get("rev_30d_0q"), e.get("rev_30d_1y")) if x is not None]
            q = [x for x in (f_.get("roe"), f_.get("net_margin"),
                             -(f_["debt_to_equity"] / 100) if f_.get("debt_to_equity") is not None else None) if x is not None]
            out[t] = {"revisions": float(_np.clip(_np.mean(revs) * 20, -1, 1)) if revs else None,   # 5% revision = 1
                      "drift": (drift.get(t) or {}).get("drift_score"),
                      "quality": float(_np.mean(q)) if q else None}
        return out

    scr = screener.screen(universe, market, holdings, regime, stage2_inputs=stage2_inputs)
    emit("screen", f"Funnel: {scr['funnel']}")
    bad = _preflight(close, scr["shortlist"])
    untradeable = {t: r for t, r in bad.items() if t in holdings}
    shortlist = [t for t in scr["shortlist"] if t not in bad or t in holdings]
    shortlist = [t for t in shortlist if t in close.columns and close[t].dropna().shape[0] > 63]
    last_px = _last_prices(close, set(shortlist) | set(holdings) | {config.BENCHMARK})
    # C8.5: a held name Yahoo didn't return today is valued at its last known close and can't trade today
    # (Oct 20: ABNB missing from a 382/513 download crashed runs 4-7 and would have dropped out of NAV)
    missing_held = [t for t in holdings if t not in last_px]
    if missing_held:
        fb = data.last_known_prices(missing_held)
        for t in missing_held:
            last_px[t] = fb.get(t, positions[t]["avg_cost"])
            untradeable[t] = "no valid live price today; valued at last known close (C8.5)"
    current_w, nav = db.current_weights(last_px)
    cash = db.get_cash()
    snaps = db.get_snapshots()
    dd_state = analytics.drawdown_state(snaps[snaps["date"] < db.today()]["total_equity"].tolist() if len(snaps) else [], nav)
    drawdown = dd_state["drawdown"]
    caps = analytics.etf_caps(close, set(shortlist) | set(holdings), asset_types)
    adv = analytics.adv20(market, set(shortlist) | set(holdings))
    prev_eq = db.last_snapshot_equity()
    nav_jump = (f"NAV moved {nav / prev_eq - 1:+.0%} vs last snapshot (> {config.NAV_JUMP_HALT:.0%}): "
                f"likely data error, manual review needed (IPS C8.6)"
                if prev_eq and abs(nav / prev_eq - 1) > config.NAV_JUMP_HALT else None)

    stocks = [t for t in shortlist if asset_types.get(t) == "stock"]
    emit("data", f"Fundamentals for {len(stocks)} stocks, news for {len(shortlist)} names")
    fundamentals = data.get_fundamentals(stocks, sectors)
    flags = data.research_flags(fundamentals)
    names = dict(zip(universe["symbol"], universe["name"]))
    ciks = dict(zip(universe["symbol"], universe["cik"]))
    # D7: no historical news exists, so in replay Sentiment is n/a and its 30% weight is redistributed (C5.2)
    news = {t: [] for t in shortlist} if replay else finbert.score_news(data.get_news(shortlist, names, ciks))
    earn_names = sorted({t for t in shortlist + holdings if asset_types.get(t) == "stock"})
    earnings = data.get_earnings(earn_names, asof=asof)
    pead = analytics.pead(earnings, close)
    if replay:
        brief = {"text": "", "model": "replay: no Morning Brief (no historical headlines)", "tickers_in_play": []}
    else:
        emit("brief", "Writing the Morning Brief")
        brief = morning_brief.build({t: news.get(t, []) for t in holdings}, earnings, holdings, llm.provider)
    tech = analytics.technicals(close, shortlist)

    emit("model", f"Training the LSTM on {len(shortlist)} shortlisted names")
    fc_cols = list(dict.fromkeys(shortlist + [config.BENCHMARK]))
    px_fc = close[fc_cols].ffill().dropna()
    day = px_fc.index[-1].date()
    # C3.5 monthly walk-forward LSTM: ~1 min on the M2, so cached per data date (3 runs/day reuse it)
    wf = (config.REPLAY["lstm_wf_days"], config.REPLAY["lstm_retrain_every"]) if replay else (forecaster.WF_DAYS, forecaster.RETRAIN_EVERY)
    fc = data._cached(data._key(f"lstm_wf_{day}_{wf[0]}", shortlist),
                      lambda: forecaster.forecast(px_fc, shortlist, market.get("volume"), wf_days=wf[0], retrain_every=wf[1]))

    race, race_sec = None, 0.0
    try:   # Kronos + horse race: optional, any failure = LSTM-only (never crashes the run)
        import time as _t
        t0 = _t.time()
        mk = {f: v[[c for c in fc_cols if c in v.columns]] for f, v in market.items()}
        kr = {} if replay else data._cached(data._key(f"kronos_{day}", shortlist),
                                            lambda: kronos_fc.forecast(mk, shortlist) or {}, keep_if=bool)
        kr_live = (kr or {}).get("forecasts") or None
        industries = dict(zip(universe["symbol"], universe["industry"]))
        industries = {t: i for t, i in industries.items() if isinstance(i, str) and asset_types.get(t) == "stock"}
        lg = data._cached(data._key(f"lgbm_{day}", shortlist),
                          lambda: lgbm.forecast(close, [t for t in shortlist if t in close.columns], industries, earnings) or {},
                          keep_if=lambda v: bool(v) and "error" not in v)
        if lg and "error" in lg:
            emit("model", f"LightGBM unavailable ({lg['error'][:80]})")
            lg = None
        race = data._cached(data._key(f"race_{day}_{bool(kr_live)}_{bool(lg)}", shortlist),
                            lambda: horse_race.run(px_fc, mk, shortlist, fc, lg, kr_live))
        race_sec = _t.time() - t0
    except Exception as e:  # noqa: BLE001
        emit("model", f"Horse race skipped ({str(e)[:100]}); LSTM-only")
        race = None

    run_id = db.start_run(llm.provider, data.DATA_SOURCE["prices"], mode)
    db.log_message(run_id, "Morning Brief", "Morning Brief (pre-open, C0.3)", brief["model"], "brief", 0, brief)
    db.log_message(run_id, "System", "Screening funnel (IPS Part B)", "rules", "tools", 0,
                   {"universe_source": data.DATA_SOURCE["universe"], "funnel": scr["funnel"],
                    "shortlist": shortlist, "preflight_excluded": bad,
                    "top_10_screen": scr["table"].head(10)[["sector", "screen_score"]].to_dict("index")})
    db.log_message(run_id, "System", "Regime rules (IPS C1.3)", "rules", "tools", 0, regime_info)
    if race:
        db.log_message(run_id, "System", "Model horse race (LSTM vs Kronos vs naive)", "rules", "tools", 0,
                       {k: v for k, v in race.items() if k not in ("live", "oos_rows")} | {"seconds": round(race_sec, 1)})
    db.log_message(run_id, "System", "Quant desk — LSTM model card", fc["model"], "tools", 0,
                   {k: v for k, v in fc.items() if k not in ("forecasts", "oos_preds")})

    ctx = {"morning_brief": brief["text"], "earnings": earnings, "pead": pead, "names": names,
           "earnings_soon": {t: {"date": e.get("next_date"), "time_et": e.get("next_time_et")}
                             for t, e in earnings.items() if e.get("reports_soon") and t in shortlist + holdings},
           "tickers": shortlist, "stocks": stocks, "sectors": sectors, "asset_types": asset_types,
           "fundamentals": fundamentals, "research_flags": flags, "news": news, "macro": macro,
           "regime_info": regime_info, "regime_rules": rules, "rule_tilts": analytics.sector_tilts(macro),
           "technicals": tech, "forecast": fc, "race": race, "current_weights": current_w, "mode": mode,
           "limits": config.RISK_LIMITS}
    failed: list[str] = []

    def call(agent_cls, stage, rnd=0, **extra):
        agent = agent_cls()
        emit(stage, f"{agent.name} is working…")
        try:
            out, model = agent.run({**ctx, **extra}, llm)
        except Exception as e:  # noqa: BLE001  IPS D11: a failed agent = neutral, never a crash
            failed.append(agent.name)
            out, model = agent.validate({}, {**ctx, **extra}), f"FAILED ({str(e)[:120]})"
            out["agent_failed"] = True
        db.log_message(run_id, agent.name, agent.title, model, stage, rnd, out)
        return out

    try:
        # -------------------------------------------- 2b. reflection memory -- #
        # grade past calls whose 5-day horizon has elapsed, using only prices <= today's as-of date
        graded = reflection.evaluate(close, asof=close.index[-1])
        if graded:
            ctx["reflect_due"] = graded
            lessons = call(ReflectionAnalyst, "reflection")
            reflection.save(graded, lessons.get("lessons", {}))
            emit("reflection", f"{len(graded)} past call(s) graded and lessons written")

        # ------------------------------------------------------ 3. analysts -- #
        ctx["macro_view"] = call(MacroStrategist, "analysis")
        ctx["research"] = call(EquityResearchAnalyst, "analysis", tickers=stocks)
        ctx["quant"] = call(QuantAnalyst, "analysis")
        ctx["sentiment"] = call(SentimentAnalyst, "analysis")

        # IPS D12: 2+ of the 4 signal agents failed or returned no signal -> no new trades
        def no_signal(name, scores):
            return name in failed or not any(abs(v.get("score", 0) or 0) > 1e-9 for v in scores.values())
        silent = [n for n, sc in (("Equity Research Analyst", ctx["research"]["ratings"]),
                                  ("Quant Analyst", ctx["quant"]["signals"]),
                                  ("Sentiment Analyst", ctx["sentiment"]["sentiment"])) if no_signal(n, sc)]
        if "Macro Strategist" in failed:
            silent.append("Macro Strategist")
        no_new_trades = len(silent) >= 2

        # --------------------------------------------------------- 4. blend -- #
        comp = analytics.composite_scores(shortlist, sectors, asset_types, ctx["research"]["ratings"],
                                          ctx["quant"]["signals"], ctx["sentiment"]["sentiment"],
                                          ctx["macro_view"]["sector_tilts"], regime)
        for t, v in pead.items():
            if t in comp:
                comp[t]["pead"] = v.get("drift_score")
        if ctx["quant"].get("quant_fallback"):
            for t in comp:
                comp[t]["quant_fallback"] = ctx["quant"]["quant_fallback"]
        ctx["composite"] = comp
        ranked = sorted(comp, key=lambda t: -comp[t]["composite"])
        rank = {t: i + 1 for i, t in enumerate(ranked)}

        forced = {}
        for t, p in positions.items():
            px, kind = last_px.get(t), asset_types.get(t, "stock")
            if px and p["avg_cost"] > 0 and px / p["avg_cost"] - 1 <= config.STOP_LOSS:
                forced[t] = f"stop-loss: {px / p['avg_cost'] - 1:.1%} from cost (IPS A17)"
            elif ((ctx["sentiment"]["sentiment"].get(t) or {}).get("score") or 0) <= config.NEWS_SELL_TRIGGER:
                forced[t] = f"news veto: sentiment {ctx['sentiment']['sentiment'][t]['score']} (IPS C4.4)"
            elif kind == "leveraged_etf" and regime == "risk_off":
                forced[t] = "leveraged long in risk-off (IPS C5.4)"
            elif kind == "leveraged_etf" and dd_state["level"] == "hard":
                forced[t] = f"fund drawdown {drawdown:.1%} at/below {config.DD_HARD:.0%}: all leveraged longs out (IPS A4/C11)"
            elif kind == "inverse_etf" and regime != "risk_off":
                forced[t] = "inverse hedge no longer needed: regime left risk-off (IPS A11)"
        forced.update({t: r for t, r in (extra_forced or {}).items() if t in positions})   # intraday news veto (C4.5)
        forced = {t: r for t, r in forced.items() if t not in untradeable}

        held_scores = [comp[t]["composite_10"] for t in holdings if t in comp]
        weakest = min(held_scores) if held_scores else 0
        unheld = [t for t in ranked if t not in positions]
        best_unheld = comp[unheld[0]]["composite_10"] if unheld else 0
        eligible_new = [] if no_new_trades else [
            t for t in unheld
            if comp[t]["composite_10"] >= 7 and (comp[t]["quant"] or 0) >= 0.2 and rank[t] <= 10
            and comp[t]["composite_10"] >= weakest + 1 and not comp[t]["insufficient_data"]
            and not (asset_types.get(t) == "leveraged_etf" and dd_state["no_new_leveraged"])]
        # IPS A7: below the 12-name target after forced exits -> replacement buys take priority
        survivors = len([t for t in holdings if t not in forced])
        if not init_mode and survivors < config.TARGET_HOLDINGS[0] and (not no_new_trades or survivors < config.MIN_HOLDINGS):
            for t in unheld:
                if len(eligible_new) >= config.TARGET_HOLDINGS[0] - survivors + 2:
                    break
                if (t not in eligible_new and asset_types.get(t) == "stock" and not comp[t]["insufficient_data"]
                        and comp[t]["composite_10"] >= 5):
                    eligible_new.append(t)
        exit_cands = [t for t in holdings if t in comp and t not in forced and (
            (comp[t]["quant"] or 0) < -0.2 or rank[t] > 20 or comp[t]["composite_10"] <= best_unheld - 2)]
        # C5.3 earnings fast track: reporting within 5 trading days, Sentiment >= +0.7 and Quant >= 5/10, max 10% NAV
        F = config.FAST_TRACK
        fast = [] if no_new_trades else [
            t for t in unheld if asset_types.get(t) == "stock" and (earnings.get(t) or {}).get("reports_soon")
            and (comp[t]["sentiment"] or -1) >= F["min_sentiment"] and analytics.to_10(comp[t]["quant"] or -1) >= F["min_quant_10"]
            and not comp[t]["insufficient_data"]]
        eligible_new += [t for t in fast if t not in eligible_new]
        caps.update({t: F["max_weight"] for t in fast})
        ctx["fast_track"] = fast
        ctx.update(eligible_new=eligible_new, exit_candidates=exit_cands, forced_exits=forced)
        relevant = set(holdings) | set(eligible_new) | set(exit_cands) | set(ranked[:10])
        ctx["lessons"] = reflection.lessons_for(relevant, {sectors.get(t) for t in relevant}, regime)
        ctx["track_record"] = reflection.track_record()
        ctx["invalidated"] = [i for i in reflection.invalidated(positions, last_px) if i["ticker"] not in forced]
        ctx["invalidated"] += [r for r in _lev_reviews(positions, close, last_px, asset_types)
                               if r["ticker"] not in forced and r["ticker"] not in {i["ticker"] for i in ctx["invalidated"]}]
        ctx["dd_state"] = dd_state

        # ---------------------------------------------- 5. PM <-> Risk loop -- #
        lo_n, hi_n = config.TARGET_HOLDINGS
        max_names = config.TARGET_HOLDINGS_INIT if init_mode else min(hi_n, max(len(holdings), lo_n))
        dd_lev = dd_state["lev_cap"]

        def build(pm, lev_override=None, risk_aversion=4.0, target=None):
            lev_lim = min(x for x in (lev_override, dd_lev, 1.0) if x is not None)
            if target is None:
                target = analytics.optimise(pm["convictions"], close, sectors, asset_types, pm["equity_exposure"],
                                            regime, macro["vix_level"], max_names, risk_aversion=risk_aversion,
                                            lev_cap_override=lev_lim,
                                            min_weight=max(0.01, config.MIN_TRADE_NOTIONAL / nav * 1.05),
                                            ensure_min=init_mode, caps=caps,
                                            inverse_min=dd_state["inverse_min_risk_off"])
            if not init_mode:   # held names the PM didn't exit stay at their weight unless resized
                for t, w in current_w.items():
                    if (t not in target and pm["convictions"].get(t, 0) > -0.5 and t not in forced
                            and asset_types.get(t, "stock") == "stock"):   # ETFs stay inside the optimiser's caps
                        target[t] = round(w, 4)
                tot = sum(target.values())
                if tot > 1:
                    target = {t: math.floor(w / tot * 1e4) / 1e4 for t, w in target.items()}
            lev_eff = min(config.RISK_LIMITS["max_leveraged_total"], rules["max_leveraged"], lev_lim)
            orders, skipped = analytics.plan_orders(target, positions, last_px, nav, cash, forced,
                                                    init_mode, comp, untradeable, sectors=sectors,
                                                    asset_types=asset_types, caps=caps, lev_cap=lev_eff,
                                                    no_new_leveraged=dd_state["no_new_leveraged"])
            post = _post_trade_weights(positions, cash, orders, last_px)
            # Prof. Manero rule 3: the Risk Manager checks the book that will ACTUALLY exist after the
            # 5-trade cap, the $25k minimum and the cash upsizing. A breach here = re-plan, never execute.
            rr = RiskManager.check_book(post, close, sectors, asset_types, regime, nav_drawdown=drawdown, adv=adv,
                                        nav=nav, caps=caps, dd_state=dd_state, current_weights=current_w)
            return target, orders, skipped, rr, post

        risk_feedback, prev = None, None
        for rnd in range(1, config.MAX_RISK_ROUNDS + 1):
            pm = call(PortfolioManager, "proposal", rnd, risk_feedback=risk_feedback, prev_proposal=prev)
            target, orders, skipped, rr, post = build(pm)
            if rnd == 1:   # C0.1 Devil's Advocate on the PM's top 3 ideas, then the PM answers each objection
                ideas = sorted([o for o in orders if o["side"] == "BUY"], key=lambda o: -o["notional"])[:3]
                if ideas:
                    da = call(DevilsAdvocate, "devils_advocate", 1, ideas=ideas, pm=pm)
                    ctx["da_objections"] = da["objections"]
                    ctx["prev_pm_rationale"] = pm["rationale"]
                    pm = call(PortfolioManager, "proposal_answers_da", 1, risk_feedback=None, prev_proposal=None)
                    target, orders, skipped, rr, post = build(pm)
            proposal = {**pm, "weights": target}
            risk = call(RiskManager, "risk_review", rnd, proposal={**proposal, "weights": post},
                        risk_report=rr, round=rnd)
            if risk["decision"] == "approve":
                break
            risk_feedback = {"issues": risk["issues"], "required_changes": risk["required_changes"]}
            prev = {"convictions": pm["convictions"], "target_weights": {t: w for t, w in target.items() if w > 0}}
        else:
            # Out of rounds (IPS C6.5): mechanical de-risk. First shrink the offending positions pro rata and
            # hand the weight to compliant names; then, if needed, cut leverage room / raise risk aversion.
            steps, lev, ra = [], rules["max_leveraged"], 4.0
            for i in range(8):
                if not rr["breaches"]:
                    break
                if i < 3:
                    target = analytics.shrink_to_limits(target, rr, sectors, asset_types, caps,
                                                        lev_cap=min(rules["max_leveraged"], dd_lev if dd_lev is not None else 1.0))
                    steps.append(f"pro-rata shrink of offending positions ({'; '.join(rr['breaches'][:2])})")
                    target, orders, skipped, rr, post = build(pm, target=target)
                else:
                    lev, ra = max(0.0, lev - 0.05), ra * 1.5
                    steps.append(f"leveraged room {lev:.0%}, risk aversion {ra:.1f}")
                    target, orders, skipped, rr, post = build(pm, lev_override=lev, risk_aversion=ra)
            proposal = {**pm, "weights": target}
            ok = not rr["breaches"]
            risk = {"decision": "approve" if ok and not rr["soft_rejects"] else ("approve" if ok else "reject"),
                    "issues": risk["issues"], "hard_breaches": rr["breaches"], "soft_rejects": rr["soft_rejects"],
                    "warnings": rr["warnings"], "limit_checks": rr["limit_checks"],
                    "required_changes": [] if ok else ["hard limits still breached after mechanical de-risk: keep current book"],
                    "override": "mechanical de-risk after 3 rejected rounds (IPS C6.5)",
                    "rationale": (f"{config.MAX_RISK_ROUNDS} rounds exhausted. Mechanical de-risk: {' -> '.join(steps) or 'none needed'}. "
                                  f"Hard limits {'now met' if ok else 'STILL breached'}"
                                  f"{'; remaining soft concerns escalated to the CIO' if ok and rr['soft_rejects'] else ''}.")}
            db.log_message(run_id, "Risk Manager", "Chief Risk Officer — mechanical de-risk (IPS C6.5)", "rules",
                           "risk_override", rnd, risk)

        turnover = sum(o["notional"] for o in orders) / nav / 2

        # ------------------------------------------------------------ 6. CIO -- #
        cio = call(ChiefInvestmentOfficer, "decision", proposal=proposal, risk_report=rr,
                   risk_decision=risk, round=rnd, turnover=round(turnover, 3), orders=orders,
                   no_new_trades=no_new_trades)
        if cio["decision"] != "approve":
            # keep the portfolio, but forced risk exits (stop-loss, news veto, ...) still go through
            orders = [o for o in orders if o["forced"]]
            if no_new_trades:
                cio["committee_summary"] = (f"Insufficient information (IPS D12): {', '.join(silent)} gave no signal. "
                                            f"No new trades; forced exits only. " + cio["committee_summary"])

        # ---------------------------------------------------------- 7. Trader -- #
        post_final = _post_trade_weights(positions, cash, orders, last_px)
        risk_line = ("Risk Manager: approved, all hard limits pass" if risk["decision"] == "approve" else "Risk Manager: rejected") + \
            (" (after mechanical de-risk)." if "mechanical" in str(risk.get("override") or "") else ".")
        for o in orders:
            t = o["ticker"]
            o["full_exit"] = o["side"] == "SELL" and o["shares"] >= positions.get(t, {}).get("shares", 0)
            o["why"] = _why(o, ctx, pm)                                     # per-trade, never the batch note (C5.5)
            o["reason"] = o["why"]
            o.update(reflection.trade_notes(o, (pm.get("trade_notes") or {}).get(t), comp.get(t)), sector=sectors.get(t))
            resp = (pm.get("invalidation_responses") or {}).get(t)
            if resp and o["side"] == "SELL" and t in {i["ticker"] for i in ctx.get("invalidated", [])}:
                o["thesis"] = f"Invalidation hit: {resp}"[:300]
            o["name"] = names.get(t, t)
            o["signals_text"] = _signals_text(comp.get(t))
            o["facts"] = {"ticker": t, "name": names.get(t, t), "sector": sectors.get(t), "side": o["side"],
                          "shares": o["shares"], "notional": o["notional"], "weight_before": o["current_weight"],
                          "weight_after": round(post_final.get(t, 0.0), 4), "why": o["why"],
                          "composite_10": (comp.get(t) or {}).get("composite_10"),
                          "news": (ctx["sentiment"]["sentiment"].get(t) or {}).get("key_driver"),
                          "quant_comment": (ctx["quant"]["signals"].get(t) or {}).get("comment"),
                          "research": (ctx["research"]["ratings"].get(t) or {}).get("thesis"),
                          "invalidation_px": o.get("invalidation_px") if o["side"] == "BUY" else None,
                          "forced_reason": o.get("forced_reason"), "replacement": o.get("replacement", False)}
            from committee.rationale_writer import template as _tmpl
            o["reason"] = _tmpl(o["facts"]) + " " + risk_line
        trader, _ = Trader().run({"orders": orders, "trades_today": db.trades_today(), "cash": cash,
                                  "init_mode": init_mode, "nav_jump_halt": nav_jump})
        trader["skipped_by_planner"] = skipped
        db.log_message(run_id, "Trader", Trader.title, "rules", "execution", 0, trader)

        approved = trader["approved_orders"] if trader["execute"] else []
        status = "proposed" if approved else ("no_trade" if not nav_jump else "halted")
        summary = cio["committee_summary"]
        db.finish_run(run_id, status, summary, proposal={
            "mode": mode, "orders": approved, "last_px": last_px, "regime": regime,
            "asof": str(close.index[-1].date()),
            "benchmark_px": last_px.get(config.BENCHMARK)})
        db.record_snapshot(run_id, last_px, last_px[config.BENCHMARK])
        emit("done", f"Proposal ready — {len(approved)} order(s), status {status}")
        result = {"run_id": run_id, "status": status, "mode": mode, "orders": approved,
                  "rejected": trader["rejected"], "nav": nav, "regime": regime,
                  "usage": llm.usage, "data_source": data.DATA_SOURCE["prices"], "failed_agents": failed}
        if execute and approved:
            result.update(execute_run(run_id))
        return result
    except Exception as e:
        db.finish_run(run_id, "error", str(e)[:500])
        raise


def intraday_recheck(provider: str | None = None, progress=None, execute: bool = False) -> dict:
    """IPS C4.5, the third daily run slot: refresh headlines for HELD names only and re-score them. A held name
    at <= -0.8 triggers a sell proposal through the normal Risk -> CIO -> Trader path; otherwise the run ends here."""
    from committee import SentimentAnalyst as _SA
    emit = progress or (lambda stage, msg: print(f"[{stage}] {msg}"))
    llm = LLMClient(provider)
    universe = data.get_universe()
    db.use_synthetic(False)
    db.init_db()
    if llm.provider != "mock" and db.runs_today() >= config.MAX_RUNS_PER_DAY:
        raise RunLimitError(f"Already {db.runs_today()} runs today; the brief allows {config.MAX_RUNS_PER_DAY}.")
    held = list(db.get_positions())
    if not held:
        return {"status": "no_holdings", "orders": []}
    names = dict(zip(universe["symbol"], universe["name"]))
    ciks = dict(zip(universe["symbol"], universe["cik"]))
    emit("news", f"Refreshing headlines for {len(held)} held names")
    news = finbert.score_news(data.get_news(held, names, ciks, refresh=True))
    run_id = db.start_run(llm.provider, "intraday news re-check", "recheck")
    try:
        out, model = _SA().run({"tickers": held, "news": news}, llm)
    except Exception as e:  # noqa: BLE001  D11
        out, model = {"sentiment": {}, "market_mood": "", "agent_failed": True}, f"FAILED ({str(e)[:100]})"
    db.log_message(run_id, "Sentiment Analyst", "Intraday news re-check (C4.5)", model, "recheck", 0, out)
    hits = {t: f"intraday news veto: sentiment {v['score']} ({v.get('key_driver', '')}) (IPS C4.4/C4.5)"
            for t, v in out.get("sentiment", {}).items() if v.get("score") is not None and v["score"] <= config.NEWS_SELL_TRIGGER}
    if not hits:
        db.finish_run(run_id, "recheck_clear", f"Intraday re-check: no held name at <= {config.NEWS_SELL_TRIGGER}.")
        emit("done", "Re-check clear: no sell proposal")
        return {"run_id": run_id, "status": "recheck_clear", "orders": [], "usage": llm.usage}
    db.finish_run(run_id, "recheck_triggered", f"Severe news on {', '.join(hits)}: full committee convened for a sell proposal.")
    emit("news", f"Severe news on {', '.join(hits)}: convening the committee")
    return run_committee(provider, progress, execute=execute, extra_forced=hits, _skip_run_limit=True)


STALE_PROPOSAL_HOURS = 2


def execute_run(run_id: int, provider: str | None = None, prices: dict | None = None) -> dict:
    """Brief §8.2 Execution Agent: commit an approved proposal to SQLite.
    D13 / A19: trades are stamped at EXECUTION time (Madrid) and priced at the latest price then, not at the
    proposal's. A proposal older than 2 hours is refused (Oct 20: proposed 08:14, executed 13:34 at stale prices).
    After execution the Trade Rationale Writer turns each trade's facts into plain English (IPS C0.4/E2)."""
    import datetime as _dt
    prop = db.get_proposal(run_id)
    runs = db.get_runs().set_index("id")
    if prop is None or runs.loc[run_id, "status"] != "proposed":
        raise RuntimeError(f"Run {run_id} has no pending proposal (status: {runs.loc[run_id, 'status']}).")
    proposed_at = runs.loc[run_id, "finished_at"] or runs.loc[run_id, "started_at"]
    if prices is None:
        age_h = (db.now() - _dt.datetime.fromisoformat(proposed_at)).total_seconds() / 3600
        if age_h > STALE_PROPOSAL_HOURS:
            db.set_run_status(run_id, "expired")
            raise RuntimeError(f"Proposal #{run_id} is {age_h:.1f} h old (proposed {db.madrid_et(proposed_at)}). "
                               f"Prices have moved: run the committee again.")
        try:   # fresh prices at execution time (cache <= 60 min)
            close = data.get_market_data(data.get_universe()["symbol"].tolist())["close"]
            prices = _last_prices(close, [o["ticker"] for o in prop["orders"]])
        except Exception:  # noqa: BLE001
            prices = {}
    init_mode = prop["mode"] == "init"
    room = len(prop["orders"]) if init_mode else max(0, config.MAX_TRADES_PER_DAY - db.trades_today())
    done, facts = [], []
    for o in prop["orders"][:room]:   # sells are first in the list, so they fund the buys
        px = float(prices.get(o["ticker"], o["price"]))
        try:
            meta = {"thesis": o.get("thesis", ""), "invalidation": o.get("invalidation", ""),
                    "invalidation_px": o.get("invalidation_px"), "entry_date": db.today(),
                    "regime": prop.get("regime"), "sector": o.get("sector"), "signals": o.get("signals", {}),
                    "horizon_days": 5}
            tr = db.execute_trade(run_id, o["ticker"], o["side"], o["shares"], px, o["reason"], meta,
                                  name=o.get("name"), signals=o.get("signals_text"), proposed_at=proposed_at)
            done.append(tr)
            facts.append({**(o.get("facts") or {"ticker": o["ticker"], "side": o["side"], "weight_before": 0.0,
                                                 "weight_after": 0.0}), "price": round(px, 2), "trade_id": tr["id"]})
        except ValueError as e:
            db.log_message(run_id, "Trader", Trader.title, "rules", "execution", 0,
                           {"execute": False, "execution_notes": f"{o['ticker']} rejected at execution: {e}"})
    if facts:
        _write_rationales(run_id, facts, prop, provider or runs.loc[run_id, "provider"])
    db.set_run_status(run_id, "executed" if done else "no_trade")
    nav = db.record_snapshot(run_id, prop["last_px"], prop["benchmark_px"])
    csv_path = db.export_trades_csv()
    rep = report.daily_report()
    return {"status": "executed" if done else "no_trade", "trades": done, "nav": nav, "trades_csv": csv_path,
            "daily_report": rep}


def _write_rationales(run_id, facts, prop, provider):
    """One call after execution; falls back to the deterministic template on any failure (D11)."""
    from committee.rationale_writer import TradeRationaleWriter, template
    risk_lines = {o["ticker"]: o["reason"].split("Risk Manager:", 1)[1] for o in prop["orders"] if "Risk Manager:" in o["reason"]}
    ctx = {"facts": facts, "all_tickers": list(data.names_map())}
    w = TradeRationaleWriter()
    try:
        out, model = w.run(ctx, LLMClient(provider))
    except Exception as e:  # noqa: BLE001
        out, model = {"rationales": {f["ticker"]: template(f) for f in facts}, "fallback_used_for": [f["ticker"] for f in facts]}, \
            f"FAILED ({str(e)[:100]}): template"
    db.log_message(run_id, w.name, w.title, model, "rationale", 0, out)
    for f in facts:
        txt = out["rationales"].get(f["ticker"]) or template(f)
        rl = risk_lines.get(f["ticker"])
        db.set_trade_reason(f["trade_id"], txt + (f" Risk Manager:{rl}" if rl else ""))


def _why(o, ctx, pm) -> str:
    """The one-line reason for THIS trade (C5.5): forced reason, the PM's per-trade thesis, else the strongest
    specific signal for this name. Never the PM's batch rationale."""
    t = o["ticker"]
    if o.get("forced"):
        return o["forced_reason"]
    note = ((pm.get("trade_notes") or {}).get(t) or {}).get("thesis")
    if note:
        return note
    sent = ctx["sentiment"]["sentiment"].get(t) or {}
    q = ctx["quant"]["signals"].get(t) or {}
    if o.get("replacement"):
        return "it is the best-ranked unheld name to replace an exit and keep at least 10 holdings"
    if o["side"] == "SELL":
        return f"its scores fell behind the best alternatives ({q.get('comment') or 'weaker momentum'})"
    if sent.get("score") is not None and abs(sent["score"]) >= 0.3 and sent.get("key_driver"):
        return f"news: {sent['key_driver']}"
    return q.get("comment") or "it ranks among the strongest names on our combined scores"


def _signals_text(c: dict | None) -> str:
    if not c:
        return ""
    f = lambda v: "n/a" if v is None else f"{v:+.2f}"
    txt = (f"composite {c.get('composite_10')}/10 · Q {f(c.get('quant'))} · S {f(c.get('sentiment'))} · "
           f"R {f(c.get('research'))} · M {f(c.get('macro'))}")
    if c.get("na"):
        txt += f" · no signal from {', '.join(c['na'])} (weight redistributed, C5.2)"
    if c.get("pead"):
        txt += f" · PEAD {c['pead']:+.2f}"
    if c.get("quant_fallback"):
        txt += f" · Q from {c['quant_fallback']}"
    return txt


def replay(days: int = 20, provider: str | None = "mock", progress=None) -> dict:
    """IPS D6-D9: run the full committee for the last `days` trading days into finance_replay.db (never finance.db).
    Day d decides on prices up to the previous close and fills at d's OPEN, stamped 15:30 Madrid (D8). Reflection
    grading runs inside the replay. Returns a summary (return vs SPY, max DD, trades, committee hit rate)."""
    import datetime as _dt
    emit = progress or (lambda stage, msg: print(f"[{stage}] {msg}"))
    R = config.REPLAY
    saved = (config.DB_PATH, config.TRADES_CSV, report.REPORT_DIR, db.now)
    config.DB_PATH, config.TRADES_CSV = R["db"], R["trades_csv"]
    report.REPORT_DIR = R["report_dir"]
    db._DB["path"] = R["db"]
    try:
        db.init_db(reset=True)
        full = data.get_market_data(data.get_universe()["symbol"].tolist())
        dates = full["close"].index[-days:]
        hh, mm = map(int, R["fill_time_madrid"].split(":"))
        for i, d in enumerate(dates):
            stamp = _dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=db.MADRID)
            db.now = lambda stamp=stamp: stamp
            emit("replay", f"Day {i + 1}/{len(dates)}: {d.date()} (data to {dates[i - 1].date() if i else 'prev close'})")
            res = run_committee(provider, progress=lambda *a: None, asof=d)
            if res["status"] == "proposed":
                opens = {t: float(full["open"][t].loc[d]) for t in {o["ticker"] for o in res["orders"]}
                         if t in full["open"].columns and pd.notna(full["open"][t].loc[d])}
                execute_run(res["run_id"], provider=provider, prices=opens)
            px_close = {t: float(full["close"][t].loc[d]) for t in full["close"].columns if pd.notna(full["close"][t].loc[d])}
            db.record_snapshot(res["run_id"], px_close, px_close[config.BENCHMARK])
            report.daily_report()
        snaps = db.get_snapshots()
        nav = snaps["total_equity"]
        spy = snaps["benchmark_px"]
        import reflection as _r
        summary = {"days": len(dates), "from": str(dates[0].date()), "to": str(dates[-1].date()),
                   "return": round(float(nav.iloc[-1] / config.STARTING_CAPITAL - 1), 4),
                   "spy_return": round(float(spy.iloc[-1] / full["close"][config.BENCHMARK].loc[:dates[0]].iloc[-2] - 1), 4),
                   "max_drawdown": round(float((nav / nav.cummax().clip(lower=config.STARTING_CAPITAL) - 1).min()), 4),
                   "trades": int(len(db.get_trades())), "track_record": _r.track_record(), "provider": provider,
                   "limitations": "prices only; fundamentals = today's snapshot (frozen); no historical news (Sentiment n/a); "
                                  "no Kronos; today's S&P 500 constituents (survivorship)"}
        summary["excess"] = round(summary["return"] - summary["spy_return"], 4)
        os.makedirs(R["report_dir"], exist_ok=True)
        with open(os.path.join(R["report_dir"], "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        return summary
    finally:
        config.DB_PATH, config.TRADES_CSV, report.REPORT_DIR, db.now = saved
        db._DB["path"] = config.DB_PATH


def snapshot_only() -> float:
    """After-close run (IPS D3, 21:45 Madrid) / Brief 'Advance Day': revalue and log, no agents."""
    universe = data.get_universe()
    pos = db.get_positions()
    tickers = list(pos) + [config.BENCHMARK]
    close = data.get_market_data(universe["symbol"].tolist())["close"]
    db.use_synthetic(data.is_synthetic())
    db.init_db()
    px = _last_prices(close, tickers)
    nav = db.record_snapshot(None, px, px[config.BENCHMARK])
    report.daily_report()
    return nav


def _lev_reviews(positions, close, last_px, asset_types) -> list[dict]:
    """IPS C6.4 (Oct 20): a leveraged/inverse ETF needs re-approval when its return since entry < 0, it is
    > 12% below its peak since entry, or it lags leverage x its underlying by more than the decay tolerance."""
    out = []
    for t, p in positions.items():
        if asset_types.get(t) not in ("leveraged_etf", "inverse_etf") or t not in last_px or t not in close.columns:
            continue
        px, entry = last_px[t], p["avg_cost"]
        since = close[t].loc[close.index >= str(p.get("opened_date") or close.index[-1].date())].dropna()
        reasons = []
        ret = px / entry - 1
        if ret < 0:
            reasons.append(f"return since entry {ret:+.1%} < 0")
        if len(since) and px / max(float(since.max()), px) - 1 < config.LEV_REVIEW_PEAK_DD:
            reasons.append(f"{px / float(since.max()) - 1:+.1%} from its peak since entry (> 12% drawdown)")
        u = config.ETF_UNDERLYING.get(t)
        if u in close.columns and len(since) > 1:
            useries = close[u].loc[since.index].dropna()
            if len(useries) > 1:
                expected = config.LEVERAGE_FACTOR.get(t, 1.0) * (float(useries.iloc[-1]) / float(useries.iloc[0]) - 1)
                actual = px / float(since.iloc[0]) - 1
                if actual < expected - config.LEV_REVIEW_DECAY_TOL:
                    reasons.append(f"decay: {actual:+.1%} vs {expected:+.1%} expected from {u} x {config.LEVERAGE_FACTOR.get(t)}")
        if reasons:
            out.append({"ticker": t, "thesis": "leveraged ETF re-approval (IPS C6.4)", "invalidation": "; ".join(reasons),
                        "entry_px": round(entry, 2), "invalidation_px": None, "last_px": round(px, 2),
                        "since_entry": round(ret, 4), "type": "leveraged_review"})
    return out


def _post_trade_weights(positions, cash, orders, px) -> dict[str, float]:
    sh = {t: p["shares"] for t, p in positions.items()}
    for o in orders:
        sh[o["ticker"]] = sh.get(o["ticker"], 0) + (o["shares"] if o["side"] == "BUY" else -o["shares"])
        cash += -o["notional"] if o["side"] == "BUY" else o["notional"]
    vals = {t: s * px[t] for t, s in sh.items() if s > 0 and t in px}
    nav = cash + sum(vals.values())
    return {t: v / nav for t, v in vals.items()}


def _reason(o, comp, pm, risk) -> str:
    """One-line reasoning stored with every trade (Brief §1, §8.1 'reason' column)."""
    c = comp.get(o["ticker"], {})
    na = c.get("na") or []
    scores = (f"composite {c.get('composite_10', '-')}/10 (Q {c.get('quant') if c.get('quant') is not None else 'n/a'}, "
              f"S {c.get('sentiment') if c.get('sentiment') is not None else 'n/a'}, "
              f"R {c.get('research') if c.get('research') is not None else 'n/a'})"
              + (f"; no signal from {', '.join(na)}, weight redistributed pro rata (C5.2)" if na else "")) \
        if c else "not in today's shortlist"
    move = f"weight {o['current_weight']:.1%} -> {o['target_weight']:.1%}"
    if o["forced"]:
        return f"FORCED EXIT — {o['forced_reason']}; {scores}"
    return f"{scores}; {move}. PM: {pm['rationale'][:160]} | Risk: {risk['decision']}"


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Run one investment committee meeting")
    ap.add_argument("--provider", help="groq | ollama | gemini | anthropic | mock")
    ap.add_argument("--execute", action="store_true", help="commit the approved trades immediately")
    ap.add_argument("--reset", action="store_true", help="wipe the fund back to $1M cash")
    ap.add_argument("--snapshot", action="store_true", help="only revalue the portfolio (after-close run)")
    ap.add_argument("--recheck", action="store_true", help="intraday news re-check on held names (C4.5)")
    ap.add_argument("--replay", type=int, metavar="DAYS", help="run the committee over the last N trading days into finance_replay.db")
    args = ap.parse_args()
    if args.reset:
        data.get_market_data(data.get_universe()["symbol"].tolist())
        db.use_synthetic(data.is_synthetic())
        db.init_db(reset=True)
        print(f"Fund reset to ${config.STARTING_CAPITAL:,.0f} cash in {db.current_db_name()}. No committee run.")
        raise SystemExit
    if args.snapshot:
        print(f"NAV ${snapshot_only():,.0f}")
        raise SystemExit
    if args.replay:
        print(json.dumps(replay(args.replay, args.provider or "mock"), indent=1, default=str))
        raise SystemExit
    res = intraday_recheck(args.provider, execute=args.execute) if args.recheck else \
        run_committee(args.provider, execute=args.execute)
    print(json.dumps({k: v for k, v in res.items() if k != "orders"}, indent=2, default=str))
    print("\nOrders:")
    for o in res["orders"]:
        print(f"  {o['side']:4} {o['shares']:>6} {o['ticker']:<6} @ {o['price']:>9.2f}  ${o['notional']:>11,.0f}  {o['reason'][:90]}")
