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

import math

import analytics
import config
import data
import database as db
import forecaster
import reflection
import report
from forecasters import horse_race, kronos_fc
import screener
from committee import (ChiefInvestmentOfficer, EquityResearchAnalyst, MacroStrategist,
                       PortfolioManager, QuantAnalyst, ReflectionAnalyst, RiskManager, SentimentAnalyst,
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


def run_committee(provider: str | None = None, progress=None, execute: bool = False) -> dict:
    emit = progress or (lambda stage, msg: print(f"[{stage}] {msg}"))
    llm = LLMClient(provider)

    # ------------------------------------------------------- 1. data + guards -- #
    emit("data", "Loading S&P 500 universe + leveraged ETFs and downloading prices")
    universe = data.get_universe()
    market = data.get_market_data(universe["symbol"].tolist())
    synthetic = data.is_synthetic()
    db.use_synthetic(synthetic)
    db.init_db()
    db.mark_stale_runs()
    if not synthetic and llm.provider != "mock" and db.runs_today() >= config.MAX_RUNS_PER_DAY:
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
    macro = data.get_macro()
    reg = analytics.macro_regime(macro)
    prev_eff, prev_raw = db.previous_regime()
    regime, regime_note = analytics.apply_hysteresis(reg["raw_regime"], prev_eff, prev_raw)
    db.save_regime(reg["raw_regime"], regime, regime_note)
    regime_info = {**reg, "regime": regime, "note": regime_note, "previous_regime": prev_eff}
    rules = config.REGIME_RULES[regime]

    # ------------------------------------------------------- screening ------- #
    scr = screener.screen(universe, market, holdings, regime)
    emit("screen", f"Funnel: {scr['funnel']}")
    bad = _preflight(close, scr["shortlist"])
    untradeable = {t: r for t, r in bad.items() if t in holdings}
    shortlist = [t for t in scr["shortlist"] if t not in bad or t in holdings]
    shortlist = [t for t in shortlist if t in close.columns and close[t].dropna().shape[0] > 63]
    last_px = _last_prices(close, set(shortlist) | set(holdings) | {config.BENCHMARK})
    current_w, nav = db.current_weights(last_px)
    cash = db.get_cash()
    drawdown = db.drawdown_from_peak(nav)
    prev_eq = db.last_snapshot_equity()
    nav_jump = (f"NAV moved {nav / prev_eq - 1:+.0%} vs last snapshot (> {config.NAV_JUMP_HALT:.0%}): "
                f"likely data error, manual review needed (IPS C8.6)"
                if prev_eq and abs(nav / prev_eq - 1) > config.NAV_JUMP_HALT else None)

    stocks = [t for t in shortlist if asset_types.get(t) == "stock"]
    emit("data", f"Fundamentals for {len(stocks)} stocks, news for {len(shortlist)} names")
    fundamentals = data.get_fundamentals(stocks, sectors)
    flags = data.research_flags(fundamentals)
    news = data.get_news(shortlist)
    tech = analytics.technicals(close, shortlist)

    emit("model", f"Training the LSTM on {len(shortlist)} shortlisted names")
    fc_cols = list(dict.fromkeys(shortlist + [config.BENCHMARK]))
    fc = forecaster.forecast(close[fc_cols].ffill().dropna(), shortlist)

    race, race_sec = None, 0.0
    try:   # Kronos + horse race: optional, any failure = LSTM-only (never crashes the run)
        import time as _t
        t0 = _t.time()
        px_fc = close[fc_cols].ffill().dropna()
        mk = {f: v[[c for c in fc_cols if c in v.columns]] for f, v in market.items()}
        kr = data._cached(data._key(f"kronos_{px_fc.index[-1].date()}", shortlist),
                          lambda: kronos_fc.forecast(mk, shortlist) or {}, keep_if=bool)
        kr_live = (kr or {}).get("forecasts") or None
        race = horse_race.run(px_fc, mk, shortlist, fc.get("oos_preds", []), fc["forecasts"], kr_live)
        race_sec = _t.time() - t0
    except Exception as e:  # noqa: BLE001
        emit("model", f"Horse race skipped ({str(e)[:100]}); LSTM-only")
        race = None

    run_id = db.start_run(llm.provider, data.DATA_SOURCE["prices"], mode)
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

    ctx = {"tickers": shortlist, "stocks": stocks, "sectors": sectors, "asset_types": asset_types,
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
        ctx["composite"] = comp
        ranked = sorted(comp, key=lambda t: -comp[t]["composite"])
        rank = {t: i + 1 for i, t in enumerate(ranked)}

        forced = {}
        for t, p in positions.items():
            px, kind = last_px.get(t), asset_types.get(t, "stock")
            if px and p["avg_cost"] > 0 and px / p["avg_cost"] - 1 <= config.STOP_LOSS:
                forced[t] = f"stop-loss: {px / p['avg_cost'] - 1:.1%} from cost (IPS A17)"
            elif (ctx["sentiment"]["sentiment"].get(t) or {}).get("score", 0) <= config.NEWS_SELL_TRIGGER:
                forced[t] = f"news veto: sentiment {ctx['sentiment']['sentiment'][t]['score']} (IPS C4.4)"
            elif kind == "leveraged_etf" and db.held_trading_days(t, p["opened_date"]) >= config.LEVERAGED_MAX_HOLD_DAYS:
                forced[t] = f"leveraged ETF held {config.LEVERAGED_MAX_HOLD_DAYS}+ trading days (IPS C6.4)"
            elif kind == "leveraged_etf" and regime == "risk_off":
                forced[t] = "leveraged long in risk-off (IPS C5.4)"
            elif kind == "leveraged_etf" and drawdown <= config.RISK_LIMITS["max_drawdown"]:
                forced[t] = f"fund drawdown {drawdown:.1%} past {config.RISK_LIMITS['max_drawdown']:.0%} (IPS A4)"
            elif kind == "inverse_etf" and regime != "risk_off":
                forced[t] = "inverse hedge no longer needed: regime left risk-off (IPS A11)"
        forced = {t: r for t, r in forced.items() if t not in untradeable}

        held_scores = [comp[t]["composite_10"] for t in holdings if t in comp]
        weakest = min(held_scores) if held_scores else 0
        unheld = [t for t in ranked if t not in positions]
        best_unheld = comp[unheld[0]]["composite_10"] if unheld else 0
        eligible_new = [] if no_new_trades else [
            t for t in unheld
            if comp[t]["composite_10"] >= 7 and comp[t]["quant"] >= 0.2 and rank[t] <= 10
            and comp[t]["composite_10"] >= weakest + 1 and not comp[t]["insufficient_data"]
            and not (asset_types.get(t) == "leveraged_etf" and drawdown <= config.RISK_LIMITS["max_drawdown"])]
        exit_cands = [t for t in holdings if t in comp and t not in forced and (
            comp[t]["quant"] < -0.2 or rank[t] > 20 or comp[t]["composite_10"] <= best_unheld - 2)]
        ctx.update(eligible_new=eligible_new, exit_candidates=exit_cands, forced_exits=forced)
        relevant = set(holdings) | set(eligible_new) | set(exit_cands) | set(ranked[:10])
        ctx["lessons"] = reflection.lessons_for(relevant, {sectors.get(t) for t in relevant}, regime)
        ctx["track_record"] = reflection.track_record()
        ctx["invalidated"] = [i for i in reflection.invalidated(positions, last_px) if i["ticker"] not in forced]

        # ---------------------------------------------- 5. PM <-> Risk loop -- #
        max_names = config.TARGET_HOLDINGS_INIT if init_mode else \
            min(config.MAX_HOLDINGS, max(len(holdings), config.TARGET_HOLDINGS_INIT))

        def build(pm, lev_override=None, risk_aversion=4.0):
            target = analytics.optimise(pm["convictions"], close, sectors, asset_types, pm["equity_exposure"],
                                        regime, macro["vix_level"], max_names, risk_aversion=risk_aversion,
                                        lev_cap_override=lev_override,
                                        min_weight=max(0.01, config.MIN_TRADE_NOTIONAL / nav * 1.05),
                                        ensure_min=init_mode)
            if not init_mode:   # held names the PM didn't exit stay at their weight unless resized
                for t, w in current_w.items():
                    if (t not in target and pm["convictions"].get(t, 0) > -0.5 and t not in forced
                            and asset_types.get(t, "stock") == "stock"):   # ETFs stay inside the optimiser's caps
                        target[t] = round(w, 4)
                tot = sum(target.values())
                if tot > 1:
                    target = {t: math.floor(w / tot * 1e4) / 1e4 for t, w in target.items()}
            orders, skipped = analytics.plan_orders(target, positions, last_px, nav, cash, forced,
                                                    init_mode, comp, untradeable)
            post = _post_trade_weights(positions, cash, orders, last_px)
            rr = analytics.risk_report(post, close, sectors, asset_types, regime, drawdown)
            if not init_mode:   # partial rebalances can't always hit the cash band in one day
                rr["soft_concerns"] += [b for b in rr["breaches"] if "cash" in b and "> maximum" in b]
                rr["breaches"] = [b for b in rr["breaches"] if not ("cash" in b and "> maximum" in b)]
            return target, orders, skipped, rr, post

        risk_feedback, prev = None, None
        for rnd in range(1, config.MAX_RISK_ROUNDS + 2):
            pm = call(PortfolioManager, "proposal", rnd, risk_feedback=risk_feedback, prev_proposal=prev)
            target, orders, skipped, rr, post = build(pm)
            proposal = {**pm, "weights": target}
            risk = call(RiskManager, "risk_review", rnd, proposal={**proposal, "weights": post},
                        risk_report=rr, round=rnd)
            if risk["decision"] == "approve":
                break
            risk_feedback = {"issues": risk["issues"], "required_changes": risk["required_changes"]}
            prev = {"convictions": pm["convictions"], "target_weights": {t: w for t, w in target.items() if w > 0}}
        else:
            # Out of rounds: mechanical de-risk that keeps >= 95% invested (IPS A12):
            # cut leveraged ETF room, raise risk aversion, until hard limits hold.
            lev = rules["max_leveraged"]
            ra = 4.0
            for _ in range(8):
                if not rr["breaches"]:
                    break
                lev, ra = max(0.0, lev - 0.05), ra * 1.5
                target, orders, skipped, rr, post = build(pm, lev_override=lev, risk_aversion=ra)
            proposal = {**pm, "weights": target}
            risk = {"decision": "approve" if not rr["breaches"] else "reject", "issues": risk["issues"],
                    "required_changes": "",
                    "rationale": (f"Negotiation rounds exhausted. Leveraged room cut to {lev:.0%}, risk aversion "
                                  f"{ra:.1f}; hard limits {'now met' if not rr['breaches'] else 'STILL breached'}."),
                    "hard_breaches": rr["breaches"]}
            db.log_message(run_id, "Risk Manager", "Chief Risk Officer — hard override", "rules",
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
        for o in orders:
            o["full_exit"] = o["side"] == "SELL" and o["shares"] >= positions.get(o["ticker"], {}).get("shares", 0)
            o["reason"] = _reason(o, comp, pm, risk)
            o.update(reflection.trade_notes(o, (pm.get("trade_notes") or {}).get(o["ticker"]), comp.get(o["ticker"])),
                     sector=sectors.get(o["ticker"]))
            resp = (pm.get("invalidation_responses") or {}).get(o["ticker"])
            if resp and o["side"] == "SELL" and o["ticker"] in {i["ticker"] for i in ctx.get("invalidated", [])}:
                o["thesis"] = f"Invalidation hit: {resp}"[:300]
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


def execute_run(run_id: int) -> dict:
    """Brief §8.2 Execution Agent: commit an approved proposal to SQLite."""
    prop = db.get_proposal(run_id)
    runs = db.get_runs().set_index("id")
    if prop is None or runs.loc[run_id, "status"] != "proposed":
        raise RuntimeError(f"Run {run_id} has no pending proposal (status: {runs.loc[run_id, 'status']}).")
    init_mode = prop["mode"] == "init"
    room = len(prop["orders"]) if init_mode else max(0, config.MAX_TRADES_PER_DAY - db.trades_today())
    done = []
    for o in prop["orders"][:room]:   # sells are first in the list, so they fund the buys
        try:
            meta = {"thesis": o.get("thesis", ""), "invalidation": o.get("invalidation", ""),
                    "invalidation_px": o.get("invalidation_px"), "entry_date": prop.get("asof"),
                    "regime": prop.get("regime"), "sector": o.get("sector"), "signals": o.get("signals", {}),
                    "horizon_days": 5}
            done.append(db.execute_trade(run_id, o["ticker"], o["side"], o["shares"], o["price"], o["reason"], meta))
        except ValueError as e:
            db.log_message(run_id, "Trader", Trader.title, "rules", "execution", 0,
                           {"execute": False, "execution_notes": f"{o['ticker']} rejected at execution: {e}"})
    db.set_run_status(run_id, "executed" if done else "no_trade")
    nav = db.record_snapshot(run_id, prop["last_px"], prop["benchmark_px"])
    csv_path = db.export_trades_csv()
    rep = report.daily_report()
    return {"status": "executed" if done else "no_trade", "trades": done, "nav": nav, "trades_csv": csv_path,
            "daily_report": rep}


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
    scores = (f"composite {c.get('composite_10', '-')}/10 (Q {c.get('quant')}, S {c.get('sentiment')}, "
              f"R {c.get('research')})") if c else "not in today's shortlist"
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
    args = ap.parse_args()
    if args.reset:
        data.get_market_data(data.get_universe()["symbol"].tolist())
        db.use_synthetic(data.is_synthetic())
        db.init_db(reset=True)
    if args.snapshot:
        print(f"NAV ${snapshot_only():,.0f}")
        raise SystemExit
    res = run_committee(args.provider, execute=args.execute)
    print(json.dumps({k: v for k, v in res.items() if k != "orders"}, indent=2, default=str))
    print("\nOrders:")
    for o in res["orders"]:
        print(f"  {o['side']:4} {o['shares']:>6} {o['ticker']:<6} @ {o['price']:>9.2f}  ${o['notional']:>11,.0f}  {o['reason'][:90]}")
