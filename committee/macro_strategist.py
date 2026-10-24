import config

from .base import Agent, as_map, compact, clip, entry

TILTS = ("overweight", "neutral", "underweight")


class MacroStrategist(Agent):
    name = "Macro Strategist"
    title = "Chief Macro Strategist"
    tier = "fast"
    role_prompt = """You own three calls (IPS C1.1): the market regime, the size of the hedge,
and which sectors to favour or avoid. The regime and the sector tilts are computed
by the fund's rules (IPS C1.3 and C1.5) and given to you; you explain them and say
what to watch. Rules: risk-off if any 2 of VIX > 25, SPY below both its 50 and 200
dma, HYG/LQD > 1% below its 20dma, VIX up > 30% in a day; any ONE of them already blocks new leveraged
buys. Risk-on only if VIX < 20,
SPY > 50dma and HYG/LQD >= its 20dma. Downgrades apply at once, upgrades need
2 days in a row. Sector tilts: top 3 sectors by 1-month ETF momentum overweight,
bottom 3 underweight. You may override AT MOST ONE sector tilt by one step, with a
stated reason grounded in the data (e.g. oil up 10% in 5 days). Never force trades."""
    output_schema = """{
  "confidence": number 0-1,
  "tilt_override": {"sector": "<sector name or empty>", "tilt": "overweight"|"neutral"|"underweight", "reason": "max 25 words"},
  "watch_list": ["max 3 short items: indicators or events to watch"],
  "rationale": "max 60 words citing the specific indicators",
  "hedge_proposal": {"pct": number 0..0.10, "reason": "max 25 words"}
}
(hedge_proposal is optional and only read when "soft_hedge" is open in the dashboard: one risk-off signal short of full risk-off.
 You propose 0-10% SPXU, the PM decides.)"""

    def build_prompt(self, ctx):
        m = {k: v for k, v in ctx["macro"].items() if k != "synthetic"}
        return (f"MACRO DASHBOARD:\n{compact(m)}\n\nRULE-BASED REGIME:\n{compact(ctx['regime_info'])}\n"
                f"RULE-BASED SECTOR TILTS:\n{compact(ctx['rule_tilts'])}"
                + (f"\nSOFT HEDGE OPEN: risk-off signal(s) {ctx.get('soft_hedge_signals')}; you may propose 0-"
                   f"{config.SOFT_HEDGE['max']:.0%} {config.SOFT_HEDGE['ticker']} via hedge_proposal." if ctx.get("soft_hedge_open") else ""))

    def mock(self, ctx):
        m, r = ctx["macro"], ctx["regime_info"]
        return {"confidence": 0.6, "tilt_override": {"sector": "", "tilt": "neutral", "reason": ""},
                "watch_list": ["[mock] VIX", "[mock] HYG/LQD"],
                "rationale": f"[mock] Regime {r['regime']}: VIX {m['vix_level']}, SPY {m['spy_vs_50dma_pct']:+.1f}% vs 50dma, "
                             f"HYG/LQD {m.get('hyg_lqd_vs_20dma_pct')}% vs 20dma."}

    def validate(self, out, ctx):
        tilts = dict(ctx["rule_tilts"])
        ov = as_map(out.get("tilt_override"))
        applied = None
        sec, tilt = str(ov.get("sector", "")), str(ov.get("tilt", "")).lower()
        if sec in tilts and tilt in TILTS and tilt != tilts[sec]:
            # one step only (overweight <-> neutral <-> underweight)
            if abs(TILTS.index(tilt) - TILTS.index(tilts[sec])) == 1:
                tilts[sec] = tilt
                applied = {"sector": sec, "tilt": tilt, "reason": str(ov.get("reason", ""))[:200]}
        r = ctx["regime_info"]
        rules = ctx["regime_rules"]
        hp = as_map(out.get("hedge_proposal"))
        hedge = ({"hedge_proposal": {"pct": round(clip(hp.get("pct"), 0.0, config.SOFT_HEDGE["max"], 0.0), 3),
                                     "reason": str(hp.get("reason", ""))[:200]}} if ctx.get("soft_hedge_open") and hp else {})
        return {
            **hedge,
            "regime": r["regime"],                       # rules decide, not the model
            "raw_regime": r["raw_regime"],
            "regime_note": r["note"],
            "confidence": clip(out.get("confidence"), 0, 1, 0.5),
            "equity_exposure_target": rules["equity_exposure"],
            "max_leveraged": rules["max_leveraged"],
            "inverse_hedge_range": rules["inverse"],
            "sector_tilts": tilts,
            "tilt_override": applied,
            "watch_list": [str(x)[:100] for x in (out.get("watch_list") if isinstance(out.get("watch_list"), list) else [])][:3],
            "rationale": str(out.get("rationale", ""))[:600],
        }
