"""
Plain-English reading of the model horse race (IPS C3.4). One wording, used by the dashboard's Model Lab, the daily
report and the Word summary, and mirrored in the comments of committee/quant_analyst.py. Pure text from the numbers.

    hit            share of 5-day direction calls a model got right (walk-forward, out of sample)
    always-up      the dumb baseline "stocks go up": how often 5-day returns were positive
    IC             rank correlation between the model's ranking and the real ranking of returns
    t              IC divided by its own noise; below 1 it could be luck
    C3.4 rule      a model gets weight only if it beats always-up AND clears 52% (25% weight; 55%+ = 50%) AND t >= 1
"""
from __future__ import annotations

NAMES = {"lstm": "LSTM", "kronos": "Kronos", "lgbm": "LightGBM", "naive": "the simple momentum rule"}

HOW_TO_READ = """- **Hit rate**: how often the model called the 5-day direction right on data it had never seen.
- **Always-up baseline**: the lazy guess "stocks usually go up". If stocks rose in 57% of the windows, a model must beat 57% to add anything.
- **IC (rank correlation)**: does the model put the future winners above the future losers? 0 = no skill, +0.05 = slight, +0.10 = good.
- **t**: the IC divided by its own noise. Below 1 the result could easily be luck.
- **The C3.4 rule**: a model only gets weight if it beats the always-up guess, clears 52% (25% weight; above 55% = 50%) and has t of at least 1. Otherwise Quant ignores it and uses price signals only."""


def _list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def plain_summary(race: dict | None) -> str:
    """One or two sentences. `race` = the horse-race message content ({"scoreboard", "weights"})."""
    if not race or not race.get("scoreboard"):
        return "No model scoreboard this run, so Quant used price signals only."
    sb, w = race["scoreboard"], race.get("weights") or {}
    base = [v["always_up"] for v in sb.values() if v.get("always_up") is not None]
    up = f" ({sum(base) / len(base):.0%})" if base else ""
    trusted = [NAMES.get(m, m) for m, x in w.items() if x and x > 0]
    if trusted:
        mix = ", ".join(f"{NAMES.get(m, m)} {x:.0%}" for m, x in w.items() if x and x > 0)
        return (f"{_list(trusted)} beat the simple 'stocks usually go up' guess{up} this week, so Quant blended "
                f"{'it' if len(trusted) == 1 else 'them'} in with the price signals ({mix}).")
    s = f"No model beat the simple 'stocks usually go up' guess{up} this week, so Quant used price signals only."
    slight = [NAMES.get(m, m) for m, v in sb.items() if m != "naive" and (v.get("ic") or 0) > 0 and (v.get("ic_t") or 0) >= 1]
    if slight:
        s += f" {_list(slight)} showed slight ranking skill but not enough."
    else:
        s += " None showed reliable ranking skill either."
    return s
