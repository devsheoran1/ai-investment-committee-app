"""Post-earnings-announcement drift (Bernard-Thomas): the last EPS surprise, fading over 60 trading days."""
from .common import event_panel

NAME = "pead_sue"
DESC = "last EPS surprise %, linear decay over 60 trading days, from the first bar after the report"
FMT = "decayed EPS surprise {raw:+.1f}%"
WINDOW = 60


def raw(d):
    ev = {t: [{"date": e["date"], "v": max(-50.0, min(50.0, float(e["surprise_pct"])))} for e in evs
              if e.get("surprise_pct") is not None] for t, evs in (d.earn_events or {}).items()}
    return event_panel(d, ev, "v", WINDOW)
