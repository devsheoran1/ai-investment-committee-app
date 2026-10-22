"""News tone (FinBERT) as a signal. The existing Sentiment pipeline scores every headline with FinBERT; this reads
the same scores deterministically (the LLM's 'surprise' judgement stays in the Sentiment agent)."""
import numpy as np

from .common import snapshot_panel

NAME = "news_surprise"
DESC = "average FinBERT tone (P(pos)-P(neg)) of the last 5 days of headlines, names with news only"
FMT = "FinBERT tone {raw:+.2f}"
HISTORY = False                 # no historical news exists (D7)
NEUTRAL = False


def raw(d):
    vals = {}
    for t, heads in (d.news or {}).items():
        fb = [h.get("finbert") for h in (heads or []) if h.get("finbert") is not None]
        if fb:
            vals[t] = float(np.mean(fb))
    return snapshot_panel(d, vals)
