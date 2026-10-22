"""52-week-high momentum (George-Hwang): stocks near their 52w high keep outperforming (anchoring)."""
NAME = "high_52w"
DESC = "price as a share of its 52-week high"
FMT = "{raw:.0%} of the 52w high"
UNIVERSE = "all"


def raw(d):
    c = d.close
    return c / c.rolling(252, min_periods=200).max()
