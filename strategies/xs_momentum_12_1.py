"""Cross-sectional 12-1 month momentum (Jegadeesh-Titman): last 12 months skipping the most recent one."""
NAME = "xs_momentum_12_1"
DESC = "buy 12-1 month winners, sell losers, ranked inside the sector"
FMT = "12-1m return {raw:+.0%}"
UNIVERSE = "all"


def raw(d):
    c = d.close
    return c.shift(21) / c.shift(252) - 1
