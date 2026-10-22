"""Short-term reversal (Jegadeesh/Lehmann): last week's losers rebound, winners give back."""
NAME = "st_reversal_1w"
DESC = "minus the last 5-day return, ranked inside the sector"
FMT = "5d return {raw:+.1%} (sign flipped)"


def raw(d):
    c = d.close
    return -(c / c.shift(5) - 1)
