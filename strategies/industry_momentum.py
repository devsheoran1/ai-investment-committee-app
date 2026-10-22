"""Industry momentum (Moskowitz-Grinblatt): 6-month return of the stock's GICS sub-industry peers, ex-itself."""
import numpy as np
import pandas as pd

NAME = "industry_momentum"
DESC = "6m return of the other stocks in the same sub-industry (sector peers if < 3 in the industry)"
FMT = "peer 6m return {raw:+.0%}"


def _peer_mean(x: pd.DataFrame, groups: pd.Series, min_n: int = 3) -> pd.DataFrame:
    """Mean of the OTHER members of each group, per date."""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for g in groups.dropna().unique():
        cols = list(groups.index[groups == g])
        if len(cols) < min_n:
            continue
        blk = x[cols]
        ok = blk.notna()
        tot = blk.fillna(0).sum(axis=1)
        n = ok.sum(axis=1)
        out[cols] = (-blk.fillna(0)).add(tot, axis=0).div((-ok.astype(float)).add(n, axis=0).replace(0, np.nan)).where(ok)
    return out


def raw(d):
    c = d.close[d.stocks()]
    r6 = c.shift(5) / c.shift(126) - 1
    by_sector = _peer_mean(r6, d.sector.reindex(r6.columns))
    by_industry = _peer_mean(r6, d.industry.reindex(r6.columns))
    return by_industry.where(by_industry.notna(), by_sector)
