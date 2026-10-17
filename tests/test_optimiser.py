"""Regression: init with a PM that marks most names 'avoid' (Oct 18 Groq run: 6 names, Energy 52%).
Run:  /opt/miniconda3/bin/python3 tests/test_optimiser.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import analytics, config


def test_init_reaches_ten_names_and_respects_caps():
    rng = np.random.default_rng(3)
    n, T = 30, 300
    secs = ["Energy"] * 8 + ["Health Care"] * 6 + ["Financials"] * 6 + ["Industrials"] * 5 + ["Utilities"] * 5
    names = [f"S{i}" for i in range(n)]
    mkt = rng.normal(0.0004, 0.01, T)
    px = pd.DataFrame(100 * np.exp(np.cumsum(np.array([0.8 * mkt + rng.normal(0, 0.012, T) for _ in names]).T, axis=0)),
                      columns=names)
    px["SPY"] = 400 * np.exp(np.cumsum(mkt))
    conv = {t: -0.9 for t in names}
    for t in names[:6]:                      # only 6 positive names, 5 of them Energy
        conv[t] = 0.8
    sectors = dict(zip(names, secs))
    kinds = {t: "stock" for t in names}
    w = analytics.optimise(conv, px, sectors, kinds, 0.98, "neutral", 18.0, 15, ensure_min=True)
    assert len(w) >= config.MIN_HOLDINGS, f"only {len(w)} names"
    by_sector = {}
    for t, x in w.items():
        by_sector[sectors[t]] = by_sector.get(sectors[t], 0) + x
        assert x <= config.RISK_LIMITS["max_single_weight"] + 1e-6
    assert max(by_sector.values()) <= config.RISK_LIMITS["max_sector_weight"] + 1e-6, by_sector
    assert min(w.values()) >= 0.0099, f"dust position {min(w.values())}"
    assert sum(w.values()) >= 1 - config.RISK_LIMITS["max_cash_weight"] - 1e-3, sum(w.values())


if __name__ == "__main__":
    test_init_reaches_ten_names_and_respects_caps(); print("ok")
