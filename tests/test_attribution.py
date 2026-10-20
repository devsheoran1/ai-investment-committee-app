"""Stage 7: attribution parts sum to the total excess (daily exact, Carino-linked within 1 bp).
Run:  /opt/miniconda3/bin/python3 tests/test_attribution.py"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import attribution as A


def _day(rng):
    w = {"AAA": 0.30, "BBB": 0.25, "CCC": 0.20, "TQQQ": 0.15, "XOM": 0.08}          # 2% cash
    rets = {t: float(rng.normal(0.001, 0.02)) for t in w}
    secs = {"AAA": "Information Technology", "BBB": "Information Technology", "CCC": "Health Care", "XOM": "Energy"}
    kinds = {t: "stock" for t in secs} | {"TQQQ": "leveraged_etf"}
    spy_w = {"Information Technology": 0.4, "Health Care": 0.1, "Energy": 0.05, "Financials": 0.45}
    sec_ret = {s: float(rng.normal(0.0005, 0.01)) for s in spy_w}
    return A.day_effects(w, rets, float(rng.normal(0.0005, 0.01)), secs, kinds, spy_w, sec_ret,
                         {"TQQQ": float(rng.normal(0.0005, 0.012))})


def test_daily_parts_sum_exactly():
    rng = np.random.default_rng(1)
    for _ in range(50):
        d = _day(rng)
        assert abs(sum(d["effects"].values()) - (d["r_fund"] - d["r_spy"])) < 1e-12


def test_carino_linked_parts_sum_to_compounded_excess_within_1bp():
    rng = np.random.default_rng(2)
    days = [_day(rng) for _ in range(15)]
    c = A.carino(days)
    Rf = math.prod(1 + d["r_fund"] for d in days) - 1
    Rb = math.prod(1 + d["r_spy"] for d in days) - 1
    assert abs(sum(c["parts"].values()) - (Rf - Rb)) < 1e-4
    assert abs(c["excess"] - (Rf - Rb)) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
