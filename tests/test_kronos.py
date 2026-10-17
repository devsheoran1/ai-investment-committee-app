"""Offline unit test for the Kronos forecaster on 3 tickers (needs the HF weights cached once).
Run:  /opt/miniconda3/bin/python3 -m pytest -q tests/test_kronos.py   (or plain python tests/test_kronos.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from forecasters import kronos_fc


def _market(n=300, tickers=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (n, len(tickers))), axis=0)),
                         index=idx, columns=list(tickers))
    return {"close": close, "open": close.shift(1).fillna(close), "high": close * 1.01, "low": close * 0.99,
            "volume": pd.DataFrame(1e6, index=idx, columns=close.columns)}


def test_predict_shapes_and_replay_safety():
    md = _market()
    out = kronos_fc.predict(md, ["AAA", "BBB", "CCC"], sample_count=3)
    assert set(out) == {"AAA", "BBB", "CCC"}
    for v in out.values():
        assert set(v) == {"exp_ret_5d", "p_up", "dispersion"} and 0 <= v["p_up"] <= 1 and v["dispersion"] >= 0
    # replay safety: wrecking every row AFTER end_pos must not change the forecast at end_pos
    md2 = {k: v.copy() for k, v in md.items()}
    for k in ("close", "open", "high", "low"):
        md2[k].iloc[250:] = 1e6
    import torch; torch.manual_seed(0); a = kronos_fc.predict(md, ["AAA"], end_pos=249, sample_count=3)
    torch.manual_seed(0); b = kronos_fc.predict(md2, ["AAA"], end_pos=249, sample_count=3)
    assert abs(a["AAA"]["exp_ret_5d"] - b["AAA"]["exp_ret_5d"]) < 1e-6


if __name__ == "__main__":
    test_predict_shapes_and_replay_safety(); print("ok")
