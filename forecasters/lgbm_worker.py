"""
Subprocess worker for forecasters/lgbm.py. Imports LightGBM only (never torch: their OpenMP runtimes clash).
Monthly walk-forward: for each block start s, train on rows whose 5-day label was known before s (t + H <= s - 1),
predict rows with s <= t < next block; the last model also predicts the live date (T-1).
Usage: python lgbm_worker.py in.npz out.npz
"""
import json
import sys

import lightgbm as lgb
import numpy as np

NAMES = ["mom_1m", "mom_3m", "mom_12_1", "vol_20d", "rsi_14", "dist_50dma", "industry_mom_1m", "reversal_1w", "pead"]
PARAMS = {"objective": "huber", "alpha": 0.9, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
          "num_threads": 4, "seed": 42}


def main(inp, outp):
    d = np.load(inp)
    X, y, t = d["X"], d["y"], d["t"]
    T, H, first, every = int(d["T"]), int(d["H"]), int(d["first"]), int(d["every"])
    pred = np.full(len(y), np.nan, dtype=np.float64)
    starts = list(range(first, T, every))
    imp = np.zeros(X.shape[1])
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else T
        tr = (t + H <= s - 1) & ~np.isnan(y)
        te = (t >= s) & (t < e)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        booster = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
        pred[te] = booster.predict(X[te])
        imp += booster.feature_importance("gain")
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    np.savez(outp, pred=pred, importance=json.dumps({n: round(float(v), 3) for n, v in zip(NAMES, imp)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
