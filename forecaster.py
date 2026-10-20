"""
The deep-learning piece (IPS C3.5, rebuilt Oct 20).

    model     LSTM (2x32) with two heads: P(up over the next 5 days) (BCE) and the 5-day return (Huber, kept for
              the IC in the horse race)
    inputs    60 days of: daily log return, log volume change, 20-day realised vol, market (SPY) return
    scaling   every series is scaled by its vol over the TRAINING window only (Oct 20 leakage fix: it used to
              be the full-sample vol, test window included)
    training  monthly walk-forward: the last 250 trading days are predicted in ~21-day blocks, each by a model
              trained only on samples whose 5-day labels were known before the block starts (gap = horizon)
    output    live forecast per ticker (from the latest block model, at most a month old), plus every
              walk-forward prediction for the C3.4 trust rule and the horse race

Expect ~50-55% directional hit rate: short-horizon returns are close to unpredictable. Say that in the presentation.
If PyTorch isn't installed, a ridge regression stands in and says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

WF_DAYS = 250          # C3.4: rolling walk-forward over the last 250 trading days
RETRAIN_EVERY = 21     # C3.5: monthly retrain


def _panel(prices: pd.DataFrame, volume: pd.DataFrame | None, tickers: list[str]):
    """Raw feature arrays (unscaled): ret, dvol, vol20, mkt; shape (T, n_tickers)."""
    lp = np.log(prices[tickers + [config.BENCHMARK]])
    r = lp.diff().iloc[1:]
    mkt = r[config.BENCHMARK].values
    R = r[tickers].values
    if volume is not None and all(t in volume.columns for t in tickers):
        v = volume[tickers].reindex(prices.index).ffill().replace(0, np.nan)
        dv = np.log(v).diff().iloc[1:].fillna(0).clip(-3, 3).values
    else:
        dv = np.zeros_like(R)
    vol20 = pd.DataFrame(R).rolling(20, min_periods=5).std().bfill().values
    return R, dv, vol20, mkt, r.index


def _samples(R, dv, vol20, mkt, lookback, horizon, train_end: int):
    """Scale with TRAINING-window statistics only (train_end = first index not available for training)."""
    sd = np.nanstd(R[:train_end], axis=0) + 1e-9
    sdm = np.nanstd(mkt[:train_end]) + 1e-9
    sdv = np.nanstd(dv[:train_end], axis=0) + 1e-9
    Xr, Xv, Xs, Xm = R / sd, dv / sdv, vol20 / sd, (mkt / sdm)[:, None].repeat(R.shape[1], 1)
    T, n = R.shape
    X, y, idx, own = [], [], [], []
    for j in range(n):
        f = np.stack([Xr[:, j], Xv[:, j], Xs[:, j], Xm[:, j]], axis=1)
        for i in range(lookback, T - horizon + 1):
            X.append(f[i - lookback:i]); y.append(R[i:i + horizon, j].sum() / sd[j]); idx.append(i); own.append(j)
    latest = [np.stack([Xr[:, j], Xv[:, j], Xs[:, j], Xm[:, j]], axis=1)[-lookback:] for j in range(n)]
    return (np.array(X, np.float32), np.array(y, np.float32), np.array(idx), np.array(own),
            np.array(latest, np.float32), sd)


def forecast(prices: pd.DataFrame, tickers: list[str], volume: pd.DataFrame | None = None,
             wf_days: int = WF_DAYS, retrain_every: int = RETRAIN_EVERY) -> dict:
    p = config.LSTM
    lookback, h = 60, p["horizon"]
    R, dv, vol20, mkt, idx_dates = _panel(prices, volume, tickers)
    T = len(R)
    first_oos = max(lookback + 120, T - h - wf_days)                 # leave >= ~120 days for the first training set
    blocks = list(range(first_oos, T - h + 1, retrain_every))
    try:
        import torch  # noqa: F401
        fit = _fit_lstm
        name = f"LSTM 2x{p['hidden']} (60d returns+volume+vol, P(up) & 5d return, monthly walk-forward)"
    except ImportError:
        fit = _fit_ridge
        name = "Ridge regression (fallback: install torch for the LSTM)"

    oos, live, last_sd = [], None, None
    for b, start in enumerate(blocks):
        end = blocks[b + 1] if b + 1 < len(blocks) else T - h + 1
        X, y, ix, own, latest, sd = _samples(R, dv, vol20, mkt, lookback, h, start)
        tr = ix + h <= start - 1                                       # labels known before the block (gap = horizon)
        te = (ix >= start) & (ix < end)
        if tr.sum() < 200 or te.sum() == 0:
            continue
        pred_ret, pred_up, model = fit(X[tr], y[tr], X[te], p)
        for k, i, j, pr, pu in zip(np.where(te)[0], ix[te], own[te], pred_ret, pred_up):
            oos.append({"ticker": tickers[j], "offset": int(T - h - i), "pred": float(pr * sd[j]),
                        "p_up": float(pu), "y": float(y[k] * sd[j])})
        if b == len(blocks) - 1:                                       # newest model = live model (<= 1 month old)
            live = model(latest)
            last_sd = sd
    if live is None:                                                   # tiny history: one fit on everything
        X, y, ix, own, latest, sd = _samples(R, dv, vol20, mkt, lookback, h, T)
        _, _, model = fit(X, y, X[:1], p)
        live, last_sd = model(latest), sd

    df = pd.DataFrame(oos)
    if len(df):
        hit = float((np.sign(df["p_up"] - 0.5) == np.sign(df["y"])).mean())
        ic = float(df["pred"].corr(df["y"])) if df["pred"].std() > 0 else 0.0
        base = float((df["y"] > 0).mean())
    else:
        hit, ic, base = 0.5, 0.0, 0.5
    per_ticker = {t: {"pred_5d_return_pct": round(float(live[0][j] * last_sd[j]) * 100, 2),
                      "p_up": round(float(live[1][j]), 3)} for j, t in enumerate(tickers)}
    return {
        "model": name,
        "train_samples": int(len(df)),
        "test_samples": int(len(df)),
        "walk_forward_days": int(min(wf_days, T - h - first_oos)),
        "retrains": len(blocks),
        "oos_directional_hit_rate": round(hit, 3),
        "always_up_hit_rate": round(base, 3),
        "oos_information_coefficient": round(ic, 3),
        "forecasts": per_ticker,
        "oos_preds": oos,
    }


# --------------------------------------------------------------------------- #
def _fit_lstm(Xtr, ytr, Xte, p):
    import torch
    from torch import nn

    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    device = config.LSTM_DEVICE if config.LSTM_DEVICE != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(Xtr.shape[2], p["hidden"], num_layers=p["layers"], batch_first=True, dropout=0.2)
            self.ret = nn.Sequential(nn.Linear(p["hidden"], 16), nn.ReLU(), nn.Linear(16, 1))
            self.up = nn.Sequential(nn.Linear(p["hidden"], 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            h = self.lstm(x)[0][:, -1]
            return self.ret(h).squeeze(-1), self.up(h).squeeze(-1)

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=1e-4)
    huber, bce = nn.HuberLoss(), nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    ut = (yt > 0).float()
    n_val = max(1, int(len(Xt) * 0.1))                              # early stopping on the newest 10% of training
    perm_all = torch.arange(len(Xt) - n_val, device=device)
    best, state, patience = float("inf"), None, 0
    for _ in range(min(p["epochs"], 15)):
        net.train()
        perm = perm_all[torch.randperm(len(perm_all), device=device)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            r_, u_ = net(Xt[b])
            loss = huber(r_, yt[b]) + bce(u_, ut[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            r_, u_ = net(Xt[-n_val:])
            v = (huber(r_, yt[-n_val:]) + bce(u_, ut[-n_val:])).item()
        if v < best - 1e-5:
            best, state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    if state:
        net.load_state_dict(state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            r_, u_ = net(torch.tensor(np.asarray(X, np.float32), device=device))
            return r_.cpu().numpy(), torch.sigmoid(u_).cpu().numpy()

    pr, pu = predict(Xte)
    return pr, pu, predict


def _fit_ridge(Xtr, ytr, Xte, p):
    A = Xtr.reshape(len(Xtr), -1)
    w = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ ytr)

    def predict(X):
        r = np.asarray(X).reshape(len(X), -1) @ w
        return r, 1 / (1 + np.exp(-r))

    pr, pu = predict(Xte)
    return pr, pu, predict
