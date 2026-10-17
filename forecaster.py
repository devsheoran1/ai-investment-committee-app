"""
The deep-learning piece: an LSTM that forecasts each stock's next-5-day return
from its last 30 days of returns (plus the market's). One model is trained on
all stocks pooled together (more data, less overfitting).

It reports an honest out-of-sample score (directional hit rate on the last 20%
of history, never seen in training) so the Quant Analyst knows how much to
trust it. Expect ~50-55%: markets are hard. Say that in your presentation.

Runs fine on CPU / an 8GB M2 (uses Apple's MPS GPU if available).
If PyTorch isn't installed, falls back to a linear model and says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _build_dataset(prices: pd.DataFrame, tickers: list[str], lookback: int, horizon: int):
    rets = np.log(prices).diff().dropna()
    mkt = rets[config.BENCHMARK].values
    X, y, t_index, owner = [], [], [], []
    latest = {}
    for tk in tickers:
        r = rets[tk].values
        sd = r.std() + 1e-9
        rn, mn = r / sd, mkt / (mkt.std() + 1e-9)      # scale-free inputs
        feats = np.stack([rn, mn], axis=1)
        for i in range(lookback, len(r) - horizon + 1):
            X.append(feats[i - lookback:i])
            y.append(r[i:i + horizon].sum() / sd)          # target also vol-scaled
            t_index.append(i)
            owner.append(tk)
        latest[tk] = (feats[-lookback:], sd)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(t_index), np.array(owner), latest)


def _split(t_index: np.ndarray, horizon: int, test_frac: float = 0.2):
    cutoff = np.quantile(t_index, 1 - test_frac)
    train = t_index < cutoff - horizon        # gap so targets never overlap the test window
    test = t_index >= cutoff
    return train, test


def forecast(prices: pd.DataFrame, tickers: list[str]) -> dict:
    p = config.LSTM
    X, y, t_idx, owner, latest = _build_dataset(prices, tickers, p["lookback"], p["horizon"])
    train, test = _split(t_idx, p["horizon"])
    try:
        import torch  # noqa: F401
        preds_test, preds_latest, model_name = _lstm(X, y, train, test, latest, p)
    except ImportError:
        preds_test, preds_latest, model_name = _linear(X, y, train, test, latest)

    hit = float(np.mean(np.sign(preds_test) == np.sign(y[test])))
    ic = float(np.corrcoef(preds_test, y[test])[0, 1]) if preds_test.std() > 0 else 0.0
    per_ticker = {}
    for tk, (pred_scaled, sd) in preds_latest.items():
        per_ticker[tk] = {"pred_5d_return_pct": round(float(pred_scaled * sd) * 100, 2)}
    return {
        "model": model_name,
        "train_samples": int(train.sum()),
        "test_samples": int(test.sum()),
        "oos_directional_hit_rate": round(hit, 3),
        "oos_information_coefficient": round(ic, 3),
        "forecasts": per_ticker,
    }


# --------------------------------------------------------------------------- #
def _lstm(X, y, train, test, latest, p):
    import torch
    from torch import nn

    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(X.shape[2], p["hidden"], num_layers=p["layers"],
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(p["hidden"], 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=1e-4)
    loss_fn = nn.HuberLoss()            # robust to fat-tailed returns
    Xtr = torch.tensor(X[train], device=device)
    ytr = torch.tensor(y[train], device=device)

    # simple early stopping on the last 10% of the training window
    n_val = max(1, int(len(Xtr) * 0.1))
    Xfit, yfit, Xval, yval = Xtr[:-n_val], ytr[:-n_val], Xtr[-n_val:], ytr[-n_val:]
    best, best_state, patience = float("inf"), None, 0
    for _ in range(p["epochs"]):
        net.train()
        perm = torch.randperm(len(Xfit), device=device)
        for i in range(0, len(Xfit), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xfit[idx]), yfit[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xval), yval).item()
        if v < best - 1e-5:
            best, best_state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        preds_test = net(torch.tensor(X[test], device=device)).cpu().numpy()
        latest_pred = {}
        for tk, (feats, sd) in latest.items():
            x = torch.tensor(feats[None].astype(np.float32), device=device)
            latest_pred[tk] = (float(net(x).cpu().numpy()[0]), sd)
    return preds_test, latest_pred, f"LSTM ({p['layers']}x{p['hidden']}, PyTorch/{device})"


def _linear(X, y, train, test, latest):
    """Ridge regression fallback if PyTorch isn't installed."""
    A = X[train].reshape(train.sum(), -1)
    lam = 10.0
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[train])
    preds_test = X[test].reshape(test.sum(), -1) @ w
    latest_pred = {tk: (float(f.reshape(-1) @ w), sd) for tk, (f, sd) in latest.items()}
    return preds_test, latest_pred, "Ridge regression (fallback — install torch for the LSTM)"
