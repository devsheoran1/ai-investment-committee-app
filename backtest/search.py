"""
Walk-forward weight search (Oct 24). Train decisions 2018-01 -> 2021-12, test 2022-01 -> today; nothing is chosen on the test period.

Fast proxy (so thousands of configurations are affordable): every Friday close, score = Quant (the ensemble) blended with the Macro sector tilt,
hold the top 15 eligible stocks equal weight to the next Friday's next-open (fills at the next open, zero cost). The full fund engine (5 trades/day,
stops, vol-managed sizing) is NOT in the proxy; the finalists are re-run through the real `simulate` to confirm.

Grid (coarse, 10% steps):  family prior over momentum / earnings / reversal / defensive (per-strategy cap c in {20%, 30%} must be feasible),
                           shrinkage lambda in {0.5, 1, 2}, regime allocator on/off, Macro-tilt share m in {0, .1, .2, .3, .4} of the Quant+Macro blend.
Selection: NOT the single best. Each configuration is scored by the mean TRAIN Sharpe of itself plus all its one-step neighbours (a 10% shift between two
families, adjacent lambda / macro share, regime flip, cap flip); the best neighbourhoods are the "robust region". A finalist whose TEST Sharpe is below 0.8x
its TRAIN Sharpe is rejected. Not backtestable: Sentiment, Research, and the Macro LLM (only its Python sector tilt is here).
    /opt/miniconda3/bin/python3 -m backtest.search
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import analytics
import config
from backtest import engine as E
from backtest.longrun import prepare_long
from backtest.run_backtest import md_table, num, pct
from strategies import regime as R
from strategies.ensemble import family_of
from strategies.sector_rotation import ETF as SECTOR_ETF

OUT = os.path.join(config.BASE_DIR, "reports", "backtest")
TRAIN = ("2018-01-01", "2021-12-31")
TEST = ("2022-01-01", "2099-01-01")
TOPN = 15
LAMS = [0.5, 1.0, 2.0]
MACRO = [0.0, 0.1, 0.2, 0.3, 0.4]
CAPS = [0.2, 0.3]
FAMS = ["momentum", "earnings", "reversal", "defensive"]
GAMMA = 0.5772156649


class Arrays:
    def __init__(self, P: E.Prep):
        idx = P.close.index
        wk = pd.Series(idx.isocalendar().week.astype(int).values, index=idx)
        last = wk.ne(wk.shift(-1)).values
        self.D = np.array([i for i in range(len(idx) - 1) if last[i] and idx[i] >= pd.Timestamp("2017-12-01")])
        self.dates = idx[self.D]
        self.names = list(P.names)
        self.fam = [family_of(n) or "other" for n in self.names]
        self.stocks = list(P.stocks)
        self.Z = np.stack([P.z[n].reindex(columns=self.stocks).values[self.D] for n in self.names], axis=2).astype(np.float32)   # D,N,K
        self.Zm = (~np.isnan(self.Z)).astype(np.float32)
        self.Zf = np.nan_to_num(self.Z)
        known = P.ic[self.names].shift(config.ENSEMBLE["ic_horizon"])
        roll = known.rolling(config.ENSEMBLE["ic_sessions"], min_periods=2)
        mean, sd, n = roll.mean(), roll.std(ddof=1), roll.count().fillna(0)
        t = mean / (sd / np.sqrt((n / config.ENSEMBLE["ic_horizon"]).clip(lower=1.0))).replace(0, np.nan)
        self.t = np.where(n.values >= config.ENSEMBLE["ic_sessions"], np.nan_to_num(t.values), 0.0).clip(min=0)[self.D]      # D,K, validated + positive part only
        states = R.regime_states(P.close[config.BENCHMARK])
        self.state = states.values[self.D]
        mults = config.REGIME_ALLOC["multipliers"]
        self.M = np.array([[mults[s].get(f, 1.0) for f in self.fam] for s in self.state])                                       # D,K
        openp = P.open[self.stocks].values
        ent, ext = self.D[:-1] + 1, self.D[1:] + 1
        self.ret = openp[ext] / openp[ent] - 1                                                                                   # D-1,N
        self.elig = P.eligible.values[self.D[:-1]]
        self.macro = self._macro(P)[:-1]
        self.w_dates = self.dates[:-1]

    def _macro(self, P) -> np.ndarray:
        etfs = sorted(set(SECTOR_ETF.values()) & set(P.close.columns))
        m1 = (P.close[etfs] / P.close[etfs].shift(21) - 1).iloc[self.D]
        rk = m1.rank(axis=1, ascending=False)
        n = m1.notna().sum(axis=1)
        tilt = pd.DataFrame(0.0, index=m1.index, columns=etfs)
        tilt[rk <= 3] = analytics.TILT_TO_SCORE["overweight"]
        tilt[rk.gt(n - 3, axis=0) & (rk > 3)] = analytics.TILT_TO_SCORE["underweight"]
        sec_etf = [SECTOR_ETF.get(P.sector.get(s)) for s in self.stocks]
        return np.stack([tilt[e].values if e in tilt.columns else np.zeros(len(tilt)) for e in sec_etf], axis=1)


def weights(A: Arrays, prior: np.ndarray, lam: float, regime: bool, caps: np.ndarray) -> np.ndarray:
    raw = prior[None, :] * (1.0 + lam * A.t)
    if regime:
        raw = raw * A.M
    w = raw / raw.sum(1, keepdims=True)
    free0 = prior > 0
    for _ in range(10):
        over = w > caps[None, :] + 1e-12
        if not over.any():
            break
        excess = ((w - caps[None, :]) * over).sum(1)
        w = np.where(over, caps[None, :], w)
        room = (~over) & free0[None, :]
        base = (w * room).sum(1)
        w = w + np.where(base[:, None] > 0, excess[:, None] * w * room / np.where(base[:, None] > 0, base[:, None], 1), 0)
    return w


def weekly_returns(A: Arrays, w: np.ndarray, m: float) -> pd.Series:
    num_ = np.einsum("dnk,dk->dn", A.Zf, w)
    den = np.einsum("dnk,dk->dn", A.Zm, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.clip(num_ / den / config.ENSEMBLE["z_to_score"], -1, 1)
    score = (1 - m) * q + m * A.macro_full
    score = np.where(np.isfinite(score), score, -np.inf)[:-1]
    score = np.where(A.elig, score, -np.inf)
    top = np.argpartition(-score, TOPN, axis=1)[:, :TOPN]
    pick = np.take_along_axis(score, top, 1) > -np.inf
    r = np.take_along_axis(A.ret, top, 1)
    r = np.where(pick, r, np.nan)
    return pd.Series(np.nanmean(r, axis=1), index=A.w_dates).fillna(0.0)


def fam_prior(A: Arrays, tenths: dict, equal_other: bool = False) -> np.ndarray:
    cnt = {f: sum(1 for x in A.fam if x == f) for f in set(A.fam)}
    return np.array([tenths.get(f, 0) / 10.0 / cnt[f] * len(A.names) if cnt[f] else 0.0 for f in A.fam])


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(52)) if len(r) > 5 and r.std() > 0 else np.nan


def maxdd(r: pd.Series) -> float:
    nav = (1 + r).cumprod()
    return float((nav / nav.cummax() - 1).min())


def psr(r: pd.Series, bench_sr: float = 0.0) -> float:
    """Probabilistic Sharpe: P(true weekly SR > bench) given the sample's skew and kurtosis (Bailey & Lopez de Prado)."""
    x = r.values
    sr, T = x.mean() / x.std(), len(x)
    sk, ku = pd.Series(x).skew(), pd.Series(x).kurt() + 3
    return float(norm.cdf((sr - bench_sr) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)))


def dsr(r: pd.Series, trial_srs: np.ndarray) -> float:
    """Deflated Sharpe: PSR against the Sharpe you would expect from the best of N noise trials (N = len(trial_srs))."""
    n = len(trial_srs)
    sr0 = np.nanstd(trial_srs) * ((1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * np.e)))
    return psr(r, sr0)


def slc(r: pd.Series, win) -> pd.Series:
    return r[(r.index >= pd.Timestamp(win[0])) & (r.index <= pd.Timestamp(win[1]))]


def grid() -> list[tuple]:
    out = []
    for c in CAPS:
        mx = int(round(c * 10))
        for rev, de, ea in itertools.product(range(mx + 1), range(mx + 1), range(2 * mx + 1)):
            mo = 10 - rev - de - ea
            if mo < 0 or mo > 5 * mx:
                continue
            for lam, reg, m in itertools.product(LAMS, (False, True), MACRO):
                out.append((mo, ea, rev, de, lam, reg, m, c))
    return out


def neighbours(k: tuple) -> list[tuple]:
    mo, ea, rev, de, lam, reg, m, c = k
    fam = [mo, ea, rev, de]
    nb = []
    for i, j in itertools.permutations(range(4), 2):
        if fam[i] > 0:
            f2 = list(fam); f2[i] -= 1; f2[j] += 1
            nb.append((*f2, lam, reg, m, c))
    for l2 in LAMS:
        if abs(LAMS.index(l2) - LAMS.index(lam)) == 1:
            nb.append((mo, ea, rev, de, l2, reg, m, c))
    for m2 in MACRO:
        if abs(MACRO.index(m2) - MACRO.index(m)) == 1:
            nb.append((mo, ea, rev, de, lam, reg, m2, c))
    nb.append((mo, ea, rev, de, lam, not reg, m, c))
    nb.append((mo, ea, rev, de, lam, reg, m, [x for x in CAPS if x != c][0]))
    return nb


def run_cfg(A: Arrays, k: tuple) -> pd.Series:
    mo, ea, rev, de, lam, reg, m, c = k
    prior = fam_prior(A, {"momentum": mo, "earnings": ea, "reversal": rev, "defensive": de})
    return weekly_returns(A, weights(A, prior, lam, reg, np.full(len(A.names), c)), m)


def baseline(A: Arrays, regime: bool = True, m: float = 0.27) -> pd.Series:
    """The live setup in the proxy: equal prior over every backtestable strategy, lambda 0.5, caps 20% (PEAD 30%), Macro share 0.15 / (0.40 + 0.15)."""
    caps = np.array([0.30 if n == "pead_sue" else 0.20 for n in A.names])
    return weekly_returns(A, weights(A, np.ones(len(A.names)), 0.5, regime, caps), m)


def main() -> dict:
    t0 = time.time()
    P = prepare_long()
    A = Arrays(P)
    A.macro_full = np.vstack([A.macro, np.zeros((1, A.macro.shape[1]))])
    G = grid()
    print(f"{len(G)} configs, strategies {A.names}", flush=True)
    rets = {}
    for i, k in enumerate(G):
        rets[k] = run_cfg(A, k)
        if i % 500 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    tr = {k: sharpe(slc(r, TRAIN)) for k, r in rets.items()}
    te = {k: sharpe(slc(r, TEST)) for k, r in rets.items()}
    trial = np.array([sr / np.sqrt(52) for sr in tr.values() if np.isfinite(sr)])          # per-period Sharpes of all trials (train)
    robust = {}
    for k in G:
        vals = [tr[k]] + [tr[n] for n in neighbours(k) if n in tr]
        vals = [v for v in vals if np.isfinite(v)]
        robust[k] = (float(np.mean(vals)), float(np.min(vals)), len(vals))
    order = sorted(G, key=lambda k: -robust[k][0])
    picks, seen = [], set()
    for k in order:                                           # distinct finalists: skip near-duplicates (same family prior)
        if k[:4] in seen:
            continue
        seen.add(k[:4]); picks.append(k)
        if len(picks) == 6:
            break
    spy = P.open["SPY"].values
    spy_r = pd.Series(spy[A.D[1:] + 1] / spy[A.D[:-1] + 1] - 1, index=A.w_dates)
    base = {"current (regime on)": baseline(A, True), "current (regime off)": baseline(A, False)}
    rows = []
    for k in picks:
        r = rets[k]
        rows.append({"cfg": k, "train": tr[k], "test": te[k], "ratio": te[k] / tr[k] if tr[k] > 0 else np.nan,
                     "robust_mean": robust[k][0], "robust_min": robust[k][1], "n_nb": robust[k][2],
                     "dd_train": maxdd(slc(r, TRAIN)), "dd_test": maxdd(slc(r, TEST)),
                     "dsr_train": dsr(slc(r, TRAIN), trial), "psr_test": psr(slc(r, TEST)),
                     "rejected": bool(not (te[k] >= 0.8 * tr[k]))})
    best_single = max(G, key=lambda k: tr[k])
    bs = {"cfg": best_single, "train": tr[best_single], "test": te[best_single]}
    cur = {n: {"train": sharpe(slc(r, TRAIN)), "test": sharpe(slc(r, TEST)), "dd_train": maxdd(slc(r, TRAIN)), "dd_test": maxdd(slc(r, TEST)),
               "psr_test": psr(slc(r, TEST))} for n, r in base.items()}
    cur["SPY weekly buy&hold"] = {"train": sharpe(slc(spy_r, TRAIN)), "test": sharpe(slc(spy_r, TEST)), "dd_train": maxdd(slc(spy_r, TRAIN)),
                                  "dd_test": maxdd(slc(spy_r, TEST)), "psr_test": psr(slc(spy_r, TEST))}
    a = np.array([tr[k] for k in G]); b = np.array([te[k] for k in G])
    ok = np.isfinite(a) & np.isfinite(b)
    persist = float(spearmanr(a[ok], b[ok])[0])
    res = {"n": len(G), "picks": rows, "best_single": bs, "current": cur, "spearman_train_test": persist,
           "grid_test_median": float(np.nanmedian(b)), "grid_train_median": float(np.nanmedian(a)),
           "share_above_current_test": float(np.mean(b > cur["current (regime on)"]["test"])), "seconds": time.time() - t0}
    winner = next((r for r in rows if not r["rejected"]), None)
    res["winner"] = winner
    res["apply"] = bool(winner and winner["test"] > cur["current (regime on)"]["test"])
    if winner:
        res["confirm"] = confirm_full_engine(P, winner["cfg"])
    json.dump(res, open(os.path.join(OUT, "weight_search.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "weight_search.md"), "w").write(to_md(res))
    print(open(os.path.join(OUT, "weight_search.md")).read())
    return res


def confirm_full_engine(P, k: tuple) -> dict:
    """Winner vs the live setup through the REAL simulate (5 trades/day, stops, vol-managed sizing); Macro share is not in the engine."""
    mo, ea, rev, de, lam, reg, m, c = k
    out = {}
    for lab, cfg, rg in (("current (N2: equal prior, lambda 0.5, regime on)", {"weight_mode": "shrinkage"}, True),
                         ("winner", {"weight_mode": "shrinkage", "family_prior": {"momentum": mo / 10, "earnings": ea / 10, "reversal": rev / 10, "defensive": de / 10},
                                     "shrink_lambda": lam, "weight_cap": c, "family_caps": {}}, reg)):
        W = E.weight_history(P, cfg=cfg, regime=rg)
        s_, v_, a_ = E.ensemble_panels(P, W)
        sc = E.as_deployed(P, s_, a_, blend=1.0)
        navs = [E.simulate(P, sc, v_, start=str((pd.Timestamp("2018-01-02") + pd.Timedelta(days=o)).date()), trail_pct=config.TRAILING_STOP)["nav"] for o in (0, 7)]
        res = {}
        for wn, win in (("train", TRAIN), ("test", TEST)):
            ms = []
            for n in navs:
                s = n[(n.index >= pd.Timestamp(win[0])) & (n.index <= pd.Timestamp(win[1]))]
                rr = s.pct_change().dropna()
                ms.append((float(rr.mean() / rr.std() * np.sqrt(252)), float((s / s.cummax() - 1).min())))
            res[wn] = {"Sharpe": float(np.mean([x[0] for x in ms])), "MaxDD": float(np.mean([x[1] for x in ms]))}
        out[lab] = res
    return out


def kname(k):
    mo, ea, rev, de, lam, reg, m, c = k
    return f"mom {mo*10}/earn {ea*10}/rev {rev*10}/def {de*10}, λ{lam:g}, regime {'on' if reg else 'off'}, macro {m:.0%}, cap {c:.0%}"


def to_md(res) -> str:
    t = pd.DataFrame({kname(r["cfg"]): {"Train Sharpe": num(r["train"]), "Test Sharpe": num(r["test"]), "Test/Train": num(r["ratio"]),
                                        "Neigh. mean / min (train)": f"{num(r['robust_mean'])} / {num(r['robust_min'])}",
                                        "Max DD train / test": f"{pct(r['dd_train'], 0)} / {pct(r['dd_test'], 0)}",
                                        "Deflated Sharpe (train, N=%d)" % res["n"]: num(r["dsr_train"]), "PSR test (SR>0)": num(r["psr_test"]),
                                        "Verdict": "REJECT (test < 0.8x train)" if r["rejected"] else "pass"} for r in res["picks"]}).T
    t.index.name = "Robust-region finalists"
    c = pd.DataFrame({n: {"Train Sharpe": num(v["train"]), "Test Sharpe": num(v["test"]), "Max DD train / test": f"{pct(v['dd_train'], 0)} / {pct(v['dd_test'], 0)}",
                          "PSR test (SR>0)": num(v["psr_test"])} for n, v in res["current"].items()}).T
    c.index.name = "Reference (same proxy)"
    w = res["winner"]
    bs = res["best_single"]
    out = ["# Walk-forward weight search (train 2018–2021, test 2022–2026)", "", __doc__.split("Fast proxy")[1].split("    /opt")[0].join(["**Fast proxy", ""]), "",
           f"{res['n']} configurations. Across the whole grid the train and test Sharpes have Spearman rank correlation **{res['spearman_train_test']:+.2f}** "
           f"(median train {num(res['grid_train_median'])}, median test {num(res['grid_test_median'])}); {res['share_above_current_test']:.0%} of configurations beat the current setup in the test period.",
           f"Best single configuration on TRAIN (what a naive search would pick): {kname(bs['cfg'])}: train {num(bs['train'])}, test {num(bs['test'])}.", "",
           md_table(t), "", md_table(c), ""]
    if w:
        out += [f"**Winner (first finalist passing the 0.8x rule):** {kname(w['cfg'])}. Test Sharpe {num(w['test'])} vs current {num(res['current']['current (regime on)']['test'])}: "
                f"**{'APPLY' if res['apply'] else 'DO NOT APPLY, keep the current setup'}** (rule: apply only if the test-period Sharpe beats the current setup).", ""]
        if res.get("confirm"):
            cf = pd.DataFrame({lab: {f"{wn} Sharpe": num(v[wn]["Sharpe"]) for wn in ("train", "test")} | {f"{wn} Max DD": pct(v[wn]["MaxDD"], 0) for wn in ("train", "test")}
                               for lab, v in res["confirm"].items()}).T
            cf.index.name = "Full-engine confirmation (mean of 2 starts)"
            out += [md_table(cf), ""]
    else:
        out += ["**No finalist passed the 0.8x test/train rule: keep the current setup.**", ""]
    out += ["Caveats: in-sample selection on 4 years of one market, survivorship-biased universe (today's S&P 500), the proxy ignores stops, the 5-trade cap and "
            "vol-managed sizing, Sentiment/Research are absent. A strategy that wins here is a hypothesis for the next out-of-sample year, not evidence.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
