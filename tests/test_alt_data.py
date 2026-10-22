"""Oct 23: SEC Form 4 insider data (A1 insider_buying) and the politician-buys context for the Sentiment agent.
All offline: fixtures + fake fetchers. Run:  /opt/miniconda3/bin/python3 tests/test_alt_data.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import date, timedelta
import config, data
from strategies import edgar, politicians
from committee.sentiment_analyst import SentimentAnalyst

FORM4 = """<?xml version="1.0"?><ownershipDocument>
<reportingOwner><reportingOwnerId><rptOwnerCik>0001111111</rptOwnerCik></reportingOwnerId></reportingOwner>
<nonDerivativeTable>
 <nonDerivativeTransaction><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>25.5</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
 <nonDerivativeTransaction><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>200</value></transactionShares><transactionPricePerShare><value>30</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
 <nonDerivativeTransaction><transactionCoding><transactionCode>A</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>5000</value></transactionShares><transactionPricePerShare><value>0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
</nonDerivativeTable></ownershipDocument>"""


def test_form4_counts_only_open_market_purchases_and_sales():
    f = edgar.parse_form4(FORM4)
    assert f["buy_usd"] == 25500.0 and f["sell_usd"] == 6000.0 and f["owner"] == "0001111111"     # grants (code A) ignored
    s = edgar.summarise([f, {"owner": "0002", "buy_usd": 0.0, "sell_usd": 10.0}, {"owner": "0003", "buy_usd": 100.0, "sell_usd": 0.0}])
    assert s["buyers"] == 2 and s["buy_usd"] == 25600.0 and s["n_filings"] == 3
    assert edgar.parse_form4("<ownershipDocument/>") == {"owner": "", "buy_usd": 0.0, "sell_usd": 0.0}


def test_edgar_failure_is_a_skipped_signal_never_an_exception(tmp_path=None):
    orig = data._cached
    data._cached = lambda name, fn, **kw: fn()
    try:
        def boom(*a, **k):
            raise RuntimeError("EDGAR 403")
        assert edgar.get_insiders({"AAA": "0000000001", "BBB": "0000000002"}, fetcher=boom) is None
        ok = lambda s, h, cik, days, deadline: {"buyers": 1, "buy_usd": 10.0, "sell_usd": 0.0, "n_filings": 1}
        got = edgar.get_insiders({"AAA": "0000000001", "BBB": "0000000002", "CCC": float("nan"), "DDD": ""}, fetcher=ok)
        assert set(got) == {"AAA", "BBB"}                                       # names without a CIK are left out, not zeroed
        half = lambda s, h, cik, days, deadline: (_ for _ in ()).throw(RuntimeError()) if cik.endswith("1") else ok(s, h, cik, days, deadline)
        got = edgar.get_insiders({"AAA": "0000000001", "BBB": "0000000002", "CCC": "0000000003"}, fetcher=half)
        assert set(got) == {"BBB", "CCC"}                                       # one name failing does not drop the rest
        assert edgar.get_insiders({}, fetcher=ok) is None
    finally:
        data._cached = orig


def test_edgar_requests_stay_polite_and_identify_us():
    assert data.SEC_USER_AGENT
    assert edgar._GAP >= 0.1 and edgar.MAX_FILINGS_PER_TICKER <= 15 and edgar.BUDGET_SECONDS <= 120


def _rows():
    t = date.today()
    d = lambda n: str(t - timedelta(days=n))
    return politicians.normalise([
        {"Representative": "Nancy Pelosi", "Ticker": "NVDA", "Transaction": "Purchase", "TransactionDate": d(10), "Range": "$1,000,001 - $5,000,000"},
        {"Representative": "Nancy Pelosi", "Ticker": "NVDA", "Transaction": "Purchase", "TransactionDate": d(12), "Range": "$250,001 - $500,000"},
        {"Representative": "Dan Crenshaw", "Ticker": "AAPL", "Transaction": "Sale (Full)", "TransactionDate": d(5)},
        {"Representative": "Random Member", "Ticker": "MSFT", "Transaction": "Purchase", "TransactionDate": d(3)},
        {"Representative": "Josh Gottheimer", "Ticker": "AMZN", "Transaction": "Purchase", "TransactionDate": d(90)},
        {"Representative": "Josh Gottheimer", "Ticker": "--", "Transaction": "Purchase", "TransactionDate": d(2)},
        {"senator": "Tommy Tuberville", "ticker": "JPM", "type": "buy", "transaction_date": d(7), "amount": "$15,001 - $50,000"},
    ])


def test_politician_buys_are_buys_by_the_watchlist_inside_the_window_only():
    out = politicians.summarise(_rows(), ["NVDA", "AAPL", "MSFT", "AMZN", "JPM"], ["Pelosi", "Crenshaw", "Gottheimer", "Tuberville"])
    assert set(out) == {"NVDA", "JPM"}                                          # sale, non-watchlist member, 90-day-old buy, '--' ticker: out
    assert out["NVDA"]["n_buys"] == 2 and out["NVDA"]["buyers"] == ["Nancy Pelosi"]
    assert politicians.summarise(_rows(), ["MSFT"], ["Pelosi"]) == {}


def test_politician_source_failure_is_empty_and_synthetic_days_skip_it():
    orig, orig_dl, orig_syn = data._cached, politicians._download, data.is_synthetic
    try:
        data._cached = lambda name, fn, **kw: fn()
        politicians._download = lambda: (_ for _ in ()).throw(RuntimeError("403"))
        data.is_synthetic = lambda: False
        assert politicians.get_politician_buys(["NVDA"]) == {}
        politicians._download = lambda: _rows()
        assert set(politicians.get_politician_buys(["NVDA", "JPM"], ["Pelosi", "Tuberville"])) == {"NVDA", "JPM"}
        data.is_synthetic = lambda: True
        assert politicians.get_politician_buys(["NVDA"]) == {}
    finally:
        data._cached, politicians._download, data.is_synthetic = orig, orig_dl, orig_syn


def test_sentiment_prompt_is_identical_without_politician_data_and_adds_one_block_with_it():
    ctx = {"tickers": ["NVDA", "AAPL"], "news": {"NVDA": [{"title": "NVDA beats", "ts": "2026-10-22 09:00", "source": "Reuters", "finbert": 0.4}], "AAPL": []},
           "earnings": {}}
    s = SentimentAnalyst()
    base = s.build_prompt(ctx)
    assert "CONGRESS" not in base
    assert s.build_prompt({**ctx, "politician_buys": {}}) == base
    with_pol = s.build_prompt({**ctx, "politician_buys": {"NVDA": {"buyers": ["Nancy Pelosi"], "n_buys": 2, "last": "2026-10-13", "amounts": []},
                                                         "XYZ": {"buyers": ["x"], "n_buys": 1, "last": "2026-10-13", "amounts": []}}})
    assert with_pol.startswith(base) and "CONGRESS BUYS" in with_pol and "Pelosi" in with_pol and "XYZ" not in with_pol
    assert "weak context" in with_pol                                              # framed as late, weak, not news
    assert config.POLITICIAN_WATCHLIST and "Pelosi" in config.POLITICIAN_WATCHLIST


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
