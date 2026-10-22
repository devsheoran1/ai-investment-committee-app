"""SEC EDGAR Form 4 (insider transactions), free, no key. Cached daily; ANY failure = the signal is skipped.

Per ticker: submissions JSON (data.sec.gov) -> Form 4 filings of the last `days` -> raw ownership XML -> open-market
purchases (transactionCode P) and sales (S) in dollars, distinct buying insiders. SEC asks for a contact in the
User-Agent (SEC_USER_AGENT in .env) and <= 10 requests/second; we stay at ~6/s.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

BUDGET_SECONDS = 75           # whole fetch; names not reached are left out (absent, never 0)
MAX_FILINGS_PER_TICKER = 12
_GAP = 0.17


def parse_form4(xml_text: str) -> dict:
    """-> {"owner": str, "buy_usd": x, "sell_usd": y} for the non-derivative open-market P / S lines of one filing."""
    root = ET.fromstring(xml_text)
    owner = (root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerCik") or "").strip()
    buy = sell = 0.0
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = (tx.findtext("transactionCoding/transactionCode") or "").strip()
        if code not in ("P", "S"):
            continue
        try:
            sh = float(tx.findtext("transactionAmounts/transactionShares/value") or 0)
            px = float(tx.findtext("transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        if code == "P":
            buy += sh * px
        else:
            sell += sh * px
    return {"owner": owner, "buy_usd": buy, "sell_usd": sell}


def summarise(filings: list[dict]) -> dict:
    buyers = {f["owner"] for f in filings if f["buy_usd"] > 0}
    return {"buyers": len(buyers), "buy_usd": round(sum(f["buy_usd"] for f in filings), 0),
            "sell_usd": round(sum(f["sell_usd"] for f in filings), 0), "n_filings": len(filings)}


def _get(session, url, headers):
    time.sleep(_GAP)
    r = session.get(url, headers=headers, timeout=12)
    r.raise_for_status()
    return r


def fetch_one(session, headers, cik: str, days: int, deadline: float) -> dict | None:
    cik10 = str(cik).zfill(10)
    sub = _get(session, f"https://data.sec.gov/submissions/CIK{cik10}.json", headers).json()
    rec = sub.get("filings", {}).get("recent", {})
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    filings = []
    for form, filed, acc, doc in zip(rec.get("form", []), rec.get("filingDate", []),
                                     rec.get("accessionNumber", []), rec.get("primaryDocument", [])):
        if form != "4" or filed < cutoff:
            continue
        if len(filings) >= MAX_FILINGS_PER_TICKER or time.time() > deadline:
            break
        raw_doc = doc.split("/", 1)[1] if doc.startswith("xsl") and "/" in doc else doc       # strip the XSL render dir
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc.replace('-', '')}/{raw_doc}"
        try:
            filings.append(parse_form4(_get(session, url, headers).text))
        except Exception:  # noqa: BLE001  one bad filing never kills the name
            continue
    return summarise(filings)


def get_insiders(ciks: dict[str, str], days: int = 90, fetcher=fetch_one) -> dict[str, dict] | None:
    """{ticker: summary}; None if EDGAR is unusable (the strategy is then skipped). Cached per day."""
    import requests
    import data
    tickers = [t for t, c in ciks.items() if c and str(c).strip() not in ("", "nan")]
    if not tickers or data.is_synthetic():
        return None
    headers = {"User-Agent": data.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

    def run():
        deadline = time.time() + BUDGET_SECONDS
        out, errors = {}, 0
        with requests.Session() as s, ThreadPoolExecutor(max_workers=3) as ex:
            futs = {t: ex.submit(fetcher, s, headers, ciks[t], days, deadline) for t in tickers}
            for t, f in futs.items():
                try:
                    r = f.result()
                    if r is not None:
                        out[t] = r
                except Exception:  # noqa: BLE001
                    errors += 1
        return out if out and errors <= len(tickers) * 0.5 else {}
    try:
        got = data._cached(data._key("insiders", tickers), run, keep_if=bool)
    except Exception:  # noqa: BLE001
        return None
    return got or None
