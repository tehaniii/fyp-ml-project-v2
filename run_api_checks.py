# run_api_checks.py
"""
Runs a minimal API test suite against your pricing code and prints a Markdown table.
No external HTTP server required. Uses monkeypatched browse_search to avoid live eBay.
"""

import os
import textwrap
from datetime import datetime

RESULTS = []

def record(id, case, desc, expected, passed, details=""):
    RESULTS.append({
        "id": id, "case": case, "desc": desc, "expected": expected,
        "outcome": "Pass" if passed else f"Fail ({details[:80]})",
        "status": "Pass" if passed else "Fail"
    })

def fake_items_happy():
    return [
        {
            "itemId": "X1",
            "title": "Ceylon 24 Stiver 1808 silver coin elephant",
            "price": {"value": "99.99", "currency": "USD"},
            "itemWebUrl": "http://example/x1"
        },
        {
            "itemId": "X3",
            "title": "Ceylon 24 Stiver 1809 coin",  # +1 year allowed
            "price": {"value": "110.00", "currency": "USD"},
            "itemWebUrl": "http://example/x3"
        },
        {
            "itemId": "X4",
            "title": "Sri Lanka 24 Stiver 1808",
            "price": {"value": "105.00", "currency": "USD"},
            "itemWebUrl": "http://example/x4"
        },
    ]

def fake_items_with_forbids():
    return [
        {
            "itemId": "Y1",
            "title": "Ceylon 24 Stiver 1808 pendant replica",  # should be filtered
            "price": {"value": "12.00", "currency": "USD"},
            "itemWebUrl": "http://example/y1"
        },
        {
            "itemId": "Y2",
            "title": "Ceylon 24 Stiver 1812",  # far year → excluded
            "price": {"value": "50.00", "currency": "USD"},
            "itemWebUrl": "http://example/y2"
        },
        {
            "itemId": "Y3",
            "title": "Ceylon 24 Stiver 1808 coin",
            "price": {"value": "120.00", "currency": "USD"},
            "itemWebUrl": "http://example/y3"
        },
    ]

def fake_browse_search_factory(mode):
    def _fake(q, category_ids=None, limit=100, sort=None):
        if mode == "happy":
            return fake_items_happy()
        if mode == "forbids":
            return fake_items_with_forbids()
        if mode == "empty":
            return []
        if mode == "error":
            raise RuntimeError("Simulated Browse API failure")
        return []
    return _fake

def run():
    # import inside so project paths resolve
    from src.ebay_module import ebay_pricing

    # IMPORTANT: ensure your Browse API call in ebay_pricing DOES NOT force category_ids="256"
    # If it does, the mock still works, but in production you should remove legacy 256.

    # API1: Happy Path (mocked)
    try:
        ebay_pricing.browse_search = fake_browse_search_factory("happy")
        out = ebay_pricing.estimate_live_price("Ceylon", "24 Stiver", 1808, None)
        ok = (
            "stats" in out and
            out.get("n_used", 0) >= 1 and
            isinstance(out.get("sample"), list) and
            len(out["sample"]) <= 8
        )
        record("API1", "Pricing – Happy Path",
               "Mocked Browse returns valid items",
               "stats.median present, n_used ≥ 1, sample ≤ 8",
               ok, details=str(out))
    except Exception as e:
        record("API1", "Pricing – Happy Path", "Exception", "Structured JSON", False, details=str(e))

    # API2: No Results
    try:
        ebay_pricing.browse_search = fake_browse_search_factory("empty")
        out = ebay_pricing.estimate_live_price("Ceylon", "24 Stiver", 1808, None)
        ok = (out["stats"]["n"] == 0 and out["n_used"] == 0 and out["sample"] == [])
        record("API2", "Pricing – No Results",
               "Mocked Browse returns 0 items",
               "stats.n = 0, n_used = 0, sample empty",
               ok, details=str(out))
    except Exception as e:
        record("API2", "Pricing – No Results", "Exception", "Graceful empty result", False, details=str(e))

    # API3 & API4: Year + Forbid filtering (combined)
    try:
        ebay_pricing.browse_search = fake_browse_search_factory("forbids")
        out = ebay_pricing.estimate_live_price("Ceylon", "24 Stiver", 1808, None)
        titles = " | ".join(s["title"].lower() for s in out.get("sample", []))
        ok_forbids = ("replica" not in titles) and ("pendant" not in titles)
        ok_year = all(("1808" in s["title"] or "1809" in s["title"]) for s in out.get("sample", []))
        record("API3", "Year Filter", "Allow year ±1; exclude far years",
               "Only 1808/1809 in sample", ok_year, details=str(out))
        record("API4", "Forbid Filter", "Exclude pendants/replicas",
               "Forbidden tokens excluded", ok_forbids, details=str(out))
    except Exception as e:
        record("API3", "Year/Forbid", "Exception", "Filters applied", False, details=str(e))
        record("API4", "Year/Forbid", "Exception", "Filters applied", False, details=str(e))

    # API5: Synonyms Expansion (we only check multiple queries don’t crash under mock)
    try:
        ebay_pricing.browse_search = fake_browse_search_factory("happy")
        out = ebay_pricing.estimate_live_price("Ceylon", "Twenty Four Stiver", 1808, None)
        ok = ("stats" in out and out.get("n_raw", 0) >= 1)
        record("API5", "Synonyms Expansion",
               "Heuristic denom variants tried",
               "At least one variant yields items", ok, details=str(out))
    except Exception as e:
        record("API5", "Synonyms Expansion", "Exception", "Multiple queries OK", False, details=str(e))

    # API6: Marketplace Switch (structure survives)
    try:
        os.environ["EBAY_MARKETPLACE"] = "EBAY_GB"
        ebay_pricing.browse_search = fake_browse_search_factory("happy")
        out = ebay_pricing.estimate_live_price("Ceylon", "24 Stiver", 1808, None, marketplace=os.getenv("EBAY_MARKETPLACE"))
        ok = ("stats" in out and isinstance(out.get("sample"), list))
        record("API6", "Marketplace Switch", "GB marketplace env",
               "Result structure intact", ok, details=str(out))
    except Exception as e:
        record("API6", "Marketplace Switch", "Exception", "Structure intact", False, details=str(e))

    # API7: Finding API helper (mocked) — optional
    try:
        from src import ebay_api as finding_api
        def fake_request(url, params, retries=2, backoff=0.7):
            class DummyR:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {
                        "findItemsByKeywordsResponse": [{
                            "ack": ["Success"],
                            "searchResult": [{
                                "item": [{
                                    "title": ["Ceylon 24 Stiver 1808"],
                                    "viewItemURL": ["http://example/find/1"],
                                    "sellingStatus": [{"currentPrice": [{"__value__": "85.00"}]}]
                                }]
                            }]
                        }]
                    }
            return DummyR()
        # monkeypatch
        finding_api._request_with_retries = fake_request
        os.environ["EBAY_APP_ID"] = "DUMMY-APP-ID"
        items = finding_api.search_ebay("Ceylon 24 Stiver 1808", entries=2)
        ok = (len(items) == 1 and "title" in items[0] and "price" in items[0] and "link" in items[0])
        record("API7", "Finding API – Happy Path", "Mocked Finding API returns items",
               "List with title/price/link", ok, details=str(items))
    except Exception as e:
        record("API7", "Finding API – Happy Path", "Exception", "List result", False, details=str(e))

    # API8: Browse error handling
    try:
        ebay_pricing.browse_search = fake_browse_search_factory("error")
        thrown = False
        try:
            ebay_pricing.estimate_live_price("Ceylon", "24 Stiver", 1808, None)
        except Exception:
            thrown = True
        record("API8", "OAuth / Network Error Handling",
               "Simulated failure from browse_search",
               "Exception is caught by caller (no app crash)", thrown, details="raised exception")
    except Exception as e:
        record("API8", "OAuth / Network Error Handling", "Exception", "Raised", False, details=str(e))

    # Write markdown table
    lines = []
    lines.append(f"# API Test Results — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("| ID | Test Case | Description | Expected Outcome | Testing Outcome | Status |")
    lines.append("|----|-----------|-------------|------------------|-----------------|--------|")
    for r in RESULTS:
        lines.append(
            f"| {r['id']} | {r['case']} | {r['desc']} | {r['expected']} | {r['outcome']} | {r['status']} |"
        )
    out_md = "\n".join(lines)
    with open("API_Test_Results.md", "w", encoding="utf-8") as f:
        f.write(out_md)
    print(out_md)
    print("\nSaved: API_Test_Results.md")

if __name__ == "__main__":
    run()
