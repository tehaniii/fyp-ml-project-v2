import os
import time
import requests
from typing import List, Dict, Any

EBAY_APP_ID = os.getenv("EBAY_APP_ID")  # e.g. TehaniGu-TehanCoi-PRD-...
EBAY_SEARCH_URL = "https://svcs.ebay.com/services/search/FindingService/v1"

def _check_env():
    if not EBAY_APP_ID:
        raise RuntimeError("Missing EBAY_APP_ID env var for Finding API.")

def _request_with_retries(url: str, params: Dict[str, Any], retries: int = 2, backoff: float = 0.7) -> requests.Response:
    last_exc = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code >= 500:
                time.sleep(backoff * (i + 1))
                last_exc = Exception(f"HTTP {r.status_code} body={r.text[:500]}")
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (i + 1))
    raise last_exc

def search_ebay(query: str, entries: int = 5) -> List[Dict[str, Any]]:
    _check_env()
    params = {
        "OPERATION-NAME": "findItemsByKeywords",
        "SERVICE-VERSION": "1.13.0",
        "SECURITY-APPNAME": EBAY_APP_ID,
        "RESPONSE-DATA-FORMAT": "JSON",
        "GLOBAL-ID": "EBAY-US",
        "categoryId": "11116",  # Coins & Paper Money
        "keywords": query,
        "paginationInput.entriesPerPage": entries
    }
    r = _request_with_retries(EBAY_SEARCH_URL, params)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON Finding API response: {r.text[:500]}")
    resp = data.get("findItemsByKeywordsResponse", [{}])[0]
    ack = resp.get("ack", [""])[0].lower()
    if ack != "success":
        errors = resp.get("errorMessage", [{}])[0].get("error", [])
        raise RuntimeError(f"Finding API not success. ack={ack}, errors={errors}")
    items = resp.get("searchResult", [{}])[0].get("item", []) or []
    results = []
    for item in items:
        try:
            title = item["title"][0]
            price = float(item["sellingStatus"][0]["currentPrice"][0]["__value__"])
            link = item["viewItemURL"][0]
            results.append({"title": title, "price": price, "link": link})
        except Exception:
            continue
    return results
