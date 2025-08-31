# src/ebay_module/browse_api.py
import os, time, requests
from typing import List, Dict, Any

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
MARKETPLACE = os.getenv("EBAY_MARKETPLACE", "EBAY_US")  # EBAY_US, EBAY_GB, ...

class EbayAuthError(Exception): pass

def _require_env(name: str, value: str | None):
    if not value:
        raise EbayAuthError(
            f"Missing env var {name}. Set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET (and optionally EBAY_MARKETPLACE)."
        )

_TOKEN = {"value": None, "exp": 0.0}  # in-memory token cache

def get_app_token() -> str:
    _require_env("EBAY_CLIENT_ID", CLIENT_ID)
    _require_env("EBAY_CLIENT_SECRET", CLIENT_SECRET)

    now = time.time()
    if _TOKEN["value"] and now < _TOKEN["exp"] - 60:
        return _TOKEN["value"]

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    resp = requests.post(EBAY_OAUTH_URL, headers=headers, data=data,
                         auth=(CLIENT_ID, CLIENT_SECRET), timeout=20)
    resp.raise_for_status()
    j = resp.json()
    _TOKEN["value"] = j["access_token"]
    _TOKEN["exp"] = now + int(j.get("expires_in", 7200))
    return _TOKEN["value"]

def browse_search(q: str, category_ids: str = "256", limit: int = 50, sort: str | None = None) -> List[Dict[str, Any]]:
    token = get_app_token()
    params = {"q": q, "category_ids": category_ids, "limit": str(limit)}
    if sort: params["sort"] = sort
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE}
    resp = requests.get(EBAY_SEARCH_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("itemSummaries", []) or []
