# src/numista_api.py
from __future__ import annotations
import os
import requests

BASE_URL = "https://api.numista.com/v3"

def _headers() -> dict:
    """
    Build headers for Numista v3.
    Reads the API key from the environment (NUMISTA_API_KEY).
    """
    api_key = os.getenv("NUMISTA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "NUMISTA_API_KEY is not set. Put it in your .env as:\n"
            "NUMISTA_API_KEY=your_real_key_here"
        )
    return {
        "Numista-API-Key": api_key,
        "User-Agent": "SriLankanCoinClassifier/1.0",
        "Accept": "application/json",
        # Optional: language for some text fields -> "en", "si", "ta" etc.
        # "Accept-Language": "en",
    }

def get_coin_details(numista_type_id: int | str) -> dict | None:
    """
    Fetch details for a coin type: /types/{id}
    Returns a JSON dict on success, or None on failure (status logged).
    """
    url = f"{BASE_URL}/types/{numista_type_id}"
    try:
        r = requests.get(url, headers=_headers(), timeout=15)
        if r.status_code == 401:
            # Make this super explicit; your app swallows errors and hides this otherwise.
            raise RuntimeError(
                "Numista returned 401 Unauthorized — "
                "check NUMISTA_API_KEY in your .env (typos/extra spaces), "
                "and confirm your key has v3 access."
            )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Numista] get_coin_details({numista_type_id}) failed: {e}")
        return None

# (Optional helpers if you ever need them later)

def get_coin_issues(numista_type_id: int | str) -> list[dict]:
    """
    Return 'issues' list from /types/{id} payload, or [] if unavailable.
    """
    js = get_coin_details(numista_type_id)
    return (js or {}).get("issues", []) or []

def get_issue_market_snapshot(numista_type_id: int | str) -> dict:
    """
    If Numista provides aggregated market info under issues, return first hit.
    This is just a convenience; safe to ignore if not present.
    """
    for issue in get_coin_issues(numista_type_id):
        market = issue.get("market")
        if market:
            return market
    return {}
