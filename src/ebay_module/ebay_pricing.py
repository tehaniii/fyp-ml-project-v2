# src/ebay_module/ebay_pricing.py
from __future__ import annotations

from typing import Dict, Any, Iterable, List, Tuple, Optional
from statistics import median
from .browse_api import browse_search

# ---- eBay title synonyms & label-specific forbids ----
LABEL_SYNONYMS = {
    # 0 — George_IV_Rixdollar_1821
    "George_IV_Rixdollar_1821": [
        "Ceylon 1 Rixdollar 1821 coin",
        "Ceylon Rix Dollar 1821 silver",
        "1821 Ceylon Rixdollar elephant coin",
        "George IV Rixdollar Ceylon 1821",
        "1 Rix Dollar Ceylon 1821 coin",
    ],

    # 1 — NissankaMalla_CopperMassa (seller titles vary: Sahassa/Sahassamalla, Kandy/Polonnaruwa/Sinhalese States)
    "NissankaMalla_CopperMassa": [
        "Ceylon Massa copper coin",
        "Sri Lanka Massa copper coin",
        "Sahassa Malla 1 Massa copper",
        "Polonnaruwa 1 Massa copper coin",
        "Kingdom of Kandy 1 Massa copper",
        "Sinhalese States 1 Massa copper",
        "Ancient Ceylon 1 Massa",
    ],

    # 2 — One_Stiver_1815_Replica (explicitly replicas/fantasy)
    "One_Stiver_1815_Replica": [
        "Ceylon 1815 One Stiver replica",
        "Ceylon 1815 1 Stiver copy coin",
        "Ceylon One Stiver reproduction coin",
        "1815 One Stiver fantasy token",
        "George III Ceylon One Stiver replica",
    ],

    # 3 — Rupee1 - New (post-1972 Sri Lanka)
    "Rupee1 - New": [
        "Sri Lanka 1 Rupee coin",
        "Sri Lankan One Rupee coin",
        "Sri Lanka Rs 1 coin",
    ],

    # 4 — Rupee1 - Old (Ceylon period)
    "Rupee1 - Old": [
        "Ceylon 1 Rupee coin",
        "Ceylon One Rupee coin",
        "Ceylon Rupee coin",
    ],

    # 5 — Rupee10 - New
    "Rupee10 - New": [
        "Sri Lanka 10 Rupees coin",
        "Sri Lankan Rs 10 coin",
        "Sri Lanka Ten Rupees coin",
    ],

    # 6 — Rupee10- Old (Ceylon period coins; avoid banknotes)
    "Rupee10- Old": [
        "Ceylon 10 Rupees coin",
        "Ceylon Ten Rupees coin",
    ],

    # 7 — Rupee2 - New
    "Rupee2 - New": [
        "Sri Lanka 2 Rupees coin",
        "Sri Lankan Rs 2 coin",
        "Sri Lanka Two Rupees coin",
    ],

    # 8 — Rupee2 - Old
    "Rupee2 - Old": [
        "Ceylon 2 Rupees coin",
        "Ceylon Two Rupees coin",
    ],

    # 9 — Rupee5 - New
    "Rupee5 - New": [
        "Sri Lanka 5 Rupees coin",
        "Sri Lankan Rs 5 coin",
        "Sri Lanka Five Rupees coin",
    ],

    # 10 — Rupee5 - Old
    "Rupee5 - Old": [
        "Ceylon 5 Rupees coin",
        "Ceylon Five Rupees coin",
    ],

    # 11 — Wekanda_Mills_Token_1881 (appears as Wekande/Wekanda; “1843” date shows on token)
    "Wekanda_Mills_Token_1881": [
        "Wekande Mills coffee token 19 cents 1881",
        "Wekanda Mills 19 cents token",
        "\"1843\" 19 cents Wekande Mills token",
        "Ceylon plantation token Wekande Mills",
    ],
    "Twenty_Four_Stiver_1808": [
    "Ceylon 24 Stiver 1808 silver coin",
    "24 Stiver 1808 Ceylon coin",
    "1808 Ceylon 24 Stiver",
    "Ceylon half Rixdollar 1808",
    "24 Stiver Elephants Ceylon 1808",
    "48 Stivers 1808 Ceylon",
    "Rixdollar 1808 Ceylon coin",
], 
"Dutch_One_Stuiver_Wreath Series_1712": [
    "Ceylon Stuiver Wreath Series 1712",
    "Dutch One Stiver 1712 Ceylon coin",
    "Stuiver 1712 Ceylon coin Wreath",
],
"Portuguese_Tanga_1631": [
    "Portuguese Tanga 1631 coin",
        "Tanga Ceylon 1631",
        "Ceylon Portuguese Tanga 1631",
        "Ceylon 1631 Tanga coin",
],
   "1998_Sri Lanka_5000Rupee_50th_Independence Anniversary": [
        "Sri Lanka 5000 Rupees 1998 50th Independence",
        "1998 Sri Lanka Rs 5000 50th Independence",
        "Sri Lanka 50th Independence Anniversary 5000 Rupees",
        "Sri Lanka 1998 5000 Rupees commemorative coin",
        "Sri Lanka 1998 5000 Rupees proof coin",
    ],
      "Darley_Butler_Token_1860": [
        "Darley Butler token 1860 Ceylon",
        "Darley & Butler token 1860 Ceylon",
        "Darley Butler & Co token Ceylon 1860",
        "Ceylon Darley Butler trading token 1860",
        "Ceylon merchant token Darley Butler 1860",
    ],
    "Dutch_One_Stuiver_Wreath Series_1712": [
        "Ceylon Stuiver Wreath Series 1712",
        "Dutch One Stuiver 1712 Ceylon coin",
        "Dutch One Stiver 1712 Ceylon coin",
        "Stuiver 1712 Ceylon wreath",
        "Ceylon 1 Stuiver 1712 Dutch",
    ],
      # 4 — Gold Kahavanu (ancient)
    "Gold_Kahavanu": [
        "Ceylon gold Kahavanu coin",
        "Sri Lanka gold Kahavanu",
        "Ancient Ceylon gold Kahavanu",
        "Medieval Sri Lanka Kahavanu gold",
        "Sinhala gold Kahavanu coin",
    ],
    
}

# Optional: extra “forbid” filters per label to reduce noise
LABEL_FORBIDS = {
    # keep genuine Rixdollar coins; avoid mentions that scream replicas/jewelry
    "George_IV_Rixdollar_1821": ["replica", "copy", "pendant", "jewelry", "note", "banknote"],

    # for ancient massās, avoid lilavati/other rulers only if you want Nissanka focus (otherwise omit)
    "NissankaMalla_CopperMassa": ["pendant", "jewelry", "replica", "copy"],

    # require replicas for the replica class (handled in query terms, but forbid genuine cues)
    "One_Stiver_1815_Replica": ["genuine", "original", "authentic", "banknote", "note"],

    # modern coins: avoid banknotes and sets
    "Rupee1 - New": ["banknote", "note", "paper", "set of notes"],
    "Rupee1 - Old": ["banknote", "note"],
    "Rupee10 - New": ["banknote", "note"],
    "Rupee10- Old": ["banknote", "note"],
    "Rupee2 - New": ["banknote", "note"],
    "Rupee2 - Old": ["banknote", "note"],
    "Rupee5 - New": ["banknote", "note"],
    "Rupee5 - Old": ["banknote", "note"],

    "Wekanda_Mills_Token_1881": ["replica", "copy", "pendant", "note", "banknote"],
}

# --------------------------- helpers ---------------------------

def build_query(country: Optional[str],
                denomination: Optional[str],
                year: Optional[int],
                km: Optional[str]) -> str:
    parts: List[str] = []
    if denomination:
        parts.append(str(denomination))
    if country:
        parts.append(str(country))
    if year:
        parts.append(str(year))
    if km:
        # KM catalog ref helps precision when sellers include it
        parts.append(str(km))
    return " ".join(p for p in parts if p).strip()


def _year_in(title: str, target_year: Optional[int]) -> bool:
    if not target_year:
        return True
    t = (title or "").lower()
    # exact match
    if str(target_year) in t:
        return True
    # tolerate ±1 (listings sometimes include adjacent dates in titles)
    for y in (target_year - 1, target_year + 1):
        if y > 0 and str(y) in t:
            return True
    return False


def _score_title(title: str, must_terms: Iterable[str]) -> int:
    t = (title or "").lower()
    return sum(1 for tok in must_terms if tok and tok.lower() in t)


def _trimmed_stats(values: List[float]) -> Dict[str, Any]:
    """
    Robust stats: drop the lowest & highest observation if we have >=5,
    then compute low / high / median. Works fine for small N.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "low": None, "high": None, "median": None}
    vals = sorted(values)
    if n >= 5:
        vals = vals[1:-1]  # trim one from each end
    return {
        "n": len(vals),
        "low": vals[0],
        "high": vals[-1],
        "median": float(median(vals)),
    }


# --------------------- filtering & aggregation ---------------------

def _filter_items(items: List[Dict[str, Any]],
                  must_year: Optional[int],
                  must_terms: Iterable[str],
                  forbid_terms: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Keep items with a valid price, no forbidden words,
    at least 2 of the must_terms in the title, and a matching year (±1 tolerated).
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        title = it.get("title", "") or ""

        # price block
        price_obj = it.get("price", {}) or {}
        try:
            price = float(price_obj.get("value"))
        except (TypeError, ValueError):
            continue

        # forbid junk / replicas / jewelry mounts, etc.
        title_l = title.lower()
        if any(tok.lower() in title_l for tok in forbid_terms):
            continue

        # soft scoring on title terms (at least 2 should appear)
        score = _score_title(title, must_terms)
        if score < 2:
            continue

        # year check
        if not _year_in(title, must_year):
            continue

        out.append(it)
    return out


# --------------------------- public API ---------------------------

def estimate_live_price(country: Optional[str],
                        denomination: Optional[str],
                        year: Optional[int],
                        km: Optional[str] = None,
                        marketplace: str = "EBAY_US",
                        limit: int = 100) -> Dict[str, Any]:
    """
    Search eBay Browse API for active listings that match a coin and compute robust stats.

    Returns:
        {
          "query": <primary query string>,
          "stats": {"n": int, "low": float|None, "high": float|None, "median": float|None},
          "n_raw": int,
          "n_used": int,
          "sample": [{"title","price","currency","url"}, ... up to 8]
        }
    """
    # Build primary + variant queries to improve recall
    q_primary = build_query(country, denomination, year, km)
    queries: List[str] = [q_primary]

    if km:
        queries.append(build_query(country, denomination, None, km))

    if denomination:
        d = str(denomination)
        variants = {
            d,
            d.replace("-", " "),
            d.replace(" ", ""),
            d.replace("One", "1"),  # e.g., "One Stiver" vs "1 Stiver"
        }
        for v in variants:
            queries.append(build_query(country, v, year, km))

    # de-duplicate while preserving order
    queries = list(dict.fromkeys(queries))

    all_items: List[Dict[str, Any]] = []
    seen_ids = set()

    for q in queries:
        # Category 256 = Coins: World (helps keep results relevant)
        items = browse_search(q=q, category_ids="256", limit=limit, sort=None)
        for it in items:
            iid = it.get("itemId") or it.get("itemWebUrl")
            if iid and iid in seen_ids:
                continue
            seen_ids.add(iid)
            all_items.append(it)

    must_terms = [denomination, country]
    forbid_terms = ["replica", "copy", "counterfeit", "token", "pendant", "necklace", "magnet"]

    filt = _filter_items(
        all_items,
        must_year=year,
        must_terms=must_terms,
        forbid_terms=forbid_terms,
    )

    prices: List[float] = []
    sample: List[Dict[str, Any]] = []
    for it in filt:
        p = float(it["price"]["value"])
        cur = it["price"]["currency"]
        title = it.get("title", "")
        url = it.get("itemWebUrl")
        prices.append(p)
        if len(sample) < 8:
            sample.append({"title": title, "price": p, "currency": cur, "url": url})

    stats = _trimmed_stats(prices)
    return {
        "query": q_primary,
        "stats": stats,
        "n_raw": len(all_items),
        "n_used": len(prices),
        "sample": sample,
    }


# --------------------------- CLI (optional) ---------------------------

if __name__ == "__main__":
    # quick manual test:
    #   python -m src.ebay_module.ebay_pricing
    import os, json
    country = os.getenv("TEST_COUNTRY", "Ceylon")
    denom = os.getenv("TEST_DENOM", "One Stiver")
    year_str = os.getenv("TEST_YEAR", "1815")
    km = os.getenv("TEST_KM", "")
    year = int(year_str) if year_str else None
    out = estimate_live_price(country, denom, year, km or None, marketplace=os.getenv("EBAY_MARKETPLACE", "EBAY_US"))
    print(json.dumps(out, indent=2))
