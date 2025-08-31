# src/ebay_module/test_ebay_api.py
import os

print("== Legacy Finding API quick test ==")
try:
    # ⬇⬇ change to relative import
    from .ebay_api import search_ebay
    res = search_ebay("One Stiver Ceylon 1815")
    if not res:
        print("No results from Finding API.")
    else:
        for r in res:
            print(f"🪙 {r['title']} - {r['price']}  -> {r['link']}")
except Exception as e:
    print("Finding API error:", e)

print("\n== Browse API price estimation test ==")
try:
    # ⬇⬇ change to relative import
    from .ebay_pricing import estimate_live_price
    est = estimate_live_price(country="Ceylon", denomination="One Stiver", year=1815)
    print("Query:", est["query"])
    print("Raw items:", est["n_raw"], "Used:", est["n_used"])
    print("Stats:", est["stats"])
    for s in est["sample"]:
        print(f" - {s['title']}  |  {s['price']} {s['currency']}  |  {s['url']}")
except Exception as e:
    print("Browse API error:", e)
