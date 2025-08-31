import requests
from bs4 import BeautifulSoup

def scrape_ebay_listing_from_numista(coin_id):
    """
    Scrapes current sale offers from Numista's 'current_sales.php' page for a given coin ID.
    Extracts eBay listings: price, title, and link.
    """
    url = f"https://en.numista.com/catalogue/current_sales.php?id={coin_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CoinPriceBot/1.0)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error connecting to Numista: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    listings = soup.select("#current_listings > div")

    if not listings:
        print("⚠️ No listings found in parsed HTML.")
        return None

    for listing in listings:
        vendor = listing.select_one(".vendor_label")
        if vendor and "ebay" in vendor.text.strip().lower():
            title_elem = listing.select_one(".item_title a")
            price_elem = listing.select_one(".item_price span")
            link_elem = listing.select_one(".item_price a")

            if title_elem and price_elem and link_elem:
                return {
                    "source": "eBay (via Numista)",
                    "title": title_elem.text.strip(),
                    "price": price_elem.text.strip(),
                    "link": link_elem["href"]
                }

    print("❌ No eBay listings found.")
    return None
