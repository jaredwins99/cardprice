"""eBay sold-listings scraper for Pokemon cards.

Scrapes completed/sold listings from eBay's search results page,
extracting price, date, and listing metadata for each sale.
"""

import logging
import random
import re
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlencode

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# eBay Pokemon TCG category
POKEMON_CATEGORY = "183454"

# Base search URL: completed + sold, sorted by most recent
SEARCH_URL = "https://www.ebay.com/sch/i.html"

DEFAULT_PARAMS = {
    "_sacat": POKEMON_CATEGORY,
    "LH_Complete": "1",
    "LH_Sold": "1",
    "_sop": "13",       # sort: newly listed (most recent sold)
    "_ipg": "60",       # items per page
}

# Rotating user agents to reduce blocking risk
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Delay between page fetches (seconds)
MIN_DELAY = 2.0
MAX_DELAY = 3.0


def _get_headers() -> dict:
    """Return request headers with a randomly chosen User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.ebay.com/",
        "DNT": "1",
    }


def _build_search_url(query: str, page: int = 1) -> str:
    """Build a full eBay search URL for sold Pokemon card listings."""
    params = {**DEFAULT_PARAMS, "_nkw": query}
    if page > 1:
        params["_pgn"] = str(page)
    return f"{SEARCH_URL}?{urlencode(params)}"


def _parse_price(price_text: str) -> float | None:
    """Extract a numeric price from eBay price strings like '$12.34'."""
    if not price_text:
        return None
    match = re.search(r"\$?([\d,]+\.?\d*)", price_text.replace(",", ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_sold_date(date_text: str) -> datetime | None:
    """Parse eBay sold-date strings like 'Sold  Feb 25, 2026'."""
    if not date_text:
        return None
    # Strip the "Sold" prefix and extra whitespace
    cleaned = re.sub(r"^Sold\s+", "", date_text.strip(), flags=re.IGNORECASE)
    for fmt in ("%b %d, %Y", "%b %d %Y", "%d %b %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.debug("Could not parse sold date: %r", date_text)
    return None


def _detect_sale_type(listing_el) -> str:
    """Determine if listing was auction, buy-it-now, or best-offer."""
    text_lower = listing_el.get_text(separator=" ").lower()
    if "best offer accepted" in text_lower or "best offer" in text_lower:
        return "best_offer"
    if "bid" in text_lower or "bids" in text_lower:
        return "auction"
    return "buy_it_now"


def _extract_item_id(url: str) -> str | None:
    """Extract eBay item ID from a listing URL."""
    match = re.search(r"/itm/(\d+)", url)
    return match.group(1) if match else None


def _parse_listing(item_el) -> dict | None:
    """Parse a single search-result item element into a dict.

    Returns None if critical fields (title, price) cannot be extracted.
    """
    result = {}

    # Title — typically in an <a> or <span> with specific classes
    title_el = (
        item_el.select_one("div.s-item__title span[role='heading']")
        or item_el.select_one("div.s-item__title span")
        or item_el.select_one("div.s-item__title")
        or item_el.select_one("h3.s-item__title")
    )
    if not title_el:
        return None
    title_text = title_el.get_text(strip=True)
    # Skip "Shop on eBay" placeholder rows
    if title_text.lower().startswith("shop on ebay"):
        return None
    result["title"] = title_text

    # Listing URL and item ID
    link_el = item_el.select_one("a.s-item__link")
    if link_el and link_el.get("href"):
        result["listing_url"] = link_el["href"].split("?")[0]
        result["item_id"] = _extract_item_id(link_el["href"])
    else:
        result["listing_url"] = None
        result["item_id"] = None

    # Sold price
    price_el = (
        item_el.select_one("span.s-item__price")
        or item_el.select_one("span.POSITIVE")
    )
    if not price_el:
        return None
    price_text = price_el.get_text(strip=True)
    # Handle price ranges (e.g. "$5.00 to $10.00") — take the first price
    result["sold_price"] = _parse_price(price_text)
    if result["sold_price"] is None:
        return None

    # Shipping price
    shipping_el = (
        item_el.select_one("span.s-item__shipping")
        or item_el.select_one("span.s-item__freeXDays")
    )
    if shipping_el:
        ship_text = shipping_el.get_text(strip=True).lower()
        if "free" in ship_text:
            result["shipping_price"] = 0.0
        else:
            result["shipping_price"] = _parse_price(ship_text)
    else:
        result["shipping_price"] = None

    # Sold date
    sold_el = (
        item_el.select_one("span.s-item__ended-date")
        or item_el.select_one("span.s-item__endedDate")
        or item_el.select_one("span.POSITIVE")
    )
    # Sometimes the date is in a separate span near the end
    date_spans = item_el.select("span.POSITIVE")
    sold_date = None
    for span in date_spans:
        txt = span.get_text(strip=True)
        if "sold" in txt.lower():
            sold_date = _parse_sold_date(txt)
            break
    if sold_date is None:
        # Fallback: look for any element with "Sold" text
        for span in item_el.find_all("span"):
            txt = span.get_text(strip=True)
            if txt.lower().startswith("sold"):
                sold_date = _parse_sold_date(txt)
                if sold_date:
                    break
    result["sold_date"] = sold_date

    # Image URL
    img_el = item_el.select_one("img.s-item__image-img") or item_el.select_one("img")
    if img_el:
        result["image_url"] = img_el.get("src") or img_el.get("data-src")
    else:
        result["image_url"] = None

    # Sale type
    result["sale_type"] = _detect_sale_type(item_el)

    return result


def scrape_sold_listings(query: str, max_pages: int = 3) -> list[dict]:
    """Scrape eBay sold listings for a given search query.

    Args:
        query: Search string (e.g. "Charizard Base Set 4/102").
        max_pages: Maximum number of result pages to scrape (1-10).

    Returns:
        List of dicts, each containing:
            title, sold_price, sold_date, item_id, shipping_price,
            image_url, sale_type, listing_url
    """
    max_pages = min(max(max_pages, 1), 10)
    all_listings: list[dict] = []
    session = requests.Session()

    for page in range(1, max_pages + 1):
        url = _build_search_url(query, page)
        logger.info("Scraping eBay page %d/%d: %s", page, max_pages, url)

        try:
            resp = session.get(url, headers=_get_headers(), timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch page %d: %s", page, e)
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Each result is an <li> with class s-item
        items = soup.select("li.s-item")
        if not items:
            logger.info("No items found on page %d, stopping.", page)
            break

        page_count = 0
        for item_el in items:
            parsed = _parse_listing(item_el)
            if parsed:
                all_listings.append(parsed)
                page_count += 1

        logger.info("Parsed %d listings from page %d", page_count, page)

        # Check if there's a next page
        next_btn = soup.select_one("a.pagination__next")
        if not next_btn and page < max_pages:
            logger.info("No next-page button found, stopping after page %d.", page)
            break

        # Polite delay between pages
        if page < max_pages:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            logger.debug("Sleeping %.1fs before next page", delay)
            time.sleep(delay)

    logger.info("Total eBay listings scraped: %d", len(all_listings))
    return all_listings
